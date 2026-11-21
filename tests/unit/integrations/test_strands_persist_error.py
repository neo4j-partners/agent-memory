"""The session manager's write failure policy, ``on_persist_error``.

Both modes keep a failed message at the head of the queue and retry it
on every later flush. Retries in the turn of the first failure do not
count. A failure in a later turn, or at close, drops the message with an
ERROR log, and later messages are written. ``"raise"`` (the default)
raises ``SessionException`` on each failure. ``"warn"`` logs instead.
Restore failures always raise.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("strands", reason="strands-agents not installed")

from strands.types.exceptions import SessionException  # noqa: E402

from neo4j_agent_memory.integrations.strands.session_manager import (  # noqa: E402
    Neo4jSessionManager,
)
from tests.unit.integrations.strands_fakes import FakeMemoryClient  # noqa: E402
from tests.unit.integrations.test_strands_nonblocking import (  # noqa: E402
    _agent,
    _buffered,
    _stored,
)

_LOGGER = "neo4j_agent_memory.integrations.strands.session_manager"
_MODES = pytest.mark.parametrize("mode", ["raise", "warn"])


def _failing_client(*, poison: str | None = None) -> tuple[FakeMemoryClient, SimpleNamespace]:
    """A fake client whose add_message raises while ``switch.down`` is True.

    With ``poison``, a message with that exact text always fails.
    """
    client = FakeMemoryClient()
    switch = SimpleNamespace(down=False)
    original = client.short_term.add_message

    async def add_message(session_id: str, role: str, content: str, **kwargs: Any) -> Any:
        if switch.down or content == poison:
            raise RuntimeError("backend down")
        return await original(session_id, role, content, **kwargs)

    client.short_term.add_message = add_message  # type: ignore[method-assign]
    return client, switch


def _records(caplog: pytest.LogCaptureFixture, level: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == _LOGGER and r.levelname == level]


async def _end_turn(manager: Neo4jSessionManager) -> None:
    """The end-of-turn flush, as Strands fires it after each agent call."""
    await manager._on_after_invocation(SimpleNamespace(agent=None))


async def _expect_failure(work: Awaitable[None], mode: str) -> None:
    """Await ``work``. ``"raise"`` must raise SessionException; ``"warn"`` must not."""
    if mode == "raise":
        with pytest.raises(SessionException, match="sess-1"):
            await work
    else:
        await work


def test_rejects_an_unknown_policy() -> None:
    with pytest.raises(ValueError, match="on_persist_error"):
        Neo4jSessionManager(
            "sess-1",
            memory_client=FakeMemoryClient(),
            on_persist_error="ignore",  # type: ignore[arg-type]
        )


@_MODES
class TestRetryBound:
    async def test_a_failed_turn_is_stored_in_order_by_the_next(
        self, mode: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        client, switch = _failing_client()
        manager = Neo4jSessionManager("sess-1", memory_client=client, on_persist_error=mode)  # type: ignore[arg-type]
        _buffered(manager, "one", "two")
        try:
            with caplog.at_level(logging.WARNING, logger=_LOGGER):
                switch.down = True
                await _expect_failure(_end_turn(manager), mode)
                assert _stored(client) == []

                switch.down = False
                await _end_turn(manager)
            assert _stored(client) == [("user", "one"), ("user", "two")]
            assert _records(caplog, "ERROR") == []
            if mode == "warn":
                (warning,) = _records(caplog, "WARNING")
                assert "sess-1" in warning.getMessage()
                assert "backend down" in warning.getMessage()
        finally:
            await manager.aclose()

    async def test_retries_in_the_failing_turn_do_not_count(
        self, mode: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        client, switch = _failing_client()
        manager = Neo4jSessionManager("sess-1", memory_client=client, on_persist_error=mode)  # type: ignore[arg-type]
        _buffered(manager, "one")
        try:
            with caplog.at_level(logging.WARNING, logger=_LOGGER):
                switch.down = True
                for _ in range(3):  # message hooks flush several times per turn
                    await _expect_failure(manager._aflush_buffer(), mode)
                await _expect_failure(_end_turn(manager), mode)

                switch.down = False
                await _end_turn(manager)
            assert _stored(client) == [("user", "one")]
            assert _records(caplog, "ERROR") == []
        finally:
            await manager.aclose()

    async def test_a_message_that_fails_again_next_turn_is_dropped(
        self, mode: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        client, _ = _failing_client(poison="secret 123")
        manager = Neo4jSessionManager("sess-1", memory_client=client, on_persist_error=mode)  # type: ignore[arg-type]
        _buffered(manager, "first", "secret 123", "after")
        try:
            with caplog.at_level(logging.WARNING, logger=_LOGGER):
                await _expect_failure(_end_turn(manager), mode)  # stores "first", then fails
                assert _stored(client) == [("user", "first")]
                assert _records(caplog, "ERROR") == []

                await _expect_failure(_end_turn(manager), mode)  # next turn: dropped
                assert _stored(client) == [("user", "first"), ("user", "after")]

                await _end_turn(manager)  # nothing is left queued
            (error,) = _records(caplog, "ERROR")
            text = error.getMessage()
            assert "sess-1" in text
            assert "user message #2" in text
            assert "backend down" in text
            assert "secret 123" not in text  # the log names the message, not its text
        finally:
            await manager.aclose()

    @pytest.mark.parametrize("use_async", [False, True], ids=["close", "aclose"])
    async def test_a_message_that_fails_again_at_close_is_dropped(
        self, mode: str, use_async: bool, caplog: pytest.LogCaptureFixture
    ) -> None:
        client, _ = _failing_client(poison="secret 123")
        manager = Neo4jSessionManager("sess-1", memory_client=client, on_persist_error=mode)  # type: ignore[arg-type]
        _buffered(manager, "secret 123", "after")
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            await _expect_failure(manager._aflush_buffer(), mode)
            close = manager.aclose() if use_async else asyncio.to_thread(manager.close)
            await _expect_failure(close, mode)
        assert _stored(client) == [("user", "after")]
        (error,) = _records(caplog, "ERROR")
        assert "user message #1" in error.getMessage()

    async def test_a_first_failure_at_close_is_not_dropped(
        self, mode: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        client, switch = _failing_client()
        manager = Neo4jSessionManager("sess-1", memory_client=client, on_persist_error=mode)  # type: ignore[arg-type]
        _buffered(manager, "last")
        switch.down = True
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            await _expect_failure(manager.aclose(), mode)
        assert _records(caplog, "ERROR") == []
        assert list(manager._ready) == [{"role": "user", "content": [{"text": "last"}]}]

    async def test_a_success_resets_the_retry(
        self, mode: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        client, switch = _failing_client()
        manager = Neo4jSessionManager("sess-1", memory_client=client, on_persist_error=mode)  # type: ignore[arg-type]
        agent = _buffered(manager, "one")
        try:
            with caplog.at_level(logging.WARNING, logger=_LOGGER):
                switch.down = True
                await _expect_failure(_end_turn(manager), mode)
                switch.down = False
                await _end_turn(manager)  # "one" stored on its retry

                manager.append_message({"role": "user", "content": [{"text": "two"}]}, agent)
                switch.down = True
                await _expect_failure(_end_turn(manager), mode)  # first failure of "two"
                switch.down = False
                await _end_turn(manager)
            assert _stored(client) == [("user", "one"), ("user", "two")]
            assert _records(caplog, "ERROR") == []
        finally:
            await manager.aclose()


class TestWarnMode:
    async def test_the_turn_completes_and_a_later_flush_writes_it(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client, switch = _failing_client()
        manager = Neo4jSessionManager("sess-1", memory_client=client, on_persist_error="warn")
        agent = _agent(manager, [{"text": "Hello!"}])
        try:
            switch.down = True
            with caplog.at_level(logging.WARNING, logger=_LOGGER):
                result = await agent.invoke_async("hi")
            assert str(result).strip() == "Hello!"
            assert _stored(client) == []
            assert _records(caplog, "WARNING")
            assert _records(caplog, "ERROR") == []  # same-turn retries do not drop
            switch.down = False
        finally:
            await manager.aclose()
        assert _stored(client) == [("user", "hi"), ("assistant", "Hello!")]

    def test_close_logs_instead_of_raising(self, caplog: pytest.LogCaptureFixture) -> None:
        client, switch = _failing_client()
        manager = Neo4jSessionManager("sess-1", memory_client=client, on_persist_error="warn")
        _buffered(manager, "last")
        switch.down = True
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            manager.close()
        assert _stored(client) == []
        assert _records(caplog, "WARNING")

    def test_restore_failures_still_raise(self) -> None:
        client = FakeMemoryClient()

        async def broken(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("backend down")

        client.short_term.get_conversation = broken  # type: ignore[method-assign]
        manager = Neo4jSessionManager("sess-1", memory_client=client, on_persist_error="warn")
        try:
            with pytest.raises(SessionException, match="initialize"):
                manager.initialize(SimpleNamespace(messages=[], agent_id="a1"))
        finally:
            manager.close()


class TestRaiseMode:
    async def test_the_default_raises_from_the_agent_call(self) -> None:
        client, switch = _failing_client()
        manager = Neo4jSessionManager("sess-1", memory_client=client)
        agent = _agent(manager, [{"text": "Hello!"}])
        switch.down = True
        try:
            with pytest.raises(Exception) as info:
                await agent.invoke_async("hi")
            # Strands may wrap a hook error raised mid-cycle in EventLoopException.
            chain = [info.value, info.value.__cause__]
            assert any(isinstance(e, SessionException) for e in chain)
        finally:
            switch.down = False
            await manager.aclose()
