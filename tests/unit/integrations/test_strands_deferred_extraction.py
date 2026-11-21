"""Fix J: entity extraction runs after the agent call, not inside it.

With ``defer_extraction=True`` the drain writes each message and starts
its extraction in the background. One session's extractions run one at a
time, in write order. A manager that owns its bridge waits for them at
close. A manager on a shared bridge leaves them to the bridge's owner.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from typing import Any

import pytest

pytest.importorskip("strands", reason="strands-agents not installed")

from strands import Agent  # noqa: E402

from neo4j_agent_memory.integrations.base import AsyncBridge  # noqa: E402
from neo4j_agent_memory.integrations.strands.session_manager import (  # noqa: E402
    Neo4jSessionManager,
)
from neo4j_agent_memory.memory.short_term import Message, MessageRole  # noqa: E402
from tests.unit.integrations.strands_fakes import FakeMemoryClient  # noqa: E402
from tests.unit.integrations.test_strands_nonblocking import ScriptedModel  # noqa: E402

_LOGGER = "neo4j_agent_memory.integrations.strands.session_manager"


class _Gate:
    """Holds every extraction of ``client`` until ``opened`` is set.

    Records the content of each extraction as it starts and as it
    finishes, and the most that ran at once.
    """

    def __init__(self, client: FakeMemoryClient) -> None:
        self.opened = threading.Event()
        self.started: list[str] = []
        self.done: list[str] = []
        self.fail: set[str] = set()
        self.most_running = 0
        self._running = 0
        client.short_term.extract_entities_from_message = self._extract

    async def _extract(self, message: Message, **kwargs: Any) -> None:
        self.started.append(message.content)
        self._running += 1
        self.most_running = max(self.most_running, self._running)
        try:
            while not self.opened.is_set():
                await asyncio.sleep(0.01)
            if message.content in self.fail:
                raise RuntimeError("extractor down")
            self.done.append(message.content)
        finally:
            self._running -= 1


async def _eventually(check: Callable[[], bool], timeout: float = 5.0) -> None:
    """Wait until ``check()`` is true, failing after ``timeout`` seconds."""
    async with asyncio.timeout(timeout):
        while not check():
            await asyncio.sleep(0.01)


def _deferring(client: FakeMemoryClient, **kwargs: Any) -> Neo4jSessionManager:
    return Neo4jSessionManager("sess-1", memory_client=client, defer_extraction=True, **kwargs)


def _agent(manager: Neo4jSessionManager, *replies: str) -> Agent:
    turns = [[{"text": reply}] for reply in replies]
    return Agent(model=ScriptedModel(*turns), session_manager=manager, callback_handler=None)


def _contents(client: FakeMemoryClient) -> list[str]:
    return [call["content"] for call in client.short_term.add_message_calls]


class TestDeferredExtraction:
    async def test_the_call_returns_before_its_extractions_finish(self) -> None:
        client = FakeMemoryClient()
        gate = _Gate(client)
        manager = _deferring(client)

        await asyncio.wait_for(_agent(manager, "A1").invoke_async("Q1"), timeout=5)
        assert _contents(client) == ["Q1", "A1"]
        assert gate.done == []

        gate.opened.set()
        await manager.aclose()
        assert gate.done == ["Q1", "A1"]

    async def test_extractions_run_one_at_a_time_in_write_order(self) -> None:
        client = FakeMemoryClient()
        gate = _Gate(client)
        manager = _deferring(client)
        agent = _agent(manager, "A1", "A2")

        await agent.invoke_async("Q1")
        await agent.invoke_async("Q2")
        await _eventually(lambda: bool(gate.started))
        assert gate.started == ["Q1"]  # the rest wait for it

        gate.opened.set()
        await manager.aclose()
        assert gate.done == ["Q1", "A1", "Q2", "A2"]
        assert gate.most_running == 1

    async def test_a_failed_extraction_is_logged_and_the_next_one_runs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING, logger=_LOGGER)
        client = FakeMemoryClient()
        gate = _Gate(client)
        gate.fail = {"Q1"}
        gate.opened.set()
        manager = _deferring(client)

        await _agent(manager, "A1").invoke_async("Q1")
        await manager.aclose()

        assert gate.done == ["A1"]
        assert "Entity extraction failed for a stored message of session 'sess-1'" in caplog.text

    async def test_each_extraction_is_bounded_by_request_timeout(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING, logger=_LOGGER)
        client = FakeMemoryClient()
        _Gate(client)  # never opens
        manager = _deferring(client, request_timeout=0.1)

        await _agent(manager, "A1").invoke_async("Q1")
        await _eventually(lambda: caplog.text.count("Entity extraction failed") == 2)
        await manager.aclose()

        assert "TimeoutError" in caplog.text

    async def test_a_late_redaction_skips_the_extraction_of_the_message_it_deleted(
        self,
    ) -> None:
        client = FakeMemoryClient()
        gate = _Gate(client)
        manager = _deferring(client)
        agent = _agent(manager, "secret 123")

        await agent.invoke_async("Q1")
        await _eventually(lambda: gate.started == ["Q1"])  # the reply's extraction waits
        manager.redact_latest_message(
            {"role": "assistant", "content": [{"text": "[REDACTED]"}]}, agent
        )
        gate.opened.set()
        await manager.aclose()

        assert _contents(client) == ["Q1", "secret 123", "[REDACTED]"]
        assert gate.done == ["Q1"]

    @pytest.mark.parametrize("backend", ["nams", "no extraction"])
    async def test_no_extraction_starts_when_the_drain_would_not_extract(
        self, backend: str
    ) -> None:
        client = FakeMemoryClient()
        manager = _deferring(client, extract_entities=backend == "nams")
        client._nams_mode = backend == "nams"  # NAMS extracts on the server
        stored = Message(role=MessageRole.USER, content="Q1")

        manager._defer_extract(stored)

        assert manager._extraction is None
        client._nams_mode = False
        await manager.aclose()


class TestClosingWithDeferredExtractions:
    @pytest.mark.parametrize("use_async", [False, True], ids=["close", "aclose"])
    async def test_a_manager_that_owns_its_bridge_waits_for_them(self, use_async: bool) -> None:
        client = FakeMemoryClient()
        gate = _Gate(client)
        manager = _deferring(client)
        await _agent(manager, "A1").invoke_async("Q1")

        threading.Timer(0.2, gate.opened.set).start()
        if use_async:
            await manager.aclose()
        else:
            await asyncio.to_thread(manager.close)

        assert gate.done == ["Q1", "A1"]

    @pytest.mark.parametrize("use_async", [False, True], ids=["close", "aclose"])
    async def test_ones_still_running_after_the_wait_are_cancelled(
        self,
        use_async: bool,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        caplog.set_level(logging.WARNING, logger=_LOGGER)
        client = FakeMemoryClient()
        gate = _Gate(client)  # never opens
        manager = _deferring(client)
        await _agent(manager, "A1").invoke_async("Q1")
        monkeypatch.setattr(manager, "_drain_timeout", lambda: 0.05)

        if use_async:
            await manager.aclose()
        else:
            await asyncio.to_thread(manager.close)

        assert gate.done == []
        assert (
            "Closing session 'sess-1' while 2 entity extraction(s) are still running" in caplog.text
        )

    async def test_a_manager_on_a_shared_bridge_does_not_wait(self) -> None:
        client = FakeMemoryClient()
        gate = _Gate(client)
        bridge = AsyncBridge()
        manager = _deferring(client, bridge=bridge)
        try:
            await _agent(manager, "A1").invoke_async("Q1")
            await asyncio.wait_for(manager.aclose(), timeout=5)
            assert gate.done == []

            gate.opened.set()  # the bridge's owner decides how long to wait
            await _eventually(lambda: gate.done == ["Q1", "A1"])
        finally:
            await asyncio.to_thread(bridge.close)
