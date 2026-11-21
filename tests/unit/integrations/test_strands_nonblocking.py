"""Fix A: the Strands session manager never blocks the agent's event loop.

Strands awaits MessageAddedEvent and AfterInvocationEvent callbacks on the
agent's loop. A hosted runtime can run every session on one shared loop, so
a hook that blocks on a memory write stalls every session. These tests drive
a real ``strands.Agent`` with a scripted model and a slow fake memory client.
A ticker task on the agent's loop must keep ticking while memory I/O runs.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import json
import logging
import threading
import time
from collections.abc import AsyncIterator, Awaitable
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("strands", reason="strands-agents not installed")

from strands import Agent, tool  # noqa: E402
from strands.models.model import Model  # noqa: E402
from strands.types.exceptions import SessionException  # noqa: E402

from neo4j_agent_memory.integrations.base import AsyncBridge  # noqa: E402
from neo4j_agent_memory.integrations.strands.session_manager import (  # noqa: E402
    Neo4jRetrievalConfig,
    Neo4jSessionManager,
)
from tests.unit.integrations.strands_fakes import FakeMemoryClient  # noqa: E402

_TICK = 0.01
_LOGGER = "neo4j_agent_memory.integrations.strands.session_manager"


class ScriptedModel(Model):
    """A Strands model that replays canned assistant turns."""

    def __init__(self, *turns: list[dict[str, Any]]) -> None:
        self._turns = list(turns)
        self.config: dict[str, Any] = {}

    def update_config(self, **model_config: Any) -> None:
        self.config.update(model_config)

    def get_config(self) -> dict[str, Any]:
        return self.config

    async def structured_output(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        raise NotImplementedError
        yield  # pragma: no cover - makes this an async generator

    async def stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        turn = self._turns.pop(0)
        yield {"messageStart": {"role": "assistant"}}
        for block in turn:
            if "text" in block:
                yield {"contentBlockStart": {"start": {}}}
                yield {"contentBlockDelta": {"delta": {"text": block["text"]}}}
            else:
                use = block["toolUse"]
                start = {"toolUse": {"toolUseId": use["toolUseId"], "name": use["name"]}}
                yield {"contentBlockStart": {"start": start}}
                yield {
                    "contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(use["input"])}}}
                }
            yield {"contentBlockStop": {}}
        stop = "tool_use" if any("toolUse" in b for b in turn) else "end_turn"
        yield {"messageStop": {"stopReason": stop}}


@tool
def lookup(city: str) -> str:
    """Look up the weather for a city."""
    return f"Sunny in {city}"


_TOOL_TURN = [
    {"text": "Let me check."},
    {"toolUse": {"toolUseId": "t1", "name": "lookup", "input": {"city": "Paris"}}},
]


def _slow_client(delay: float) -> FakeMemoryClient:
    """A fake client whose add_message sleeps first, like a real write."""
    client = FakeMemoryClient()
    original = client.short_term.add_message

    async def slow_add(*args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(delay)
        return await original(*args, **kwargs)

    client.short_term.add_message = slow_add  # type: ignore[method-assign]
    return client


def _stored(client: FakeMemoryClient) -> list[tuple[str, str]]:
    return [(c["role"], c["content"]) for c in client.short_term.add_message_calls]


def _agent(manager: Neo4jSessionManager, *turns: list[dict[str, Any]]) -> Agent:
    return Agent(
        model=ScriptedModel(*turns),
        tools=[lookup],
        session_manager=manager,
        callback_handler=None,
    )


async def _ticks_during(work: Awaitable[Any]) -> int:
    """Count ticks a sibling task gets on this loop while ``work`` runs."""
    ticks = 0
    done = asyncio.Event()

    async def ticker() -> None:
        nonlocal ticks
        while not done.is_set():
            await asyncio.sleep(_TICK)
            ticks += 1

    task = asyncio.create_task(ticker())
    try:
        await work
    finally:
        done.set()
        await task
    return ticks


# ---------------------------------------------------------------------------
# AsyncBridge.run_async
# ---------------------------------------------------------------------------


class TestBridgeRunAsync:
    async def test_runs_on_the_bridge_loop_and_returns_the_result(self) -> None:
        bridge = AsyncBridge()

        async def which_loop() -> int:
            return id(asyncio.get_running_loop())

        try:
            bridge_loop = await bridge.run_async(which_loop())
            assert bridge_loop != id(asyncio.get_running_loop())
            assert bridge_loop == bridge.run(which_loop())
        finally:
            bridge.close()

    async def test_caller_loop_keeps_running_while_work_runs(self) -> None:
        bridge = AsyncBridge()
        try:
            ticks = await _ticks_during(bridge.run_async(asyncio.sleep(0.3)))
        finally:
            bridge.close()
        assert ticks >= 10

    async def test_timeout_raises_and_cancels_the_bridge_task(self) -> None:
        bridge = AsyncBridge(timeout=0.05)
        cancelled = threading.Event()

        async def slow() -> None:
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        try:
            with pytest.raises(TimeoutError):
                await bridge.run_async(slow())
            assert await asyncio.to_thread(cancelled.wait, 2)
        finally:
            bridge.close()

    async def test_run_async_after_close_raises_and_closes_the_coroutine(self) -> None:
        bridge = AsyncBridge()
        bridge.close()

        async def noop() -> None:
            return None

        coro = noop()
        with pytest.raises(RuntimeError, match="AsyncBridge is closed"):
            await bridge.run_async(coro)
        assert coro.cr_frame is None
        assert bridge._thread is None

    @pytest.mark.parametrize("use_async", [False, True], ids=["run", "run_async"])
    async def test_dead_thread_is_fatal(self, use_async: bool) -> None:
        """A loop thread that died without close() is not restarted."""
        bridge = AsyncBridge()

        async def answer() -> int:
            return 42

        try:
            bridge.run(answer())
            dead = bridge._thread
            assert dead is not None and bridge._loop is not None
            bridge._loop.call_soon_threadsafe(bridge._loop.stop)
            dead.join(timeout=2)
            assert not dead.is_alive()

            for _ in range(2):  # it stays broken
                coro = answer()
                with pytest.raises(RuntimeError, match="stopped unexpectedly"):
                    if use_async:
                        await bridge.run_async(coro)
                    else:
                        bridge.run(coro)
                assert coro.cr_frame is None  # closed, so no "never awaited"
            assert bridge._thread is dead
        finally:
            bridge.close()


def _destroyed_pending(caplog: pytest.LogCaptureFixture) -> list[str]:
    """asyncio's "Task was destroyed but it is pending!" reports, after a GC pass."""
    gc.collect()
    return [r.getMessage() for r in caplog.records if "Task was destroyed" in r.getMessage()]


class TestBridgeShutdown:
    def test_close_lets_pending_tasks_unwind(self, caplog: pytest.LogCaptureFixture) -> None:
        bridge = AsyncBridge(timeout=0.05)
        cleaned = threading.Event()

        async def stuck() -> None:
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)  # cleanup that itself awaits
                cleaned.set()

        with pytest.raises(TimeoutError):
            bridge.run(stuck())  # a run() timeout leaves the task pending
        loop = bridge._loop
        assert loop is not None

        with caplog.at_level(logging.ERROR, logger="asyncio"):
            bridge.close()
            assert cleaned.is_set()
            assert loop.is_closed()
            assert _destroyed_pending(caplog) == []

    def test_run_in_flight_at_close_raises_runtime_error(self) -> None:
        bridge = AsyncBridge(timeout=5)
        outcome: list[BaseException] = []

        def call() -> None:
            try:
                bridge.run(asyncio.sleep(5))
            except BaseException as e:  # noqa: BLE001 - recorded for the assertion
                outcome.append(e)

        caller = threading.Thread(target=call)
        caller.start()
        time.sleep(0.1)
        bridge.close()
        caller.join(timeout=2)

        assert not caller.is_alive()
        assert len(outcome) == 1
        assert isinstance(outcome[0], RuntimeError)
        assert "work in flight" in str(outcome[0])

    @pytest.mark.parametrize("shield", [False, True], ids=["plain", "shielded"])
    async def test_bridge_side_cancel_reaches_the_caller_as_runtime_error(
        self, shield: bool
    ) -> None:
        bridge = AsyncBridge(timeout=5)
        future = bridge.submit(asyncio.sleep(5))
        waiter = asyncio.create_task(bridge.wait(future, shield=shield))
        await asyncio.sleep(0.1)
        try:
            await asyncio.to_thread(bridge.close)
            with pytest.raises(RuntimeError, match="work in flight"):
                await waiter
        finally:
            await asyncio.to_thread(bridge.close)

    async def test_cancelling_the_caller_still_raises_cancelled_error(self) -> None:
        bridge = AsyncBridge(timeout=5)
        try:
            waiter = asyncio.create_task(bridge.run_async(asyncio.sleep(5)))
            await asyncio.sleep(0.1)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
        finally:
            await asyncio.to_thread(bridge.close)


# ---------------------------------------------------------------------------
# Hooks run off the agent's loop
# ---------------------------------------------------------------------------


class TestHooksDoNotBlockTheLoop:
    async def test_agent_loop_keeps_running_during_memory_writes(self) -> None:
        """The core regression: each write sleeps 0.3 s on the bridge loop."""
        client = _slow_client(0.3)
        manager = Neo4jSessionManager("sess-1", memory_client=client)
        agent = _agent(manager, [{"text": "Hello!"}])
        try:
            ticks = await _ticks_during(agent.invoke_async("hi"))
            assert _stored(client) == [("user", "hi"), ("assistant", "Hello!")]
            # Two writes take 0.6 s. A blocked loop would barely tick.
            assert ticks >= 30
        finally:
            await manager.aclose()

    async def test_retrieval_does_not_block_the_loop(self) -> None:
        client = FakeMemoryClient()
        client.long_term.entities = [
            SimpleNamespace(
                display_name="Acme", type="ORGANIZATION", full_type="ORGANIZATION", description=None
            )
        ]
        original = client.long_term.search_entities

        async def slow_search(*args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(0.3)
            return await original(*args, **kwargs)

        client.long_term.search_entities = slow_search  # type: ignore[method-assign]
        manager = Neo4jSessionManager(
            "sess-1",
            memory_client=client,
            retrieval_config=Neo4jRetrievalConfig(include_preferences=False),
        )
        agent = _agent(manager, [{"text": "Acme is a customer."}])
        try:
            ticks = await _ticks_during(agent.invoke_async("tell me about Acme"))
            assert ticks >= 15
            # The agent saw the injected block; the stored text is the original.
            assert agent.messages[0]["content"][0]["text"].startswith("<user_context>")
            assert _stored(client)[0] == ("user", "tell me about Acme")
        finally:
            await manager.aclose()

    async def test_aclose_does_not_block_the_loop(self) -> None:
        client = _slow_client(0.3)
        manager = Neo4jSessionManager("sess-1", memory_client=client)
        agent = SimpleNamespace(messages=[], agent_id="a1")
        manager.initialize(agent)
        manager.append_message({"role": "user", "content": [{"text": "last"}]}, agent)

        ticks = await _ticks_during(manager.aclose())

        assert ticks >= 15
        assert _stored(client) == [("user", "last")]
        assert client.close_calls == 1
        assert manager._bridge._thread is None


class TestPersistenceThroughTheAgent:
    async def test_tool_turn_persists_text_messages_in_order(self) -> None:
        client = _slow_client(0.1)
        manager = Neo4jSessionManager("sess-1", memory_client=client)
        agent = _agent(manager, _TOOL_TURN, [{"text": "Sunny."}])
        try:
            ticks = await _ticks_during(agent.invoke_async("weather in Paris?"))
            # Three writes take 0.3 s. A blocked loop would barely tick.
            assert ticks >= 15
            assert _stored(client) == [
                ("user", "weather in Paris?"),
                ("assistant", "Let me check."),
                ("assistant", "Sunny."),
            ]
        finally:
            await manager.aclose()

    def test_sync_agent_call_still_persists(self) -> None:
        client = FakeMemoryClient()
        manager = Neo4jSessionManager("sess-1", memory_client=client)
        agent = _agent(manager, [{"text": "Hello!"}])
        try:
            agent("hi")
            assert _stored(client) == [("user", "hi"), ("assistant", "Hello!")]
        finally:
            manager.close()

    def test_append_message_does_no_io(self) -> None:
        client = FakeMemoryClient()
        manager = Neo4jSessionManager("sess-1", memory_client=client)
        agent = SimpleNamespace(messages=[], agent_id="a1")
        manager.initialize(agent)

        def no_io(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("append_message must not block on the bridge")

        try:
            manager._bridge.run = no_io  # type: ignore[method-assign]
            manager.append_message({"role": "user", "content": [{"text": "one"}]}, agent)
            manager.append_message({"role": "assistant", "content": [{"text": "two"}]}, agent)
        finally:
            del manager._bridge.run
            manager.close()
        assert _stored(client) == [("user", "one"), ("assistant", "two")]

    async def test_write_timeout_surfaces_session_exception_without_hanging(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = _slow_client(5)
        manager = Neo4jSessionManager("sess-1", memory_client=client, request_timeout=0.5)
        agent = _agent(manager, [{"text": "Hello!"}])
        errors: list[Exception] = []

        async def invoke() -> None:
            try:
                await agent.invoke_async("hi")
            except Exception as e:  # noqa: BLE001 - checked below
                errors.append(e)

        started = time.monotonic()
        try:
            ticks = await _ticks_during(invoke())
            assert time.monotonic() - started < 2
            # The 0.5 s wait must not block the loop.
            assert ticks >= 20
            assert len(errors) == 1
            # Strands may wrap a hook error raised mid-cycle in EventLoopException.
            chain = [errors[0], errors[0].__cause__]
            failure = next(e for e in chain if isinstance(e, SessionException))
            assert isinstance(failure.__cause__, TimeoutError)
        finally:
            with caplog.at_level(logging.ERROR, logger="asyncio"):
                with contextlib.suppress(SessionException):
                    await manager.aclose()  # the write still runs; close cancels it
                assert _destroyed_pending(caplog) == []


# ---------------------------------------------------------------------------
# Writes run to completion, one drain at a time
# ---------------------------------------------------------------------------


def _slowing_client(*delays: float) -> FakeMemoryClient:
    """A fake client whose Nth add_message sleeps ``delays[N]`` (0 after)."""
    client = FakeMemoryClient()
    original = client.short_term.add_message
    remaining = list(delays)

    async def add(*args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(remaining.pop(0) if remaining else 0)
        return await original(*args, **kwargs)

    client.short_term.add_message = add  # type: ignore[method-assign]
    return client


def _buffered(manager: Neo4jSessionManager, *texts: str) -> SimpleNamespace:
    """Initialize ``manager`` on a bare agent and buffer user ``texts``."""
    agent = SimpleNamespace(messages=[], agent_id="a1")
    manager.initialize(agent)
    for text in texts:
        manager.append_message({"role": "user", "content": [{"text": text}]}, agent)
    return agent


def _stores_then(client: FakeMemoryClient, *, delay: float = 0.0, fail: bool = False) -> None:
    """Make the first add_message store its message, then reply late or fail.

    A write whose reply is lost after it committed. Later calls are normal.
    """
    original = client.short_term.add_message
    calls = 0

    async def add(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        stored = await original(*args, **kwargs)
        if calls == 1:
            await asyncio.sleep(delay)
            if fail:
                raise RuntimeError("connection reset after commit")
        return stored

    client.short_term.add_message = add  # type: ignore[method-assign]


class TestOneStoredCopyPerMessage:
    """A write that commits but replies late or fails is stored once."""

    async def test_a_bolt_write_that_replies_late_is_stored_once(self) -> None:
        client = FakeMemoryClient()
        _stores_then(client, delay=0.4)
        manager = Neo4jSessionManager(
            "sess-1", memory_client=client, request_timeout=0.2, on_persist_error="warn"
        )
        _buffered(manager, "one")
        manager._release_pending()
        await manager._aflush_ready()  # the wait runs out at 0.2 s
        await asyncio.sleep(0.3)  # the write replies at 0.4 s
        await manager._aflush_ready()  # takes that reply instead of writing again
        await manager.aclose()
        assert _stored(client) == [("user", "one")]

    async def test_a_bolt_write_that_fails_after_it_committed_is_stored_once(self) -> None:
        client = FakeMemoryClient()
        _stores_then(client, fail=True)
        manager = Neo4jSessionManager("sess-1", memory_client=client, on_persist_error="warn")
        _buffered(manager, "one")
        manager._release_pending()
        await manager._aflush_ready()  # fails after the message was stored
        await manager._aflush_ready()  # the retry carries the same id
        await manager.aclose()
        ids = [c["message_id"] for c in client.short_term.add_message_calls]
        assert len(ids) == 2 and ids[0] == ids[1]
        (conversation,) = client.short_term.conversations.values()
        assert [m.content for m in conversation.messages] == ["one"]

    async def test_a_nams_write_that_replies_late_is_stored_once(self) -> None:
        client = FakeMemoryClient(nams_mode=True)
        wire = client.wire
        original = wire.request

        async def request(spec: Any, **kwargs: Any) -> Any:
            reply = await original(spec, **kwargs)  # the service has stored it
            if spec.bridge_method == "add_message":
                await asyncio.sleep(0.4)
            return reply

        wire.request = request  # type: ignore[method-assign]
        manager = Neo4jSessionManager(
            "sess-1",
            memory_client=client,
            request_timeout=0.2,
            on_persist_error="warn",
            extract_entities=False,
        )
        _buffered(manager, "one")
        manager._release_pending()
        await manager._aflush_ready()  # the wait runs out at 0.2 s
        await asyncio.sleep(0.3)  # the write replies at 0.4 s
        await manager._aflush_ready()  # takes that reply instead of writing again
        await manager.aclose()
        assert [c.json["content"] for c in wire.calls_for("add_message")] == ["one"]


class TestWritesRunToCompletion:
    async def test_cancelling_the_caller_mid_write_keeps_the_turn(self) -> None:
        client = _slow_client(0.3)
        manager = Neo4jSessionManager("sess-1", memory_client=client)
        agent = _agent(manager, [{"text": "A1"}])
        task = asyncio.create_task(agent.invoke_async("hi"))
        await asyncio.sleep(0.15)  # the model is done; the user write is in flight
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        await manager.aclose()

        assert _stored(client) == [("user", "hi"), ("assistant", "A1")]

    async def test_a_batch_longer_than_the_timeout_does_not_fail(self) -> None:
        client = _slow_client(0.2)  # each write fits in the timeout; all four do not
        manager = Neo4jSessionManager("sess-1", memory_client=client, request_timeout=0.5)
        _buffered(manager, "a", "b", "c", "d")
        try:
            await manager._on_after_invocation(SimpleNamespace(agent=None))
            assert _stored(client) == [("user", t) for t in "abcd"]
        finally:
            await manager.aclose()

    async def test_a_wait_that_runs_out_leaves_the_write_running(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Each write takes 0.2 s, inside request_timeout. The hook waits
        # 0.5 * (1 + 1) = 1.0 s for its one message. Seven more messages
        # arrive while its drain runs, and the drain writes them too. So the
        # drain needs 1.6 s, and the hook's wait runs out near 1.0 s.
        client = _slow_client(0.2)
        manager = Neo4jSessionManager(
            "sess-1", memory_client=client, request_timeout=0.5, extract_entities=False
        )
        agent = _buffered(manager, "one")
        texts = ["two", "three", "four", "five", "six", "seven", "eight", "nine"]
        try:
            with caplog.at_level(logging.WARNING, logger=_LOGGER):
                hook = asyncio.create_task(
                    manager._on_after_invocation(SimpleNamespace(agent=None))
                )
                await asyncio.sleep(0.05)  # the drain is writing "one"
                for text in texts:  # "two" to "eight" join the queue; "nine" stays buffered
                    manager.append_message({"role": "user", "content": [{"text": text}]}, agent)
                await hook  # no raise
            (warning,) = [r for r in caplog.records if r.name == _LOGGER]
            assert "sess-1" in warning.getMessage()
            assert "background" in warning.getMessage()
            assert 0 < len(_stored(client)) < 8
        finally:
            await manager.aclose()
        assert _stored(client) == [("user", text) for text in ["one", *texts]]

    async def test_concurrent_drains_write_in_order(self) -> None:
        client = _slowing_client(0.3)  # the first write is the slowest
        manager = Neo4jSessionManager("sess-1", memory_client=client)
        _buffered(manager, "one", "two", "three")

        await asyncio.gather(manager._aflush_buffer(), manager._aflush_buffer())

        assert _stored(client) == [("user", "one"), ("user", "two"), ("user", "three")]
        await manager.aclose()

    async def test_sync_close_racing_a_hook_writes_in_order(self) -> None:
        client = _slowing_client(0.3)
        manager = Neo4jSessionManager("sess-1", memory_client=client)
        _buffered(manager, "one", "two", "three")

        hook = asyncio.create_task(manager._aflush_ready())  # writes one, two
        await asyncio.sleep(0.05)
        await asyncio.to_thread(manager.close)  # writes the rest
        await hook

        assert _stored(client) == [("user", "one"), ("user", "two"), ("user", "three")]

    @pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
    async def test_a_failed_message_stays_queued_for_the_next_flush(self, use_async: bool) -> None:
        client = FakeMemoryClient()
        manager = Neo4jSessionManager("sess-1", memory_client=client)
        _buffered(manager, "one", "two")

        async def flush() -> None:
            if use_async:
                await manager._aflush_buffer()
            else:
                await asyncio.to_thread(manager._flush_buffer)

        client.short_term.fail_next_add = True
        try:
            with pytest.raises(SessionException):
                await flush()
            assert _stored(client) == []

            await flush()
            assert _stored(client) == [("user", "one"), ("user", "two")]
        finally:
            await manager.aclose()

    async def test_aclose_during_a_turn_keeps_order(self) -> None:
        client = _slow_client(0.3)
        manager = Neo4jSessionManager("sess-1", memory_client=client)
        agent = _agent(manager, [{"text": "A1"}])
        task = asyncio.create_task(agent.invoke_async("hi"))
        await asyncio.sleep(0.1)

        await manager.aclose()
        with contextlib.suppress(Exception):
            await task

        stored = _stored(client)
        assert stored[:1] == [("user", "hi")]
        assert stored in ([("user", "hi")], [("user", "hi"), ("assistant", "A1")])


# ---------------------------------------------------------------------------
# aopen(): initialize does no I/O
# ---------------------------------------------------------------------------


class TestAopen:
    async def test_aopen_preloads_history_so_initialize_does_no_io(self) -> None:
        client = FakeMemoryClient()
        await client.short_term.create_conversation(session_id="sess-1")
        await client.short_term.add_message("sess-1", "user", "earlier")
        await client.short_term.add_message("sess-1", "assistant", "noted")
        manager = Neo4jSessionManager("sess-1", memory_client=client)

        await manager.aopen()

        def no_io(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("initialize must not block on the bridge after aopen()")

        manager._bridge.run = no_io  # type: ignore[method-assign]
        try:
            agent = _agent(manager, [{"text": "Hello again."}])
        finally:
            del manager._bridge.run
        try:
            assert [m["content"][0]["text"] for m in agent.messages] == ["earlier", "noted"]
            await agent.invoke_async("hi")
            assert _stored(client)[-2:] == [("user", "hi"), ("assistant", "Hello again.")]
        finally:
            await manager.aclose()

    async def test_aopen_failure_raises_session_exception(self) -> None:
        client = FakeMemoryClient()

        async def boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("no database")

        client.short_term.get_conversation = boom  # type: ignore[method-assign]
        manager = Neo4jSessionManager("sess-1", memory_client=client)
        try:
            with pytest.raises(SessionException, match="initialize"):
                await manager.aopen()
        finally:
            await manager.aclose()

    async def test_aopen_after_close_raises_session_exception(self) -> None:
        manager = Neo4jSessionManager("sess-1", memory_client=FakeMemoryClient())
        await manager.aclose()
        with pytest.raises(SessionException, match="closed"):
            await manager.aopen()

    async def test_seeded_messages_are_written_by_the_first_hook(self) -> None:
        client = FakeMemoryClient()
        manager = Neo4jSessionManager("sess-1", memory_client=client)
        await manager.aopen()
        agent = Agent(
            model=ScriptedModel([{"text": "Hello!"}]),
            session_manager=manager,
            callback_handler=None,
            messages=[
                {"role": "user", "content": [{"text": "seed q"}]},
                {"role": "assistant", "content": [{"text": "seed a"}]},
            ],
        )
        try:
            assert _stored(client) == []
            await agent.invoke_async("hi")
            assert _stored(client) == [
                ("user", "seed q"),
                ("assistant", "seed a"),
                ("user", "hi"),
                ("assistant", "Hello!"),
            ]
        finally:
            await manager.aclose()


# ---------------------------------------------------------------------------
# Redaction rewrites the queue before anything reaches the backend
# ---------------------------------------------------------------------------


class RedactingModel(ScriptedModel):
    """A model whose guardrail redacts the user's message before replying."""

    async def stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        yield {"redactContent": {"redactUserContentMessage": "[REDACTED]"}}
        async for event in super().stream(*args, **kwargs):
            yield event


def _contents(client: FakeMemoryClient) -> list[str]:
    """The text of every message the backend holds now, after any deletes."""
    return [m.content for c in client.short_term.conversations.values() for m in c.messages]


def _commits_then_waits(target: Any, method: str, delay: float) -> None:
    """Make the first ``method`` call on ``target`` commit, then reply late."""
    original = getattr(target, method)
    calls = 0

    async def late(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        result = await original(*args, **kwargs)
        if calls == 1:
            await asyncio.sleep(delay)
        return result

    setattr(target, method, late)


def _uses_tools(manager: Neo4jSessionManager, agent: SimpleNamespace, *names: str) -> None:
    """Queue an assistant message that calls the tools ``names``."""
    content: list[dict[str, Any]] = [{"text": "calling tools"}]
    content += [{"toolUse": {"toolUseId": n, "name": n, "input": {}}} for n in names]
    manager.append_message({"role": "assistant", "content": content}, agent)
    manager._release_pending()


class TestToolMirroringOutlivesItsWait:
    """A wait that runs out never cuts tool-call mirroring short."""

    async def test_a_trace_that_starts_late_is_started_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = FakeMemoryClient()
        _commits_then_waits(client.reasoning, "start_trace", 0.3)
        manager = Neo4jSessionManager(
            "sess-1",
            memory_client=client,
            request_timeout=0.2,
            extract_entities=False,
            record_tool_calls=True,
        )
        agent = _buffered(manager)
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            _uses_tools(manager, agent, "lookup")
            await manager._aflush_ready()  # the wait runs out at 0.2 s
        await asyncio.sleep(0.2)  # the trace start replies at 0.3 s
        _uses_tools(manager, agent, "search")
        await manager._aflush_ready()
        await manager.aclose()

        reasoning = client.reasoning
        (trace,) = reasoning.traces
        assert [(s["trace_id"], s["action"]) for s in reasoning.steps] == [
            (trace["id"], "lookup"),
            (trace["id"], "search"),
        ]
        assert len(reasoning.tool_calls) == 2
        assert any("background" in r.getMessage() for r in caplog.records)

    async def test_a_late_step_gets_its_tool_call_and_keeps_its_place(self) -> None:
        client = FakeMemoryClient()
        _commits_then_waits(client.reasoning, "add_step", 0.3)
        manager = Neo4jSessionManager(
            "sess-1",
            memory_client=client,
            request_timeout=0.2,
            extract_entities=False,
            record_tool_calls=True,
        )
        agent = _buffered(manager)
        _uses_tools(manager, agent, "lookup", "fetch")
        await manager._aflush_ready()  # the wait runs out while "lookup" replies
        _uses_tools(manager, agent, "search")
        await manager._aflush_ready()  # waits for "lookup" and "fetch" first
        await manager.aclose()

        reasoning = client.reasoning
        assert [s["action"] for s in reasoning.steps] == ["lookup", "fetch", "search"]
        assert [c["step_id"] for c in reasoning.tool_calls] == [s["id"] for s in reasoning.steps]


class TestRedactionDuringAWrite:
    async def test_the_redacted_text_replaces_the_written_message(self) -> None:
        client = _slow_client(0.3)
        manager = Neo4jSessionManager("sess-1", memory_client=client)
        agent = _buffered(manager, "secret 123")
        manager._release_pending()
        drain = asyncio.wrap_future(manager._submit_drain())
        await asyncio.sleep(0.1)  # the write of "secret 123" is in flight

        manager.redact_latest_message({"role": "user", "content": [{"text": "[REDACTED]"}]}, agent)
        await drain

        assert _contents(client) == ["[REDACTED]"]
        assert len(client.short_term.deleted_message_ids) == 1
        await manager.aclose()

    @pytest.mark.parametrize("latest", ["no text", "dropped"])
    async def test_a_late_redaction_never_rewrites_an_earlier_message(
        self, latest: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = FakeMemoryClient()
        manager = Neo4jSessionManager("sess-1", memory_client=client, on_persist_error="warn")
        agent = _buffered(manager, "kept")
        if latest == "no text":
            tool_use = {"toolUseId": "t1", "name": "lookup", "input": {}}
            manager.append_message({"role": "assistant", "content": [{"toolUse": tool_use}]}, agent)
        else:
            original = client.short_term.add_message

            async def lost_fails(session_id: str, role: Any, content: str, **kw: Any) -> Any:
                if content == "lost":
                    raise RuntimeError("backend down")
                return await original(session_id, role, content, **kw)

            client.short_term.add_message = lost_fails  # type: ignore[method-assign]
            manager.append_message({"role": "user", "content": [{"text": "lost"}]}, agent)
        manager._release_pending()
        await manager._aflush_ready()  # writes "kept"; a write of "lost" fails
        if latest == "dropped":
            manager._turns_ended += 1  # "lost" is now from an earlier turn
            await manager._aflush_ready()  # fails again, and drops "lost"
            client.short_term.add_message = original  # type: ignore[method-assign]

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            manager.redact_latest_message(
                {"role": "user", "content": [{"text": "[REDACTED]"}]}, agent
            )

        assert _contents(client) == ["kept"]
        assert client.short_term.deleted_message_ids == []
        assert any("nothing to redact" in r.getMessage() for r in caplog.records)
        await manager.aclose()

    async def test_a_late_redaction_waits_for_the_one_the_drain_applies(self) -> None:
        client = _slow_client(0.3)
        manager = Neo4jSessionManager("sess-1", memory_client=client, extract_entities=False)
        agent = _buffered(manager, "secret 123")
        manager._release_pending()
        drain = asyncio.wrap_future(manager._submit_drain())
        await asyncio.sleep(0.1)  # the write of "secret 123" is in flight
        first = {"role": "user", "content": [{"text": "[REDACTED 1]"}]}
        manager.redact_latest_message(first, agent)  # rewrites the queue
        await asyncio.sleep(0.3)  # the drain is storing "[REDACTED 1]"

        second = {"role": "user", "content": [{"text": "[REDACTED 2]"}]}
        await asyncio.to_thread(manager.redact_latest_message, second, agent)
        await drain

        assert _contents(client) == ["[REDACTED 2]"]
        await manager.aclose()

    async def test_late_redactions_at_once_store_one_message(self) -> None:
        client = FakeMemoryClient()
        manager = Neo4jSessionManager("sess-1", memory_client=client, extract_entities=False)
        agent = _buffered(manager, "secret 123")
        await manager._aflush_buffer()
        _commits_then_waits(client.short_term, "add_message", 0.3)

        await asyncio.gather(
            *(
                asyncio.to_thread(
                    manager.redact_latest_message,
                    {"role": "user", "content": [{"text": f"[REDACTED {n}]"}]},
                    agent,
                )
                for n in (1, 2)
            )
        )

        assert len(_contents(client)) == 1
        assert _contents(client)[0].startswith("[REDACTED")
        await manager.aclose()

    async def test_a_failed_redaction_is_logged_as_an_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = _slow_client(0.3)

        async def delete_fails(*args: Any, **kwargs: Any) -> bool:
            raise RuntimeError("delete refused")

        client.short_term.delete_message = delete_fails  # type: ignore[method-assign]
        manager = Neo4jSessionManager("sess-1", memory_client=client)
        agent = _buffered(manager, "secret 123")
        manager._release_pending()
        drain = asyncio.wrap_future(manager._submit_drain())
        await asyncio.sleep(0.1)

        manager.redact_latest_message({"role": "user", "content": [{"text": "[REDACTED]"}]}, agent)
        with caplog.at_level(logging.ERROR, logger=_LOGGER):
            await drain  # the write succeeded, so the flush does not fail

        (error,) = [r for r in caplog.records if r.levelname == "ERROR"]
        assert "sess-1" in error.getMessage()
        assert "delete refused" in error.getMessage()
        assert manager._last_persisted is None  # never retarget a failed redaction
        await manager.aclose()


class TestRedactionThroughTheAgent:
    async def test_redaction_rewrites_a_queued_seed_message(self) -> None:
        client = FakeMemoryClient()
        manager = Neo4jSessionManager("sess-1", memory_client=client)
        await manager.aopen()
        agent = SimpleNamespace(
            messages=[
                {"role": "user", "content": [{"text": "seed q"}]},
                {"role": "user", "content": [{"text": "my SSN is 123-45-6789"}]},
            ],
            agent_id="a1",
        )
        manager.initialize(agent)  # queues the seed; aopen promised no I/O here
        manager.redact_latest_message({"role": "user", "content": [{"text": "[REDACTED]"}]}, agent)
        try:
            await manager._aflush_buffer()
            assert _stored(client) == [("user", "seed q"), ("user", "[REDACTED]")]
        finally:
            await manager.aclose()

    @pytest.mark.parametrize("seeded", [False, True], ids=["fresh", "seeded"])
    async def test_guardrail_redaction_stores_only_the_redacted_text(self, seeded: bool) -> None:
        client = FakeMemoryClient()
        manager = Neo4jSessionManager("sess-1", memory_client=client)
        seed = [
            {"role": "user", "content": [{"text": "seed q"}]},
            {"role": "assistant", "content": [{"text": "seed a"}]},
        ]
        if seeded:
            await manager.aopen()
        agent = Agent(
            model=RedactingModel(_TOOL_TURN, [{"text": "done"}]),
            tools=[lookup],
            session_manager=manager,
            callback_handler=None,
            messages=seed if seeded else None,
        )
        try:
            await agent.invoke_async("secret 123")
        finally:
            await manager.aclose()

        stored = _stored(client)
        assert all("secret 123" not in text for _, text in stored)
        expected_turn = [
            ("user", "[REDACTED]"),
            ("assistant", "Let me check."),
            ("assistant", "done"),
        ]
        expected_seed = [("user", "seed q"), ("assistant", "seed a")] if seeded else []
        assert stored == expected_seed + expected_turn


# ---------------------------------------------------------------------------
# Closed manager: hooks raise SessionException, not a bridge RuntimeError
# ---------------------------------------------------------------------------


class TestHooksAfterClose:
    @pytest.mark.parametrize("hook", ["_on_after_invocation", "_on_message_flush"])
    async def test_hook_after_close_raises_session_exception(self, hook: str) -> None:
        client = FakeMemoryClient()
        manager = Neo4jSessionManager("sess-1", memory_client=client)
        agent = SimpleNamespace(messages=[], agent_id="a1")
        manager.initialize(agent)
        manager.append_message({"role": "user", "content": [{"text": "x"}]}, agent)
        client.short_term.fail_next_add = True
        with pytest.raises(SessionException):
            manager.close()  # the close-time write fails

        event = SimpleNamespace(agent=agent, message={"role": "user", "content": []})
        with pytest.raises(SessionException, match="closed"):
            await getattr(manager, hook)(event)

    async def test_after_invocation_in_warn_mode_logs_what_close_left_unwritten(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = FakeMemoryClient()
        manager = Neo4jSessionManager("sess-1", memory_client=client, on_persist_error="warn")
        agent = _buffered(manager, "x")
        client.short_term.fail_next_add = True
        with caplog.at_level(logging.ERROR, logger=_LOGGER):
            manager.close()  # the close-time write fails, so "x" stays queued
            caplog.clear()
            await manager._on_after_invocation(SimpleNamespace(agent=agent))  # no raise

        (error,) = [r for r in caplog.records if r.levelname == "ERROR"]
        assert "sess-1" in error.getMessage()
        assert "1 message" in error.getMessage()
        assert _stored(client) == []

    async def test_after_invocation_with_nothing_buffered_is_a_noop_after_close(
        self,
    ) -> None:
        client = FakeMemoryClient()
        manager = Neo4jSessionManager("sess-1", memory_client=client)
        agent = SimpleNamespace(messages=[], agent_id="a1")
        manager.initialize(agent)
        manager.append_message({"role": "user", "content": [{"text": "x"}]}, agent)
        await manager.aclose()  # writes "x"; nothing is left buffered

        await manager._on_after_invocation(SimpleNamespace(agent=agent))
        assert _stored(client) == [("user", "x")]

    async def test_abandoned_stream_finalizes_cleanly_after_aclose(self) -> None:
        client = FakeMemoryClient()
        manager = Neo4jSessionManager("sess-1", memory_client=client)
        agent = _agent(manager, [{"text": "Hello!"}])
        stream = agent.stream_async("hi")
        async for _ in stream:
            if agent.messages:
                break  # the caller walks away mid-turn
        await manager.aclose()  # writes "hi"; nothing is left buffered

        # Strands fires AfterInvocationEvent from the generator's finally.
        await stream.aclose()
        assert _stored(client) == [("user", "hi")]

    async def test_a_second_aclose_waits_for_the_first(self) -> None:
        client = _slow_client(0.2)
        manager = Neo4jSessionManager("sess-1", memory_client=client)
        _buffered(manager, "one")
        first = asyncio.create_task(manager.aclose())
        await asyncio.sleep(0.05)

        await manager.aclose()
        assert _stored(client) == [("user", "one")]
        assert first.done()
        await first

    async def test_a_second_aclose_waits_but_does_not_reraise(self) -> None:
        client = _slow_client(0.2)
        client.short_term.fail_next_add = True
        manager = Neo4jSessionManager("sess-1", memory_client=client)
        _buffered(manager, "one")
        first = asyncio.create_task(manager.aclose())
        await asyncio.sleep(0.05)

        await manager.aclose()  # the first caller owns the error
        assert first.done()
        with pytest.raises(SessionException):
            await first

    async def test_aclose_waits_for_a_close_on_another_thread(self) -> None:
        client = _slow_client(0.2)
        manager = Neo4jSessionManager("sess-1", memory_client=client)
        _buffered(manager, "one")
        closer = asyncio.create_task(asyncio.to_thread(manager.close))
        await asyncio.sleep(0.05)

        await manager.aclose()
        assert _stored(client) == [("user", "one")]
        await closer

    async def test_aclose_is_idempotent_and_close_after_aclose_is_a_noop(self) -> None:
        client = FakeMemoryClient()
        manager = Neo4jSessionManager("sess-1", memory_client=client)
        manager.initialize(SimpleNamespace(messages=[], agent_id="a1"))
        await manager.aclose()
        await manager.aclose()
        manager.close()
        assert client.close_calls == 1
