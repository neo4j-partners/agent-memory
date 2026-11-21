"""Fix C: the Strands session manager writes a turn only when it succeeds.

The manager buffers each message of an agent call and writes them at
AfterInvocationEvent if the call returned a result. A call that raised or
was cancelled stores nothing, so the backend never holds a prompt without
its reply. A turn paused by an interrupt stays buffered until it resumes.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import tomllib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

pytest.importorskip("strands", reason="strands-agents not installed")

from strands import Agent, ToolContext, tool  # noqa: E402
from strands.hooks import BeforeInvocationEvent  # noqa: E402
from strands.types.agent import ConcurrentInvocationMode  # noqa: E402

from neo4j_agent_memory.integrations.strands.session_manager import (  # noqa: E402
    Neo4jSessionManager,
)
from tests.unit.integrations.strands_fakes import FakeMemoryClient, turn_ended  # noqa: E402
from tests.unit.integrations.test_strands_nonblocking import (  # noqa: E402
    _TOOL_TURN,
    ScriptedModel,
    _agent,
    _buffered,
    _stored,
    lookup,
)
from tests.unit.integrations.test_strands_persist_error import _failing_client  # noqa: E402

_LOGGER = "neo4j_agent_memory.integrations.strands.session_manager"
_REPO = Path(__file__).resolve().parents[3]
_NO_TURN = "pop from empty list"  # a ScriptedModel with no turn left raises this


def _records(caplog: pytest.LogCaptureFixture, level: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == _LOGGER and r.levelname == level]


@tool(context=True)
def approve(tool_context: ToolContext) -> str:
    """Ask a human to approve the action."""
    return f"approved: {tool_context.interrupt('approval', reason='proceed?')}"


_APPROVAL_TURN = [
    {"text": "Asking for approval."},
    {"toolUse": {"toolUseId": "t1", "name": "approve", "input": {}}},
]


@tool(context=True)
def abort(tool_context: ToolContext) -> str:
    """Stop the agent now."""
    tool_context.agent.cancel()
    return "aborting"


def _approval_agent(manager: Neo4jSessionManager) -> Agent:
    return Agent(
        model=ScriptedModel(_APPROVAL_TURN, [{"text": "Done."}]),
        tools=[approve],
        session_manager=manager,
        callback_handler=None,
    )


def _answer(paused: Any) -> Any:
    """The prompt that resumes ``paused`` with a yes."""
    (interrupt,) = paused.interrupts or []
    return [{"interruptResponse": {"interruptId": interrupt.id, "response": "yes"}}]


def _once(hook: Any) -> Any:
    """A BeforeInvocationEvent callback that runs ``hook`` on its first call only."""
    calls: list[None] = []

    def callback(event: BeforeInvocationEvent) -> None:
        if not calls:
            calls.append(None)
            hook(event)

    return callback


def _deny(event: BeforeInvocationEvent) -> None:
    event.cancel = "denied"


def _crash(event: BeforeInvocationEvent) -> None:
    raise RuntimeError("hook failed")


_FULL_APPROVAL_TURN = [
    ("user", "delete the file"),
    ("assistant", "Asking for approval."),
    ("assistant", "Done."),
]


class _StalledModel(ScriptedModel):
    """A model whose first call never answers. Later calls replay ``turns``."""

    def __init__(self, *turns: list[dict[str, Any]]) -> None:
        super().__init__(*turns)
        self.started = asyncio.Event()

    async def stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        if not self.started.is_set():
            self.started.set()
            await asyncio.Event().wait()
        async for event in super().stream(*args, **kwargs):
            yield event


class _RedactingModel(ScriptedModel):
    """A model whose guardrail redacts the prompt before it answers."""

    async def stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        yield {"redactContent": {"redactUserContentMessage": "[REDACTED]"}}
        async for event in super().stream(*args, **kwargs):
            yield event


class TestASuccessfulTurn:
    async def test_nothing_is_written_until_the_turn_ends(self) -> None:
        client = FakeMemoryClient()
        manager = Neo4jSessionManager("sess-1", memory_client=client, extract_entities=False)
        seen_mid_turn: list[list[tuple[str, str]]] = []

        @tool
        def peek(city: str) -> str:
            """Look up the weather for a city."""
            seen_mid_turn.append(_stored(client))
            return f"Sunny in {city}"

        turn = [
            {"text": "Let me check."},
            {"toolUse": {"toolUseId": "t1", "name": "peek", "input": {"city": "Paris"}}},
        ]
        agent = Agent(
            model=ScriptedModel(turn, [{"text": "It is sunny."}]),
            tools=[peek],
            session_manager=manager,
            callback_handler=None,
        )
        try:
            await agent.invoke_async("weather?")
        finally:
            await manager.aclose()

        assert seen_mid_turn == [[]]
        assert _stored(client) == [
            ("user", "weather?"),
            ("assistant", "Let me check."),
            ("assistant", "It is sunny."),
        ]

    async def test_each_call_writes_its_own_turn(self) -> None:
        client = FakeMemoryClient()
        manager = Neo4jSessionManager("sess-1", memory_client=client, extract_entities=False)
        agent = _agent(manager, [{"text": "A1"}], [{"text": "A2"}])
        try:
            await agent.invoke_async("Q1")
            assert _stored(client) == [("user", "Q1"), ("assistant", "A1")]
            await agent.invoke_async("Q2")
        finally:
            await manager.aclose()
        assert _stored(client) == [
            ("user", "Q1"),
            ("assistant", "A1"),
            ("user", "Q2"),
            ("assistant", "A2"),
        ]

    async def test_a_redact_content_chunk_rewrites_the_prompt(self) -> None:
        client = FakeMemoryClient()
        manager = Neo4jSessionManager("sess-1", memory_client=client, extract_entities=False)
        agent = Agent(
            model=_RedactingModel([{"text": "I cannot help with that."}]),
            session_manager=manager,
            callback_handler=None,
        )
        try:
            await agent.invoke_async("my card is 4111")
        finally:
            await manager.aclose()
        assert _stored(client) == [
            ("user", "[REDACTED]"),
            ("assistant", "I cannot help with that."),
        ]

    async def test_redaction_rewrites_the_latest_buffered_message(self) -> None:
        client = FakeMemoryClient()
        manager = Neo4jSessionManager("sess-1", memory_client=client, extract_entities=False)
        agent = _buffered(manager, "first", "secret 123")
        manager.redact_latest_message({"role": "user", "content": [{"text": "[REDACTED]"}]}, agent)
        try:
            await manager._on_after_invocation(turn_ended(agent))
            assert _stored(client) == [("user", "first"), ("user", "[REDACTED]")]
        finally:
            await manager.aclose()


class TestATurnWithoutAResult:
    async def test_a_failed_call_stores_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        client = FakeMemoryClient()
        manager = Neo4jSessionManager("sess-1", memory_client=client, extract_entities=False)
        agent = _agent(manager)  # the model has no turn to give, so it raises
        with (
            caplog.at_level(logging.INFO, logger=_LOGGER),
            pytest.raises(Exception, match=_NO_TURN),
        ):
            await agent.invoke_async("hi")
        await manager.aclose()

        assert _stored(client) == []
        (info,) = _records(caplog, "INFO")
        assert "sess-1" in info
        assert "1 message" in info

    async def test_a_call_that_fails_after_a_tool_call_stores_nothing(self) -> None:
        client = FakeMemoryClient()
        manager = Neo4jSessionManager("sess-1", memory_client=client, extract_entities=False)
        agent = _agent(manager, _TOOL_TURN)  # the model raises after the tool runs
        with pytest.raises(Exception, match=_NO_TURN):
            await agent.invoke_async("weather?")
        await manager.aclose()
        assert _stored(client) == []

    async def test_the_call_after_a_failure_writes_only_its_own_turn(self) -> None:
        client = FakeMemoryClient()
        manager = Neo4jSessionManager("sess-1", memory_client=client, extract_entities=False)
        model = ScriptedModel()
        agent = Agent(model=model, tools=[lookup], session_manager=manager, callback_handler=None)
        with pytest.raises(Exception, match=_NO_TURN):
            await agent.invoke_async("Q1")
        model._turns.append([{"text": "A2"}])
        try:
            await agent.invoke_async("Q2")
        finally:
            await manager.aclose()
        assert _stored(client) == [("user", "Q2"), ("assistant", "A2")]

    async def test_a_cancelled_call_stores_nothing(self) -> None:
        client = FakeMemoryClient()
        manager = Neo4jSessionManager("sess-1", memory_client=client, extract_entities=False)
        agent = _agent(manager)

        def deny(event: BeforeInvocationEvent) -> None:
            event.cancel = "denied"

        agent.hooks.add_callback(BeforeInvocationEvent, deny)
        try:
            await agent.invoke_async("hi")
        finally:
            await manager.aclose()
        assert _stored(client) == []

    async def test_agent_cancel_stores_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        client = FakeMemoryClient()
        manager = Neo4jSessionManager("sess-1", memory_client=client, extract_entities=False)
        turn = [
            {"text": "Stopping."},
            {"toolUse": {"toolUseId": "t1", "name": "abort", "input": {}}},
        ]
        agent = Agent(
            model=ScriptedModel(turn),
            tools=[abort],
            session_manager=manager,
            callback_handler=None,
        )
        try:
            with caplog.at_level(logging.INFO, logger=_LOGGER):
                result = await agent.invoke_async("hi")
        finally:
            await manager.aclose()

        assert result.stop_reason == "cancelled"
        assert _stored(client) == []
        (info,) = _records(caplog, "INFO")
        assert "cancelled" in info
        assert "3 message" in info  # the prompt, the tool call, and its result

    async def test_a_task_cancelled_before_its_result_stores_nothing(self) -> None:
        client = FakeMemoryClient()
        manager = Neo4jSessionManager("sess-1", memory_client=client, extract_entities=False)
        model = _StalledModel([{"text": "A2"}])
        agent = Agent(model=model, session_manager=manager, callback_handler=None)
        try:
            call = asyncio.create_task(agent.invoke_async("Q1"))
            await model.started.wait()
            call.cancel()
            with pytest.raises(asyncio.CancelledError):
                await call
            assert _stored(client) == []
            await agent.invoke_async("Q2")
        finally:
            await manager.aclose()
        assert _stored(client) == [("user", "Q2"), ("assistant", "A2")]

    def test_the_strands_floor_closes_an_abandoned_stream(self) -> None:
        """Strands before 1.55 leaves an abandoned stream's AfterInvocationEvent to the GC."""
        pyproject = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))
        strands = pyproject["project"]["optional-dependencies"]["strands"]
        assert "strands-agents>=1.55,<2" in strands
        assert "events.aclose()" in inspect.getsource(Agent.stream_async)

    async def test_an_abandoned_stream_stores_nothing(self) -> None:
        client = FakeMemoryClient()
        manager = Neo4jSessionManager("sess-1", memory_client=client, extract_entities=False)
        agent = _agent(manager, [{"text": "A2"}])
        try:
            stream = agent.stream_async("Q1")
            async for _ in stream:
                if agent.messages:
                    break  # the caller walks away before the model answers
            await stream.aclose()
            assert _stored(client) == []
            await agent.invoke_async("Q2")
        finally:
            await manager.aclose()
        assert _stored(client) == [("user", "Q2"), ("assistant", "A2")]


class TestAStopThatAnswers:
    async def test_the_stop_tool_writes_the_turn(self) -> None:
        stop_module = pytest.importorskip("strands.vended_tools.stop")
        client = FakeMemoryClient()
        manager = Neo4jSessionManager("sess-1", memory_client=client, extract_entities=False)
        turn = [{"toolUse": {"toolUseId": "t1", "name": "stop", "input": {"message": "Bye."}}}]
        agent = Agent(
            model=ScriptedModel(turn),
            tools=[stop_module.stop],
            session_manager=manager,
            callback_handler=None,
        )
        try:
            result = await agent.invoke_async("hi")
        finally:
            await manager.aclose()

        assert result.stop_reason == "cancelled"
        assert _stored(client) == [("user", "hi"), ("assistant", "Bye.")]

    @pytest.mark.parametrize(
        "stop_reason",
        ["limit_turns", "limit_total_tokens", "limit_output_tokens", "max_tokens", "cancelled"],
    )
    async def test_a_result_without_a_cancel_signal_writes_the_turn(self, stop_reason: str) -> None:
        client = FakeMemoryClient()
        manager = Neo4jSessionManager("sess-1", memory_client=client, extract_entities=False)
        agent = _buffered(manager)
        try:
            await manager._on_before_invocation(turn_ended(agent))
            manager.append_message({"role": "user", "content": [{"text": "Q1"}]}, agent)
            await manager._on_after_invocation(turn_ended(agent, stop_reason))
        finally:
            await manager.aclose()
        assert _stored(client) == [("user", "Q1")]


class TestAnInterruptedTurn:
    async def test_an_interrupt_keeps_the_turn_until_it_resumes(self) -> None:
        client = FakeMemoryClient()
        manager = Neo4jSessionManager("sess-1", memory_client=client, extract_entities=False)
        agent = Agent(
            model=ScriptedModel(_APPROVAL_TURN, [{"text": "Done."}]),
            tools=[approve],
            session_manager=manager,
            callback_handler=None,
        )
        try:
            paused = await agent.invoke_async("delete the file")
            assert paused.stop_reason == "interrupt"
            assert _stored(client) == []

            (interrupt,) = paused.interrupts or []
            response: Any = [
                {"interruptResponse": {"interruptId": interrupt.id, "response": "yes"}}
            ]
            await agent.invoke_async(response)
        finally:
            await manager.aclose()
        assert _stored(client) == [
            ("user", "delete the file"),
            ("assistant", "Asking for approval."),
            ("assistant", "Done."),
        ]

    async def test_a_cancelled_resume_keeps_the_turn_paused(self) -> None:
        client = FakeMemoryClient()
        manager = Neo4jSessionManager("sess-1", memory_client=client, extract_entities=False)
        agent = _approval_agent(manager)
        try:
            paused = await agent.invoke_async("delete the file")
            agent.hooks.add_callback(BeforeInvocationEvent, _once(_deny))
            await agent.invoke_async(_answer(paused))  # the resume is cancelled
            assert _stored(client) == []

            await agent.invoke_async(_answer(paused))
        finally:
            await manager.aclose()
        assert _stored(client) == _FULL_APPROVAL_TURN

    async def test_a_resume_after_a_failed_before_hook_writes_the_whole_turn(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = FakeMemoryClient()
        manager = Neo4jSessionManager("sess-1", memory_client=client, extract_entities=False)
        agent = _approval_agent(manager)
        try:
            paused = await agent.invoke_async("delete the file")
            agent.hooks.add_callback(BeforeInvocationEvent, _once(_crash))
            with pytest.raises(RuntimeError, match="hook failed"):
                await agent.invoke_async(_answer(paused))  # no AfterInvocationEvent fires

            with caplog.at_level(logging.WARNING, logger=_LOGGER):
                await agent.invoke_async(_answer(paused))
        finally:
            await manager.aclose()

        assert _stored(client) == _FULL_APPROVAL_TURN
        (warning,) = _records(caplog, "WARNING")
        assert "2 message" in warning

    @pytest.mark.parametrize("use_async", [False, True], ids=["close", "aclose"])
    async def test_closing_during_an_interrupt_warns_and_writes_nothing(
        self, use_async: bool, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = FakeMemoryClient()
        manager = Neo4jSessionManager("sess-1", memory_client=client, extract_entities=False)
        agent = _buffered(manager)
        await manager._on_before_invocation(turn_ended(agent))
        manager.append_message({"role": "user", "content": [{"text": "delete it"}]}, agent)
        await manager._on_after_invocation(turn_ended(agent, "interrupt"))

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            if use_async:
                await manager.aclose()
            else:
                manager.close()

        assert _stored(client) == []
        (warning,) = _records(caplog, "WARNING")
        assert "sess-1" in warning
        assert "interrupt" in warning
        assert "1 message" in warning


class TestACallThatStartsWhileATurnRuns:
    async def test_it_warns_and_keeps_the_open_turn_buffered(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = FakeMemoryClient()
        manager = Neo4jSessionManager("sess-1", memory_client=client, extract_entities=False)
        agent = _buffered(manager)
        try:
            await manager._on_before_invocation(turn_ended(agent))
            manager.append_message({"role": "user", "content": [{"text": "Q1"}]}, agent)
            with caplog.at_level(logging.WARNING, logger=_LOGGER):
                await manager._on_before_invocation(turn_ended(agent))
            assert _stored(client) == []

            manager.append_message({"role": "user", "content": [{"text": "Q2"}]}, agent)
            await manager._on_after_invocation(turn_ended(agent))
        finally:
            await manager.aclose()

        assert _stored(client) == [("user", "Q1"), ("user", "Q2")]
        (warning,) = _records(caplog, "WARNING")
        assert "sess-1" in warning
        assert "1 message" in warning

    @pytest.mark.parametrize("use_async", [False, True], ids=["close", "aclose"])
    async def test_closing_during_it_logs_an_error(
        self, use_async: bool, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = FakeMemoryClient()
        manager = Neo4jSessionManager("sess-1", memory_client=client, extract_entities=False)
        agent = _buffered(manager)
        await manager._on_before_invocation(turn_ended(agent))
        manager.append_message({"role": "user", "content": [{"text": "Q1"}]}, agent)

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            if use_async:
                await manager.aclose()
            else:
                manager.close()
            (error,) = _records(caplog, "ERROR")
            assert "sess-1" in error
            assert "1 message" in error

            await manager._on_after_invocation(turn_ended(agent))  # the call ends late

        assert _stored(client) == []
        assert len(_records(caplog, "ERROR")) == 1


class _Summary(BaseModel):
    text: str


class _HeldModel(ScriptedModel):
    """A model that holds its first answer until ``release`` is set.

    A second call raises while the first is held. Structured output
    answers at once.
    """

    def __init__(self, *turns: list[dict[str, Any]]) -> None:
        super().__init__(*turns)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        if self.started.is_set() and not self.release.is_set():
            raise RuntimeError("model down")
        self.started.set()
        await self.release.wait()
        async for event in super().stream(*args, **kwargs):
            yield event

    async def structured_output(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        yield {"output": _Summary(text="a summary")}


def _held_agent(manager: Neo4jSessionManager, model: _HeldModel) -> Agent:
    return Agent(
        model=model,
        session_manager=manager,
        callback_handler=None,
        concurrent_invocation_mode=ConcurrentInvocationMode.UNSAFE_REENTRANT,
    )


class TestCallsThatOverlap:
    async def test_structured_output_during_a_call_leaves_its_turn(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = FakeMemoryClient()
        manager = Neo4jSessionManager("sess-1", memory_client=client, extract_entities=False)
        model = _HeldModel([{"text": "A1"}])
        agent = _held_agent(manager, model)
        try:
            call = asyncio.create_task(agent.invoke_async("Q1"))
            await model.started.wait()
            with caplog.at_level(logging.WARNING, logger=_LOGGER):
                summary = await agent.structured_output_async(_Summary, "Summarize.")
            model.release.set()
            await call
        finally:
            await manager.aclose()

        assert summary.text == "a summary"
        assert _stored(client) == [("user", "Q1"), ("assistant", "A1")]
        (warning,) = _records(caplog, "WARNING")
        assert "first call's result decides" in warning

    async def test_a_failed_call_leaves_the_turn_it_overlapped(self) -> None:
        client = FakeMemoryClient()
        manager = Neo4jSessionManager("sess-1", memory_client=client, extract_entities=False)
        model = _HeldModel([{"text": "A1"}])
        agent = _held_agent(manager, model)
        try:
            call = asyncio.create_task(agent.invoke_async("Q1"))
            await model.started.wait()
            with pytest.raises(RuntimeError, match="model down"):
                await agent.invoke_async("Q2")
            model.release.set()
            await call
        finally:
            await manager.aclose()

        assert _stored(client) == [("user", "Q1"), ("user", "Q2"), ("assistant", "A1")]

    async def test_structured_output_alone_ends_its_turn(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = FakeMemoryClient()
        manager = Neo4jSessionManager("sess-1", memory_client=client, extract_entities=False)
        model = _HeldModel([{"text": "A1"}])
        model.release.set()
        agent = _held_agent(manager, model)
        try:
            with caplog.at_level(logging.WARNING, logger=_LOGGER):
                await agent.structured_output_async(_Summary, "Summarize.")
                await agent.invoke_async("Q1")
        finally:
            await manager.aclose()

        assert _stored(client) == [("user", "Q1"), ("assistant", "A1")]
        assert _records(caplog, "WARNING") == []

    async def test_a_call_after_a_failed_before_hook_in_another_task_writes_its_turn(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = FakeMemoryClient()
        manager = Neo4jSessionManager("sess-1", memory_client=client, extract_entities=False)
        model = _HeldModel([{"text": "A1"}])
        model.release.set()
        agent = _held_agent(manager, model)
        agent.hooks.add_callback(BeforeInvocationEvent, _once(_crash))
        try:
            with pytest.raises(RuntimeError, match="hook failed"):
                await asyncio.create_task(agent.invoke_async("Q0"))

            with caplog.at_level(logging.WARNING, logger=_LOGGER):
                await agent.invoke_async("Q1")
        finally:
            await manager.aclose()

        assert _stored(client) == [("user", "Q1"), ("assistant", "A1")]
        (warning,) = _records(caplog, "WARNING")
        assert "before the previous call's turn ended" in warning


class TestTheNextCallWritesWhatIsQueued:
    async def test_messages_appended_outside_a_call_are_written_when_the_next_starts(
        self,
    ) -> None:
        client = FakeMemoryClient()
        manager = Neo4jSessionManager("sess-1", memory_client=client, extract_entities=False)
        agent = _buffered(manager, "direct tool call")
        try:
            await manager._on_before_invocation(turn_ended(agent))
            assert _stored(client) == [("user", "direct tool call")]
        finally:
            await manager.aclose()

    async def test_a_failed_write_is_retried_once_before_the_next_turn(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client, switch = _failing_client()
        manager = Neo4jSessionManager(
            "sess-1", memory_client=client, on_persist_error="warn", extract_entities=False
        )
        agent = _buffered(manager)
        try:
            with caplog.at_level(logging.WARNING, logger=_LOGGER):
                await manager._on_before_invocation(turn_ended(agent))
                manager.append_message({"role": "user", "content": [{"text": "Q1"}]}, agent)
                switch.down = True
                await manager._on_after_invocation(turn_ended(agent))  # warns, keeps Q1
                assert _stored(client) == []

                switch.down = False
                await manager._on_before_invocation(turn_ended(agent))
                assert _stored(client) == [("user", "Q1")]
            assert _records(caplog, "ERROR") == []
            (warning,) = _records(caplog, "WARNING")
            assert "backend down" in warning
        finally:
            await manager.aclose()

    async def test_a_write_that_fails_after_the_wait_runs_out_is_retried(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = FakeMemoryClient()
        original = client.short_term.add_message
        failures: list[None] = []

        async def slow_then_fail_once(*args: Any, **kwargs: Any) -> Any:
            if not failures:
                failures.append(None)
                await asyncio.sleep(0.2)
                raise RuntimeError("backend down")
            return await original(*args, **kwargs)

        client.short_term.add_message = slow_then_fail_once  # type: ignore[method-assign]
        manager = Neo4jSessionManager(
            "sess-1", memory_client=client, on_persist_error="warn", extract_entities=False
        )
        monkeypatch.setattr(manager, "_drain_timeout", lambda: 0.05)
        agent = _buffered(manager)
        try:
            with caplog.at_level(logging.WARNING, logger=_LOGGER):
                await manager._on_before_invocation(turn_ended(agent))
                manager.append_message({"role": "user", "content": [{"text": "Q1"}]}, agent)
                await manager._on_after_invocation(turn_ended(agent))  # stops waiting
                await asyncio.sleep(0.4)  # the write fails after the wait ran out

                await manager._on_before_invocation(turn_ended(agent))  # retries Q1
        finally:
            await manager.aclose()

        assert _stored(client) == [("user", "Q1")]
        assert _records(caplog, "ERROR") == []
