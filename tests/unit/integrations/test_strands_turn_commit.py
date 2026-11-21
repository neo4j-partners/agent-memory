"""Fix C: the Strands session manager writes a turn only when it succeeds.

The manager buffers each message of an agent call and writes them at
AfterInvocationEvent if the call returned a result. A call that raised or
was cancelled stores nothing, so the backend never holds a prompt without
its reply. A turn paused by an interrupt stays buffered until it resumes.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

pytest.importorskip("strands", reason="strands-agents not installed")

from strands import Agent, ToolContext, tool  # noqa: E402
from strands.hooks import BeforeInvocationEvent  # noqa: E402

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
