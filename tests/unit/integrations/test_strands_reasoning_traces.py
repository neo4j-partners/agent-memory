"""The Strands session manager records each turn's tool uses as one trace.

With ``record_tool_calls`` on, a turn that used tools becomes one reasoning
trace whose task is the turn's prompt. The trace is completed when the turn
ends, with ``success=True`` or, for a call that raised, ``success=False``.
NAMS keeps no trace outcome, so there the outcome is one more step.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

pytest.importorskip("strands", reason="strands-agents not installed")

from strands.hooks import AfterToolsEvent  # noqa: E402

from neo4j_agent_memory.integrations.strands.session_manager import (  # noqa: E402
    _FAILED,
    _OUTCOME_ACTION,
    _SUCCEEDED,
    Neo4jSessionManager,
)
from neo4j_agent_memory.nams.reasoning import NamsReasoningMemory  # noqa: E402
from tests.unit.integrations.strands_fakes import FakeMemoryClient, StubCall  # noqa: E402
from tests.unit.integrations.test_strands_nonblocking import (  # noqa: E402
    _TOOL_TURN,
    _agent,
    _stored,
)
from tests.unit.integrations.test_strands_turn_commit import (  # noqa: E402
    _NO_TURN,
    _answer,
    _approval_agent,
    _crash,
    _once,
)

_NOW = "2026-11-01T00:00:00Z"


def _manager(client: FakeMemoryClient, *, record_tool_calls: bool = True) -> Neo4jSessionManager:
    return Neo4jSessionManager(
        "sess-1",
        memory_client=client,
        extract_entities=False,
        record_tool_calls=record_tool_calls,
    )


def _steps(client: FakeMemoryClient) -> list[tuple[Any, str]]:
    return [(s["trace_id"], s["action"]) for s in client.reasoning.steps]


class TestATurnWithTools:
    async def test_a_successful_turn_is_one_completed_trace(self) -> None:
        client = FakeMemoryClient()
        manager = _manager(client)
        agent = _agent(manager, _TOOL_TURN, [{"text": "Sunny."}])
        try:
            await agent.invoke_async("weather?")
        finally:
            await manager.aclose()

        (trace,) = client.reasoning.traces
        assert trace["task"] == "weather?"
        assert trace["completed"]
        assert trace["success"] is True
        assert trace["outcome"] == _SUCCEEDED
        assert _steps(client) == [(trace["id"], "lookup")]
        (call,) = client.reasoning.tool_calls
        assert call["tool_name"] == "lookup"
        assert call["arguments"] == {"city": "Paris"}

    async def test_each_turn_gets_its_own_trace(self) -> None:
        client = FakeMemoryClient()
        manager = _manager(client)
        agent = _agent(manager, _TOOL_TURN, [{"text": "A1"}], _TOOL_TURN, [{"text": "A2"}])
        try:
            await agent.invoke_async("Q1")
            await agent.invoke_async("Q2")
        finally:
            await manager.aclose()

        first, second = client.reasoning.traces
        assert (first["task"], second["task"]) == ("Q1", "Q2")
        assert first["completed"] and second["completed"]
        assert _steps(client) == [(first["id"], "lookup"), (second["id"], "lookup")]

    async def test_a_failed_turn_is_a_failed_trace_and_stores_no_messages(self) -> None:
        client = FakeMemoryClient()
        manager = _manager(client)
        agent = _agent(manager, _TOOL_TURN)  # the model raises after the tool runs
        try:
            with pytest.raises(Exception, match=_NO_TURN):
                await agent.invoke_async("weather?")
        finally:
            await manager.aclose()

        assert _stored(client) == []
        (trace,) = client.reasoning.traces
        assert trace["task"] == "weather?"
        assert trace["completed"]
        assert trace["success"] is False
        assert trace["outcome"] == _FAILED
        assert _steps(client) == [(trace["id"], "lookup")]

    async def test_an_interrupted_turn_is_one_trace_once_it_resumes(self) -> None:
        client = FakeMemoryClient()
        manager = _manager(client)
        agent = _approval_agent(manager)
        try:
            paused = await agent.invoke_async("delete the file")
            assert client.reasoning.traces == []  # the turn is still open
            await agent.invoke_async(_answer(paused))
        finally:
            await manager.aclose()

        (trace,) = client.reasoning.traces
        assert trace["task"] == "delete the file"
        assert trace["success"] is True
        assert _steps(client) == [(trace["id"], "approve")]

    async def test_a_call_that_fails_at_an_interrupt_is_traced_with_its_resume(self) -> None:
        client = FakeMemoryClient()
        manager = _manager(client)
        agent = _approval_agent(manager)
        agent.hooks.add_callback(AfterToolsEvent, _once(_crash))  # Strands keeps the interrupt
        try:
            with pytest.raises(Exception, match="hook failed"):
                await agent.invoke_async("delete the file")
            assert client.reasoning.traces == []  # the turn waits for its resume
            interrupts = list(agent._interrupt_state.interrupts.values())
            (interrupt,) = interrupts
            await agent.invoke_async(
                [{"interruptResponse": {"interruptId": interrupt.id, "response": "yes"}}]
            )
        finally:
            await manager.aclose()

        (trace,) = client.reasoning.traces
        assert trace["task"] == "delete the file"
        assert trace["success"] is True


class TestATurnWithoutTraces:
    async def test_a_turn_without_tools_has_no_trace(self) -> None:
        client = FakeMemoryClient()
        manager = _manager(client)
        agent = _agent(manager, [{"text": "Hello."}])
        try:
            await agent.invoke_async("hi")
        finally:
            await manager.aclose()

        assert _stored(client) == [("user", "hi"), ("assistant", "Hello.")]
        assert client.reasoning.traces == []

    async def test_nothing_is_recorded_when_mirroring_is_off(self) -> None:
        client = FakeMemoryClient()
        manager = _manager(client, record_tool_calls=False)
        agent = _agent(manager, _TOOL_TURN, [{"text": "Sunny."}])
        try:
            await agent.invoke_async("weather?")
        finally:
            await manager.aclose()

        assert client.reasoning.traces == []
        assert client.reasoning.steps == []


def _step_reply(call: StubCall) -> dict[str, Any]:
    body = call.json if isinstance(call.json, dict) else {}
    return {"id": str(uuid.uuid4()), "createdAt": _NOW, **body}


def _tool_call_reply(call: StubCall) -> dict[str, Any]:
    body = call.json if isinstance(call.json, dict) else {}
    return {"id": str(uuid.uuid4()), "status": "success", "createdAt": _NOW, **body}


def _nams_client() -> FakeMemoryClient:
    """A NAMS-mode client whose reasoning is the real NAMS class on the stub wire."""
    client = FakeMemoryClient(nams_mode=True)
    client.reasoning = NamsReasoningMemory(client.wire)  # type: ignore[assignment]
    client.wire.responses["record_step"] = _step_reply
    client.wire.responses["record_tool_call"] = _tool_call_reply
    return client


def _wire_steps(client: FakeMemoryClient) -> list[tuple[str, str, str | None]]:
    return [
        (c.json["actionTaken"], c.json["reasoning"], c.json.get("result"))
        for c in client.wire.calls_for("record_step")
    ]


def _completed(client: FakeMemoryClient) -> list[dict[str, Any]]:
    reasoning: NamsReasoningMemory = client.reasoning  # type: ignore[assignment]
    return list(reasoning._traces.values())


class TestOnNams:
    async def test_the_outcome_is_the_last_step_of_the_turn(self) -> None:
        client = _nams_client()
        manager = _manager(client)
        agent = _agent(manager, _TOOL_TURN, [{"text": "Sunny."}])
        try:
            await agent.invoke_async("weather?")
        finally:
            await manager.aclose()

        assert _wire_steps(client) == [
            ("lookup", "Tool use: lookup", None),
            (_OUTCOME_ACTION, _SUCCEEDED, "success"),
        ]
        (tool_call,) = client.wire.calls_for("record_tool_call")
        assert tool_call.json["toolName"] == "lookup"
        (trace,) = _completed(client)
        assert trace["task"] == "weather?"
        assert trace["success"] is True
        assert trace["completed_at"] is not None

    async def test_a_failed_turn_ends_with_a_failure_step(self) -> None:
        client = _nams_client()
        manager = _manager(client)
        agent = _agent(manager, _TOOL_TURN)  # the model raises after the tool runs
        try:
            with pytest.raises(Exception, match=_NO_TURN):
                await agent.invoke_async("weather?")
        finally:
            await manager.aclose()

        assert client.wire.calls_for("add_message") == []
        assert client.wire.calls_for("bulk_add_messages") == []
        assert _wire_steps(client) == [
            ("lookup", "Tool use: lookup", None),
            (_OUTCOME_ACTION, _FAILED, "failure"),
        ]
        (trace,) = _completed(client)
        assert trace["success"] is False
