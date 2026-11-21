"""Optional temperature and the BedrockProvider request overrides.

``temperature`` defaults to ``None`` across the provider protocol, the
adapters, and ``LLMEntityExtractor``. ``None`` means "use the model's
default", so the field is left out of the request. Some models, such as
Claude Sonnet 5.5 on Bedrock, reject a temperature outright.

``BedrockProvider`` also takes ``send_temperature`` (False never sends one)
and ``tool_choice`` (None keeps forced tool use, "auto" lets the model
choose). Each test asserts the request a stubbed client received.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from neo4j_agent_memory.llm.types import ChatMessage
from tests.unit.llm.test_provider_request_shape import _FakeOpenAIClient, _Recorder

_SONNET = "bedrock/us.anthropic.claude-sonnet-5-5-v1:0"
_TITAN = "bedrock/amazon.titan-text-premier-v1:0"
_MESSAGES = [ChatMessage(role="user", content="hi")]


class _Extraction(BaseModel):
    value: str


def _text_response(text: str) -> dict[str, Any]:
    return {
        "output": {"message": {"content": [{"text": text}]}},
        "usage": {},
        "stopReason": "end_turn",
    }


def _tool_response(tool_input: dict[str, Any]) -> dict[str, Any]:
    return {
        "output": {
            "message": {
                "content": [{"toolUse": {"name": "submit_extraction", "input": tool_input}}]
            }
        },
        "usage": {},
        "stopReason": "tool_use",
    }


class _FakeConverseClient:
    """Stands in for the boto3 bedrock-runtime client."""

    def __init__(self, *responses: dict[str, Any]) -> None:
        self.calls: list[dict[str, Any]] = []
        self._responses = list(responses)

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]


def _bedrock(model: str = _SONNET, *responses: dict[str, Any], **kwargs: Any):
    from neo4j_agent_memory.llm.adapters.bedrock import BedrockProvider

    provider = BedrockProvider(model, aws_region="us-west-2", **kwargs)
    client = _FakeConverseClient(*(responses or (_text_response("ok"),)))
    provider._client = client
    return provider, client


# ---------------------------------------------------------------------------
# BedrockProvider.complete
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("send_temperature", "temperature", "expected"),
    [
        (True, None, None),
        (True, 0.0, 0.0),
        (True, 0.7, 0.7),
        (False, None, None),
        (False, 0.0, None),
        (False, 0.7, None),
    ],
)
async def test_bedrock_complete_temperature(send_temperature, temperature, expected) -> None:
    provider, client = _bedrock(send_temperature=send_temperature)

    await provider.complete(_MESSAGES, temperature=temperature)

    config = client.calls[-1]["inferenceConfig"]
    assert config.get("temperature") == expected
    assert ("temperature" in config) is (expected is not None)
    assert "maxTokens" in config


async def test_bedrock_complete_default_sends_no_temperature() -> None:
    provider, client = _bedrock()

    await provider.complete(_MESSAGES)

    assert "temperature" not in client.calls[-1]["inferenceConfig"]


# ---------------------------------------------------------------------------
# BedrockProvider.complete_structured
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tool_choice", "expected_choice"),
    [
        (None, {"tool": {"name": "submit_extraction"}}),
        ("tool", {"tool": {"name": "submit_extraction"}}),
        ("auto", {"auto": {}}),
    ],
)
@pytest.mark.parametrize(
    ("send_temperature", "temperature", "expected_temperature"),
    [
        (True, None, None),
        (True, 0.0, 0.0),
        (False, 0.0, None),
    ],
)
async def test_bedrock_structured_request_shape(
    tool_choice, expected_choice, send_temperature, temperature, expected_temperature
) -> None:
    provider, client = _bedrock(
        _SONNET,
        _tool_response({"value": "ok"}),
        tool_choice=tool_choice,
        send_temperature=send_temperature,
    )

    result = await provider.complete_structured(_MESSAGES, _Extraction, temperature=temperature)

    assert result.value == "ok"
    sent = client.calls[-1]
    assert sent["toolConfig"]["toolChoice"] == expected_choice
    config = sent["inferenceConfig"]
    assert config.get("temperature") == expected_temperature
    assert ("temperature" in config) is (expected_temperature is not None)


async def test_bedrock_sonnet_5_5_shape_falls_back_to_sap_without_temperature() -> None:
    """With tool_choice="auto" the model may answer in text; SAP takes over."""
    provider, client = _bedrock(
        _SONNET,
        _text_response("no tool call"),
        _text_response('{"value": "ok"}'),
        tool_choice="auto",
        send_temperature=False,
    )

    result = await provider.complete_structured(_MESSAGES, _Extraction, temperature=0.0)

    assert result.value == "ok"
    assert len(client.calls) == 2
    assert client.calls[0]["toolConfig"]["toolChoice"] == {"auto": {}}
    assert "toolConfig" not in client.calls[1]
    assert all("temperature" not in call["inferenceConfig"] for call in client.calls)


async def test_bedrock_non_anthropic_structured_honors_send_temperature() -> None:
    provider, client = _bedrock(_TITAN, _text_response('{"value": "ok"}'), send_temperature=False)

    await provider.complete_structured(_MESSAGES, _Extraction, temperature=0.0)

    assert "temperature" not in client.calls[-1]["inferenceConfig"]


def test_bedrock_rejects_unknown_tool_choice() -> None:
    from neo4j_agent_memory.llm.adapters.bedrock import BedrockProvider

    with pytest.raises(ValueError, match="tool_choice"):
        BedrockProvider(_SONNET, tool_choice="any")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Other adapters omit temperature when it is None
# ---------------------------------------------------------------------------


async def test_openai_omits_temperature_by_default() -> None:
    pytest.importorskip("openai")
    from neo4j_agent_memory.llm.adapters.openai import OpenAIProvider

    recorder = _Recorder()
    provider = OpenAIProvider("openai/gpt-5-mini", api_key="x")
    provider._client = _FakeOpenAIClient(recorder, content='{"value": "ok"}')  # type: ignore[assignment]

    await provider.complete(_MESSAGES)
    assert "temperature" not in recorder.last

    await provider.complete_structured(_MESSAGES, _Extraction)
    assert "temperature" not in recorder.last

    await provider.complete(_MESSAGES, temperature=0.2)
    assert recorder.last["temperature"] == 0.2


async def test_litellm_omits_temperature_by_default(monkeypatch) -> None:
    from types import SimpleNamespace

    from neo4j_agent_memory.llm.adapters import litellm as adapter

    calls: list[dict[str, Any]] = []

    async def acompletion(**kwargs: Any) -> Any:
        calls.append(kwargs)
        message = SimpleNamespace(content="ok")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            model="m",
            usage=None,
        )

    monkeypatch.setattr(
        adapter, "_ensure_litellm", lambda: SimpleNamespace(acompletion=acompletion)
    )
    provider = adapter.LiteLLMProvider("openai/gpt-5-mini")

    await provider.complete(_MESSAGES)
    assert "temperature" not in calls[-1]

    await provider.complete(_MESSAGES, temperature=0.3)
    assert calls[-1]["temperature"] == 0.3


async def test_instructor_omits_temperature_by_default() -> None:
    from neo4j_agent_memory.llm.adapters.instructor import InstructorProvider

    calls: list[dict[str, Any]] = []

    class _Client:
        async def create(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return _Extraction(value="ok")

    provider = object.__new__(InstructorProvider)
    provider._client = _Client()

    await provider.complete_structured(_MESSAGES, _Extraction)
    assert "temperature" not in calls[-1]

    await provider.complete_structured(_MESSAGES, _Extraction, temperature=0.1)
    assert calls[-1]["temperature"] == 0.1


def test_anthropic_does_not_warn_for_default_temperature(caplog) -> None:
    from neo4j_agent_memory.llm.adapters import anthropic as adapter

    adapter._sampling_warning_emitted = False
    try:
        with caplog.at_level("WARNING", logger=adapter.__name__):
            adapter._note_dropped_temperature(None)
        assert not caplog.records
    finally:
        adapter._sampling_warning_emitted = False


# ---------------------------------------------------------------------------
# Callers default to None
# ---------------------------------------------------------------------------


async def test_llm_extractor_default_sends_no_temperature_to_bedrock() -> None:
    from neo4j_agent_memory.extraction.llm_extractor import LLMEntityExtractor

    provider, client = _bedrock(_SONNET, _tool_response({"entities": []}))
    extractor = LLMEntityExtractor(provider=provider)

    await extractor.extract("Alice works at Acme.")

    assert client.calls, "expected a Converse request"
    assert all("temperature" not in call["inferenceConfig"] for call in client.calls)


async def test_llm_summarizer_default_sends_no_temperature() -> None:
    from neo4j_agent_memory.memory.short_term import _llm_summarizer

    provider, client = _bedrock(_SONNET, _text_response("summary"))

    assert await _llm_summarizer(provider)("transcript") == "summary"
    assert "temperature" not in client.calls[-1]["inferenceConfig"]
