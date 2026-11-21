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
from tests.unit.llm._mocks import (
    FakeOpenAIClient,
    Recorder,
    bedrock_provider,
    converse_text,
    converse_tool_use,
)

_SONNET = "bedrock/us.anthropic.claude-sonnet-5-5-v1:0"
_TITAN = "bedrock/amazon.titan-text-premier-v1:0"
_MESSAGES = [ChatMessage(role="user", content="hi")]


class _Extraction(BaseModel):
    value: str


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
async def test_bedrock_complete_temperature(
    send_temperature: bool, temperature: float | None, expected: float | None
) -> None:
    provider, client = bedrock_provider(_SONNET, send_temperature=send_temperature)

    await provider.complete(_MESSAGES, temperature=temperature)

    config = client.calls[-1]["inferenceConfig"]
    assert config.get("temperature") == expected
    assert ("temperature" in config) is (expected is not None)
    assert "maxTokens" in config


async def test_bedrock_complete_default_sends_no_temperature() -> None:
    provider, client = bedrock_provider(_SONNET)

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
    tool_choice: str | None,
    expected_choice: dict[str, Any],
    send_temperature: bool,
    temperature: float | None,
    expected_temperature: float | None,
) -> None:
    provider, client = bedrock_provider(
        _SONNET,
        converse_tool_use({"value": "ok"}),
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
    provider, client = bedrock_provider(
        _SONNET,
        converse_text("no tool call"),
        converse_text('{"value": "ok"}'),
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
    provider, client = bedrock_provider(
        _TITAN, converse_text('{"value": "ok"}'), send_temperature=False
    )

    await provider.complete_structured(_MESSAGES, _Extraction, temperature=0.0)

    assert "temperature" not in client.calls[-1]["inferenceConfig"]


def test_bedrock_rejects_unknown_tool_choice() -> None:
    from neo4j_agent_memory.llm.adapters.bedrock import BedrockProvider

    with pytest.raises(ValueError, match="tool_choice"):
        BedrockProvider(_SONNET, tool_choice="any")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# BedrockProvider falls back to toolChoice auto when a forced choice is rejected
# ---------------------------------------------------------------------------

_GLOBAL_SONNET = "bedrock/global.anthropic.claude-sonnet-5-5-v1:0"
_FORCED = {"tool": {"name": "submit_extraction"}}
_AUTO = {"auto": {}}


def _validation_error(message: str) -> Exception:
    from botocore.exceptions import ClientError

    return ClientError({"Error": {"Code": "ValidationException", "Message": message}}, "Converse")


class _RejectsForcedToolChoice:
    """A Converse client that rejects a forced toolChoice and accepts auto."""

    def __init__(self, rejection: str) -> None:
        self.calls: list[dict[str, Any]] = []
        self._rejection = rejection

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if kwargs["toolConfig"]["toolChoice"] != _AUTO:
            raise _validation_error(self._rejection)
        return converse_tool_use({"value": "ok"})


_TOOL_CHOICE_REJECTION = (
    "This model doesn't support the toolConfig.toolChoice.tool field. "
    "Remove toolConfig.toolChoice.tool and try again."
)


def _rejecting_provider(rejection: str, **kwargs: Any) -> tuple[Any, _RejectsForcedToolChoice]:
    from neo4j_agent_memory.llm.adapters.bedrock import BedrockProvider

    provider = BedrockProvider(_GLOBAL_SONNET, aws_region="us-west-2", **kwargs)
    client = _RejectsForcedToolChoice(rejection)
    provider._client = client
    return provider, client


async def test_bedrock_rejected_forced_tool_choice_falls_back_to_auto(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from neo4j_agent_memory.llm.adapters import bedrock as adapter

    provider, client = _rejecting_provider(_TOOL_CHOICE_REJECTION)

    with caplog.at_level("WARNING", logger=adapter.__name__):
        first = await provider.complete_structured(_MESSAGES, _Extraction)
        second = await provider.complete_structured(_MESSAGES, _Extraction)

    assert (first.value, second.value) == ("ok", "ok")
    choices = [call["toolConfig"]["toolChoice"] for call in client.calls]
    assert choices == [_FORCED, _AUTO, _AUTO]
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "toolChoice" in warnings[0].getMessage()


# The message Bedrock returns for a forced choice on Claude Sonnet 5.5.
_BEDROCK_TOOL_CHOICE_MESSAGE = 'tool_choice: type "tool" and "any" are not supported for this model'


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (_BEDROCK_TOOL_CHOICE_MESSAGE, True),
        (_TOOL_CHOICE_REJECTION, True),
        ("The tool choice is not supported for this model.", True),
        ("Input is too long for requested model.", False),
    ],
)
def test_bedrock_recognizes_a_tool_choice_rejection(message: str, expected: bool) -> None:
    from neo4j_agent_memory.llm.adapters.bedrock import _rejects_tool_choice

    assert _rejects_tool_choice(_validation_error(message)) is expected


async def test_bedrock_real_tool_choice_message_falls_back_to_auto() -> None:
    provider, client = _rejecting_provider(_BEDROCK_TOOL_CHOICE_MESSAGE)

    result = await provider.complete_structured(_MESSAGES, _Extraction)

    assert result.value == "ok"
    choices = [call["toolConfig"]["toolChoice"] for call in client.calls]
    assert choices == [_FORCED, _AUTO]


async def test_bedrock_failed_auto_retry_keeps_the_forced_choice(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A toolChoice-mentioning error that auto also fails does not switch to auto."""
    from neo4j_agent_memory.llm.adapters import bedrock as adapter
    from neo4j_agent_memory.llm.errors import ProviderInvalidRequestError

    provider, client = _rejecting_provider("unused")
    rejection = _validation_error("toolChoice could not be applied to this request.")
    retry_error = _validation_error("toolChoice auto could not be applied either.")
    errors = iter([rejection, retry_error])

    def converse(**kwargs: Any) -> dict[str, Any]:
        client.calls.append(kwargs)
        raise next(errors)

    client.converse = converse  # type: ignore[method-assign]

    with (
        caplog.at_level("WARNING", logger=adapter.__name__),
        pytest.raises(ProviderInvalidRequestError) as info,
    ):
        await provider.complete_structured(_MESSAGES, _Extraction)

    choices = [call["toolConfig"]["toolChoice"] for call in client.calls]
    assert choices == [_FORCED, _AUTO]
    assert provider._fall_back_to_auto is False
    assert not [r for r in caplog.records if r.levelname == "WARNING"]
    assert info.value.__cause__ is retry_error
    assert retry_error.__cause__ is rejection


async def test_bedrock_other_validation_errors_do_not_fall_back() -> None:
    from neo4j_agent_memory.llm.errors import ProviderInvalidRequestError

    provider, client = _rejecting_provider("Input is too long for requested model.")

    with pytest.raises(ProviderInvalidRequestError, match="too long"):
        await provider.complete_structured(_MESSAGES, _Extraction)
    assert len(client.calls) == 1

    with pytest.raises(ProviderInvalidRequestError, match="too long"):
        await provider.complete_structured(_MESSAGES, _Extraction)
    choices = [call["toolConfig"]["toolChoice"] for call in client.calls]
    assert choices == [_FORCED, _FORCED]  # the next call is still forced


async def test_bedrock_explicit_forced_tool_choice_does_not_fall_back() -> None:
    from neo4j_agent_memory.llm.errors import ProviderInvalidRequestError

    provider, client = _rejecting_provider(_TOOL_CHOICE_REJECTION, tool_choice="tool")

    with pytest.raises(ProviderInvalidRequestError, match="toolChoice"):
        await provider.complete_structured(_MESSAGES, _Extraction)

    assert len(client.calls) == 1


# ---------------------------------------------------------------------------
# Other adapters omit temperature when it is None
# ---------------------------------------------------------------------------


async def test_openai_omits_temperature_by_default() -> None:
    pytest.importorskip("openai")
    from neo4j_agent_memory.llm.adapters.openai import OpenAIProvider

    recorder = Recorder()
    provider = OpenAIProvider("openai/gpt-5-mini", api_key="x")
    provider._client = FakeOpenAIClient(recorder, content='{"value": "ok"}')

    await provider.complete(_MESSAGES)
    assert "temperature" not in recorder.last

    await provider.complete_structured(_MESSAGES, _Extraction)
    assert "temperature" not in recorder.last

    await provider.complete(_MESSAGES, temperature=0.2)
    assert recorder.last["temperature"] == 0.2


async def test_litellm_omits_temperature_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
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


def _record_litellm(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Stub LiteLLM's ``acompletion`` and return the list of its kwargs."""
    from types import SimpleNamespace

    from neo4j_agent_memory.llm.adapters import litellm as adapter

    calls: list[dict[str, Any]] = []

    async def acompletion(**kwargs: Any) -> Any:
        calls.append(kwargs)
        message = SimpleNamespace(content='{"value": "ok"}')
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            model="m",
            usage=None,
        )

    monkeypatch.setattr(
        adapter, "_ensure_litellm", lambda: SimpleNamespace(acompletion=acompletion)
    )
    return calls


async def test_litellm_honors_send_temperature_false(monkeypatch: pytest.MonkeyPatch) -> None:
    from neo4j_agent_memory.llm.adapters.litellm import LiteLLMProvider

    calls = _record_litellm(monkeypatch)
    provider = LiteLLMProvider(
        "bedrock/us.anthropic.claude-sonnet-5-5-v1:0", send_temperature=False
    )

    await provider.complete(_MESSAGES, temperature=0.3)
    await provider.complete_structured(_MESSAGES, _Extraction, temperature=0.3)

    assert all("temperature" not in call for call in calls)
    assert all("send_temperature" not in call for call in calls)


async def test_litellm_send_temperature_false_drops_constructor_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from neo4j_agent_memory.llm.adapters.litellm import LiteLLMProvider

    calls = _record_litellm(monkeypatch)
    provider = LiteLLMProvider(
        "bedrock/us.anthropic.claude-sonnet-5-5-v1:0", send_temperature=False, temperature=0.2
    )

    await provider.complete(_MESSAGES)
    await provider.complete_structured(_MESSAGES, _Extraction)

    assert all("temperature" not in call for call in calls)


async def test_litellm_fallback_drops_bedrock_only_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from neo4j_agent_memory.llm import factory

    calls = _record_litellm(monkeypatch)
    monkeypatch.setattr(factory, "_has", lambda extra: extra == "litellm")
    provider = factory.from_provider(
        "bedrock/us.anthropic.claude-sonnet-5-5-v1:0",
        send_temperature=False,
        tool_choice="auto",
    )

    await provider.complete(_MESSAGES, temperature=0.3)

    assert "tool_choice" not in calls[-1]
    assert "send_temperature" not in calls[-1]
    assert "temperature" not in calls[-1]


def test_litellm_fallback_rejects_unknown_tool_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    from neo4j_agent_memory.llm import factory

    monkeypatch.setattr(factory, "_has", lambda extra: extra == "litellm")

    with pytest.raises(ValueError, match="tool_choice"):
        factory.from_provider("bedrock/us.anthropic.claude-sonnet-5-5-v1:0", tool_choice="any")


def test_litellm_fallback_warns_once_about_dropped_tool_choice(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from neo4j_agent_memory.llm import factory

    monkeypatch.setattr(factory, "_has", lambda extra: extra == "litellm")
    monkeypatch.setattr(factory, "_tool_choice_drop_warned", False, raising=False)

    with caplog.at_level("WARNING", logger=factory.__name__):
        for _ in range(2):
            factory.from_provider("bedrock/us.anthropic.claude-sonnet-5-5-v1:0", tool_choice="auto")

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "tool_choice" in warnings[0].getMessage()


_NATIVE_ROUTES = [
    ("openai", "openai/gpt-5", "OpenAIProvider"),
    ("anthropic", "anthropic/claude-sonnet-5-5", "AnthropicProvider"),
]


def _native_route(monkeypatch: pytest.MonkeyPatch, extra: str, class_name: str) -> list[str]:
    """Route ``from_provider`` to a stub of the native ``class_name``.

    Like the real class, the stub takes no ``tool_choice``. Returns the
    models it was built with.
    """
    import importlib

    from neo4j_agent_memory.llm import factory

    built: list[str] = []

    class _Native:
        def __init__(self, model: str, *, api_key: str | None = None) -> None:
            built.append(model)

    adapter = importlib.import_module(f"neo4j_agent_memory.llm.adapters.{extra}")
    monkeypatch.setattr(adapter, class_name, _Native)
    monkeypatch.setattr(factory, "_has", lambda name: name == extra)
    monkeypatch.setattr(factory, "_tool_choice_drop_warned", False, raising=False)
    return built


@pytest.mark.parametrize(("extra", "model", "class_name"), _NATIVE_ROUTES)
def test_native_routes_warn_once_about_dropped_tool_choice(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    extra: str,
    model: str,
    class_name: str,
) -> None:
    from neo4j_agent_memory.llm import factory

    built = _native_route(monkeypatch, extra, class_name)

    with caplog.at_level("WARNING", logger=factory.__name__):
        for _ in range(2):
            factory.from_provider(model, tool_choice="auto")

    assert built == [model, model]
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert class_name in warnings[0].getMessage()


@pytest.mark.parametrize(("extra", "model", "class_name"), _NATIVE_ROUTES)
def test_native_routes_reject_unknown_tool_choice(
    monkeypatch: pytest.MonkeyPatch, extra: str, model: str, class_name: str
) -> None:
    from neo4j_agent_memory.llm import factory

    _native_route(monkeypatch, extra, class_name)

    with pytest.raises(ValueError, match="tool_choice"):
        factory.from_provider(model, tool_choice="any")


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


def test_anthropic_does_not_warn_for_default_temperature(
    caplog: pytest.LogCaptureFixture,
) -> None:
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

    provider, client = bedrock_provider(_SONNET, converse_tool_use({"entities": []}))
    extractor = LLMEntityExtractor(provider=provider)

    await extractor.extract("Alice works at Acme.")

    assert client.calls, "expected a Converse request"
    assert all("temperature" not in call["inferenceConfig"] for call in client.calls)


async def test_llm_summarizer_default_sends_no_temperature() -> None:
    from neo4j_agent_memory.memory.short_term import _llm_summarizer

    provider, client = bedrock_provider(_SONNET, converse_text("summary"))

    assert await _llm_summarizer(provider)("transcript") == "summary"
    assert "temperature" not in client.calls[-1]["inferenceConfig"]
