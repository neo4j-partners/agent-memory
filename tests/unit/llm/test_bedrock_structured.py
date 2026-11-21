"""BedrockProvider.complete_structured follow-ups to the G fix.

Each test stubs the Converse client and counts the requests it received.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from neo4j_agent_memory.llm.errors import StructuredExtractionError
from neo4j_agent_memory.llm.types import ChatMessage
from tests.unit.llm._mocks import (
    bedrock_provider,
    converse_response,
    converse_text,
    converse_tool_use,
)

_SONNET = "bedrock/us.anthropic.claude-sonnet-5-5-v1:0"
_TITAN = "bedrock/amazon.titan-text-premier-v1:0"
_MESSAGES = [ChatMessage(role="user", content="hi")]


class _Extraction(BaseModel):
    value: str


async def test_sap_retry_never_sends_an_empty_assistant_block() -> None:
    provider, client = bedrock_provider(
        _TITAN, converse_response(), converse_text('{"value": "ok"}')
    )

    result = await provider.complete_structured(_MESSAGES, _Extraction)

    assert result.value == "ok"
    retry = client.calls[1]["messages"]
    texts = [b["text"] for m in retry if m["role"] == "assistant" for b in m["content"]]
    assert texts and all(t.strip() for t in texts)


def _sonnet(*responses: dict[str, Any]) -> Any:
    return bedrock_provider(_SONNET, *responses, tool_choice="auto", send_temperature=False)


def _assistant_texts(call: dict[str, Any]) -> list[str]:
    return [b["text"] for m in call["messages"] if m["role"] == "assistant" for b in m["content"]]


# ---------------------------------------------------------------------------
# tool_choice="auto": a text reply is parsed before any retry
# ---------------------------------------------------------------------------


async def test_valid_text_json_reply_costs_one_request() -> None:
    provider, client = _sonnet(
        converse_text('{"value": "first"}'), converse_text('{"value": "second"}')
    )

    result = await provider.complete_structured(_MESSAGES, _Extraction)

    assert result.value == "first"
    assert len(client.calls) == 1


@pytest.mark.parametrize("max_retries", [0, 1, 2])
async def test_invalid_text_reply_counts_against_max_retries(max_retries: int) -> None:
    provider, client = _sonnet(converse_text("not json"))

    with pytest.raises(StructuredExtractionError) as info:
        await provider.complete_structured(_MESSAGES, _Extraction, max_retries=max_retries)

    assert len(client.calls) == max_retries + 1
    assert len(info.value.last_attempts) == max_retries + 1


async def test_retry_is_seeded_with_the_first_reply() -> None:
    provider, client = _sonnet(converse_text("not json"), converse_text('{"value": "ok"}'))

    result = await provider.complete_structured(_MESSAGES, _Extraction)

    assert result.value == "ok"
    assert len(client.calls) == 2
    retry = client.calls[1]
    assert "toolConfig" not in retry
    assert _assistant_texts(retry) == ["not json"]
    assert "not valid JSON" in retry["messages"][-1]["content"][0]["text"]


_MIXED = converse_response(
    {"text": "Sure, here:"},
    {"toolUse": {"toolUseId": "t1", "name": "submit_extraction", "input": {"value": "ok"}}},
    stop="tool_use",
)


@pytest.mark.parametrize(
    ("first", "expected_calls", "retry_text"),
    [
        pytest.param(_MIXED, 1, None, id="mixed-text-and-tool-use"),
        pytest.param(converse_response(), 2, "[empty response]", id="empty-reply"),
        pytest.param(converse_tool_use({"v": 1}), 2, '{"v": 1}', id="invalid-tool-input"),
    ],
)
async def test_auto_reply_shapes(
    first: dict[str, Any], expected_calls: int, retry_text: str | None
) -> None:
    provider, client = _sonnet(first, converse_text('{"value": "ok"}'))

    result = await provider.complete_structured(_MESSAGES, _Extraction)

    assert result.value == "ok"
    assert len(client.calls) == expected_calls
    if retry_text is not None:
        assert _assistant_texts(client.calls[1]) == [retry_text]


# ---------------------------------------------------------------------------
# Anthropic detection across inference-profile prefixes and ARNs
# ---------------------------------------------------------------------------

_ARN = "arn:aws:bedrock:us-west-2:123456789012"


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("anthropic.claude-3-5-sonnet-20241022-v2:0", True),
        ("us.anthropic.claude-sonnet-5-5-v1:0", True),
        ("global.anthropic.claude-sonnet-5-5-v1:0", True),
        ("eu.anthropic.claude-sonnet-5-5-v1:0", True),
        ("apac.anthropic.claude-sonnet-5-5-v1:0", True),
        ("jp.anthropic.claude-sonnet-5-5-v1:0", True),
        ("au.anthropic.claude-sonnet-5-5-v1:0", True),
        ("us-gov.anthropic.claude-sonnet-5-5-v1:0", True),
        (f"{_ARN}:inference-profile/global.anthropic.claude-sonnet-5-5-v1:0", True),
        (f"{_ARN}::foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0", True),
        ("amazon.titan-text-premier-v1:0", False),
        ("us.meta.llama3-3-70b-instruct-v1:0", False),
        ("global.amazon.nova-pro-v1:0", False),
        ("us.notanthropic.model-v1:0", False),
        (f"{_ARN}:inference-profile/us.amazon.nova-pro-v1:0", False),
        (f"{_ARN}:application-inference-profile/abc123", False),
    ],
)
def test_is_anthropic_on_bedrock(model: str, expected: bool) -> None:
    from neo4j_agent_memory.llm.adapters.bedrock import _is_anthropic_on_bedrock

    assert _is_anthropic_on_bedrock(model) is expected


async def test_global_profile_honors_tool_choice() -> None:
    provider, client = bedrock_provider(
        "bedrock/global.anthropic.claude-sonnet-5-5-v1:0",
        converse_text('{"value": "ok"}'),
        tool_choice="auto",
    )

    await provider.complete_structured(_MESSAGES, _Extraction)

    assert client.calls[0]["toolConfig"]["toolChoice"] == {"auto": {}}
