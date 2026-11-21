"""BedrockProvider.complete_structured follow-ups to the G fix.

Each test stubs the Converse client and counts the requests it received.
"""

from __future__ import annotations

from pydantic import BaseModel

from neo4j_agent_memory.llm.types import ChatMessage
from tests.unit.llm._mocks import bedrock_provider, converse_response, converse_text

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
