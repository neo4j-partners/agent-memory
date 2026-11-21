"""Shared test doubles for the llm test suite.

The contract test harness and the foundation tests both consume these,
so they live in a private module rather than getting redefined per-file.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from neo4j_agent_memory.llm.types import ChatMessage, Completion, Usage

if TYPE_CHECKING:
    from neo4j_agent_memory.llm.adapters.bedrock import BedrockProvider


class CannedLLMProvider:
    """Minimal :class:`LLMProvider` implementation backed by a queue of strings.

    Returns the next queued response on each :meth:`complete` call. Useful
    for exercising :func:`schema_aligned_extract` retry behavior without
    hitting a real model.
    """

    def __init__(self, responses: Sequence[str], *, model: str = "test/canned") -> None:
        self.model = model
        self._responses = list(responses)
        self.calls: list[list[ChatMessage]] = []

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        timeout: float | None = None,
    ) -> Completion:
        # Record what the caller sent so retry tests can assert the
        # validation-feedback message is appended.
        self.calls.append(list(messages))
        if not self._responses:
            raise RuntimeError("CannedLLMProvider exhausted")
        content = self._responses.pop(0)
        return Completion(
            content=content,
            model=self.model,
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


class CannedEmbeddingProvider:
    """Minimal :class:`EmbeddingProvider` returning deterministic test vectors."""

    def __init__(self, *, model: str = "test/canned-embed", dimensions: int = 4) -> None:
        self.model = model
        self.dimensions = dimensions

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        # Each vector is just [i, 0, 0, ...] of length ``dimensions`` so
        # tests can distinguish individual results without caring about
        # actual semantic content.
        return [[float(i), *([0.0] * (self.dimensions - 1))] for i, _ in enumerate(texts)]

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]


class SimplePayload(BaseModel):
    """Trivial Pydantic model used to exercise structured extraction."""

    name: str
    age: int


# ---------------------------------------------------------------------------
# Recording SDK clients
# ---------------------------------------------------------------------------


class Recorder:
    """Collects the kwargs each fake ``create()`` call received."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @property
    def last(self) -> dict[str, Any]:
        assert self.calls, "expected the adapter to issue a request"
        return self.calls[-1]


class Block:
    def __init__(self, type_: str, **fields: Any) -> None:
        self.type = type_
        for key, value in fields.items():
            setattr(self, key, value)


class _OpenAIUsage:
    prompt_tokens = 5
    completion_tokens = 3
    total_tokens = 8
    prompt_tokens_details = None


class _OpenAIChoice:
    def __init__(self, content: str) -> None:
        self.message = Block("message", content=content)
        self.finish_reason = "stop"


class _OpenAIResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_OpenAIChoice(content)]
        self.model = "gpt-5-mini"
        self.usage = _OpenAIUsage()

    def model_dump(self) -> dict[str, Any]:
        return {}


class _FakeOpenAICompletions:
    def __init__(self, recorder: Recorder, content: str) -> None:
        self._recorder = recorder
        self._content = content

    async def create(self, **kwargs: Any) -> _OpenAIResponse:
        self._recorder.calls.append(kwargs)
        return _OpenAIResponse(self._content)


class _FakeOpenAIEmbeddings:
    def __init__(self, recorder: Recorder, dimensions: int) -> None:
        self._recorder = recorder
        self._dimensions = dimensions

    async def create(self, **kwargs: Any) -> Any:
        self._recorder.calls.append(kwargs)
        inputs = kwargs["input"]
        data = [
            Block("embedding", index=i, embedding=[0.1] * self._dimensions)
            for i in range(len(inputs))
        ]
        return Block("response", data=data)


class FakeOpenAIClient:
    def __init__(self, recorder: Recorder, *, content: str = "hi", dimensions: int = 1536) -> None:
        self.chat = Block("chat", completions=_FakeOpenAICompletions(recorder, content))
        self.embeddings = _FakeOpenAIEmbeddings(recorder, dimensions)


class FakeConverseClient:
    """Stands in for the boto3 bedrock-runtime client.

    Returns the queued responses in order and repeats the last one.
    """

    def __init__(self, *responses: dict[str, Any]) -> None:
        self.calls: list[dict[str, Any]] = []
        self._responses = list(responses)

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]


def converse_response(*blocks: dict[str, Any], stop: str = "end_turn") -> dict[str, Any]:
    """A Converse response whose message holds ``blocks``."""
    return {
        "output": {"message": {"content": list(blocks)}},
        "usage": {},
        "stopReason": stop,
    }


def converse_text(text: str) -> dict[str, Any]:
    """A Converse response with one text block."""
    return converse_response({"text": text})


def converse_tool_use(tool_input: dict[str, Any]) -> dict[str, Any]:
    """A Converse response with one ``submit_extraction`` tool call."""
    use = {"toolUseId": "t1", "name": "submit_extraction", "input": tool_input}
    return converse_response({"toolUse": use}, stop="tool_use")


def bedrock_provider(
    model: str, *responses: dict[str, Any], **kwargs: Any
) -> tuple[BedrockProvider, FakeConverseClient]:
    """A :class:`BedrockProvider` wired to a :class:`FakeConverseClient`."""
    from neo4j_agent_memory.llm.adapters.bedrock import BedrockProvider

    provider = BedrockProvider(model, aws_region="us-west-2", **kwargs)
    client = FakeConverseClient(*(responses or (converse_text("ok"),)))
    provider._client = client
    return provider, client


__all__ = [
    "Block",
    "CannedEmbeddingProvider",
    "CannedLLMProvider",
    "FakeConverseClient",
    "FakeOpenAIClient",
    "Recorder",
    "SimplePayload",
    "bedrock_provider",
    "converse_response",
    "converse_text",
    "converse_tool_use",
]
