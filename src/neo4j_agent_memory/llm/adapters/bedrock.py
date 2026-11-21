"""Native AWS Bedrock adapter for the Provider Protocol.

Uses Bedrock's Converse API for a unified interface across Anthropic,
Amazon Titan, and Meta Llama models hosted on Bedrock. Treated as a
separate adapter from the Anthropic-direct adapter because:

* The auth model is AWS SigV4, not an API key.
* Configuration (``aws_region``, ``aws_profile``) differs substantively.
* Bedrock-hosted Anthropic models have a different model id format
  (``anthropic.claude-3-5-sonnet-20241022-v2:0``).

Implements:

* :class:`BedrockProvider` — :class:`LLMProvider` + :class:`StructuredExtractor`
* :class:`BedrockEmbeddingProvider` — :class:`EmbeddingProvider` (wraps
  :class:`~neo4j_agent_memory.embeddings.bedrock.BedrockEmbedder`)

Install with::

    pip install 'neo4j-agent-memory[bedrock]'
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from pydantic import BaseModel, ValidationError

from neo4j_agent_memory.llm.defaults import lookup_embedding_dimensions
from neo4j_agent_memory.llm.errors import (
    ProviderAuthError,
    ProviderInvalidRequestError,
    ProviderRateLimitError,
    ProviderServiceError,
    ProviderTimeoutError,
)
from neo4j_agent_memory.llm.structured import schema_aligned_extract
from neo4j_agent_memory.llm.types import ChatMessage, Completion, Usage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from neo4j_agent_memory.embeddings.bedrock import BedrockEmbedder


logger = logging.getLogger(__name__)


T = TypeVar("T", bound=BaseModel)


_DEFAULT_MAX_TOKENS = 4096

_TOOL_CHOICES = ("tool", "auto")


# An optional geography prefix (``us.``, ``us-gov.``, ``global.``), then
# ``anthropic.``.
_ANTHROPIC_MODEL_RE = re.compile(r"^(?:[a-z]+(?:-[a-z]+)*\.)?anthropic\.")

# Bedrock names the rejected field in the message. For a forced choice,
# Anthropic models on Bedrock return 'tool_choice: type "tool" and "any" are
# not supported for this model'. Other models write "toolConfig.toolChoice".
# Bedrock names the field as toolChoice, tool_choice, tool-choice, or tool
# choice. Some messages name only the rejected types.
_TOOL_CHOICE_ERROR_RE = re.compile(r'tool[-_ ]?choice|type "tool" and "any"', re.IGNORECASE)


def _strip_provider_prefix(model: str) -> str:
    if model.startswith("bedrock/"):
        return model[len("bedrock/") :]
    return model


def _is_anthropic_on_bedrock(bare_model: str) -> bool:
    """Whether ``bare_model`` is an Anthropic model on Bedrock.

    Anthropic model ids start with ``anthropic.``. Cross-region inference
    profiles add a geography prefix, such as ``us.``, ``eu.``, ``apac.``
    or ``global.``. For an ARN, the id after the last ``/`` is checked.

    An application inference profile ARN does not name its model, so it
    is not detected. Such a model uses the generic SAP loop.
    """
    return _ANTHROPIC_MODEL_RE.match(bare_model.rsplit("/", 1)[-1]) is not None


def _rejects_tool_choice(exc: Exception) -> bool:
    """Whether ``exc`` is a Bedrock ValidationException that rejects ``toolChoice``."""
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    error = response.get("Error", {})
    if error.get("Code") != "ValidationException":
        return False
    return _TOOL_CHOICE_ERROR_RE.search(error.get("Message", "")) is not None


def _translate_bedrock_exception(exc: Exception) -> Exception:
    """Map a :mod:`botocore.exceptions` exception to a :class:`ProviderError`."""
    try:
        from botocore.exceptions import (
            ClientError,
            ConnectTimeoutError,
            EndpointConnectionError,
            ReadTimeoutError,
        )
    except ImportError:
        return exc

    if isinstance(exc, (ReadTimeoutError, ConnectTimeoutError)):
        return ProviderTimeoutError(str(exc))
    if isinstance(exc, EndpointConnectionError):
        return ProviderServiceError(str(exc))
    if isinstance(exc, ClientError):
        # ClientError has a typed .response dict; getattr appeases type checkers
        # that only see the bare Exception base class.
        response_dict = getattr(exc, "response", {}) or {}
        code = response_dict.get("Error", {}).get("Code", "")
        if code in {
            "UnrecognizedClientException",
            "InvalidSignatureException",
            "AccessDeniedException",
        }:
            return ProviderAuthError(str(exc))
        if code in {"ThrottlingException", "TooManyRequestsException"}:
            return ProviderRateLimitError(str(exc))
        if code in {"ValidationException"}:
            return ProviderInvalidRequestError(str(exc))
        if code in {
            "InternalServerException",
            "ServiceUnavailableException",
            "ModelTimeoutException",
        }:
            return ProviderServiceError(str(exc))
    return exc


def _converse_messages(
    messages: Sequence[ChatMessage],
) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]]]:
    """Split messages into ``(system, messages)`` for the Converse API.

    Bedrock's Converse API expects:
    - ``system=[{"text": "..."}, ...]`` — a list
    - ``messages=[{"role": "user|assistant", "content": [{"text": "..."}]}]``
    """
    system: list[dict[str, Any]] = []
    converse_messages: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "system":
            system.append({"text": msg.content})
            continue
        role = msg.role if msg.role in ("user", "assistant") else "user"
        converse_messages.append({"role": role, "content": [{"text": msg.content}]})
    return (system if system else None), converse_messages


class BedrockProvider:
    """Native Bedrock LLM provider via the Converse API.

    Example::

        provider = BedrockProvider(
            "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
            aws_region="us-west-2",
        )

    Two options adjust the Converse request for models that reject parts
    of it:

    * ``send_temperature``: When False, the provider never sends a
      temperature, even when a caller passes one. A ``temperature`` of
      ``None`` is never sent either way.
    * ``tool_choice``: Controls ``complete_structured`` on Anthropic
      models. ``None`` forces the ``submit_extraction`` tool. If Bedrock
      rejects the forced choice, the provider retries that call with
      ``"auto"``. Once a retry works, it logs one warning and sends
      ``"auto"`` from then on. This switch is a one-way latch shared by
      all calls on the provider. Calls made concurrently before it is set
      can each pay one rejected request. One warning is logged per provider.
      ``"tool"`` always forces the tool and never falls back. ``"auto"``
      lets the model choose. If the model answers in
      text instead, the provider parses that text as JSON. When it does
      not validate, ``schema_aligned_extract`` retries, and the first
      reply counts toward ``max_retries``.

    Claude Sonnet 5.5 and similar models reject both a temperature and a
    forced tool choice. Configure them like this::

        provider = BedrockProvider(
            "bedrock/us.anthropic.claude-sonnet-5-5-v1:0",
            send_temperature=False,
            tool_choice="auto",
        )
    """

    def __init__(
        self,
        model: str,
        *,
        aws_region: str | None = None,
        aws_profile: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        timeout: float = 60.0,
        return_raw: bool = False,
        default_max_tokens: int = _DEFAULT_MAX_TOKENS,
        send_temperature: bool = True,
        tool_choice: Literal["tool", "auto"] | None = None,
    ) -> None:
        if tool_choice is not None and tool_choice not in _TOOL_CHOICES:
            raise ValueError(f"tool_choice must be None, 'tool', or 'auto', not {tool_choice!r}")
        bare = _strip_provider_prefix(model)
        self.model = f"bedrock/{bare}"
        self._bare_model = bare
        self._aws_region = aws_region
        self._aws_profile = aws_profile
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key
        self._timeout = timeout
        self._return_raw = return_raw
        self._default_max_tokens = default_max_tokens
        self._send_temperature = send_temperature
        self._tool_choice = tool_choice
        # Set when Bedrock rejects a forced toolChoice under tool_choice=None.
        # The lock makes the check-and-set atomic for calls on other threads.
        self._fall_back_to_auto = False
        self._fall_back_lock = threading.Lock()
        self._client: Any | None = None

    def _inference_config(self, temperature: float | None, max_tokens: int) -> dict[str, Any]:
        """Build ``inferenceConfig``, leaving temperature out when it must not be sent."""
        config: dict[str, Any] = {"maxTokens": max_tokens}
        if self._send_temperature and temperature is not None:
            config["temperature"] = temperature
        return config

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import boto3
                from botocore.config import Config
            except ImportError as exc:
                raise ImportError(
                    "boto3 not installed. Install with: pip install 'neo4j-agent-memory[bedrock]'"
                ) from exc
            session_kwargs: dict[str, Any] = {}
            if self._aws_profile:
                session_kwargs["profile_name"] = self._aws_profile
            if self._aws_region:
                session_kwargs["region_name"] = self._aws_region
            session = boto3.Session(**session_kwargs)
            client_kwargs: dict[str, Any] = {
                "config": Config(read_timeout=self._timeout),
            }
            if self._aws_access_key_id:
                client_kwargs["aws_access_key_id"] = self._aws_access_key_id
            if self._aws_secret_access_key:
                client_kwargs["aws_secret_access_key"] = self._aws_secret_access_key
            self._client = session.client("bedrock-runtime", **client_kwargs)
        return self._client

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        timeout: float | None = None,
    ) -> Completion:
        client = self._ensure_client()
        system, conv_messages = _converse_messages(messages)
        inference_config = self._inference_config(
            temperature,
            max_tokens if max_tokens is not None else self._default_max_tokens,
        )
        if stop is not None:
            inference_config["stopSequences"] = list(stop)
        kwargs: dict[str, Any] = {
            "modelId": self._bare_model,
            "messages": conv_messages,
            "inferenceConfig": inference_config,
        }
        if system is not None:
            kwargs["system"] = system

        try:
            # boto3 is sync; run in a thread.
            response = await asyncio.to_thread(client.converse, **kwargs)
        except Exception as exc:
            translated = _translate_bedrock_exception(exc)
            if translated is exc:
                raise
            raise translated from exc

        output_message = response.get("output", {}).get("message", {})
        content_blocks = output_message.get("content", [])
        text_parts = [b.get("text", "") for b in content_blocks if "text" in b]
        usage_data = response.get("usage") or {}
        return Completion(
            content="".join(text_parts),
            model=self._bare_model,
            usage=Usage(
                prompt_tokens=usage_data.get("inputTokens", 0) or 0,
                completion_tokens=usage_data.get("outputTokens", 0) or 0,
                total_tokens=usage_data.get("totalTokens", 0) or 0,
            ),
            finish_reason=response.get("stopReason"),
            raw=response if self._return_raw else None,
        )

    async def complete_structured(
        self,
        messages: Sequence[ChatMessage],
        response_model: type[T],
        *,
        temperature: float | None = None,
        max_retries: int = 2,
        timeout: float | None = None,
    ) -> T:
        """Use Converse toolConfig for Anthropic-on-Bedrock; SAP for everything else.

        Only Anthropic models on Bedrock reliably support forced tool use
        through Converse. For Titan and Llama models we delegate to the
        generic SAP retry loop. When the model replies without a valid
        ``submit_extraction`` call, which ``tool_choice="auto"`` allows,
        the SAP loop takes over. SAP first validates that reply, either
        the reply text or the tool input. It counts as the first of the
        ``max_retries + 1`` attempts.
        """
        if not _is_anthropic_on_bedrock(self._bare_model):
            return await schema_aligned_extract(
                self,
                messages,
                response_model,
                temperature=temperature,
                max_retries=max_retries,
                timeout=timeout,
            )

        client = self._ensure_client()
        system, conv_messages = _converse_messages(messages)
        schema = response_model.model_json_schema()
        tool_spec = {
            "toolSpec": {
                "name": "submit_extraction",
                "description": (
                    f"Submit a structured extraction matching the {response_model.__name__} schema."
                ),
                "inputSchema": {"json": schema},
            }
        }
        kwargs: dict[str, Any] = {
            "modelId": self._bare_model,
            "messages": conv_messages,
            "inferenceConfig": self._inference_config(temperature, self._default_max_tokens),
            "toolConfig": {
                "tools": [tool_spec],
                "toolChoice": (
                    {"auto": {}}
                    if self._tool_choice == "auto" or self._fall_back_to_auto
                    else {"tool": {"name": "submit_extraction"}}
                ),
            },
        }
        if system is not None:
            kwargs["system"] = system

        try:
            response = await self._converse_with_tool(client, kwargs)
        except Exception as exc:
            translated = _translate_bedrock_exception(exc)
            if translated is exc:
                raise
            raise translated from exc

        content = response.get("output", {}).get("message", {}).get("content", [])
        tool_uses = [block["toolUse"] for block in content if "toolUse" in block]
        if tool_uses:
            tool_input = tool_uses[0].get("input", {})
            try:
                return response_model.model_validate(tool_input)
            except ValidationError as exc:
                logger.debug(
                    "Bedrock tool-use response failed validation; retrying with SAP: %s",
                    exc,
                )
            # A string input is already the reply text. botocore decodes any
            # other input from JSON, so it encodes back as is.
            first_reply = tool_input if isinstance(tool_input, str) else json.dumps(tool_input)
        else:
            # tool_choice="auto" lets the model answer in text. SAP checks
            # that text first, so a valid answer costs no second request.
            first_reply = "".join(block.get("text", "") for block in content if "text" in block)

        return await schema_aligned_extract(
            self,
            messages,
            response_model,
            temperature=temperature,
            max_retries=max_retries,
            timeout=timeout,
            first_reply=first_reply,
        )

    async def _converse_with_tool(self, client: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Send a tool-use Converse request, falling back to ``toolChoice`` auto.

        Only the default ``tool_choice=None`` falls back, and only when
        Bedrock rejects the forced ``toolChoice`` itself. Any other error
        is raised unchanged. The provider switches to auto only after the
        auto retry succeeds. If the retry fails, its error is raised with
        the rejection as its cause, and the next call is forced again.
        """
        try:
            return await asyncio.to_thread(client.converse, **kwargs)
        except Exception as exc:
            forced = "tool" in kwargs["toolConfig"]["toolChoice"]
            if self._tool_choice is not None or not forced or not _rejects_tool_choice(exc):
                raise
            rejection = exc
        auto = {**kwargs, "toolConfig": {**kwargs["toolConfig"], "toolChoice": {"auto": {}}}}
        try:
            response: dict[str, Any] = await asyncio.to_thread(client.converse, **auto)
        except Exception as exc:
            raise exc from rejection
        with self._fall_back_lock:
            first = not self._fall_back_to_auto
            self._fall_back_to_auto = True
        if first:
            logger.warning(
                "Bedrock model %s rejected a forced toolChoice, so this provider "
                "now sends toolChoice auto. Pass tool_choice='auto' to skip the "
                "rejected call. Error: %s",
                self._bare_model,
                rejection,
            )
        return response


class BedrockEmbeddingProvider:
    """Bedrock embedding provider.

    Wraps the existing
    :class:`~neo4j_agent_memory.embeddings.bedrock.BedrockEmbedder` for the
    new Protocol surface.

    Implements :class:`~neo4j_agent_memory.llm.protocol.EmbeddingProvider`.
    """

    def __init__(
        self,
        model: str = "bedrock/amazon.titan-embed-text-v2:0",
        *,
        aws_region: str | None = None,
        aws_profile: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        dimensions: int | None = None,
        batch_size: int = 25,
        normalize: bool = True,
    ) -> None:
        bare = _strip_provider_prefix(model)
        self.model = f"bedrock/{bare}"
        self._bare_model = bare
        self._aws_region = aws_region
        self._aws_profile = aws_profile
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key
        self._batch_size = batch_size
        self._normalize = normalize
        self._underlying: BedrockEmbedder | None = None

        if dimensions is not None:
            self.dimensions = dimensions
        else:
            known = lookup_embedding_dimensions(self.model)
            if known is None:
                raise ValueError(
                    f"Could not determine dimensions for Bedrock model {self.model!r}. "
                    f"Pass dimensions=N explicitly or use a model in the defaults table."
                )
            self.dimensions = known

    def _ensure_underlying(self) -> BedrockEmbedder:
        if self._underlying is None:
            try:
                from neo4j_agent_memory.embeddings.bedrock import BedrockEmbedder
            except ImportError as exc:
                raise ImportError(
                    "boto3 not installed. Install with: pip install 'neo4j-agent-memory[bedrock]'"
                ) from exc
            self._underlying = BedrockEmbedder(
                model=self._bare_model,
                region_name=self._aws_region,
                profile_name=self._aws_profile,
                aws_access_key_id=self._aws_access_key_id,
                aws_secret_access_key=self._aws_secret_access_key,
                batch_size=self._batch_size,
                normalize=self._normalize,
            )
        return self._underlying

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        underlying = self._ensure_underlying()
        return await underlying.embed_batch(list(texts))

    async def embed_one(self, text: str) -> list[float]:
        underlying = self._ensure_underlying()
        return await underlying.embed(text)


__all__ = [
    "BedrockProvider",
    "BedrockEmbeddingProvider",
]
