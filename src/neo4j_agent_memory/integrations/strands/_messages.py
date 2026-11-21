"""Strands ↔ neo4j-agent-memory message mapping (pure functions).

One concern: converting between Strands' content-block message dicts and
the library's stored ``Message`` model. No session state, no I/O.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from strands.types.content import Message as StrandsMessage

    from neo4j_agent_memory.memory.short_term import Message as StoredMessage


def _message_text(message: StrandsMessage) -> str:
    """Concatenate the text blocks of a Strands message (tool blocks ignored)."""
    blocks = message.get("content") or []
    texts = [b["text"] for b in blocks if isinstance(b, dict) and b.get("text")]
    return "\n".join(texts)


def _to_strands_message(stored: StoredMessage) -> StrandsMessage:
    """Convert a stored neo4j-agent-memory Message to a Strands message dict."""
    role = stored.role.value if hasattr(stored.role, "value") else str(stored.role)
    if role not in ("user", "assistant"):
        role = "assistant"
    return {
        "role": cast("Literal['user', 'assistant']", role),
        "content": [{"text": stored.content}],
    }


def _normalize_history(
    messages: list[StrandsMessage],
) -> tuple[list[StrandsMessage], StrandsMessage | None]:
    """Shape restored text history so roles alternate user, assistant, user.

    Restored messages hold text only, so a tool turn with preamble text
    leaves consecutive assistant messages, and a failed turn leaves
    consecutive user messages. Each same-role run is merged into one
    message, with the non-empty texts joined by a blank line. Leading assistant
    messages are dropped, so history starts with a user message. A
    trailing user message never got a reply, so it is dropped too.

    Returns the normalized history and the dropped trailing user message,
    or None when nothing was dropped from the end.
    """
    merged: list[StrandsMessage] = []
    for message in messages:
        if merged and merged[-1]["role"] == message["role"]:
            parts = (_message_text(merged[-1]), _message_text(message))
            text = "\n\n".join(part for part in parts if part)
            merged[-1] = {"role": message["role"], "content": [{"text": text}]}
        else:
            merged.append(message)
    if merged and merged[0]["role"] == "assistant":
        merged.pop(0)
    trailing = merged.pop() if merged and merged[-1]["role"] == "user" else None
    return merged, trailing
