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
) -> tuple[list[StrandsMessage], list[StrandsMessage]]:
    """Shape restored text history so roles alternate user, assistant, user.

    Restored messages hold text only, so a tool turn with preamble text
    leaves consecutive assistant messages. Each assistant run is merged
    into one message, with the non-empty texts joined by a blank line.

    Consecutive user messages come only from failed or abandoned turns.
    Only the last message of a user run is kept, so an unanswered prompt
    is never glued onto the next question. Leading assistant messages are
    dropped, so history starts with a user message. A trailing user
    message never got a reply, so it is dropped too.

    Returns the normalized history and the user messages left out, oldest
    first.
    """
    history: list[StrandsMessage] = []
    dropped: list[StrandsMessage] = []
    for message in messages:
        previous = history[-1] if history else None
        if previous is None or previous["role"] != message["role"]:
            history.append(message)
        elif message["role"] == "user":
            dropped.append(previous)
            history[-1] = message
        else:
            parts = (_message_text(previous), _message_text(message))
            text = "\n\n".join(part for part in parts if part)
            history[-1] = {"role": message["role"], "content": [{"text": text}]}
    if history and history[0]["role"] == "assistant":
        history.pop(0)
    if history and history[-1]["role"] == "user":
        dropped.append(history.pop())
    return history, dropped
