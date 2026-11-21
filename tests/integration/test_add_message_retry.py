"""Integration tests: add_message with a message_id stores the message once."""

from __future__ import annotations

from uuid import uuid4

import pytest

from neo4j_agent_memory.memory.short_term import MessageRole
from tests.integration.test_delete_message import _chain


async def _add(memory_client, session_id: str, content: str, message_id=None):
    return await memory_client.short_term.add_message(
        session_id,
        MessageRole.USER,
        content,
        extract_entities=False,
        generate_embedding=False,
        message_id=message_id,
    )


@pytest.mark.integration
class TestAddMessageRetry:
    @pytest.mark.asyncio
    async def test_a_retry_with_the_same_id_stores_one_message(self, memory_client, session_id):
        await _add(memory_client, session_id, "a")
        message_id = uuid4()
        first = await _add(memory_client, session_id, "b", message_id)
        retry = await _add(memory_client, session_id, "b", str(message_id))

        assert first.id == retry.id == message_id
        assert await _chain(memory_client, session_id) == ["a", "b"]
        await _add(memory_client, session_id, "c")
        assert await _chain(memory_client, session_id) == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_an_id_held_by_another_conversation_is_not_reused(
        self, memory_client, session_id
    ):
        message_id = uuid4()
        await _add(memory_client, session_id, "a", message_id)

        with pytest.raises(Exception, match="already exists"):
            await _add(memory_client, f"{session_id}-other", "b", message_id)
        assert await _chain(memory_client, f"{session_id}-other") == []

    @pytest.mark.asyncio
    async def test_a_retry_returns_the_time_stored_by_the_first_write(
        self, memory_client, session_id
    ):
        message_id = uuid4()
        first = await _add(memory_client, session_id, "b", message_id)
        retry = await _add(memory_client, session_id, "b", message_id)

        conversation = await memory_client.short_term.get_conversation(session_id)
        assert [m.created_at for m in conversation.messages] == [first.created_at]
        assert retry.created_at == first.created_at
