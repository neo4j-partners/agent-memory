"""Integration tests: ShortTermMemory.delete_message keeps the message chain intact."""

from __future__ import annotations

from uuid import uuid4

import pytest

from neo4j_agent_memory.memory.short_term import MessageRole

CHAIN_STATE = """
MATCH (c:Conversation {session_id: $session_id})
OPTIONAL MATCH (c)-[:FIRST_MESSAGE]->(first:Message)
WITH c, collect(first.content) AS firsts
OPTIONAL MATCH (c)-[:HAS_MESSAGE]->(m:Message)
OPTIONAL MATCH (m)-[:NEXT_MESSAGE]->(n:Message)
WITH firsts, m, collect(n.content) AS nexts
RETURN firsts, collect({content: m.content, nexts: nexts}) AS messages
"""


async def _add(memory_client, session_id: str, contents: list[str]) -> list:
    return [
        await memory_client.short_term.add_message(
            session_id,
            MessageRole.USER,
            content,
            extract_entities=False,
            generate_embedding=False,
        )
        for content in contents
    ]


async def _chain(memory_client, session_id: str) -> list[str]:
    """Walk FIRST_MESSAGE then NEXT_MESSAGE, checking the chain invariants."""
    rows = await memory_client._client.execute_read(CHAIN_STATE, {"session_id": session_id})
    firsts = rows[0]["firsts"]
    messages = [m for m in rows[0]["messages"] if m["content"] is not None]
    if not messages:
        assert firsts == []
        return []
    assert len(firsts) == 1, f"expected one FIRST_MESSAGE, got {firsts}"
    successors = {}
    for message in messages:
        assert len(message["nexts"]) <= 1, f"forked chain at {message['content']}"
        successors[message["content"]] = message["nexts"][0] if message["nexts"] else None
    order = [firsts[0]]
    while (following := successors[order[-1]]) is not None:
        order.append(following)
    assert len(order) == len(messages), "chain does not reach every message"
    conversation = await memory_client.short_term.get_conversation(session_id)
    assert [m.content for m in conversation.messages] == order
    return order


@pytest.mark.integration
class TestDeleteMessageChain:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("index", "expected"),
        [
            (0, ["b", "c"]),
            (1, ["a", "c"]),
            (2, ["a", "b"]),
        ],
        ids=["head", "middle", "tail"],
    )
    async def test_delete_relinks_chain(self, memory_client, session_id, index, expected):
        stored = await _add(memory_client, session_id, ["a", "b", "c"])

        assert await memory_client.short_term.delete_message(stored[index].id) is True

        assert await _chain(memory_client, session_id) == expected
        await _add(memory_client, session_id, ["d"])
        assert await _chain(memory_client, session_id) == [*expected, "d"]

    @pytest.mark.asyncio
    async def test_delete_only_message(self, memory_client, session_id):
        stored = await _add(memory_client, session_id, ["only"])

        assert await memory_client.short_term.delete_message(stored[0].id) is True

        assert await _chain(memory_client, session_id) == []
        await _add(memory_client, session_id, ["next"])
        assert await _chain(memory_client, session_id) == ["next"]

    @pytest.mark.asyncio
    async def test_delete_unknown_message_returns_false(self, memory_client):
        assert await memory_client.short_term.delete_message(uuid4()) is False
        assert await memory_client.short_term.delete_message("not-a-uuid") is False


_LINKS = {
    "MENTIONS": ("Entity", "(m)-[:MENTIONS]->(x)"),
    "EXTRACTED_FROM": ("Entity", "(x)-[:EXTRACTED_FROM]->(m)"),
    "INITIATED_BY": ("ReasoningTrace", "(x)-[:INITIATED_BY]->(m)"),
    "TRIGGERED_BY": ("ToolCall", "(x)-[:TRIGGERED_BY]->(m)"),
}


async def _link(memory_client, message_id: str, rel_type: str) -> None:
    label, pattern = _LINKS[rel_type]
    await memory_client._client.execute_write(
        f"MATCH (m:Message {{id: $id}}) CREATE (x:{label} {{id: $xid, name: 'linked'}}) "
        f"CREATE {pattern}",
        {"id": message_id, "xid": str(uuid4())},
    )


@pytest.mark.integration
class TestDeleteMessageCascade:
    @pytest.mark.asyncio
    async def test_cascade_removes_links_and_keeps_linked_nodes(self, memory_client, session_id):
        stored = await _add(memory_client, session_id, ["a", "b"])
        for rel_type in _LINKS:
            await _link(memory_client, str(stored[1].id), rel_type)

        assert await memory_client.short_term.delete_message(stored[1].id) is True

        assert await _chain(memory_client, session_id) == ["a"]
        rows = await memory_client._client.execute_read(
            "MATCH (x {name: 'linked'}) RETURN count(x) AS n", {}
        )
        assert rows[0]["n"] == len(_LINKS)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("rel_type", list(_LINKS))
    async def test_no_cascade_refuses_linked_message_and_changes_nothing(
        self, memory_client, session_id, rel_type
    ):
        stored = await _add(memory_client, session_id, ["a", "b", "c"])
        await _link(memory_client, str(stored[1].id), rel_type)

        with pytest.raises(ValueError, match=rel_type):
            await memory_client.short_term.delete_message(stored[1].id, cascade=False)

        assert await _chain(memory_client, session_id) == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_no_cascade_deletes_unlinked_message(self, memory_client, session_id):
        stored = await _add(memory_client, session_id, ["a", "b", "c"])

        assert await memory_client.short_term.delete_message(stored[1].id, cascade=False) is True

        assert await _chain(memory_client, session_id) == ["a", "c"]
