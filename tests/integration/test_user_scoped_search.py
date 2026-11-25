"""Integration tests for user-scoped long-term search.

``search_entities(user_identifier=...)`` returns only the entities mentioned
in that user's conversations, reached through
``(:Conversation {user_identifier})-[:HAS_MESSAGE]->(:Message)-[:MENTIONS]->(:Entity)``.
Without ``user_identifier`` it searches every user's entities, as before.

``add_fact(user_identifier=...)`` links the fact with
``(:User)-[:HAS_FACT]->(:Fact)``, and ``search_facts(user_identifier=...)``
returns only the facts with that edge.
"""

from __future__ import annotations

import pytest

from neo4j_agent_memory.schema.models import EntityRef

ALICE = "alice@example.com"
BOB = "bob@example.com"


async def _mention(client, user: str, session_id: str, *names: str) -> None:
    """Store one message from ``user`` that mentions each named OBJECT entity."""
    await client.short_term.add_message(
        session_id,
        "user",
        f"Notes on {', '.join(names)}",
        user_identifier=user,
        extraction_mode="explicit",
        explicit_mentions=[EntityRef(name=name, type="OBJECT") for name in names],
    )


async def _names(client, query: str, **kwargs) -> set[str]:
    entities = await client.long_term.search_entities(query, threshold=0.0, **kwargs)
    return {entity.name for entity in entities}


@pytest.mark.integration
@pytest.mark.asyncio
class TestUserScopedEntitySearch:
    async def test_each_user_finds_only_the_entities_they_mentioned(self, clean_memory_client):
        client = clean_memory_client
        await _mention(client, ALICE, "alice-session", "hydraulic pump")
        await _mention(client, BOB, "bob-session", "hydraulic valve")

        assert await _names(client, "hydraulic", user_identifier=ALICE) == {"hydraulic pump"}
        assert await _names(client, "hydraulic", user_identifier=BOB) == {"hydraulic valve"}

    async def test_without_a_user_the_search_covers_every_user(self, clean_memory_client):
        client = clean_memory_client
        await _mention(client, ALICE, "alice-session", "hydraulic pump")
        await _mention(client, BOB, "bob-session", "hydraulic valve")

        assert await _names(client, "hydraulic") == {"hydraulic pump", "hydraulic valve"}

    async def test_an_entity_both_users_mention_is_found_for_each(self, clean_memory_client):
        client = clean_memory_client
        await _mention(client, ALICE, "alice-session", "hydraulic pump")
        await _mention(client, BOB, "bob-session", "hydraulic pump", "hydraulic valve")

        assert await _names(client, "hydraulic", user_identifier=ALICE) == {"hydraulic pump"}
        assert await _names(client, "hydraulic", user_identifier=BOB) == {
            "hydraulic pump",
            "hydraulic valve",
        }

    async def test_a_user_with_no_conversations_finds_nothing(self, clean_memory_client):
        client = clean_memory_client
        await _mention(client, BOB, "bob-session", "hydraulic valve")

        assert await _names(client, "hydraulic", user_identifier=ALICE) == set()

    async def test_unscoped_messages_belong_to_no_user(self, clean_memory_client, session_id):
        client = clean_memory_client
        await client.short_term.add_message(
            session_id,
            "user",
            "Notes on hydraulic pump",
            extraction_mode="explicit",
            explicit_mentions=[EntityRef(name="hydraulic pump", type="OBJECT")],
        )

        assert await _names(client, "hydraulic", user_identifier=ALICE) == set()
        assert await _names(client, "hydraulic") == {"hydraulic pump"}

    async def test_other_users_nearer_entities_do_not_crowd_out_a_users_own(
        self, clean_memory_client
    ):
        """The index limit applies before the user filter, so the scoped
        search asks the index for more candidates than ``limit``."""
        client = clean_memory_client
        await _mention(
            client,
            BOB,
            "bob-session",
            *(f"hydraulic pump seal {n}" for n in range(5)),
        )
        await _mention(client, ALICE, "alice-session", "hydraulic gauge")

        found = await client.long_term.search_entities(
            "hydraulic pump seal", limit=1, threshold=0.0, user_identifier=ALICE
        )

        assert [entity.name for entity in found] == ["hydraulic gauge"]

    async def test_scoped_hits_keep_their_similarity_and_the_limit(self, clean_memory_client):
        client = clean_memory_client
        await _mention(
            client,
            ALICE,
            "alice-session",
            *(f"hydraulic line {n}" for n in range(4)),
        )

        found = await client.long_term.search_entities(
            "hydraulic line", limit=2, threshold=0.0, user_identifier=ALICE
        )

        assert len(found) == 2
        scores = [entity.metadata["similarity"] for entity in found]
        assert scores == sorted(scores, reverse=True)


async def _fact_triples(client, query: str, **kwargs) -> set[tuple[str, str, str]]:
    facts = await client.long_term.search_facts(query, threshold=0.0, **kwargs)
    return {(fact.subject, fact.predicate, fact.object) for fact in facts}


async def _fact_owners(client, fact_id) -> set[str]:
    rows = await client._client.execute_read(
        "MATCH (u:User)-[:HAS_FACT]->(:Fact {id: $id}) RETURN u.identifier AS user",
        {"id": str(fact_id)},
    )
    return {row["user"] for row in rows}


@pytest.mark.integration
@pytest.mark.asyncio
class TestUserScopedFactSearch:
    async def test_add_fact_with_a_user_writes_the_has_fact_edge(self, clean_memory_client):
        client = clean_memory_client
        fact = await client.long_term.add_fact(
            "pump 7", "installed_on", "N123", user_identifier=ALICE
        )

        assert await _fact_owners(client, fact.id) == {ALICE}

    async def test_add_fact_without_a_user_writes_no_user_edge(self, clean_memory_client):
        client = clean_memory_client
        fact = await client.long_term.add_fact("pump 7", "installed_on", "N123")

        assert await _fact_owners(client, fact.id) == set()

    async def test_each_user_finds_only_their_own_facts(self, clean_memory_client):
        client = clean_memory_client
        await client.long_term.add_fact("pump 7", "installed_on", "N123", user_identifier=ALICE)
        await client.long_term.add_fact("valve 9", "installed_on", "N456", user_identifier=BOB)

        assert await _fact_triples(client, "installed on", user_identifier=ALICE) == {
            ("pump 7", "installed_on", "N123")
        }
        assert await _fact_triples(client, "installed on", user_identifier=BOB) == {
            ("valve 9", "installed_on", "N456")
        }

    async def test_without_a_user_the_search_covers_every_fact(self, clean_memory_client):
        client = clean_memory_client
        await client.long_term.add_fact("pump 7", "installed_on", "N123", user_identifier=ALICE)
        await client.long_term.add_fact("valve 9", "installed_on", "N456", user_identifier=BOB)
        await client.long_term.add_fact("gauge 3", "installed_on", "N789")

        assert await _fact_triples(client, "installed on") == {
            ("pump 7", "installed_on", "N123"),
            ("valve 9", "installed_on", "N456"),
            ("gauge 3", "installed_on", "N789"),
        }

    async def test_a_fact_with_no_owner_is_not_returned_to_a_user(self, clean_memory_client):
        client = clean_memory_client
        await client.long_term.add_fact("gauge 3", "installed_on", "N789")

        assert await _fact_triples(client, "installed on", user_identifier=ALICE) == set()

    async def test_a_deduplicated_fact_is_linked_to_the_second_user(self, clean_memory_client):
        client = clean_memory_client
        first = await client.long_term.add_fact(
            "pump 7", "installed_on", "N123", user_identifier=ALICE
        )
        second = await client.long_term.add_fact(
            "pump 7", "installed_on", "N123", user_identifier=BOB
        )

        assert second.id == first.id
        assert second.metadata.get("deduplicated") is True
        assert await _fact_owners(client, first.id) == {ALICE, BOB}
        assert await _fact_triples(client, "installed on", user_identifier=BOB) == {
            ("pump 7", "installed_on", "N123")
        }

    async def test_other_users_nearer_facts_do_not_crowd_out_a_users_own(self, clean_memory_client):
        client = clean_memory_client
        for n in range(5):
            await client.long_term.add_fact(
                f"pump seal {n}", "installed_on", f"N{n}", user_identifier=BOB
            )
        await client.long_term.add_fact("gauge 3", "reads", "pressure", user_identifier=ALICE)

        found = await client.long_term.search_facts(
            "pump seal installed on", limit=1, threshold=0.0, user_identifier=ALICE
        )

        assert [fact.subject for fact in found] == ["gauge 3"]
