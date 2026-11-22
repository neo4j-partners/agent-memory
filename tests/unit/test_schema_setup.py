"""Tests for SchemaManager.setup_all's existence check.

setup_all reads the existing constraint and index names once, with one
SHOW CONSTRAINTS and one SHOW INDEXES, and creates only the missing ones.
A stub client records every query, so the tests count the round trips.
"""

from __future__ import annotations

from typing import Any

import pytest

from neo4j_agent_memory.core.exceptions import SchemaError
from neo4j_agent_memory.graph import queries
from neo4j_agent_memory.graph.schema import SchemaManager


class _StubClient:
    """Answers SHOW queries from fixed name lists and records every query."""

    def __init__(
        self,
        constraints: set[str] | None = None,
        indexes: set[str] | None = None,
        *,
        fail_show: bool = False,
    ) -> None:
        self._constraints = constraints or set()
        self._indexes = indexes or set()
        self._fail_show = fail_show
        self.reads: list[str] = []
        self.writes: list[str] = []

    async def execute_read(self, query: str, params: dict | None = None) -> list[dict[str, Any]]:
        self.reads.append(query)
        if query in (queries.SHOW_CONSTRAINTS, queries.SHOW_INDEXES) and self._fail_show:
            raise RuntimeError("show failed")
        if query == queries.SHOW_CONSTRAINTS:
            return [{"name": name} for name in sorted(self._constraints)]
        if query == queries.SHOW_INDEXES:
            return [{"name": name} for name in sorted(self._indexes)]
        if query == queries.GET_SCHEMA_MIGRATION:
            return [{"name": params["name"]}]  # backfills already ran
        return []

    async def execute_write(self, query: str, params: dict | None = None) -> list[dict[str, Any]]:
        self.writes.append(query)
        return []

    async def check_constraint_exists(self, name: str) -> bool:
        raise AssertionError("setup must not check names one at a time")

    async def check_index_exists(self, name: str) -> bool:
        raise AssertionError("setup must not check names one at a time")


async def _created_names() -> tuple[set[str], set[str]]:
    """Return the constraint and index names setup_all creates on an empty database."""
    client = _StubClient()
    await SchemaManager(client).setup_all()  # type: ignore[arg-type]
    constraints, indexes = set(), set()
    for query in client.writes:
        words = query.split()
        name = words[words.index("CONSTRAINT" if "CONSTRAINT" in words else "INDEX") + 1]
        (constraints if "CONSTRAINT" in words else indexes).add(name)
    return constraints, indexes


@pytest.mark.asyncio
async def test_empty_database_creates_everything_with_if_not_exists():
    client = _StubClient()
    await SchemaManager(client).setup_all()  # type: ignore[arg-type]

    assert client.reads.count(queries.SHOW_CONSTRAINTS) == 1
    assert client.reads.count(queries.SHOW_INDEXES) == 1
    assert len(client.writes) == 40  # 15 constraints, 18 indexes, 6 vector, 1 point
    assert all("IF NOT EXISTS" in query for query in client.writes)


@pytest.mark.asyncio
async def test_set_up_database_takes_two_schema_reads_and_no_writes():
    constraints, indexes = await _created_names()
    # A constraint's backing index shares its name, as SHOW INDEXES lists it.
    client = _StubClient(constraints, indexes | constraints)

    await SchemaManager(client).setup_all()  # type: ignore[arg-type]

    assert client.writes == []
    schema_reads = [q for q in client.reads if q != queries.GET_SCHEMA_MIGRATION]
    assert schema_reads == [queries.SHOW_CONSTRAINTS, queries.SHOW_INDEXES]


@pytest.mark.asyncio
async def test_creates_only_the_missing_names():
    constraints, indexes = await _created_names()
    client = _StubClient(constraints - {"tool_name"}, indexes - {"entity_embedding_idx"})

    await SchemaManager(client).setup_all()  # type: ignore[arg-type]

    assert len(client.writes) == 2
    assert any("CONSTRAINT tool_name " in query for query in client.writes)
    assert any("INDEX entity_embedding_idx " in query for query in client.writes)


@pytest.mark.asyncio
async def test_setup_method_alone_reads_names_once():
    client = _StubClient()
    await SchemaManager(client).setup_indexes()  # type: ignore[arg-type]

    assert client.reads == [queries.SHOW_INDEXES]
    assert len(client.writes) == 18


@pytest.mark.asyncio
async def test_failed_name_read_raises_schema_error():
    client = _StubClient(fail_show=True)
    with pytest.raises(SchemaError, match="Failed to read existing constraints"):
        await SchemaManager(client).setup_all()  # type: ignore[arg-type]
