"""Tests for opting out of memory subsystems in ``SchemaManager``.

The default must install exactly the schema it has always installed. A
caller that names subsystems in ``SchemaConfig.skip_subsystems`` gets that
schema minus those subsystems, and nothing else changes.

Unit tests have no live Neo4j, so ``Neo4jClient`` is stubbed and the
``CREATE ...`` statements the manager issues are recorded and parsed back
into object names.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from neo4j_agent_memory.config.settings import MemorySettings, MemorySubsystem, SchemaConfig
from neo4j_agent_memory.graph.schema import SchemaManager
from neo4j_agent_memory.llm.errors import EmbeddingDimensionMismatchError

_CREATE_PATTERN = re.compile(
    r"CREATE\s+(CONSTRAINT|INDEX|VECTOR INDEX|POINT INDEX)\s+(\w+)\s+IF NOT EXISTS"
)


class _RecordingClient:
    """Neo4jClient stand-in that records the schema objects created."""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.created: dict[str, set[str]] = {
            "CONSTRAINT": set(),
            "INDEX": set(),
            "VECTOR INDEX": set(),
            "POINT INDEX": set(),
        }
        self._rows = rows or []

    async def check_constraint_exists(self, name: str) -> bool:
        return False

    async def check_index_exists(self, name: str) -> bool:
        return False

    async def execute_write(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        match = _CREATE_PATTERN.search(" ".join(query.split()))
        assert match is not None, f"unrecognised schema statement: {query}"
        self.created[match.group(1)].add(match.group(2))
        return []

    async def execute_read(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return self._rows


async def _setup(*skip: MemorySubsystem) -> _RecordingClient:
    """Run ``setup_all`` against a recording client and return it."""
    client = _RecordingClient()
    manager = SchemaManager(client, skip_subsystems=skip)  # type: ignore[arg-type]
    await manager.setup_all()
    return client


# ---------------------------------------------------------------------------
# The managed tables
# ---------------------------------------------------------------------------


def test_every_managed_object_is_tagged_with_a_subsystem():
    """The four tables partition the schema — no object is untagged."""
    tables = (
        SchemaManager._MANAGED_CONSTRAINTS,
        SchemaManager._MANAGED_INDEXES,
        SchemaManager._MANAGED_VECTOR_INDEXES,
        SchemaManager._MANAGED_POINT_INDEXES,
    )
    names: list[str] = []
    for table in tables:
        for name, label, property_name, subsystem in table:
            assert label and property_name
            assert isinstance(subsystem, MemorySubsystem)
            names.append(name)
    # Names are unique across every table, so each belongs to exactly one
    # subsystem.
    assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# Default behaviour — the full schema
# ---------------------------------------------------------------------------


async def test_default_installs_the_full_schema():
    client = await _setup()

    assert client.created["CONSTRAINT"] == {
        "conversation_id",
        "message_id",
        "entity_id",
        "preference_id",
        "fact_id",
        "reasoning_trace_id",
        "reasoning_step_id",
        "tool_name",
        "tool_call_id",
        "user_identifier",
        "consolidation_run_id",
        "memory_read_audit_id",
    }
    assert client.created["INDEX"] == {
        "conversation_session_idx",
        "conversation_archived_idx",
        "message_timestamp_idx",
        "message_role_idx",
        "entity_type_idx",
        "entity_name_idx",
        "entity_canonical_idx",
        "preference_category_idx",
        "trace_session_idx",
        "trace_success_idx",
        "trace_error_kind_idx",
        "tool_call_status_idx",
        "consolidation_run_kind_idx",
        "memory_read_audit_kind_idx",
    }
    assert client.created["VECTOR INDEX"] == {
        "message_embedding_idx",
        "entity_embedding_idx",
        "preference_embedding_idx",
        "fact_embedding_idx",
        "task_embedding_idx",
        "step_embedding_idx",
    }
    assert client.created["POINT INDEX"] == {"entity_location_idx"}

    # 12 constraints, 14 property indexes, 6 vector indexes, 1 point index.
    assert len(client.created["CONSTRAINT"]) == 12
    total_indexes = (
        len(client.created["CONSTRAINT"])  # each constraint has a backing index
        + len(client.created["INDEX"])
        + len(client.created["VECTOR INDEX"])
        + len(client.created["POINT INDEX"])
    )
    assert total_indexes == 33


def test_schema_config_defaults_to_no_skips():
    assert SchemaConfig().skip_subsystems == frozenset()
    assert MemorySettings().schema_config.skip_subsystems == frozenset()


# ---------------------------------------------------------------------------
# Opting out
# ---------------------------------------------------------------------------


async def test_opt_out_installs_the_reduced_schema():
    """The set a lean deployment skips: facts, hygiene, audit, geospatial."""
    client = await _setup(
        MemorySubsystem.FACTS,
        MemorySubsystem.CONSOLIDATION,
        MemorySubsystem.READ_AUDIT,
        MemorySubsystem.GEOSPATIAL,
    )

    assert client.created["CONSTRAINT"] == {
        "conversation_id",
        "message_id",
        "entity_id",
        "preference_id",
        "reasoning_trace_id",
        "reasoning_step_id",
        "tool_name",
        "tool_call_id",
        "user_identifier",
    }
    assert client.created["INDEX"] == {
        "conversation_session_idx",
        "message_timestamp_idx",
        "message_role_idx",
        "entity_type_idx",
        "entity_name_idx",
        "entity_canonical_idx",
        "preference_category_idx",
        "trace_session_idx",
        "trace_success_idx",
        "trace_error_kind_idx",
        "tool_call_status_idx",
    }
    assert client.created["VECTOR INDEX"] == {
        "message_embedding_idx",
        "entity_embedding_idx",
        "preference_embedding_idx",
        "task_embedding_idx",
        "step_embedding_idx",
    }
    assert client.created["POINT INDEX"] == set()

    # 9 constraints, 11 property indexes, 5 vector indexes, 0 point indexes.
    assert len(client.created["CONSTRAINT"]) == 9
    total_indexes = (
        len(client.created["CONSTRAINT"])
        + len(client.created["INDEX"])
        + len(client.created["VECTOR INDEX"])
        + len(client.created["POINT INDEX"])
    )
    assert total_indexes == 25


@pytest.mark.parametrize("subsystem", list(MemorySubsystem))
async def test_skipping_one_subsystem_leaves_the_others_intact(subsystem: MemorySubsystem):
    full = await _setup()
    reduced = await _setup(subsystem)

    for kind, created in reduced.created.items():
        # Never creates anything the full run did not create.
        assert created <= full.created[kind]

    dropped = {name for kind in full.created for name in full.created[kind] - reduced.created[kind]}
    expected_dropped = {
        name
        for table in (
            SchemaManager._MANAGED_CONSTRAINTS,
            SchemaManager._MANAGED_INDEXES,
            SchemaManager._MANAGED_VECTOR_INDEXES,
            SchemaManager._MANAGED_POINT_INDEXES,
        )
        for name, _, _, tagged in table
        if tagged is subsystem
    }
    assert dropped == expected_dropped
    assert dropped, f"{subsystem} owns no schema objects"


async def test_skipping_everything_creates_nothing():
    client = await _setup(*MemorySubsystem)
    assert all(created == set() for created in client.created.values())


def test_skipped_subsystems_is_exposed():
    manager = SchemaManager(
        _RecordingClient(),  # type: ignore[arg-type]
        skip_subsystems=[MemorySubsystem.FACTS],
    )
    assert manager.skipped_subsystems == frozenset({MemorySubsystem.FACTS})
    assert SchemaManager(_RecordingClient()).skipped_subsystems == frozenset()  # type: ignore[arg-type]


def test_skip_subsystems_accepts_plain_strings():
    config = SchemaConfig(skip_subsystems={"facts", "geospatial"})  # type: ignore[arg-type]
    assert config.skip_subsystems == frozenset({MemorySubsystem.FACTS, MemorySubsystem.GEOSPATIAL})


def test_skip_subsystems_rejects_unknown_names():
    with pytest.raises(ValueError):
        SchemaConfig(skip_subsystems={"not_a_subsystem"})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Vector dimension validation under a skip
# ---------------------------------------------------------------------------


async def test_validation_ignores_a_skipped_subsystems_index():
    """A leftover index we no longer manage must not fail the connect."""
    rows = [
        {
            "name": "fact_embedding_idx",
            "options": {"indexConfig": {"vector.dimensions": 1536}},
        },
    ]
    manager = SchemaManager(
        _RecordingClient(rows=rows),  # type: ignore[arg-type]
        vector_dimensions=384,
        skip_subsystems=[MemorySubsystem.FACTS],
    )
    await manager.validate_vector_index_dimensions(384)


async def test_validation_still_catches_an_active_subsystems_index():
    rows = [
        {
            "name": "fact_embedding_idx",
            "options": {"indexConfig": {"vector.dimensions": 1536}},
        },
        {
            "name": "message_embedding_idx",
            "options": {"indexConfig": {"vector.dimensions": 1536}},
        },
    ]
    manager = SchemaManager(
        _RecordingClient(rows=rows),  # type: ignore[arg-type]
        vector_dimensions=384,
        skip_subsystems=[MemorySubsystem.FACTS],
    )
    with pytest.raises(EmbeddingDimensionMismatchError) as excinfo:
        await manager.validate_vector_index_dimensions(384)
    message = str(excinfo.value)
    assert "message_embedding_idx" in message
    assert "fact_embedding_idx" not in message
