"""Neo4j schema management for indexes and constraints."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, cast

from neo4j_agent_memory.config.settings import MemorySubsystem
from neo4j_agent_memory.core.exceptions import SchemaError
from neo4j_agent_memory.graph import queries
from neo4j_agent_memory.schema.models import (
    AdoptionLabelReport,
    AdoptionReport,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from neo4j_agent_memory.graph.client import Neo4jClient


# A Cypher identifier we are willing to interpolate into a label position.
# Matches names like ``Person``, ``Person_v2``, ``Client``. We deliberately
# disallow backticks so the caller cannot break out of the quoted label.
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
logger = logging.getLogger(__name__)


# Default vector dimensions
DEFAULT_VECTOR_DIMENSIONS = 1536


def _extract_vector_dimensions(options: object) -> int | None:
    """Pull ``vector.dimensions`` from a ``SHOW VECTOR INDEXES`` options map.

    Neo4j's vector index ``options`` is a nested map of shape::

        {"indexConfig": {"vector.dimensions": 1536, ...}, ...}

    Drivers may return the inner map as a ``dict`` or a Neo4j map object.
    Returns ``None`` if the structure does not match or the value is not
    a positive integer (e.g. older Neo4j versions, user-edited indexes).
    """
    if not isinstance(options, dict):
        return None
    # cast: Neo4j driver returns untyped dicts; isinstance confirmed it is a dict.
    options_d = cast("dict[str, Any]", options)
    config = options_d.get("indexConfig")
    if not isinstance(config, dict):
        return None
    config_d = cast("dict[str, Any]", config)
    dims = config_d.get("vector.dimensions")
    if isinstance(dims, int) and dims > 0:
        return dims
    return None


class SchemaManager:
    """
    Manages Neo4j schema for agent memory.

    Handles creation of indexes and constraints for all memory types.
    """

    # Unique constraints the library manages, each tagged with the
    # subsystem it belongs to. Every constraint also creates a backing
    # index, so a skipped constraint removes two entries from
    # ``SHOW INDEXES``-style counts: one constraint and one index.
    _MANAGED_CONSTRAINTS: tuple[tuple[str, str, str, MemorySubsystem], ...] = (
        # Short-term memory
        ("conversation_id", "Conversation", "id", MemorySubsystem.SHORT_TERM),
        ("message_id", "Message", "id", MemorySubsystem.SHORT_TERM),
        # Long-term memory
        ("entity_id", "Entity", "id", MemorySubsystem.ENTITIES),
        ("preference_id", "Preference", "id", MemorySubsystem.PREFERENCES),
        ("fact_id", "Fact", "id", MemorySubsystem.FACTS),
        # Reasoning memory
        ("reasoning_trace_id", "ReasoningTrace", "id", MemorySubsystem.REASONING),
        ("reasoning_step_id", "ReasoningStep", "id", MemorySubsystem.REASONING),
        ("tool_name", "Tool", "name", MemorySubsystem.REASONING),
        ("tool_call_id", "ToolCall", "id", MemorySubsystem.REASONING),
        # Multi-tenant (v0.4)
        ("user_identifier", "User", "identifier", MemorySubsystem.USERS),
        # Hygiene + privacy (v0.5)
        ("consolidation_run_id", "ConsolidationRun", "id", MemorySubsystem.CONSOLIDATION),
        ("memory_read_audit_id", "MemoryReadAudit", "id", MemorySubsystem.READ_AUDIT),
    )

    # Regular property indexes the library manages.
    _MANAGED_INDEXES: tuple[tuple[str, str, str, MemorySubsystem], ...] = (
        # Short-term memory
        ("conversation_session_idx", "Conversation", "session_id", MemorySubsystem.SHORT_TERM),
        ("message_timestamp_idx", "Message", "timestamp", MemorySubsystem.SHORT_TERM),
        ("message_role_idx", "Message", "role", MemorySubsystem.SHORT_TERM),
        # Long-term memory
        ("entity_type_idx", "Entity", "type", MemorySubsystem.ENTITIES),
        ("entity_name_idx", "Entity", "name", MemorySubsystem.ENTITIES),
        ("entity_canonical_idx", "Entity", "canonical_name", MemorySubsystem.ENTITIES),
        ("preference_category_idx", "Preference", "category", MemorySubsystem.PREFERENCES),
        # Reasoning memory
        ("trace_session_idx", "ReasoningTrace", "session_id", MemorySubsystem.REASONING),
        ("trace_success_idx", "ReasoningTrace", "success", MemorySubsystem.REASONING),
        ("trace_error_kind_idx", "ReasoningTrace", "error_kind", MemorySubsystem.REASONING),
        ("tool_call_status_idx", "ToolCall", "status", MemorySubsystem.REASONING),
        # Hygiene (v0.5)
        ("conversation_archived_idx", "Conversation", "archived", MemorySubsystem.CONSOLIDATION),
        ("consolidation_run_kind_idx", "ConsolidationRun", "kind", MemorySubsystem.CONSOLIDATION),
        ("memory_read_audit_kind_idx", "MemoryReadAudit", "kind", MemorySubsystem.READ_AUDIT),
    )

    # Vector indexes the library manages. Used by both
    # :meth:`setup_vector_indexes` and
    # :meth:`validate_vector_index_dimensions`.
    _MANAGED_VECTOR_INDEXES: tuple[tuple[str, str, str, MemorySubsystem], ...] = (
        ("message_embedding_idx", "Message", "embedding", MemorySubsystem.SHORT_TERM),
        ("entity_embedding_idx", "Entity", "embedding", MemorySubsystem.ENTITIES),
        ("preference_embedding_idx", "Preference", "embedding", MemorySubsystem.PREFERENCES),
        ("fact_embedding_idx", "Fact", "embedding", MemorySubsystem.FACTS),
        ("task_embedding_idx", "ReasoningTrace", "task_embedding", MemorySubsystem.REASONING),
        ("step_embedding_idx", "ReasoningStep", "embedding", MemorySubsystem.REASONING),
    )

    # Point indexes the library manages.
    _MANAGED_POINT_INDEXES: tuple[tuple[str, str, str, MemorySubsystem], ...] = (
        # Location entities have a 'location' Point property for coordinates
        ("entity_location_idx", "Entity", "location", MemorySubsystem.GEOSPATIAL),
    )

    def __init__(
        self,
        client: Neo4jClient,
        *,
        vector_dimensions: int = DEFAULT_VECTOR_DIMENSIONS,
        skip_subsystems: Iterable[MemorySubsystem] = (),
    ):
        """
        Initialize schema manager.

        Args:
            client: Neo4j client
            vector_dimensions: Dimensions for vector indexes
            skip_subsystems: Memory subsystems whose constraints and
                indexes should not be created. Empty by default, which
                installs the full schema. A skipped subsystem still
                accepts writes — it just loses uniqueness enforcement and
                index-backed search for its node types.
        """
        self._client = client
        self._vector_dimensions = vector_dimensions
        self._skip_subsystems = frozenset(skip_subsystems)

    @property
    def skipped_subsystems(self) -> frozenset[MemorySubsystem]:
        """Subsystems this manager will not create schema objects for."""
        return self._skip_subsystems

    def _is_active(self, subsystem: MemorySubsystem) -> bool:
        """Whether schema objects for ``subsystem`` should be created."""
        return subsystem not in self._skip_subsystems

    async def setup_all(self) -> None:
        """Set up all indexes and constraints."""
        await self.setup_constraints()
        await self.setup_indexes()
        await self.setup_vector_indexes()
        await self.setup_point_indexes()

    async def setup_constraints(self) -> None:
        """Create unique constraints for all node types."""
        for constraint_name, label, property_name, subsystem in self._MANAGED_CONSTRAINTS:
            if not self._is_active(subsystem):
                continue
            await self._create_constraint(constraint_name, label, property_name)

    async def setup_indexes(self) -> None:
        """Create regular indexes for common queries."""
        for index_name, label, property_name, subsystem in self._MANAGED_INDEXES:
            if not self._is_active(subsystem):
                continue
            await self._create_index(index_name, label, property_name)

    async def setup_vector_indexes(self) -> None:
        """Create vector indexes for semantic search."""
        for index_name, label, property_name, subsystem in self._MANAGED_VECTOR_INDEXES:
            if not self._is_active(subsystem):
                continue
            await self._create_vector_index(index_name, label, property_name)

    async def validate_vector_index_dimensions(self, expected: int) -> None:
        """Verify existing vector indexes match ``expected`` dimensions.

        Called from :meth:`MemoryClient.connect` after schema setup so the
        first sign of a misconfigured embedder is a clear error, not a
        downstream cosine-similarity failure or a silently-wrong search
        result.

        Only indexes the library manages
        (:attr:`_MANAGED_VECTOR_INDEXES`) are checked. User-created vector
        indexes outside that set are ignored, and so are indexes belonging
        to a skipped subsystem — this manager did not create those and
        does not size them, so a leftover from an earlier full setup must
        not fail the connect.

        Args:
            expected: Dimensionality declared by the configured embedder.

        Raises:
            EmbeddingDimensionMismatchError: Any managed vector index
                has a dimension count that disagrees with ``expected``.
                The error message lists every mismatching index and points
                at the migration runbook.
        """
        # Lazy import — avoids a circular dependency between
        # :mod:`graph.schema` and :mod:`llm.errors`.
        from neo4j_agent_memory.llm.errors import EmbeddingDimensionMismatchError

        try:
            rows = await self._client.execute_read(queries.SHOW_VECTOR_INDEXES_WITH_OPTIONS)
        except Exception as exc:
            # Vector indexes require Neo4j 5.11+. On older servers the
            # ``SHOW VECTOR INDEXES`` syntax fails; skip validation in
            # that case — :meth:`setup_vector_indexes` already swallows
            # the equivalent CREATE failure.
            if getattr(exc, "code", None) == "Neo.ClientError.Statement.SyntaxError":
                return
            logger.warning(
                "Failed to validate vector index dimensions using SHOW VECTOR INDEXES.",
                exc_info=True,
            )
            raise

        managed_names = {
            name
            for name, _, _, subsystem in self._MANAGED_VECTOR_INDEXES
            if self._is_active(subsystem)
        }
        mismatches: list[tuple[str, int, int]] = []
        for row in rows:
            name = row.get("name")
            if name not in managed_names:
                continue
            actual = _extract_vector_dimensions(row.get("options"))
            if actual is None:
                continue
            if actual != expected:
                mismatches.append((name, expected, actual))

        if not mismatches:
            return

        # Build a multi-line, actionable error message.
        lines = ["Vector index dimension mismatch.", ""]
        for name, want, got in mismatches:
            lines.append(f"  Index {name!r}: expected {want}, found {got}")
        lines.extend(
            [
                "",
                "This usually means you've changed the embedding model since the",
                "indexes were created. Two options:",
                "",
                "  1. Drop and recreate the indexes (loses existing embeddings).",
                "     See docs/how-to/migrate-embedding-model.adoc.",
                "  2. Revert to the previous embedding model.",
                "",
                f"Configured embedder dimensions: {expected}",
            ]
        )
        raise EmbeddingDimensionMismatchError(
            "\n".join(lines),
            expected_dimensions=expected,
            actual_dimensions=mismatches[0][2],
            index_name=mismatches[0][0],
        )

    async def setup_point_indexes(self) -> None:
        """Create point indexes for geospatial queries."""
        for index_name, label, property_name, subsystem in self._MANAGED_POINT_INDEXES:
            if not self._is_active(subsystem):
                continue
            await self._create_point_index(index_name, label, property_name)

    async def _create_constraint(
        self,
        constraint_name: str,
        label: str,
        property_name: str,
    ) -> None:
        """Create a unique constraint if it doesn't exist."""
        try:
            exists = await self._client.check_constraint_exists(constraint_name)
            if exists:
                return

            query = queries.create_constraint_query(constraint_name, label, property_name)
            await self._client.execute_write(query)
        except Exception as e:
            raise SchemaError(f"Failed to create constraint {constraint_name}: {e}") from e

    async def _create_index(
        self,
        index_name: str,
        label: str,
        property_name: str,
    ) -> None:
        """Create a regular index if it doesn't exist."""
        try:
            exists = await self._client.check_index_exists(index_name)
            if exists:
                return

            query = queries.create_index_query(index_name, label, property_name)
            await self._client.execute_write(query)
        except Exception as e:
            raise SchemaError(f"Failed to create index {index_name}: {e}") from e

    async def _create_vector_index(
        self,
        index_name: str,
        label: str,
        property_name: str,
    ) -> None:
        """Create a vector index if it doesn't exist."""
        try:
            exists = await self._client.check_index_exists(index_name)
            if exists:
                return

            query = queries.create_vector_index_query(
                index_name, label, property_name, self._vector_dimensions
            )
            await self._client.execute_write(query)
        except Exception:
            # Vector indexes require Neo4j 5.11+, log warning but don't fail
            # as the package can still work without vector search
            pass

    async def _create_point_index(
        self,
        index_name: str,
        label: str,
        property_name: str,
    ) -> None:
        """Create a point index for geospatial queries if it doesn't exist."""
        try:
            exists = await self._client.check_index_exists(index_name)
            if exists:
                return

            query = queries.create_point_index_query(index_name, label, property_name)
            await self._client.execute_write(query)
        except Exception:
            # Point indexes require Neo4j 5.0+, log warning but don't fail
            pass

    async def drop_all(self) -> None:
        """Drop all memory-related indexes and constraints."""
        # Get all constraints
        constraints = await self._client.execute_read(queries.SHOW_CONSTRAINTS)
        for constraint in constraints:
            name = constraint["name"]
            if self._is_memory_schema(name):
                await self._client.execute_write(queries.drop_constraint_query(name))

        # Get all indexes
        indexes = await self._client.execute_read(queries.SHOW_INDEXES)
        for index in indexes:
            name = index["name"]
            if self._is_memory_schema(name):
                await self._client.execute_write(queries.drop_index_query(name))

    def _is_memory_schema(self, name: str) -> bool:
        """Check if a schema element belongs to agent memory."""
        memory_prefixes = [
            "conversation_",
            "message_",
            "entity_",
            "preference_",
            "fact_",
            "reasoning_",
            "trace_",
            "tool_",
            "task_",
            "step_",
            "user_",
            "consolidation_",
            "memory_read_",
        ]
        return any(name.startswith(prefix) for prefix in memory_prefixes)

    async def adopt_existing_graph(
        self,
        label_to_type: dict[str, str],
        *,
        name_property_per_label: dict[str, str] | None = None,
        dry_run: bool = False,
    ) -> AdoptionReport:
        """Adopt an existing domain graph as long-term memory entities.

        For each input label, attaches the ``:Entity`` super-label and the
        library's required ``id`` / ``type`` / ``name`` properties to nodes
        already present in the database. After adoption, library APIs that
        MERGE on ``(:Entity {name, type})`` (e.g. message-mention extraction,
        relation writes) link to the existing nodes instead of creating
        duplicates.

        The operation is idempotent — re-running on already-adopted nodes
        is a no-op. Nodes that lack the configured name property are
        skipped and reported.

        Args:
            label_to_type: Map from existing Neo4j label to the library
                entity type to assign, e.g.
                ``{"Person": "PERSON", "Movie": "OBJECT"}``.
            name_property_per_label: Optional per-label override for the
                property to use as the entity name. Defaults to ``"name"``
                for every label not in the map.
            dry_run: When True, do not mutate the graph — return a report
                describing what *would* happen.

        Returns:
            An :class:`AdoptionReport` with per-label counts.

        Raises:
            SchemaError: If a label or name property contains characters
                that cannot be safely interpolated as a Cypher identifier.
        """
        name_property_per_label = name_property_per_label or {}
        per_label: list[AdoptionLabelReport] = []

        for label, entity_type in label_to_type.items():
            name_property = name_property_per_label.get(label, "name")

            if not _SAFE_IDENTIFIER.match(label):
                raise SchemaError(
                    f"Refusing to adopt label {label!r}: must match [A-Za-z_][A-Za-z0-9_]*"
                )
            if not _SAFE_IDENTIFIER.match(name_property):
                raise SchemaError(
                    f"Refusing to use name property {name_property!r} for "
                    f"label {label!r}: must match [A-Za-z_][A-Za-z0-9_]*"
                )

            already_rows = await self._client.execute_read(
                queries.count_already_adopted_query(label)
            )
            already_adopted = int(already_rows[0]["already_adopted_count"]) if already_rows else 0

            skipped_rows = await self._client.execute_read(
                queries.count_adoption_skipped_query(label, name_property)
            )
            skipped = int(skipped_rows[0]["skipped_count"]) if skipped_rows else 0

            if dry_run:
                # Count what *would* migrate without mutating.
                projected_rows = await self._client.execute_read(
                    f"""
                    MATCH (n:`{label}`)
                    WHERE NOT n:Entity AND n.`{name_property}` IS NOT NULL
                    RETURN count(n) AS migrated_count
                    """
                )
                migrated = int(projected_rows[0]["migrated_count"]) if projected_rows else 0
            else:
                rows = await self._client.execute_write(
                    queries.adopt_label_to_entity_query(label, name_property),
                    {"label_lc": label.lower(), "type": entity_type},
                )
                # ``execute_write`` returns the records produced by the
                # query — it's a list of dicts here.
                migrated = int(rows[0]["migrated_count"]) if rows else 0

            per_label.append(
                AdoptionLabelReport(
                    label=label,
                    type=entity_type,
                    name_property=name_property,
                    migrated_count=migrated,
                    already_adopted_count=already_adopted,
                    skipped_count=skipped,
                )
            )

        return AdoptionReport(by_label=per_label, dry_run=dry_run)

    async def get_schema_info(self) -> dict[str, Any]:
        """Get information about the current schema."""
        constraints = await self._client.execute_read(queries.SHOW_CONSTRAINTS_DETAIL)
        indexes = await self._client.execute_read(queries.SHOW_INDEXES_DETAIL)

        return {
            "constraints": [
                {
                    "name": c["name"],
                    "type": c["type"],
                    "labels": c["labelsOrTypes"],
                    "properties": c["properties"],
                }
                for c in constraints
                if self._is_memory_schema(c["name"])
            ],
            "indexes": [
                {
                    "name": i["name"],
                    "type": i["type"],
                    "labels": i["labelsOrTypes"],
                    "properties": i["properties"],
                }
                for i in indexes
                if self._is_memory_schema(i["name"])
            ],
        }
