"""Datetime normalization for values sent to Neo4j as query parameters."""

from __future__ import annotations

from datetime import UTC, datetime


def as_neo4j_utc(value: datetime) -> datetime:
    """Return an aware datetime as the same instant at a fixed UTC offset.

    Neo4j stores an aware datetime with its zone. A named zone, such as
    pytz's ``UTC`` or ``ZoneInfo("UTC")``, then compares unequal to the same
    instant at offset ``Z`` and sorts after it. The driver returns stored
    datetimes with pytz's ``UTC``, so a value read back and sent again carries
    a named zone. Converting to ``datetime.UTC`` sends a fixed offset.

    A naive datetime is returned unchanged. Neo4j reads it in the server's
    default zone, which is a fixed offset.
    """
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC)
