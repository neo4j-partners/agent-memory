"""Tests for normalizing datetimes sent to Neo4j as query parameters."""

from datetime import UTC, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
import pytz

from neo4j_agent_memory.graph._temporal import as_neo4j_utc

INSTANT = datetime(2026, 1, 1, 12, 0, 0, 123456, tzinfo=UTC)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(INSTANT.astimezone(pytz.UTC), id="pytz-utc"),
        pytest.param(INSTANT.astimezone(ZoneInfo("UTC")), id="zoneinfo-utc"),
        pytest.param(INSTANT.astimezone(ZoneInfo("America/Los_Angeles")), id="named-zone"),
        pytest.param(
            INSTANT.astimezone(timezone(timedelta(hours=5, minutes=30))), id="offset-0530"
        ),
        pytest.param(INSTANT, id="datetime-utc"),
    ],
)
def test_an_aware_datetime_becomes_the_same_instant_in_fixed_utc(value: datetime) -> None:
    normalized = as_neo4j_utc(value)

    assert normalized.tzinfo is UTC
    assert normalized == INSTANT
    assert normalized.utcoffset() == timedelta(0)


def test_a_naive_datetime_is_returned_unchanged() -> None:
    naive = INSTANT.replace(tzinfo=None)

    assert as_neo4j_utc(naive) is naive
