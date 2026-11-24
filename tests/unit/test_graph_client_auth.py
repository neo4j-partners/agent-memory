"""``Neo4jConfig.auth_manager`` replaces the fixed login when it is set."""

from unittest.mock import AsyncMock, MagicMock

import neo4j
import pytest
from neo4j.auth_management import AsyncAuthManager, AuthManager
from pydantic import SecretStr, ValidationError

from neo4j_agent_memory import MemorySettings
from neo4j_agent_memory.config.settings import Neo4jConfig
from neo4j_agent_memory.graph import client as client_module
from neo4j_agent_memory.graph.client import Neo4jClient


class _Manager(AsyncAuthManager):
    """An async auth manager that always gives one login."""

    async def get_auth(self) -> neo4j.Auth:
        return neo4j.basic_auth("rotating", "current")

    async def handle_security_exception(
        self, auth: neo4j.Auth, error: neo4j.exceptions.Neo4jError
    ) -> bool:
        return False


class _SyncManager(AuthManager):
    """A sync auth manager, which the async driver cannot use."""

    def get_auth(self) -> neo4j.Auth:
        return neo4j.basic_auth("rotating", "current")

    def handle_security_exception(
        self, auth: neo4j.Auth, error: neo4j.exceptions.Neo4jError
    ) -> bool:
        return False


def _capture_driver(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace the driver factory with one that records its arguments."""
    driver = MagicMock()
    driver.verify_connectivity = AsyncMock()
    driver.close = AsyncMock()
    factory = MagicMock(return_value=driver)
    monkeypatch.setattr(client_module.AsyncGraphDatabase, "driver", factory)
    return factory


def test_auth_manager_defaults_to_none() -> None:
    assert Neo4jConfig(password=SecretStr("x")).auth_manager is None


def test_auth_manager_is_kept_as_given() -> None:
    manager = _Manager()
    settings = MemorySettings(
        _env_file=None,
        neo4j=Neo4jConfig(password=SecretStr("x"), auth_manager=manager),
    )
    assert settings.neo4j.auth_manager is manager


def test_auth_manager_must_be_async() -> None:
    with pytest.raises(ValidationError):
        Neo4jConfig(password=SecretStr("x"), auth_manager=_SyncManager())


def test_unknown_fields_are_still_rejected() -> None:
    with pytest.raises(ValidationError):
        Neo4jConfig(password=SecretStr("x"), auth="manager")


def test_auth_manager_stays_out_of_dumps_repr_and_schema() -> None:
    config = Neo4jConfig(password=SecretStr("x"), auth_manager=_Manager())
    assert "auth_manager" not in config.model_dump()
    assert "auth_manager" not in config.model_dump_json()
    assert "auth_manager" not in repr(config)
    assert "auth_manager" not in Neo4jConfig.model_json_schema()["properties"]
    assert config.model_dump() == Neo4jConfig(password=SecretStr("x")).model_dump()


@pytest.mark.asyncio
async def test_connect_passes_the_auth_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = _capture_driver(monkeypatch)
    manager = _Manager()
    client = Neo4jClient(Neo4jConfig(password=SecretStr("unused"), auth_manager=manager))

    await client.connect()

    assert factory.call_args.kwargs["auth"] is manager


@pytest.mark.asyncio
async def test_connect_without_a_manager_passes_the_fixed_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _capture_driver(monkeypatch)
    client = Neo4jClient(Neo4jConfig(username="reader", password=SecretStr("secret")))

    await client.connect()

    assert factory.call_args.kwargs["auth"] == ("reader", "secret")


@pytest.mark.asyncio
async def test_the_driver_asks_the_manager_for_its_login() -> None:
    """A real async driver accepts the manager as is, without wrapping it."""
    manager = _Manager()
    driver = neo4j.AsyncGraphDatabase.driver("bolt://localhost:7687", auth=manager)
    try:
        # The driver wraps anything that is not an AsyncAuthManager in a
        # static manager, so finding ours means it will call get_auth.
        assert driver._pool.pool_config.auth is manager
    finally:
        await driver.close()
