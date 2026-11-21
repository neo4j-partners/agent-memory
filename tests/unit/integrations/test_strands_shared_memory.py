"""Fix E: many Strands sessions share one memory client and one loop.

``SharedMemory.open(settings)`` connects one client on one background
loop. Each manager it builds runs on that loop, writes only its own
session, and leaves the client and the loop running when it closes.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

import pytest

pytest.importorskip("strands", reason="strands-agents not installed")

from strands import Agent  # noqa: E402

from neo4j_agent_memory.integrations.base import AsyncBridge  # noqa: E402
from neo4j_agent_memory.integrations.strands import SharedMemory  # noqa: E402
from neo4j_agent_memory.integrations.strands.session_manager import (  # noqa: E402
    Neo4jSessionManager,
    SessionException,
)
from tests.unit.integrations.strands_fakes import FakeMemoryClient  # noqa: E402
from tests.unit.integrations.test_strands_deferred_extraction import (  # noqa: E402
    _Gate,
    _Hook,
    _record_running_at_close,
)
from tests.unit.integrations.test_strands_nonblocking import ScriptedModel  # noqa: E402

_SETTINGS: Any = object()  # open() hands it to the patched MemoryClient only
_SHARED_LOGGER = "neo4j_agent_memory.integrations.strands.shared"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> FakeMemoryClient:
    """The client ``SharedMemory.open`` builds, whatever the settings."""
    fake = FakeMemoryClient()
    monkeypatch.setattr("neo4j_agent_memory.MemoryClient", lambda _settings: fake)
    return fake


def _bridge_threads() -> int:
    """Background loop threads running now, across every bridge."""
    return sum(t.name == "neo4j-strands-session-manager" for t in threading.enumerate())


def _stored_by_session(client: FakeMemoryClient) -> dict[str, list[tuple[str, str]]]:
    stored: dict[str, list[tuple[str, str]]] = {}
    for call in client.short_term.add_message_calls:
        stored.setdefault(call["session_id"], []).append((call["role"], call["content"]))
    return stored


def _agent(manager: Neo4jSessionManager, *replies: str) -> Agent:
    turns = [[{"text": reply}] for reply in replies]
    return Agent(model=ScriptedModel(*turns), session_manager=manager, callback_handler=None)


class TestManagersShareOneClient:
    async def test_each_manager_writes_only_its_own_session(self, client: FakeMemoryClient) -> None:
        shared = SharedMemory.open(_SETTINGS)
        first = shared.session_manager("sess-1", extract_entities=False)
        second = shared.session_manager("sess-2", extract_entities=False)
        try:
            await _agent(first, "A1").invoke_async("Q1")
            await _agent(second, "B1").invoke_async("P1")
            await first.aclose()
            await second.aclose()
        finally:
            await shared.aclose()

        assert client.connect_calls == 1
        assert _stored_by_session(client) == {
            "sess-1": [("user", "Q1"), ("assistant", "A1")],
            "sess-2": [("user", "P1"), ("assistant", "B1")],
        }

    @pytest.mark.parametrize("use_async", [False, True], ids=["close", "aclose"])
    async def test_closing_one_manager_leaves_the_client_and_loop_running(
        self, client: FakeMemoryClient, use_async: bool
    ) -> None:
        shared = SharedMemory.open(_SETTINGS)
        first = shared.session_manager("sess-1", extract_entities=False)
        second = shared.session_manager("sess-2", extract_entities=False)
        try:
            await _agent(first, "A1").invoke_async("Q1")
            if use_async:
                await first.aclose()
            else:
                first.close()

            assert client.is_connected
            assert client.close_calls == 0
            await _agent(second, "B1").invoke_async("P1")
            await second.aclose()
        finally:
            await shared.aclose()

        assert _stored_by_session(client)["sess-2"] == [("user", "P1"), ("assistant", "B1")]

    async def test_twenty_concurrent_sessions_share_one_thread(
        self, client: FakeMemoryClient
    ) -> None:
        threads_before = _bridge_threads()
        shared = SharedMemory.open(_SETTINGS)
        managers = [shared.session_manager(f"sess-{n}", extract_entities=False) for n in range(20)]
        try:
            agents = [_agent(m, f"A{n}") for n, m in enumerate(managers)]
            await asyncio.gather(*(a.invoke_async(f"Q{n}") for n, a in enumerate(agents)))
            assert _bridge_threads() == threads_before + 1
            await asyncio.gather(*(m.aclose() for m in managers))
        finally:
            await shared.aclose()

        stored = _stored_by_session(client)
        assert stored == {
            f"sess-{n}": [("user", f"Q{n}"), ("assistant", f"A{n}")] for n in range(20)
        }
        assert client.connect_calls == 1
        assert client.close_calls == 1


class TestClosingSharedMemory:
    async def test_aclose_closes_the_client_once(self, client: FakeMemoryClient) -> None:
        shared = SharedMemory.open(_SETTINGS)
        await shared.aclose()
        await shared.aclose()
        shared.close()

        assert shared.closed
        assert client.close_calls == 1

    def test_close_closes_the_client_once(self, client: FakeMemoryClient) -> None:
        shared = SharedMemory.open(_SETTINGS)
        shared.close()
        shared.close()
        assert client.close_calls == 1

    async def test_no_manager_is_built_after_close(self, client: FakeMemoryClient) -> None:
        shared = SharedMemory.open(_SETTINGS)
        await shared.aclose()
        with pytest.raises(RuntimeError, match="closed"):
            shared.session_manager("sess-1")

    def test_close_on_the_loop_thread_raises_and_stays_open(self, client: FakeMemoryClient) -> None:
        shared = SharedMemory.open(_SETTINGS)

        async def close_on_the_loop() -> None:
            shared.close()

        with pytest.raises(RuntimeError, match="loop thread"):
            shared._bridge.run(close_on_the_loop())
        assert not shared.closed
        assert client.is_connected

        shared.close()
        assert client.close_calls == 1

    def test_a_failed_connect_stops_the_loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeMemoryClient()

        async def refuse() -> None:
            raise ConnectionError("no route to host")

        monkeypatch.setattr(fake, "connect", refuse)
        monkeypatch.setattr("neo4j_agent_memory.MemoryClient", lambda _settings: fake)
        threads_before = _bridge_threads()

        with pytest.raises(ConnectionError, match="no route"):
            SharedMemory.open(_SETTINGS)
        assert _bridge_threads() == threads_before

    def test_a_connect_that_fails_after_opening_the_driver_closes_the_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeMemoryClient()

        async def open_then_fail() -> None:
            fake._connected = True  # the driver is open
            raise ConnectionError("schema setup failed")

        monkeypatch.setattr(fake, "connect", open_then_fail)
        monkeypatch.setattr("neo4j_agent_memory.MemoryClient", lambda _settings: fake)
        threads_before = _bridge_threads()

        with pytest.raises(ConnectionError, match="schema setup"):
            SharedMemory.open(_SETTINGS)
        assert fake.close_calls == 1
        assert not fake.is_connected
        assert _bridge_threads() == threads_before

    def test_a_close_that_fails_after_a_failed_connect_is_logged(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        fake = FakeMemoryClient()

        async def refuse() -> None:
            raise ConnectionError("no route to host")

        async def broken_close() -> None:
            raise RuntimeError("driver gone")

        monkeypatch.setattr(fake, "connect", refuse)
        monkeypatch.setattr(fake, "close", broken_close)
        monkeypatch.setattr("neo4j_agent_memory.MemoryClient", lambda _settings: fake)
        threads_before = _bridge_threads()

        with (
            caplog.at_level(logging.WARNING, logger=_SHARED_LOGGER),
            pytest.raises(ConnectionError, match="no route"),
        ):
            SharedMemory.open(_SETTINGS)
        assert "driver gone" in caplog.text
        assert _bridge_threads() == threads_before


class TestClosingWithBackgroundWork:
    @pytest.mark.parametrize("use_async", [False, True], ids=["close", "aclose"])
    async def test_close_waits_for_deferred_extractions(
        self, client: FakeMemoryClient, use_async: bool
    ) -> None:
        gate = _Gate(client)
        shared = SharedMemory.open(_SETTINGS)
        manager = shared.session_manager("sess-1", defer_extraction=True)
        await _agent(manager, "A1").invoke_async("Q1")
        await manager.aclose()
        assert gate.done == []

        threading.Timer(0.2, gate.opened.set).start()
        if use_async:
            await shared.aclose()
        else:
            await asyncio.to_thread(shared.close)

        assert gate.done == ["Q1", "A1"]
        assert client.close_calls == 1

    @pytest.mark.parametrize("use_async", [False, True], ids=["close", "aclose"])
    async def test_tasks_still_running_after_the_timeout_are_cancelled(
        self,
        client: FakeMemoryClient,
        use_async: bool,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        caplog.set_level(logging.WARNING, logger="neo4j_agent_memory.integrations.strands.shared")
        gate = _Gate(client)  # never opens
        threads_before = _bridge_threads()
        shared = SharedMemory.open(_SETTINGS)
        manager = shared.session_manager("sess-1", defer_extraction=True)
        await _agent(manager, "A1").invoke_async("Q1")
        await manager.aclose()
        running_at_close = _record_running_at_close(client, gate, monkeypatch)

        if use_async:
            await asyncio.wait_for(shared.aclose(timeout=0.1), timeout=5)
        else:
            await asyncio.wait_for(asyncio.to_thread(shared.close, 0.1), timeout=5)

        assert gate.done == []
        assert "Closing SharedMemory while 2 background task(s) are still running" in caplog.text
        assert running_at_close == [0]  # cancelled before the client closed
        assert client.close_calls == 1
        assert _bridge_threads() == threads_before

    @pytest.mark.parametrize("timeout", [0.0, -1.0, float("nan"), float("inf")])
    @pytest.mark.parametrize("use_async", [False, True], ids=["close", "aclose"])
    async def test_a_timeout_that_is_not_positive_and_finite_is_refused(
        self, client: FakeMemoryClient, use_async: bool, timeout: float
    ) -> None:
        shared = SharedMemory.open(_SETTINGS)
        with pytest.raises(ValueError, match="timeout must be a positive finite number"):
            if use_async:
                await shared.aclose(timeout=timeout)
            else:
                shared.close(timeout=timeout)

        assert not shared.closed
        await shared.aclose()
        assert client.close_calls == 1


class TestAManagerOnASharedBridge:
    def test_bridge_needs_a_memory_client(self) -> None:
        settings: Any = object()
        with pytest.raises(ValueError, match="memory_client"):
            Neo4jSessionManager("sess-1", settings=settings, bridge=AsyncBridge())

    def test_bridge_needs_a_connected_client(self) -> None:
        bridge = AsyncBridge()
        try:
            with pytest.raises(ValueError, match="connected"):
                Neo4jSessionManager("sess-1", memory_client=FakeMemoryClient(), bridge=bridge)
        finally:
            bridge.close()

    async def test_it_does_not_close_the_client(self) -> None:
        client = FakeMemoryClient()
        bridge = AsyncBridge()
        await bridge.run_async(client.connect())
        manager = Neo4jSessionManager(
            "sess-1", memory_client=client, bridge=bridge, extract_entities=False
        )
        try:
            await manager.aopen()
            await manager.aclose()
            assert client.is_connected
            assert not bridge.closed
        finally:
            await asyncio.to_thread(bridge.close)

    async def test_it_keeps_its_own_request_timeout(self) -> None:
        client = FakeMemoryClient()

        async def hang(*args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(30)

        client.short_term.get_conversation = hang
        bridge = AsyncBridge(timeout=30.0)
        await bridge.run_async(client.connect())
        manager = Neo4jSessionManager(
            "sess-1", memory_client=client, bridge=bridge, request_timeout=0.1
        )
        try:
            with pytest.raises(SessionException):
                await asyncio.wait_for(manager.aopen(), timeout=5)
            await manager.aclose()
        finally:
            await asyncio.to_thread(bridge.close)


class TestTheBackgroundHook:
    async def test_every_manager_reports_its_background_tasks(
        self, client: FakeMemoryClient
    ) -> None:
        _Gate(client).opened.set()
        hook = _Hook()
        shared = SharedMemory.open(_SETTINGS, background_hook=hook)
        first = shared.session_manager("sess-1", defer_extraction=True)
        second = shared.session_manager("sess-2", defer_extraction=True)
        try:
            await _agent(first, "A1").invoke_async("Q1")
            await _agent(second, "B1").invoke_async("P1")
            await first.aclose()
            await second.aclose()
        finally:
            await shared.aclose()

        assert sorted(hook.names("start")) == ["extract:sess-1"] * 2 + ["extract:sess-2"] * 2
        hook.assert_paired()

    async def test_a_task_cancelled_when_the_loop_stops_ends_once(
        self, client: FakeMemoryClient
    ) -> None:
        _Gate(client)  # never opens
        hook = _Hook()
        shared = SharedMemory.open(_SETTINGS, background_hook=hook)
        manager = shared.session_manager("sess-1", defer_extraction=True)
        await _agent(manager, "A1").invoke_async("Q1")
        await manager.aclose()

        await shared.aclose(timeout=0.05)

        assert len(hook.names("start")) == 2
        hook.assert_paired()

    async def test_a_manager_cannot_be_given_its_own_hook(self, client: FakeMemoryClient) -> None:
        shared = SharedMemory.open(_SETTINGS)
        try:
            with pytest.raises(TypeError, match="background_hook"):
                shared.session_manager("sess-1", background_hook=_Hook())
        finally:
            await shared.aclose()
