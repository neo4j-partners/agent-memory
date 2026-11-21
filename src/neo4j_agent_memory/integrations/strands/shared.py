"""One memory client and one background loop shared by many Strands sessions.

Each ``Neo4jSessionManager`` built from ``settings=`` owns a client and a
background loop. The async Neo4j driver and the NAMS HTTP client bind to
the loop they connect on, so every session then holds its own connection
pool and thread. A server that runs many sessions builds one
``SharedMemory`` at startup instead, and asks it for a manager per session.
"""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, Any

from neo4j_agent_memory.integrations.base import AsyncBridge
from neo4j_agent_memory.integrations.strands.session_manager import Neo4jSessionManager

if TYPE_CHECKING:
    from neo4j_agent_memory import MemoryClient, MemorySettings

__all__ = ["SharedMemory"]


class SharedMemory:
    """A memory client and a background loop that session managers share.

    Build it once with ``SharedMemory.open(settings)``, then call
    ``session_manager(session_id)`` for each session. The managers share
    the client, its connection pool, and one background thread. A manager
    built this way closes neither: its ``close()`` and ``aclose()`` write
    its own messages and return.

    Close every manager before closing this object. Closing stops the
    loop, so a write still running is cancelled. ``aclose()`` and
    ``close()`` close the client once, and a second call does nothing.

    Example::

        shared = SharedMemory.open(settings)
        manager = shared.session_manager("session-1", user_id="alice")
        agent = Agent(model=model, session_manager=manager)
        ...
        await manager.aclose()
        await shared.aclose()
    """

    def __init__(self, client: MemoryClient, bridge: AsyncBridge, request_timeout: float) -> None:
        """Wrap a client connected on ``bridge``. Use ``open`` instead."""
        self._client = client
        self._bridge = bridge
        self._request_timeout = request_timeout
        self._lock = threading.Lock()
        self._closed = False

    @classmethod
    def open(cls, settings: MemorySettings, *, request_timeout: float = 30.0) -> SharedMemory:
        """Build a client from ``settings`` and connect it on a new loop.

        Blocks until the client is connected. ``request_timeout`` bounds the
        connect and closing the client. It is also the default
        ``request_timeout`` of each manager this object builds.

        Raises:
            Exception: Connecting failed. The loop is stopped before the
                error propagates.
        """
        from neo4j_agent_memory import MemoryClient

        bridge = AsyncBridge(timeout=request_timeout)
        client = MemoryClient(settings)
        try:
            bridge.run(client.connect(), request_timeout)
        except BaseException:
            bridge.close()
            raise
        return cls(client, bridge, request_timeout)

    @property
    def client(self) -> MemoryClient:
        """The shared client. Run its coroutines on this object's loop only."""
        return self._client

    @property
    def closed(self) -> bool:
        """True once ``close()`` or ``aclose()`` has been called."""
        return self._closed

    def session_manager(self, session_id: str, **kwargs: Any) -> Neo4jSessionManager:
        """Return a manager for ``session_id`` on the shared client and loop.

        ``kwargs`` go to ``Neo4jSessionManager``. ``request_timeout``
        defaults to this object's. Passing ``settings``, ``memory_client``
        or ``bridge`` raises ``TypeError``.

        Raises:
            RuntimeError: This object is closed.
        """
        if self._closed:
            raise RuntimeError("SharedMemory is closed")
        kwargs.setdefault("request_timeout", self._request_timeout)
        return Neo4jSessionManager(
            session_id, memory_client=self._client, bridge=self._bridge, **kwargs
        )

    def _begin_close(self) -> bool:
        """Mark this object closed. Returns False when it already was."""
        with self._lock:
            if self._closed:
                return False
            self._closed = True
            return True

    async def aclose(self) -> None:
        """Close the client and stop the loop without blocking the caller's loop."""
        if not self._begin_close():
            return
        try:
            if self._client.is_connected:
                await self._bridge.run_async(self._client.close(), self._request_timeout)
        finally:
            await asyncio.to_thread(self._bridge.close)

    def close(self) -> None:
        """Close the client and stop the loop, blocking until done."""
        if not self._begin_close():
            return
        try:
            if self._client.is_connected:
                self._bridge.run(self._client.close(), self._request_timeout)
        finally:
            self._bridge.close()
