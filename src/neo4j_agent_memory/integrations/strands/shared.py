"""One memory client and one background loop shared by many Strands sessions.

Each ``Neo4jSessionManager`` built from ``settings=`` owns a client and a
background loop. The async Neo4j driver and the NAMS HTTP client bind to
the loop they connect on, so every session then holds its own connection
pool and thread. A server that runs many sessions builds one
``SharedMemory`` at startup instead, and asks it for a manager per session.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Any

from neo4j_agent_memory.integrations.base import AsyncBridge
from neo4j_agent_memory.integrations.strands.session_manager import Neo4jSessionManager

if TYPE_CHECKING:
    from collections.abc import Callable

    from neo4j_agent_memory import MemoryClient, MemorySettings

__all__ = ["SharedMemory"]

logger = logging.getLogger(__name__)


class SharedMemory:
    """A memory client and a background loop that session managers share.

    Build it once with ``SharedMemory.open(settings)``, then call
    ``session_manager(session_id)`` for each session. The managers share
    the client, its connection pool, and one background thread. A manager
    built this way closes neither: its ``close()`` and ``aclose()`` write
    its own messages and return.

    Close every manager before closing this object. A manager's close
    leaves its background work running, such as deferred entity
    extraction. ``aclose()`` and ``close()`` wait for every task still
    running on the loop, up to ``timeout`` seconds. Then they close the
    client and stop the loop, which cancels the tasks still running. A
    WARNING gives their number. A second call does nothing.

    Example::

        shared = SharedMemory.open(settings)
        manager = shared.session_manager("session-1", user_id="alice")
        agent = Agent(model=model, session_manager=manager)
        ...
        await manager.aclose()
        await shared.aclose()
    """

    def __init__(
        self,
        client: MemoryClient,
        bridge: AsyncBridge,
        request_timeout: float,
        background_hook: Callable[[str], Callable[[], None]] | None = None,
    ) -> None:
        """Wrap a client connected on ``bridge``. Use ``open`` instead."""
        self._client = client
        self._bridge = bridge
        self._request_timeout = request_timeout
        self._background_hook = background_hook
        self._lock = threading.Lock()
        self._closed = False

    @classmethod
    def open(
        cls,
        settings: MemorySettings,
        *,
        request_timeout: float = 30.0,
        background_hook: Callable[[str], Callable[[], None]] | None = None,
    ) -> SharedMemory:
        """Build a client from ``settings`` and connect it on a new loop.

        Blocks until the client is connected. ``request_timeout`` bounds the
        connect and closing the client. It is also the default
        ``request_timeout`` of each manager this object builds.

        ``background_hook`` goes to every manager this object builds. It
        is called with a short name when a deferred extraction or a
        tool-call recording starts, and the function it returns runs once
        when that task ends. Both run on the loop thread, so they must be
        thread-safe and fast. See ``Neo4jSessionManager``.

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
        return cls(client, bridge, request_timeout, background_hook)

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
        defaults to this object's. Passing ``settings``, ``memory_client``,
        ``bridge`` or ``background_hook`` raises ``TypeError``.

        Raises:
            RuntimeError: This object is closed.
        """
        if self._closed:
            raise RuntimeError("SharedMemory is closed")
        kwargs.setdefault("request_timeout", self._request_timeout)
        return Neo4jSessionManager(
            session_id,
            memory_client=self._client,
            bridge=self._bridge,
            background_hook=self._background_hook,
            **kwargs,
        )

    def _begin_close(self) -> bool:
        """Mark this object closed. Returns False when it already was."""
        with self._lock:
            if self._closed:
                return False
            self._closed = True
            return True

    async def aclose(self, timeout: float = 30.0) -> None:
        """Like ``close()``, without blocking the caller's loop."""
        _check_timeout(timeout)
        if not self._begin_close():
            return
        try:
            try:
                await self._bridge.run_async(
                    self._await_tasks(timeout), timeout + self._request_timeout
                )
            finally:
                if self._client.is_connected:
                    await self._bridge.run_async(self._client.close(), self._request_timeout)
        finally:
            await asyncio.to_thread(self._bridge.close)

    def close(self, timeout: float = 30.0) -> None:
        """Wait for the loop's tasks, close the client, and stop the loop.

        Blocks until done. ``timeout`` bounds the wait for the tasks still
        running on the loop. It is separate from ``request_timeout``,
        which bounds closing the client.

        Raises:
            ValueError: ``timeout`` is not positive. The object stays open.
        """
        _check_timeout(timeout)
        if not self._begin_close():
            return
        try:
            try:
                self._bridge.run(self._await_tasks(timeout), timeout + self._request_timeout)
            finally:
                if self._client.is_connected:
                    self._bridge.run(self._client.close(), self._request_timeout)
        finally:
            self._bridge.close()

    async def _await_tasks(self, timeout: float) -> None:
        """Wait up to ``timeout`` for the loop's other tasks. Runs on the loop.

        A task can start another while this waits, so the wait looks again
        until no task is left or the time runs out. A WARNING gives the
        number still running.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        this = asyncio.current_task()
        while pending := asyncio.all_tasks() - {this}:
            remaining = deadline - loop.time()
            if remaining <= 0:
                logger.warning(
                    "Closing SharedMemory while %d background task(s) are still "
                    "running after %ss. They are cancelled.",
                    len(pending),
                    timeout,
                )
                return
            await asyncio.wait(pending, timeout=remaining)


def _check_timeout(timeout: float) -> None:
    """Raise ``ValueError`` unless ``timeout`` is a positive number of seconds."""
    if not timeout > 0:
        raise ValueError(f"timeout must be positive, not {timeout!r}.")
