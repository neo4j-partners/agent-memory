"""Shared utilities for agent framework integrations.

This module provides common functionality used across all framework integrations,
including async/sync bridging, input validation, and helper functions.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import logging
import threading
from collections.abc import Callable, Coroutine
from functools import wraps
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")

logger = logging.getLogger(__name__)

# Shared executor for all integrations to avoid per-call overhead
# Using a module-level executor with lazy initialization
_executor: concurrent.futures.ThreadPoolExecutor | None = None
_EXECUTOR_MAX_WORKERS = 4
_SYNC_TIMEOUT_SECONDS = 30

#: How long AsyncBridge shutdown lets cancelled tasks unwind.
_SHUTDOWN_TIMEOUT_SECONDS = 5.0

_CLOSED_IN_FLIGHT = "AsyncBridge closed with work in flight"
_THREAD_DIED = "AsyncBridge loop thread stopped unexpectedly; the bridge cannot be reused"
_ON_LOOP_THREAD = (
    "AsyncBridge.{method}() cannot run on the bridge's own loop thread, "
    "because it would wait on that loop. Call it from another thread, or "
    "await run_async() on the loop."
)


def _cancel_tasks(loop: asyncio.AbstractEventLoop) -> None:
    """Cancel the loop's pending tasks and let them unwind, within a bound."""
    tasks = asyncio.all_tasks(loop)
    for task in tasks:
        task.cancel()
    if tasks:
        loop.run_until_complete(asyncio.wait(tasks, timeout=_SHUTDOWN_TIMEOUT_SECONDS))


def _shutdown_executor(executor: concurrent.futures.ThreadPoolExecutor) -> None:
    """Shut down ``executor``, waiting up to ``_SHUTDOWN_TIMEOUT_SECONDS`` for its jobs.

    ``loop.shutdown_default_executor()`` waits for every job. On Python
    3.11 it takes no timeout and joins its thread even when cancelled, so
    one hung ``asyncio.to_thread`` job would hang the loop's shutdown. A
    job still running after the wait is left to finish on its own thread,
    jobs not yet started are cancelled, and a WARNING says so.
    """
    waiter = threading.Thread(target=executor.shutdown, name="asyncbridge-executor", daemon=True)
    waiter.start()
    waiter.join(_SHUTDOWN_TIMEOUT_SECONDS)
    if waiter.is_alive():
        logger.warning(
            "An AsyncBridge executor job is still running after %ss. The loop "
            "closes without waiting for it.",
            _SHUTDOWN_TIMEOUT_SECONDS,
        )
        executor.shutdown(wait=False, cancel_futures=True)


class AsyncBridge:
    """Persistent background-event-loop bridge for sync code driving async clients.

    Unlike :func:`run_sync` (one-shot ``asyncio.run`` per call), AsyncBridge keeps
    ONE loop alive across calls — required when the async client holds loop-bound
    state (httpx/bolt transports), e.g. a long-lived integration object making
    many sequential calls. Use run_sync for stateless one-shot calls; use
    AsyncBridge when the same client instance must serve multiple calls.

    ``run()`` blocks the calling thread until the result arrives. From async
    code, await ``run_async()`` instead, so the caller's event loop keeps
    running.

    The loop thread starts lazily on first use. ``close()`` is final: a
    later ``run()`` raises ``RuntimeError`` instead of starting a new
    thread, so a closed owner cannot leak a loop thread or reuse
    transports bound to the old loop. ``run_async()`` behaves the same way.
    A loop thread that stops without ``close()`` is fatal too. The bridge is
    marked broken, and later calls raise ``RuntimeError`` rather than start
    a fresh loop that the owner's clients were never bound to.

    When the loop stops, its thread shuts it down the way ``asyncio.run``
    does. It cancels the tasks still pending, gives them
    ``_SHUTDOWN_TIMEOUT_SECONDS`` to unwind, and shuts down async
    generators. It waits up to the same bound for the loop's default
    executor, which runs ``asyncio.to_thread`` jobs, and closes the loop. A task can start after the
    first cancel, so the tasks still pending are cancelled once more just
    before the loop closes. A caller still waiting on cancelled work gets
    ``RuntimeError`` rather than ``CancelledError``.

    ``run()`` and ``close()`` wait on the loop, so they raise
    ``RuntimeError`` when called on the loop thread itself. Without the
    check, ``run()`` would block until its timeout.
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._closed = False
        self._broken = False
        # Futures from submit() that are not done yet. Shutdown cancels any
        # whose task never started, so no caller waits on them forever.
        # Once the loop stops, _stopped makes submit() refuse new work, so
        # no future is added after that sweep.
        self._futures: set[concurrent.futures.Future[Any]] = set()
        self._futures_lock = threading.Lock()
        self._stopped = False  # under _futures_lock
        # The last task chained under each key. Loop thread only.
        self._tails: dict[str, asyncio.Task[Any]] = {}
        # Callbacks from when_done() that have not run yet. Loop thread only.
        self._when_done: dict[object, tuple[asyncio.Task[Any], Callable[[], None]]] = {}

    @property
    def closed(self) -> bool:
        """True once ``close()`` has been called."""
        return self._closed

    def chain(
        self,
        key: str,
        start: Callable[[asyncio.Task[Any] | None], asyncio.Task[T]],
    ) -> asyncio.Task[T]:
        """Start a task that follows the last task chained under ``key``.

        ``start`` gets that task, or None when none is running, and returns
        the new task. The new task should wait for the one it gets. Every
        user of this bridge that chains under one key then shares one
        order. Runs on the loop thread only.
        """
        task = start(self._tails.get(key))
        self._tails[key] = task
        task.add_done_callback(lambda done: self._drop_tail(key, done))
        return task

    def _drop_tail(self, key: str, task: asyncio.Task[Any]) -> None:
        """Forget ``task`` as the tail of ``key``, unless a later task replaced it."""
        if self._tails.get(key) is task:
            del self._tails[key]

    def when_done(self, task: asyncio.Task[Any], callback: Callable[[], None]) -> None:
        """Call ``callback`` exactly once, when ``task`` ends.

        A task that ignores the shutdown cancel is still pending when the
        loop closes, and its done callbacks never run. Shutdown then calls
        ``callback`` itself and logs a WARNING that names the task. If that
        call raises, a WARNING logs the error. Runs on the loop thread only.
        """
        token = object()
        self._when_done[token] = (task, callback)
        task.add_done_callback(lambda _: self._call_when_done(token))

    def _call_when_done(self, token: object) -> None:
        """Run the callback registered under ``token``, unless it already ran."""
        if (entry := self._when_done.pop(token, None)) is not None:
            entry[1]()

    def _call_stranded(self) -> None:
        """Run the callbacks of tasks that are still pending at shutdown.

        A callback that raises is logged as a WARNING, and the rest still run.
        """
        for token in list(self._when_done):
            task, _ = self._when_done[token]
            logger.warning(
                "Task %r ignored cancellation and is dropped as the AsyncBridge loop "
                "closes. Its done callback runs now.",
                task.get_name(),
            )
            try:
                self._call_when_done(token)
            except Exception:
                logger.warning(
                    "The done callback of task %r raised.", task.get_name(), exc_info=True
                )

    @property
    def on_loop_thread(self) -> bool:
        """True when the caller runs on the bridge's loop thread."""
        thread = self._thread
        return thread is not None and thread is threading.current_thread()

    def _check_off_loop(self, method: str) -> None:
        if self.on_loop_thread:
            raise RuntimeError(_ON_LOOP_THREAD.format(method=method))

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._closed:
                raise RuntimeError("AsyncBridge is closed")
            if self._thread is not None and not self._thread.is_alive():
                self._broken = True
            if self._broken:
                raise RuntimeError(_THREAD_DIED)
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
                # The loop thread shuts this executor down itself, with a
                # bound, so a hung job cannot hang the loop's shutdown.
                executor = concurrent.futures.ThreadPoolExecutor(
                    thread_name_prefix="asyncbridge-worker"
                )
                self._loop.set_default_executor(executor)
                self._thread = threading.Thread(
                    target=self._run_loop,
                    args=(self._loop, executor),
                    name="neo4j-strands-session-manager",
                    daemon=True,
                )
                self._thread.start()
            return self._loop

    def _run_loop(
        self, loop: asyncio.AbstractEventLoop, executor: concurrent.futures.ThreadPoolExecutor
    ) -> None:
        """Loop thread body: run until stopped, then shut down like ``asyncio.run``."""
        asyncio.set_event_loop(loop)
        try:
            loop.run_forever()
        finally:
            if not self._closed:
                # Not lock-guarded: close() holds _lock while it joins us.
                self._broken = True
            with self._futures_lock:
                self._stopped = True
            try:
                _cancel_tasks(loop)
                loop.run_until_complete(loop.shutdown_asyncgens())
                _shutdown_executor(executor)
            finally:
                # Cancel before loop.close(). A cancelled future cancels its
                # task through this loop, which must still be open.
                with self._futures_lock:
                    stranded = [f for f in self._futures if not f.done()]
                for future in stranded:
                    future.cancel()
                try:
                    # A task can start during the steps above. Cancel it now,
                    # or loop.close() destroys it while it is pending.
                    _cancel_tasks(loop)
                finally:
                    try:
                        self._call_stranded()
                    finally:
                        loop.close()

    def _forget(self, future: concurrent.futures.Future[Any]) -> None:
        with self._futures_lock:
            self._futures.discard(future)

    def submit(self, coro: Coroutine[Any, Any, T]) -> concurrent.futures.Future[T]:
        """Schedule ``coro`` on the background loop and return its future.

        Raises:
            RuntimeError: The bridge is closed. ``coro`` is closed unscheduled,
                unless the loop stopped after it was handed over. Then its
                future is cancelled, and so is its task if it starts.
        """
        try:
            loop = self._ensure_loop()
        except RuntimeError:
            coro.close()  # never scheduled; avoid a "never awaited" warning
            raise
        try:
            future = asyncio.run_coroutine_threadsafe(coro, loop)
        except RuntimeError as e:  # the loop closed after _ensure_loop returned
            coro.close()
            raise RuntimeError("AsyncBridge is closed") from e
        with self._futures_lock:
            stopped = self._stopped
            if not stopped:
                self._futures.add(future)
        if stopped:  # the loop stopped after _ensure_loop returned
            future.cancel()
            raise RuntimeError("AsyncBridge is closed")
        future.add_done_callback(self._forget)
        return future

    def run(self, coro: Coroutine[Any, Any, T], timeout: float | None = None) -> T:
        """Submit ``coro`` to the background loop and block for its result.

        Generic over the coroutine's result type, so callers get the
        awaited value's real type back (e.g. ``run(client.add_message(...))``
        returns ``Message``, not ``Any``).

        Raises:
            RuntimeError: The bridge is closed, or the caller runs on the
                loop thread. ``coro`` is closed unscheduled.
        """
        try:
            self._check_off_loop("run")
        except RuntimeError:
            coro.close()
            raise
        future = self.submit(coro)
        try:
            return future.result(timeout=timeout if timeout is not None else self._timeout)
        except concurrent.futures.CancelledError as e:
            raise RuntimeError(_CLOSED_IN_FLIGHT) from e

    async def run_async(self, coro: Coroutine[Any, Any, T], timeout: float | None = None) -> T:
        """Submit ``coro`` to the background loop and await its result.

        Use this from async code. The caller's event loop stays free while
        ``coro`` runs, so its other tasks keep running. ``coro`` still runs
        on the bridge loop, next to the client's loop-bound transports.

        On timeout this raises ``TimeoutError`` and cancels the bridge
        task. Cancelling the caller also cancels the bridge task.
        """
        return await self.wait(self.submit(coro), timeout)

    async def wait(
        self,
        future: concurrent.futures.Future[T],
        timeout: float | None = None,
        *,
        shield: bool = False,
    ) -> T:
        """Await a future from ``submit()`` without blocking the caller's loop.

        On timeout this raises ``TimeoutError``. Without ``shield``, the
        timeout or cancelling the caller also cancels the bridge task. With
        ``shield``, only the wait stops, and the task runs to completion.

        Raises:
            RuntimeError: The bridge shut down and cancelled the task.
        """
        waiter: asyncio.Future[T] = asyncio.wrap_future(future)
        if shield:
            waiter = asyncio.shield(waiter)
        try:
            return await asyncio.wait_for(waiter, timeout if timeout is not None else self._timeout)
        except asyncio.CancelledError as e:
            task = asyncio.current_task()
            if future.cancelled() and (task is None or not task.cancelling()):
                raise RuntimeError(_CLOSED_IN_FLIGHT) from e
            raise

    def close(self) -> None:
        """Stop the loop and join its thread. Safe to call repeatedly.

        The loop thread then cancels the work still in flight, lets it
        unwind, and closes the loop. Callers waiting on that work get
        ``RuntimeError``. Drain work you need before closing (the session
        manager waits for its writes first).

        If the thread does not finish within the join timeout, we drop our
        references and let the daemon thread finish its shutdown on its
        own. The bridge is always marked closed, so a later ``run()``
        raises instead of restarting.

        Raises:
            RuntimeError: The caller runs on the loop thread. The bridge
                stays open.
        """
        self._check_off_loop("close")
        with self._lock:
            self._closed = True
            loop = self._loop
            thread = self._thread
            self._loop = None
            self._thread = None
            if loop is None or thread is None or not thread.is_alive():
                return
            with contextlib.suppress(RuntimeError):  # its thread closed it already
                loop.call_soon_threadsafe(loop.stop)
            # Shutdown cancels tasks twice and waits for the executor, each
            # time up to the bound.
            thread.join(timeout=3 * _SHUTDOWN_TIMEOUT_SECONDS + 5)


def _get_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Get or create the shared thread pool executor."""
    global _executor
    if _executor is None:
        _executor = concurrent.futures.ThreadPoolExecutor(max_workers=_EXECUTOR_MAX_WORKERS)
    return _executor


def run_sync(func: Callable[P, Coroutine[Any, Any, T]]) -> Callable[P, T]:
    """
    Decorator to run an async function synchronously.

    This decorator wraps an async function to make it callable from sync code.
    It handles the case where the code is already running in an async context
    by using a thread pool executor.

    Args:
        func: The async function to wrap

    Returns:
        A sync wrapper function

    Example:
        @run_sync
        async def my_async_function(arg: str) -> str:
            return await some_async_operation(arg)

        # Now callable synchronously
        result = my_async_function("test")
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # Running in async context - use thread pool
            executor = _get_executor()
            coro = func(*args, **kwargs)
            future = executor.submit(lambda: asyncio.run(coro))
            return future.result(timeout=_SYNC_TIMEOUT_SECONDS)
        else:
            # Not in async context - run directly
            return asyncio.run(func(*args, **kwargs))

    return wrapper


def validate_session_id(session_id: str) -> str:
    """
    Validate and normalize a session ID.

    Args:
        session_id: The session ID to validate

    Returns:
        The normalized session ID

    Raises:
        ValueError: If session_id is invalid
    """
    if not session_id:
        raise ValueError("session_id must be a non-empty string")
    if not isinstance(session_id, str):
        raise ValueError(f"session_id must be a string, got {type(session_id).__name__}")
    return session_id.strip()


def validate_query(query: str, allow_empty: bool = False) -> str:
    """
    Validate a search query.

    Args:
        query: The query string to validate
        allow_empty: Whether to allow empty queries

    Returns:
        The normalized query string

    Raises:
        ValueError: If query is invalid
    """
    if not isinstance(query, str):
        raise ValueError(f"query must be a string, got {type(query).__name__}")

    normalized = query.strip()
    if not allow_empty and not normalized:
        raise ValueError("query must be a non-empty string")

    return normalized


def validate_limit(limit: int, max_limit: int = 1000) -> int:
    """
    Validate a limit parameter.

    Args:
        limit: The limit value to validate
        max_limit: Maximum allowed limit

    Returns:
        The validated limit

    Raises:
        ValueError: If limit is invalid
    """
    if not isinstance(limit, int):
        raise ValueError(f"limit must be an integer, got {type(limit).__name__}")
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if limit > max_limit:
        raise ValueError(f"limit must be at most {max_limit}")
    return limit


def validate_threshold(threshold: float) -> float:
    """
    Validate a similarity threshold.

    Args:
        threshold: The threshold value (0.0 to 1.0)

    Returns:
        The validated threshold

    Raises:
        ValueError: If threshold is invalid
    """
    if not isinstance(threshold, (int, float)):
        raise ValueError(f"threshold must be a number, got {type(threshold).__name__}")
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0")
    return float(threshold)


def format_context_section(title: str, items: list[str], max_items: int = 10) -> str:
    """
    Format a section of context for inclusion in prompts.

    Args:
        title: Section title
        items: List of context items
        max_items: Maximum items to include

    Returns:
        Formatted section string
    """
    if not items:
        return ""

    limited_items = items[:max_items]
    formatted = [f"## {title}"]
    for item in limited_items:
        formatted.append(f"- {item}")

    if len(items) > max_items:
        formatted.append(f"- ... and {len(items) - max_items} more")

    return "\n".join(formatted)


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to add when truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


class IntegrationError(Exception):
    """Base exception for integration errors."""

    pass


class ValidationError(IntegrationError):
    """Exception raised for validation errors."""

    pass


class ConnectionError(IntegrationError):
    """Exception raised for connection errors."""

    pass


class TimeoutError(IntegrationError):
    """Exception raised for timeout errors."""

    pass
