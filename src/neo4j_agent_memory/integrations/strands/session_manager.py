"""Strands SessionManager backed by neo4j-agent-memory (bolt or NAMS).

Maps a Strands session onto one ``Conversation`` — no Strands-specific
node types are written to the graph. Persistence is memory-grade: text
turns are stored (and feed entity extraction / the shared brain);
tool-use blocks and ``agent.state`` are not round-tripped.

Sibling modules own the non-session concerns: ``_messages`` (Strands ↔
Memory message mapping), ``_retrieval`` (long-term search + context-block
formatting), and ``integrations.base`` (the sync↔async bridge).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import logging
import threading
from collections import deque
from typing import TYPE_CHECKING, Any
from uuid import UUID

from neo4j_agent_memory.integrations.base import AsyncBridge as _AsyncBridge
from neo4j_agent_memory.integrations.strands._messages import (
    _message_text,
    _normalize_history,
    _to_strands_message,
)
from neo4j_agent_memory.integrations.strands._retrieval import (
    Neo4jRetrievalConfig,
    _retrieve_context,
)

try:
    from strands.hooks import (
        AfterInvocationEvent,
        HookRegistry,
        MessageAddedEvent,
    )
    from strands.session.session_manager import SessionManager
    from strands.types.exceptions import SessionException
except ImportError as import_error:  # pragma: no cover - exercised via package __init__
    raise ImportError(
        "strands-agents is required for the Strands session manager. "
        "Install with: pip install neo4j-agent-memory[strands]"
    ) from import_error

if TYPE_CHECKING:
    from types import TracebackType

    from strands import Agent
    from strands.types.content import Message as StrandsMessage
    from strands.types.tools import ToolUse

    from neo4j_agent_memory import MemoryClient, MemorySettings
    from neo4j_agent_memory.memory.short_term import Message as StoredMessage
    from neo4j_agent_memory.nams.endpoints import TransportMode

logger = logging.getLogger(__name__)

#: Conversation-metadata key linking a Conversation to a Strands session id.
_SESSION_KEY = "strands_session_id"

#: Shared by the double-extraction ValueError and the double-injection warning.
_COEXISTENCE_HINT = (
    "Set extraction=False on Neo4jMemoryStore (recall only, recommended), or "
    "extract_entities=False on Neo4jSessionManager (let the store own extraction)."
)

__all__ = ["Neo4jRetrievalConfig", "Neo4jSessionManager"]


class Neo4jSessionManager(SessionManager):
    """Strands SessionManager persisting conversations to neo4j-agent-memory.

    Memory-grade persistence (see design spec): text turns are stored
    and restored; tool-use blocks and ``agent.state`` are not. One
    Strands session maps to one ``Conversation``.

    Provide exactly one of ``memory_client`` (bolt or NAMS; left open on
    close unless we connected it) or ``settings`` (a client is
    constructed and owned by the manager).

    Single-``Agent`` scope. The base class registers multi-agent
    (``MultiAgentInitializedEvent`` / ``AfterNodeCallEvent`` /
    ``AfterMultiAgentInvocationEvent``) and bidirectional-agent hooks
    unconditionally, and its implementations of them raise
    ``NotImplementedError`` naming this class — so attaching this manager to a
    Strands Graph/Swarm or a ``BidiAgent`` fails at the first dispatch. Those
    topologies want one of Strands' own repository-backed managers
    (``FileSessionManager`` / ``S3SessionManager`` / ``RepositorySessionManager``);
    the "shared brain" pattern this manager is built for is N independent
    ``Agent`` instances, each with its own manager, over one graph.
    """

    def __init__(
        self,
        session_id: str,
        memory_client: MemoryClient | None = None,
        settings: MemorySettings | None = None,
        *,
        user_id: str | None = None,
        retrieval_config: Neo4jRetrievalConfig | None = None,
        extract_entities: bool = True,
        record_tool_calls: bool = False,
        request_timeout: float = 30.0,
        restore_limit: int | None = None,
    ) -> None:
        if (memory_client is None) == (settings is None):
            raise ValueError(
                "Provide exactly one of memory_client= or settings= to Neo4jSessionManager."
            )
        self.session_id = session_id
        self._user_id = user_id
        self._retrieval_config = retrieval_config
        self._extract_entities = extract_entities
        self._record_tool_calls = record_tool_calls
        self._restore_limit = restore_limit
        self._request_timeout = request_timeout
        self._bridge = _AsyncBridge(timeout=request_timeout)
        # True when this manager owns the client lifecycle (settings-constructed
        # client, or when we performed the connect on an externally-supplied client).
        self._should_close_client: bool = settings is not None
        self._client: MemoryClient
        if settings is not None:
            from neo4j_agent_memory import MemoryClient as _MemoryClient

            self._client = _MemoryClient(settings)
        else:
            assert memory_client is not None  # guaranteed by the XOR check above
            self._client = memory_client
        self._conversation_key: str | None = None
        self._pending: StrandsMessage | None = None  # latest message, still redactable
        self._ready: deque[StrandsMessage] = deque()  # older messages waiting for a hook to write
        # _pending and _ready are read and changed on the agent's thread and
        # on the bridge thread, so every access holds this lock.
        self._queue_lock = threading.Lock()
        self._drain_lock: asyncio.Lock | None = None  # bridge loop only; one drain at a time
        self._inflight: concurrent.futures.Future[None] | None = None  # latest drain
        self._restored: list[StrandsMessage] | None = None  # history preloaded by aopen()
        self._last_persisted: StoredMessage | None = None  # last stored (late redaction)
        self._trace_id: UUID | None = None  # lazy reasoning trace (record_tool_calls)
        self._closed = False
        self._closing: concurrent.futures.Future[None] | None = None  # done when close ends
        self._warned_double_injection = False

    @property
    def _is_nams(self) -> bool:
        """Single home for the backend dichotomy: NAMS lacks preference/fact
        search and message delete, and keys conversations by server-issued
        UUID. Replace with a client-level capability probe when one exists."""
        return bool(self._client.is_nams)

    # --------------------------------------------------------- factory methods

    @classmethod
    def for_nams(
        cls,
        session_id: str,
        *,
        endpoint: str | None = None,
        api_key: str | None = None,
        transport_mode: TransportMode = "auto",
        **kwargs: Any,
    ) -> Neo4jSessionManager:
        """Build a NAMS-backed manager using the same env-var conventions as
        ``nams_context_graph_tools()`` (MEMORY_ENDPOINT / MEMORY_API_KEY)."""
        from neo4j_agent_memory.integrations.strands.config import (
            build_nams_settings,
            resolve_nams_connection,
        )

        endpoint, api_key = resolve_nams_connection(endpoint, api_key)
        settings = build_nams_settings(endpoint, api_key, transport_mode)
        return cls(session_id, settings=settings, **kwargs)

    # ------------------------------------------------------------ lifecycle

    async def _aconnect(self) -> None:
        if not self._client.is_connected:
            await self._client.connect()
            self._should_close_client = True

    async def _aresolve_conversation(self) -> str:
        """Return the backend session key for short_term calls.

        NAMS issues conversation UUIDs, so we locate (or create) the
        conversation whose metadata carries our Strands session id. Bolt
        keys conversations by session_id directly (auto-created on first
        message).
        """
        if not self._is_nams:
            return self.session_id
        short_term = self._client.short_term
        # Narrow server-side to this user; the NAMS client pages past the
        # service's 200-per-page cap with next_cursor up to this limit.
        conversations = await short_term.list_conversations(
            user_identifier=self._user_id, limit=1000
        )
        for conversation in conversations:
            if (conversation.metadata or {}).get(_SESSION_KEY) == self.session_id:
                return str(conversation.id)
        created = await short_term.create_conversation(
            session_id=self.session_id,
            metadata={_SESSION_KEY: self.session_id, "session_type": "AGENT"},
            user_identifier=self._user_id,
        )
        return str(created.id)

    async def _aresolve_key(self) -> str:
        """Connect and resolve the conversation key (without loading history)."""
        await self._aconnect()
        key = self._conversation_key
        if key is None:
            key = await self._aresolve_conversation()
            self._conversation_key = key
        return key

    async def _ainitialize(self) -> list[StrandsMessage]:
        key = await self._aresolve_key()
        conversation = await self._client.short_term.get_conversation(
            key, limit=self._restore_limit
        )
        return [_to_strands_message(m) for m in conversation.messages]

    def _check_open(self) -> None:
        """Raise SessionException if ``close()`` has already run."""
        if self._closed:
            raise SessionException(f"Session manager for session {self.session_id!r} is closed")

    async def aopen(self) -> None:
        """Connect and load history without blocking the caller's event loop.

        Optional. Await it before building the ``Agent``. ``initialize`` then
        uses the preloaded history and does no I/O, and any seeded messages
        are written by the first hook instead. Without ``aopen()``,
        ``initialize`` loads history itself and blocks until it arrives.

        Raises:
            SessionException: The manager is closed, or loading failed.
        """
        self._check_open()
        try:
            self._restored = await self._bridge.run_async(self._ainitialize())
        except Exception as e:
            raise SessionException(f"Failed to initialize session {self.session_id!r}") from e

    # ----------------------------------------------------- SessionManager API

    def _our_stores(self, agent: Agent) -> list[Any]:
        """Neo4jMemoryStore instances registered on the agent's MemoryManager.

        Reads ``MemoryManager._stores`` — a private, pinned by
        ``test_strands_coexistence.py::TestPrivateAttributeCoupling``.
        """
        manager = getattr(agent, "memory_manager", None)
        if manager is None:
            return []
        from neo4j_agent_memory.integrations.strands.memory_store import Neo4jMemoryStore

        stores = getattr(manager, "_stores", None) or []
        return [store for store in stores if isinstance(store, Neo4jMemoryStore)]

    def _check_memory_manager_coexistence(self, agent: Agent) -> None:
        """Guard the two overlaps possible with a paired Neo4jMemoryStore.

        Pairing itself is supported: this manager persists/restores the
        transcript, the store feeds the agent loop. Not supported: both
        sides extracting the same turns, both sides injecting into the
        same user message.
        """
        stores = self._our_stores(agent)
        if not stores:
            return

        extracting = [store for store in stores if store.extraction]
        if extracting:
            # NAMS extracts every write server-side (add_message drops
            # extraction kwargs there), so our own extract_entities flag
            # cannot avert the duplication on that backend.
            if self._extract_entities or self._is_nams:
                raise ValueError(
                    f"Neo4jSessionManager and Neo4jMemoryStore "
                    f"'{extracting[0].name}' would write and extract the same "
                    f"turns twice. {_COEXISTENCE_HINT}"
                )

        if self._retrieval_config is not None and not self._warned_double_injection:
            manager = getattr(agent, "memory_manager", None)
            if getattr(manager, "_injection_config", None) is not False:
                self._warned_double_injection = True
                logger.warning(
                    "Memory is being injected twice: Neo4jRetrievalConfig on this "
                    "session manager and MemoryManager injection from "
                    "Neo4jMemoryStore '%s'. Drop retrieval_config, or pass "
                    "injection=False to MemoryManager.",
                    stores[0].name,
                )

    def initialize(self, agent: Agent, **kwargs: Any) -> None:
        """Restore the agent's conversation history from the graph.

        Restored history is normalized so roles alternate, starting with a
        user message. User messages that never got a reply are left out of
        the agent's history, with a warning. Stored messages are not
        changed.
        """
        self._check_open()
        self._check_memory_manager_coexistence(agent)
        preloaded = self._restored is not None
        if self._restored is not None:
            stored, self._restored = self._restored, None
        else:
            try:
                stored = self._bridge.run(self._ainitialize())
            except Exception as e:
                raise SessionException(f"Failed to initialize session {self.session_id!r}") from e
        if stored:
            # Normalizing can leave nothing, for example a lone stale user
            # message. The session still exists, so do not seed it below.
            restored, dropped = _normalize_history(stored)
            if dropped:
                logger.warning(
                    "Session %r: leaving %d user message(s) with no reply out "
                    "of the restored history.",
                    self.session_id,
                    len(dropped),
                )
            agent.messages.clear()
            agent.messages.extend(restored)
        elif agent.messages:
            # New session seeded with pre-existing in-memory history.
            for message in list(agent.messages):
                self.append_message(message, agent)
            if preloaded:
                # aopen() promised no I/O here; the first hook writes these.
                self._release_pending()
            else:
                self._flush_buffer()

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Wire base persistence hooks, then our flush + injection hooks.

        Our hooks are coroutines. Strands awaits them on the agent's loop,
        and they await memory I/O on the bridge loop, so the agent's loop
        keeps running other tasks meanwhile.

        Order matters: Strands fires ``MessageAddedEvent`` callbacks in
        registration order, so buffering (base ``MessageAddedEvent`` ->
        append_message) always runs before the write of older messages and
        before context injection.
        AfterInvocationEvent dispatches in reverse registration order, so our
        flush runs before the base no-op sync_agent. That same reverse ordering
        means any user hooks registered after this manager run BEFORE our flush,
        so external AfterInvocationEvent hooks observe the final turn while it is
        still un-persisted; hooks that need the persisted message should read it
        on the next turn instead.

        Both orderings hold *within one ``HookOrder`` priority group*: since
        strands 1.5x, ``HookRegistry.add_callback`` takes a keyword-only
        ``order`` and sorts by it first (lower runs first), preserving —
        or, for reverse-dispatch events, inverting — registration order only
        inside a group. Every callback registered here uses the default
        priority, so a hook deliberately registered at ``HookOrder.SDK_FIRST``
        or ``SDK_LAST`` can still straddle our flush.
        """
        super().register_hooks(registry, **kwargs)
        registry.add_callback(AfterInvocationEvent, self._on_after_invocation)
        registry.add_callback(MessageAddedEvent, self._on_message_flush)
        if self._retrieval_config is not None:
            registry.add_callback(MessageAddedEvent, self._on_message_added)

    async def _on_after_invocation(self, event: AfterInvocationEvent) -> None:
        """Write the rest of the turn.

        Strands fires this from its ``finally``, so it also runs when a
        caller abandons a stream and the generator is finalized after
        ``aclose()``. With nothing buffered there is nothing to lose, so a
        closed manager returns quietly. It raises only when messages would
        be dropped.
        """
        if self._closed:
            with self._queue_lock:
                buffered = self._pending is not None or bool(self._ready)
            if not buffered:
                return
            self._check_open()
        await self._aflush_buffer()

    async def _on_message_flush(self, event: MessageAddedEvent) -> None:
        """Write the messages ``append_message`` moved out of the slot."""
        self._check_open()
        await self._aflush_ready()

    async def _on_message_added(self, event: MessageAddedEvent) -> None:
        await self._inject_context(event.message)

    async def _inject_context(self, message: StrandsMessage) -> None:
        """Prepend relevant long-term memories to a user message (in-memory only).

        Failures degrade: a memory lookup must never break the agent's turn.
        """
        cfg = self._retrieval_config
        if cfg is None or message.get("role") != "user":
            return
        query = _message_text(message)
        if not query:
            return
        target = next(
            (b for b in message.get("content") or [] if isinstance(b, dict) and "text" in b),
            None,
        )
        if target is None or target["text"].startswith(f"<{cfg.context_tag}>\nRelevant memory:\n"):
            return  # no text block, or already injected (event re-fired)
        try:
            block = await self._bridge.run_async(
                _retrieve_context(
                    self._client.long_term,
                    query,
                    cfg,
                    nams=self._is_nams,
                    user_id=self._user_id,
                )
            )
        except Exception as e:
            logger.warning(
                "Memory retrieval failed for session %s; continuing without injected context: %s",
                self.session_id,
                e,
            )
            return
        if block:
            target["text"] = f"{block}\n{target['text']}"

    def append_message(self, message: StrandsMessage, agent: Agent, **kwargs: Any) -> None:
        """Buffer the new message. Does no I/O.

        The latest message waits in a one-slot buffer so guardrail
        redaction can rewrite it before it ever reaches the backend (NAMS
        has no message update/delete endpoint). The message it replaces
        moves to the ready queue, which the next hook writes.
        """
        self._check_open()
        latest = copy.deepcopy(message)
        with self._queue_lock:
            self._release_pending_locked()
            self._pending = latest

    def _release_pending(self) -> None:
        """Move the slot's message to the ready queue."""
        with self._queue_lock:
            self._release_pending_locked()

    def _release_pending_locked(self) -> None:
        if self._pending is not None:
            self._ready.append(self._pending)
            self._pending = None

    def redact_latest_message(
        self, redact_message: StrandsMessage, agent: Agent, **kwargs: Any
    ) -> None:
        """Replace the latest message with redacted content.

        Normal path: the latest message is still in the write-behind
        buffer, so we rewrite the buffer and the original never reaches
        the backend. Late path (buffer already flushed — defensive; the
        Strands lifecycle redacts within the same invocation): bolt
        deletes the stored message and re-adds the redacted text (the
        re-added message gets a fresh timestamp, so it moves to the end
        of restored history); NAMS has no delete/update endpoint, so we
        log a warning.
        """
        self._check_open()
        redacted = copy.deepcopy(redact_message)
        with self._queue_lock:
            if self._pending is not None:
                self._pending = redacted
                return
            if self._ready:
                # Seeded by initialize() after aopen() and not written yet.
                self._ready[-1] = redacted
                return
        if self._last_persisted is None:
            logger.warning(
                "redact_latest_message called for session %s but no message "
                "has been stored yet; nothing to redact.",
                self.session_id,
            )
            return
        if self._is_nams:
            logger.warning(
                "Cannot redact already-persisted message %s on NAMS (no "
                "message update/delete endpoint). The redacted content was "
                "NOT applied server-side.",
                self._last_persisted.id,
            )
            return
        text = _message_text(redact_message) or "[REDACTED]"
        try:
            self._bridge.run(self._client.short_term.delete_message(self._last_persisted.id))
            self._store_message(redact_message.get("role", "user"), text, extract=False)
        except Exception as e:
            self._last_persisted = None  # id may already be deleted; don't reuse it
            raise SessionException(
                f"Failed to redact latest message for session {self.session_id!r}"
            ) from e

    def sync_agent(self, agent: Agent, **kwargs: Any) -> None:
        """Agent state is not persisted (design decision: no Strands-specific
        nodes in the graph). Also fired on MessageAddedEvent by the base
        class, so it must not flush the buffer."""
        return None

    def _flush_buffer(self) -> None:
        """Persist every buffered message, blocking until done.

        For ``close()`` and other sync callers. Hooks await
        ``_aflush_buffer`` instead. Raises SessionException on failure, and
        the failed message stays queued (see ``_adrain``).
        """
        self._release_pending()
        try:
            future = self._submit_drain()
            if future is not None:
                future.result(timeout=self._request_timeout)
        except Exception as e:
            raise SessionException(
                f"Failed to persist message for session {self.session_id!r}"
            ) from e

    async def _aflush_buffer(self) -> None:
        """Persist every buffered message without blocking the caller's loop."""
        self._release_pending()
        await self._aflush_ready()

    async def _aflush_ready(self) -> None:
        """Persist the ready queue in order, leaving the latest message buffered.

        The write runs on the bridge and is shielded: cancelling the caller,
        or a timeout, stops only the wait. The write itself runs to
        completion, and ``close()`` waits for it.

        Raises SessionException on failure. The failed message stays at the
        head of the queue, and the next flush retries it (see ``_adrain``).
        """
        try:
            future = self._submit_drain()
            if future is not None:
                await self._bridge.wait(future, shield=True)
        except Exception as e:
            raise SessionException(
                f"Failed to persist message for session {self.session_id!r}"
            ) from e

    def _submit_drain(self) -> concurrent.futures.Future[None] | None:
        """Start a drain of the ready queue on the bridge, or None if it is empty."""
        with self._queue_lock:
            if not self._ready:
                return None
            future = self._bridge.submit(self._adrain())
            self._inflight = future
            return future

    async def _adrain(self) -> None:
        """Write the ready queue in order. Runs on the bridge loop.

        One drain runs at a time, so concurrent flushes (a hook, ``aclose()``
        and a sync ``close()`` on another thread) cannot reorder or
        double-write messages. A message leaves the queue only once its
        write has finished.

        Delivery is at least once. A failed write leaves its message at the
        head of the queue for the next drain to retry, ahead of every later
        message. Neither backend lets the caller set a message id, so a
        retry after a write that failed late, for example once the backend
        had stored it, can store the message twice.
        """
        if self._drain_lock is None:
            self._drain_lock = asyncio.Lock()
        async with self._drain_lock:
            while True:
                with self._queue_lock:
                    if not self._ready:
                        return
                    message = self._ready[0]
                if text := _message_text(message):
                    self._last_persisted = await self._aadd_message(
                        message["role"], text, extract=self._extract_entities
                    )
                with self._queue_lock:
                    self._ready.popleft()
                if blocks := self._tool_uses(message):
                    try:
                        await self._arecord_tool_uses(blocks)
                    except Exception as e:
                        logger.warning("Failed to mirror tool calls to reasoning memory: %s", e)

    def _wait_for_inflight(self) -> None:
        """Let the latest drain finish before the client closes. Blocking."""
        future = self._inflight
        if future is None:
            return
        try:
            future.result(timeout=self._request_timeout)
        except TimeoutError:
            logger.warning(
                "A memory write for session %r was still running at close.", self.session_id
            )
        except Exception:  # noqa: S110 - the flush that started it already reported it
            pass

    async def _await_inflight(self) -> None:
        """Like ``_wait_for_inflight``, without blocking the caller's loop."""
        future = self._inflight
        if future is None:
            return
        try:
            await self._bridge.wait(future, shield=True)
        except TimeoutError:
            logger.warning(
                "A memory write for session %r was still running at close.", self.session_id
            )
        except Exception:  # noqa: S110 - the flush that started it already reported it
            pass

    def _tool_uses(self, message: StrandsMessage) -> list[ToolUse]:
        """The toolUse blocks to mirror into reasoning memory, if enabled.

        A message with no text (pure tool use or result) is not memory, so
        only its tool uses are recorded.
        """
        if not self._record_tool_calls:
            return []
        return [
            b["toolUse"]
            for b in (message.get("content") or [])
            if isinstance(b, dict) and "toolUse" in b
        ]

    def _store_message(self, role: str, text: str, *, extract: bool) -> None:
        """Persist one message, blocking (tracked for late redaction)."""
        self._last_persisted = self._bridge.run(self._aadd_message(role, text, extract=extract))

    async def _aadd_message(self, role: str, text: str, *, extract: bool) -> StoredMessage:
        """Resolve the session key and add one message. Runs on the bridge loop."""
        return await self._client.short_term.add_message(
            await self._aresolve_key(),
            role,
            text,
            extract_entities=extract,
            user_identifier=self._user_id,
            metadata={_SESSION_KEY: self.session_id},
        )

    async def _arecord_tool_uses(self, blocks: list[ToolUse]) -> None:
        """Mirror toolUse blocks into reasoning memory. Runs on the bridge loop."""
        key = await self._aresolve_key()
        trace_id = self._trace_id
        if trace_id is None:
            trace = await self._client.reasoning.start_trace(key, task="Strands agent session")
            trace_id = trace.id
            self._trace_id = trace_id
        for block in blocks:
            name = block.get("name") or "unknown"
            step = await self._client.reasoning.add_step(
                trace_id, thought=f"Tool use: {name}", action=name
            )
            await self._client.reasoning.record_tool_call(step.id, name, block.get("input") or {})

    def _begin_close(self) -> concurrent.futures.Future[None] | None:
        """Mark the manager closed and return the future this close completes.

        Returns None when another ``close()`` or ``aclose()`` got there first.
        """
        with self._queue_lock:
            if self._closed:
                return None
            self._closed = True
            self._closing = concurrent.futures.Future()
            return self._closing

    async def aclose(self) -> None:
        """Like ``close()``, but awaits the I/O so the caller's loop keeps running.

        A call made while another close is still running waits for it to
        finish, then returns. The first caller gets any error.
        """
        closing = self._begin_close()
        if closing is None:
            if self._closing is not None:
                # asyncio.wait neither re-raises the error nor cancels the future.
                await asyncio.wait({asyncio.wrap_future(self._closing)})
            return
        try:
            await self._aflush_buffer()
        finally:
            try:
                await self._await_inflight()
                if self._should_close_client and self._client.is_connected:
                    await self._bridge.run_async(self._client.close())
            finally:
                try:
                    await asyncio.to_thread(self._bridge.close)
                finally:
                    closing.set_result(None)

    def close(self) -> None:
        """Flush, release the client (if owned/connected by us), stop the bridge.

        A call made while another close is running returns at once. Waiting
        here could block the event loop that close is running on.
        """
        closing = self._begin_close()
        if closing is None:
            return
        try:
            self._flush_buffer()
        finally:
            try:
                self._wait_for_inflight()
                if self._should_close_client and self._client.is_connected:
                    self._bridge.run(self._client.close())
            finally:
                try:
                    self._bridge.close()
                finally:
                    closing.set_result(None)

    def __enter__(self) -> Neo4jSessionManager:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
