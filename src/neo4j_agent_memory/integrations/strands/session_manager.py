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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import UUID, uuid4

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
    from neo4j_agent_memory.memory.short_term import ShortTermMemory
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

_PERSIST_ERROR_POLICIES = ("raise", "warn")


@dataclass(slots=True)
class _Queued:
    """A message in the write queue, with the state its drop rule needs."""

    message: StrandsMessage
    turn: int  # the value of _turns_ended when the message was queued
    failed: bool = False  # a write of it has failed
    # Stable across retries, so Bolt stores the message once however often
    # its write is retried.
    id: str = field(default_factory=lambda: str(uuid4()))
    # The latest write and the content it writes. Bridge loop only. A drain
    # that stops waiting leaves the write running, and the next drain
    # awaits it instead of writing again.
    write: asyncio.Task[StoredMessage] | None = None
    written: StrandsMessage | None = None


def _describe(error: BaseException) -> str:
    """The error's message, or its type name when it has none (TimeoutError)."""
    return str(error) or type(error).__name__


class _MessageDropped(Exception):
    """A drain dropped a message after its retry failed. Carries that error."""

    def __init__(self, error: Exception) -> None:
        super().__init__(str(error))
        self.error = error


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

    ``on_persist_error`` sets what a failed memory write does. In both
    modes the failed message stays at the head of the write queue, and
    the next flush retries it before any later message. A turn flushes
    several times, so failures in the turn that queued the message do not
    count. A message queued in an earlier turn is dropped if its write
    fails. At close, a message whose write already failed once is dropped
    if it fails again. A dropped message gets an ERROR log that names it
    by session, role, and position, and the later messages are written.
    So after a failed flush, only messages from the current turn stay
    queued. A failure at close leaves messages unwritten for good, so it
    is logged as an ERROR with their number in both modes. With
    ``"raise"`` (the default) each failure
    raises ``SessionException`` to the caller, which fails the agent call
    when it happens in a hook. With ``"warn"`` each failure is logged and
    the agent call goes on. Failures to restore history always raise, and
    failures to retrieve context are always warnings.
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
        on_persist_error: Literal["raise", "warn"] = "raise",
    ) -> None:
        if (memory_client is None) == (settings is None):
            raise ValueError(
                "Provide exactly one of memory_client= or settings= to Neo4jSessionManager."
            )
        if on_persist_error not in _PERSIST_ERROR_POLICIES:
            raise ValueError(
                f"on_persist_error must be 'raise' or 'warn', not {on_persist_error!r}."
            )
        self._on_persist_error = on_persist_error
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
        self._ready: deque[_Queued] = deque()  # older messages waiting for a hook to write
        # _pending and _ready are read and changed on the agent's thread and
        # on the bridge thread, so every access holds this lock.
        self._queue_lock = threading.Lock()
        self._drain_lock: asyncio.Lock | None = None  # bridge loop only; one drain at a time
        self._latest_drain: concurrent.futures.Future[None] | None = None  # under _queue_lock
        self._dequeued = 0  # messages that left _ready, stored or dropped; drain only
        self._turns_ended = 0  # AfterInvocationEvents seen
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
        closed manager returns quietly. When messages would be lost,
        ``"raise"`` raises SessionException. ``"warn"`` logs one ERROR with
        their number, as for any other write that fails at close.
        """
        if self._closed:
            with self._queue_lock:
                unwritten = int(self._pending is not None) + len(self._ready)
            if not unwritten:
                return
            if self._on_persist_error == "raise":
                self._check_open()
            logger.error(
                "Session %r was closed before its turn ended. %d message(s) were not written.",
                self.session_id,
                unwritten,
            )
            return
        try:
            await self._aflush_buffer()
        finally:
            self._turns_ended += 1  # messages queued before now are dropped if they fail

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
            self._ready.append(_Queued(self._pending, self._turns_ended))
            self._pending = None

    def redact_latest_message(
        self, redact_message: StrandsMessage, agent: Agent, **kwargs: Any
    ) -> None:
        """Replace the latest message with redacted content.

        Normal path: the latest message is still queued, so the queue is
        rewritten and the original never reaches the backend. A message
        stays queued while its write runs, so a redaction that lands then
        also rewrites the queue, and the drain applies it once the write
        finishes. Late path: the message is already stored. The Strands
        lifecycle redacts within the same invocation, so this path is
        defensive. See ``_aredact_stored`` for what each backend does.
        """
        self._check_open()
        redacted = copy.deepcopy(redact_message)
        with self._queue_lock:
            if self._pending is not None:
                self._pending = redacted
                return
            if self._ready:
                self._ready[-1].message = redacted
                return
            stored = self._last_persisted
        if stored is None:
            logger.warning(
                "redact_latest_message called for session %s but no message "
                "has been stored yet; nothing to redact.",
                self.session_id,
            )
            return
        try:
            self._last_persisted = self._bridge.run(self._aredact_stored(stored, redacted))
        except Exception as e:
            self._last_persisted = None  # id may already be deleted; don't reuse it
            raise SessionException(
                f"Failed to redact latest message for session {self.session_id!r}"
            ) from e

    async def _aredact_stored(
        self, stored: StoredMessage, redacted: StrandsMessage
    ) -> StoredMessage:
        """Replace a stored message with redacted content. Runs on the bridge loop.

        Bolt deletes the stored message and adds the redacted text. The new
        message gets a fresh timestamp, so it moves to the end of restored
        history. NAMS has no endpoint to update or delete a message. There
        the original stays, a warning is logged, and ``stored`` is returned.
        """
        if self._is_nams:
            logger.warning(
                "Cannot redact stored message %s of session %r on NAMS. NAMS "
                "has no endpoint to update or delete a message, so the "
                "original stays.",
                stored.id,
                self.session_id,
            )
            return stored
        await self._client.short_term.delete_message(stored.id)
        text = _message_text(redacted) or "[REDACTED]"
        return await self._aadd_message(redacted.get("role", "user"), text)

    def sync_agent(self, agent: Agent, **kwargs: Any) -> None:
        """Agent state is not persisted (design decision: no Strands-specific
        nodes in the graph). Also fired on MessageAddedEvent by the base
        class, so it must not flush the buffer."""
        return None

    def _flush_buffer(self) -> None:
        """Persist every buffered message, blocking until done.

        For ``close()`` and other sync callers. Hooks await
        ``_aflush_buffer`` instead. A failure is handled by
        ``on_persist_error`` (see ``_wait_failed``).
        """
        self._release_pending()
        future: concurrent.futures.Future[None] | None = None
        try:
            future = self._submit_drain()
            if future is not None:
                future.result(timeout=self._drain_timeout())
        except Exception as e:
            self._wait_failed(future, e)

    async def _aflush_buffer(self) -> None:
        """Persist every buffered message without blocking the caller's loop."""
        self._release_pending()
        await self._aflush_ready()

    async def _aflush_ready(self) -> None:
        """Persist the ready queue in order, leaving the latest message buffered.

        The writes run on the bridge and are shielded: cancelling the
        caller stops only the wait. The writes run to completion, and
        ``close()`` waits for them.

        A failure is handled by ``on_persist_error`` (see
        ``_wait_failed``).
        """
        future: concurrent.futures.Future[None] | None = None
        try:
            future = self._submit_drain()
            if future is not None:
                await self._bridge.wait(future, self._drain_timeout(), shield=True)
        except Exception as e:
            self._wait_failed(future, e)

    def _drain_timeout(self) -> float:
        """How long a caller waits for a drain.

        Each step of a drain is bounded by ``request_timeout``, so a drain
        can take that long per step for each queued message. One message
        more covers a drain that is still finishing a popped message.
        """
        steps = 1 + int(self._extract_entities and not self._is_nams) + int(self._record_tool_calls)
        with self._queue_lock:
            queued = len(self._ready)
        return self._request_timeout * steps * (queued + 1)

    def _wait_failed(
        self, future: concurrent.futures.Future[None] | None, error: Exception
    ) -> None:
        """Handle an error from a flush's wait for its drain.

        A drain that is still running means the wait ran out, not a write.
        The drain goes on. A drain that succeeded means it finished just
        after the wait ran out, so nothing failed. Any other error is a
        failed flush.
        """
        if future is None or future.done():
            if future is not None and not future.cancelled() and future.exception() is None:
                return
            self._persist_failed(error)
            return
        with self._queue_lock:
            unwritten = len(self._ready)
        if self._closed:
            logger.error(
                "Closing session %r before its memory writes finished. "
                "%d message(s) were not written.",
                self.session_id,
                unwritten,
            )
        else:
            logger.warning(
                "Memory writes for session %r are taking longer than expected. "
                "They go on in the background.",
                self.session_id,
            )

    def _persist_failed(self, error: Exception) -> None:
        """Apply ``on_persist_error`` to a failed flush.

        A failure at close leaves its messages unwritten for good, so it is
        logged as an ERROR in both modes. ``"raise"`` then raises
        SessionException. ``"warn"`` logs a warning for a failure before
        close. A drop gets no warning, since the drain logged it as an
        error.
        """
        dropped = isinstance(error, _MessageDropped)
        cause = error.error if isinstance(error, _MessageDropped) else error
        if dropped:
            detail = "a message was dropped after its write failed"
        elif self._closed:
            with self._queue_lock:
                unwritten = len(self._ready)
            detail = f"{unwritten} message(s) were not written before close"
            logger.error(
                "Closing session %r with %d message(s) not written: %s",
                self.session_id,
                unwritten,
                _describe(cause),
            )
        else:
            detail = "the message stays queued for the next flush"
        if self._on_persist_error == "raise":
            raise SessionException(
                f"Failed to persist message for session {self.session_id!r}; {detail}"
            ) from cause
        if not dropped and not self._closed:
            logger.warning(
                "Failed to persist message for session %r. It stays queued, "
                "and the next flush retries it: %s",
                self.session_id,
                _describe(cause),
            )

    def _submit_drain(self) -> concurrent.futures.Future[None] | None:
        """Start a drain of the ready queue on the bridge.

        Returns None when the queue is empty, unless the manager is
        closing while a drain still runs. Drains run one at a time, so the
        drain a close submits also waits for that one.
        """
        with self._queue_lock:
            running = self._latest_drain is not None and not self._latest_drain.done()
            if not self._ready and not (self._closed and running):
                return None
            self._latest_drain = self._bridge.submit(self._adrain())
            return self._latest_drain

    async def _adrain(self) -> None:
        """Write the ready queue in order. Runs on the bridge loop.

        One drain runs at a time, so concurrent flushes (a hook, ``aclose()``
        and a sync ``close()`` on another thread) cannot reorder or
        double-write messages. A message leaves the queue only once its
        write has finished.

        A message is written without entity extraction, then extracted as
        a separate step. A failed extraction is logged and leaves the
        message stored, so it never causes a second write.

        A failed write leaves its message at the head of the queue for the
        next drain to retry, ahead of every later message. A retry does
        not store the message twice. See ``_awrite``.

        Each wait is bounded by ``request_timeout``. A write that times
        out is a failed write, though it keeps running.

        A failed write drops its message when the message was queued in an
        earlier turn, or when the manager is closing and the message failed
        before. A drop is logged as an ERROR, and the drain goes on with
        the later messages. If every later message is written, the drain
        raises ``_MessageDropped`` at the end. Any other failure marks the
        message as failed, stops the drain, and raises the error as is.
        """
        if self._drain_lock is None:
            self._drain_lock = asyncio.Lock()
        dropped: _MessageDropped | None = None
        async with self._drain_lock:
            while True:
                with self._queue_lock:
                    if not self._ready:
                        break
                    entry = self._ready[0]
                try:
                    written, stored = await self._awrite(entry)
                except Exception as e:
                    stale = entry.turn < self._turns_ended
                    if not stale and not (self._closed and entry.failed):
                        entry.failed = True
                        raise
                    if entry.write is not None:
                        entry.write.cancel()  # a drop ends the write a timeout left running
                    logger.error(
                        "Dropped %s message #%d of session %r. Its write failed: %s",
                        entry.message.get("role", "unknown"),
                        self._dequeued + 1,
                        self.session_id,
                        _describe(e),
                    )
                    dropped = _MessageDropped(e)
                    self._pop_head()
                    continue
                latest = self._pop_head(stored)
                if stored is not None and latest is not written:
                    stored = await self._aredact_written(stored, latest)
                if stored is not None:
                    await self._aextract(stored)
                if blocks := self._tool_uses(latest):
                    try:
                        await asyncio.wait_for(
                            self._arecord_tool_uses(blocks), self._request_timeout
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to mirror tool calls to reasoning memory: %s", _describe(e)
                        )
        if dropped is not None:
            raise dropped

    async def _awrite(self, entry: _Queued) -> tuple[StrandsMessage, StoredMessage | None]:
        """Write the entry's message, waiting at most ``request_timeout``.

        Returns the content written and the stored message. The stored
        message is None for a message with no text, which is not written.

        The write runs as a shielded task kept on the entry. A wait that
        runs out leaves the write running, and the next drain awaits that
        write rather than starting another. NAMS assigns message ids on the
        server, so this is what keeps a slow write from being stored twice
        there. Bolt also stores the message under the entry's id, so a
        write that failed after it committed is not stored twice either.
        """
        task = entry.write
        if task is None or (task.done() and (task.cancelled() or task.exception() is not None)):
            message = entry.message
            if not (text := _message_text(message)):
                return message, None
            task = asyncio.ensure_future(self._aadd_message(message["role"], text, entry.id))
            entry.write, entry.written = task, message
        written = cast("StrandsMessage", entry.written)
        return written, await asyncio.wait_for(asyncio.shield(task), self._request_timeout)

    def _pop_head(self, stored: StoredMessage | None = None) -> StrandsMessage:
        """Remove the head of the ready queue, count it, and return its content.

        The content is read under the lock, so it includes a redaction that
        landed while the write ran. ``stored`` becomes the target of a late
        redaction in the same step. So a redaction finds the message either
        in the queue or in ``_last_persisted``.
        """
        with self._queue_lock:
            entry = self._ready.popleft()
            if stored is not None:
                self._last_persisted = stored
        self._dequeued += 1
        return entry.message

    async def _aredact_written(
        self, stored: StoredMessage, redacted: StrandsMessage
    ) -> StoredMessage | None:
        """Apply a redaction that landed while its message was being written.

        Returns the message that is now stored, or None when the redaction
        failed. A failure is logged as an ERROR, since the original text
        may still be stored. The message itself was written, so it is not
        a persist failure.
        """
        try:
            result: StoredMessage | None = await asyncio.wait_for(
                self._aredact_stored(stored, redacted), self._request_timeout
            )
        except Exception as e:
            logger.error(
                "Failed to redact message %s of session %r, so its original "
                "text may still be stored. The redaction arrived while the "
                "message was being written: %s",
                stored.id,
                self.session_id,
                _describe(e),
            )
            result = None
        with self._queue_lock:
            self._last_persisted = result
        return result

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

    async def _aadd_message(
        self, role: str, text: str, message_id: str | None = None
    ) -> StoredMessage:
        """Resolve the session key and add one message. Runs on the bridge loop.

        Never extracts. Bolt's ``add_message`` extracts after the write and
        raises with the message stored, so the drain extracts separately
        (see ``_aextract``). Bolt stores the message under ``message_id``,
        and NAMS ignores it.
        """
        return await self._client.short_term.add_message(
            await self._aresolve_key(),
            role,
            text,
            extract_entities=False,
            user_identifier=self._user_id,
            metadata={_SESSION_KEY: self.session_id},
            message_id=message_id,
        )

    async def _aextract(self, stored: StoredMessage) -> None:
        """Extract entities from a stored message, logging a failure.

        Runs on the bridge loop. NAMS extracts every write server-side, so
        this does nothing there.
        """
        if not self._extract_entities or self._is_nams:
            return
        bolt_short_term = cast("ShortTermMemory", self._client.short_term)
        try:
            await asyncio.wait_for(
                bolt_short_term.extract_entities_from_message(
                    stored, user_identifier=self._user_id
                ),
                self._request_timeout,
            )
        except Exception as e:
            logger.warning(
                "Entity extraction failed for a stored message of session %r. "
                "The message is stored without entities: %s",
                self.session_id,
                _describe(e),
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
        failure: BaseException | None = None
        try:
            try:
                await self._aflush_buffer()
            except BaseException as e:
                failure = e
            try:
                if self._should_close_client and self._client.is_connected:
                    await self._bridge.run_async(self._client.close())
            except BaseException as e:
                failure = self._first_failure(failure, e, "close the memory client")
            try:
                await asyncio.to_thread(self._bridge.close)
            except BaseException as e:
                failure = self._first_failure(failure, e, "stop the bridge")
        finally:
            closing.set_result(None)
        if failure is not None:
            raise failure

    def close(self) -> None:
        """Flush, release the client (if owned/connected by us), stop the bridge.

        A call made while another close is running returns at once. Waiting
        here could block the event loop that close is running on.

        Raises:
            RuntimeError: The caller runs on the bridge's loop thread, where
                waiting for the writes would block them. The manager stays
                open.
        """
        if self._bridge.on_loop_thread:
            raise RuntimeError(
                f"Neo4jSessionManager.close() for session {self.session_id!r} "
                "cannot run on its bridge loop thread. Call it from another thread."
            )
        closing = self._begin_close()
        if closing is None:
            return
        failure: BaseException | None = None
        try:
            try:
                self._flush_buffer()
            except BaseException as e:
                failure = e
            try:
                if self._should_close_client and self._client.is_connected:
                    self._bridge.run(self._client.close())
            except BaseException as e:
                failure = self._first_failure(failure, e, "close the memory client")
            try:
                self._bridge.close()
            except BaseException as e:
                failure = self._first_failure(failure, e, "stop the bridge")
        finally:
            closing.set_result(None)
        if failure is not None:
            raise failure

    def _first_failure(
        self, failure: BaseException | None, error: BaseException, step: str
    ) -> BaseException:
        """Return the error a close raises, which is its first one.

        Every close step runs even after one fails. The flush comes first,
        so its error wins over an error from releasing the client or the
        bridge. A later error is logged instead of raised.
        """
        if failure is None:
            return error
        logger.error(
            "Failed to %s while closing session %r: %s",
            step,
            self.session_id,
            _describe(error),
            exc_info=error,
        )
        return failure

    def __enter__(self) -> Neo4jSessionManager:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
