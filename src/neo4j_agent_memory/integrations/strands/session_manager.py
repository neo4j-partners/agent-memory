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
import weakref
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import UUID, uuid4

from neo4j_agent_memory.integrations.base import _SHUTDOWN_TIMEOUT_SECONDS as _UNWIND_SECONDS
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
        BeforeInvocationEvent,
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
    from collections.abc import Callable, Coroutine
    from types import TracebackType

    from strands import Agent
    from strands.agent import AgentResult
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

#: Where the agent is in its turn. "running" lasts from BeforeInvocationEvent
#: to AfterInvocationEvent. "paused" is a turn stopped by an interrupt.
_TurnState = Literal["idle", "running", "paused"]


@dataclass(slots=True)
class _Queued:
    """A message in the write queue, with the state its drop rule needs."""

    message: StrandsMessage
    turn: int  # the value of _turns_ended when the message was queued
    failed: bool = False  # a write of it has failed
    # Stable across retries, so Bolt stores the message once however often
    # its write is retried.
    id: str = field(default_factory=lambda: str(uuid4()))
    # The latest write, and the content the first write sent. Bridge loop
    # only. A drain that stops waiting leaves the write running, and the
    # next drain awaits it instead of writing again.
    write: asyncio.Task[StoredMessage] | None = None
    written: StrandsMessage | None = None


def _cancel_requested(agent: Agent | None) -> bool:
    """True when ``agent.cancel()`` or its ``cancel_signal`` stopped the call.

    The ``stop`` tool also ends a call with stop reason ``"cancelled"``,
    but it leaves the signal clear. Strands clears the signal after the
    AfterInvocationEvent hooks run. Strands 1.52 keeps the signal private,
    and every cancel there sets it.
    """
    signal = getattr(agent, "cancel_signal", None) or getattr(agent, "_cancel_signal", None)
    return isinstance(signal, threading.Event) and signal.is_set()


def _holds_invocation_lock(event: BeforeInvocationEvent) -> bool:
    """True when the call that fired ``event`` holds the agent's invocation lock.

    In Strands' default ``THROW`` mode, ``stream_async`` takes the lock
    before its BeforeInvocationEvent and passes the call's messages with
    it. ``structured_output_async`` takes no lock and passes no messages.
    In ``UNSAFE_REENTRANT`` mode no call takes the lock.
    """
    if getattr(event, "messages", None) is None:
        return False
    mode = getattr(event.agent, "concurrent_invocation_mode", "throw")
    return bool(mode != "unsafe_reentrant")


def _interrupt_active(agent: Agent | None) -> bool:
    """True while Strands holds an interrupt the next call must answer."""
    state = getattr(agent, "_interrupt_state", None)
    return bool(getattr(state, "activated", False))


def _describe(error: BaseException) -> str:
    """The error's message, or its type name when it has none (TimeoutError)."""
    return str(error) or type(error).__name__


def _report_done(name: str, done: Callable[[], None]) -> None:
    """Call the done function ``background_hook`` returned, logging an error."""
    try:
        done()
    except Exception as e:
        logger.warning(
            "The done function background_hook returned for task %r failed: %s",
            name,
            _describe(e),
        )


def _copy_outcome(
    source: concurrent.futures.Future[None], target: concurrent.futures.Future[None]
) -> None:
    """Give ``target`` the result, error, or cancellation of ``source``."""
    if source.cancelled():
        target.cancel()
    elif (error := source.exception()) is not None:
        target.set_exception(error)
    else:
        target.set_result(None)


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

    ``bridge`` runs the manager's memory I/O on a loop it does not own,
    and it needs ``memory_client``. The client must already be connected
    on that bridge's loop, or the manager raises ``ValueError``. A manager
    given a bridge
    closes neither the bridge nor the client. Its ``close()`` writes its
    own messages and returns. ``SharedMemory`` builds managers this way,
    so many sessions share one client and one loop.

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

    Messages are written a turn at a time. A turn runs from Strands'
    ``BeforeInvocationEvent`` to its ``AfterInvocationEvent``, and its
    messages stay buffered until it ends. A turn that succeeds is written
    in order. A turn that raises or is cancelled is discarded with an INFO
    log, so the graph never holds a prompt without its reply. An interrupt
    pauses the turn until the call that resumes it. Messages appended
    outside an agent call are written by the next hook or by close. A
    turn whose AfterInvocationEvent never reached this manager, because a
    hook raised, is discarded with a warning when the next call starts. A
    process that stops in the middle of a turn loses that turn.

    ``on_persist_error`` sets what a failed memory write does. In both
    modes the failed message stays at the head of the write queue, and the
    next flush retries it before any later message. The next flush runs at
    the start of the next agent call, or at close. A message queued in an
    earlier turn is dropped if its write fails. At close, a message whose
    write already failed once is dropped if it fails again. A dropped
    message gets an ERROR log that names it by session, role, and
    position, and the later messages are written. So after a failed flush,
    only messages from the current turn stay queued. A failure at close
    leaves messages unwritten for good, so it is logged as an ERROR with
    their number in both modes. With ``"raise"`` (the default) each
    failure raises ``SessionException`` to the caller, which fails the
    agent call when it happens in a hook. With ``"warn"`` each failure is
    logged and the agent call goes on. Failures to restore history always
    raise, and failures to retrieve context are always warnings.

    ``defer_extraction`` takes entity extraction off the write path. The
    drain writes each message and starts its extraction in the
    background, so the agent call does not wait for it. The extractions of
    one session run one at a time, in write order. Each is bounded by
    ``request_timeout``, and a failure is logged as a WARNING. A late
    redaction that deletes a message before its extraction starts skips
    that extraction. The backlog has no cap. Extractions slower than the
    message rate let it grow for as long as the manager lives. A manager
    that owns its bridge waits for the extractions at close, as long as
    it waits for a drain. It then cancels the rest before it closes the
    client, and a WARNING gives their number. A manager on a shared
    bridge does not wait. ``SharedMemory`` waits for them when it closes.
    NAMS extracts on the server, so the option does nothing there.

    ``background_hook`` reports the work that outlives a write. It is
    called with a short name, such as ``"extract:<session_id>"``, when a
    deferred extraction or a tool-call recording starts. It returns a
    function that runs once when that task ends, whether it succeeded,
    failed, or was cancelled. Both run on the bridge thread, so they must
    be thread-safe and fast. An error from either is logged as a WARNING
    and never fails a write. ``SharedMemory`` passes its own hook to the
    managers it builds.
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
        defer_extraction: bool = False,
        record_tool_calls: bool = False,
        request_timeout: float = 30.0,
        restore_limit: int | None = None,
        on_persist_error: Literal["raise", "warn"] = "raise",
        bridge: _AsyncBridge | None = None,
        background_hook: Callable[[str], Callable[[], None]] | None = None,
    ) -> None:
        if (memory_client is None) == (settings is None):
            raise ValueError(
                "Provide exactly one of memory_client= or settings= to Neo4jSessionManager."
            )
        if bridge is not None:
            if memory_client is None:
                raise ValueError(
                    "bridge= needs memory_client=. A client built from settings= "
                    "connects on the manager's own bridge."
                )
            if not memory_client.is_connected:
                raise ValueError(
                    "bridge= needs a memory_client connected on that bridge's loop. "
                    "The bridge's owner connects the client and closes it."
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
        self._defer_extraction = defer_extraction
        self._background_hook = background_hook
        self._record_tool_calls = record_tool_calls
        self._restore_limit = restore_limit
        self._request_timeout = request_timeout
        # A shared bridge belongs to its creator, who closes it and the client.
        self._owns_bridge = bridge is None
        self._bridge = _AsyncBridge(timeout=request_timeout) if bridge is None else bridge
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
        self._turn: list[StrandsMessage] = []  # the current turn, written when it succeeds
        self._turn_state: _TurnState = "idle"
        self._ready: deque[_Queued] = deque()  # messages waiting for a hook to write
        # _turn, _turn_state and _ready are read and changed on the agent's
        # thread and on the bridge thread, so every access holds this lock.
        self._queue_lock = threading.Lock()
        self._drain_lock: asyncio.Lock | None = None  # bridge loop only; one drain at a time
        self._redact_lock: asyncio.Lock | None = None  # bridge loop only; one redaction at a time
        self._drain_run: concurrent.futures.Future[None] | None = None  # under _queue_lock
        self._dequeued = 0  # messages that left _ready, stored or dropped; drain only
        self._turns_ended = 0  # AfterInvocationEvents seen; numbers the queued turns
        # Turns whose end-of-turn drain has finished. A message queued before
        # then is dropped if a later write of it fails.
        self._turns_written = 0  # under _queue_lock
        # The paused turn's length when the running call resumed it, else
        # None. A resume that fails drops what it added. Under _queue_lock.
        self._resume_from: int | None = None
        # The call that owns the open turn: its invocation_state, the task
        # its BeforeInvocationEvent ran in, and whether a message joined the
        # turn since. Calls that overlap it fold into its turn. _open_calls
        # counts calls whose BeforeInvocationEvent ran and whose
        # AfterInvocationEvent has not. All under _queue_lock.
        self._owner: dict[str, Any] | None = None
        self._owner_task: asyncio.Task[Any] | None = None
        self._owner_appended = False
        self._owner_locks = False  # the owner holds the agent's invocation lock
        self._open_calls = 0
        self._restored: list[StrandsMessage] | None = None  # history preloaded by aopen()
        # Set once initialize() succeeds. A weak reference, since the agent
        # holds this manager through its hooks.
        self._agent_ref: weakref.ref[Agent] | None = None
        self._last_persisted: StoredMessage | None = None  # last stored (late redaction)
        self._trace_id: UUID | None = None  # lazy reasoning trace (record_tool_calls)
        self._tool_record: asyncio.Task[None] | None = None  # latest mirroring; bridge loop only
        self._live_tool_records: set[asyncio.Task[None]] = set()  # unfinished; bridge loop only
        self._extraction: asyncio.Task[None] | None = None  # own latest deferred; bridge loop only
        self._live_extractions: set[asyncio.Task[None]] = set()  # unfinished; bridge loop only
        self._awaiting_extraction: set[UUID] = set()  # deferred, not yet checked; bridge loop only
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
            self._should_close_client = self._owns_bridge

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
            self._restored = await self._bridge.run_async(
                self._ainitialize(), self._request_timeout
            )
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

        Once the previous Agent is freed, another can attach. A turn the
        freed Agent left open is dropped, with a warning. See
        ``_drop_freed_agents_turn``.

        Raises:
            SessionException: The manager is closed, loading failed, or the
                manager already serves another Agent. One manager tracks
                one Agent's turns, so a second Agent would discard them.
        """
        self._check_open()
        attached = self._agent_ref() if self._agent_ref is not None else None
        if attached is not None and attached is not agent:
            raise SessionException(
                f"Session manager for session {self.session_id!r} is already "
                "attached to another Agent. Give each Agent its own manager."
            )
        if self._agent_ref is not None and attached is None:
            self._drop_freed_agents_turn()
        self._check_memory_manager_coexistence(agent)
        preloaded = self._restored is not None
        if self._restored is not None:
            stored, self._restored = self._restored, None
        else:
            try:
                stored = self._bridge.run(self._ainitialize(), self._request_timeout)
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
                self._release_turn()
            else:
                self._flush_buffer()
        self._agent_ref = weakref.ref(agent)

    def _drop_freed_agents_turn(self) -> None:
        """Drop the turn that a freed Agent left open.

        A turn paused at an interrupt can be resumed only by the Agent that
        holds the interrupt. Once that Agent is freed, nothing can resume
        the turn. The next Agent's first call would otherwise write it
        ahead of its own turn, with a tool call that has no result. A turn
        left running is dropped too, as the next call would discard it.
        Messages appended outside a call stay buffered.
        """
        with self._queue_lock:
            if self._turn_state == "idle":
                return
            dropped = len(self._turn)
            self._turn.clear()
            self._turn_state = "idle"
            self._resume_from = None
            self._owner = None
            self._owner_task = None
            self._owner_appended = False
            self._owner_locks = False
            self._open_calls = 0
        logger.warning(
            "Session %r: a new agent attached after the agent with an open "
            "turn was freed. Nothing can resume that turn, so its %d "
            "message(s) are not written.",
            self.session_id,
            dropped,
        )

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Wire base persistence hooks, then our turn and injection hooks.

        Our hooks are coroutines. Strands awaits them on the agent's loop,
        and they await memory I/O on the bridge loop, so the agent's loop
        keeps running other tasks meanwhile.

        ``BeforeInvocationEvent`` opens a turn, and ``AfterInvocationEvent``
        ends it. The base class buffers each message in the turn from its
        ``MessageAddedEvent`` hook (``append_message``). The turn's messages
        are written only when the turn succeeds.

        Order matters: Strands fires ``MessageAddedEvent`` callbacks in
        registration order, so buffering (base ``MessageAddedEvent`` ->
        append_message) copies a user message before context injection
        changes it.
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
        registry.add_callback(BeforeInvocationEvent, self._on_before_invocation)
        registry.add_callback(AfterInvocationEvent, self._on_after_invocation)
        if self._retrieval_config is not None:
            registry.add_callback(MessageAddedEvent, self._on_message_added)

    async def _on_before_invocation(self, event: BeforeInvocationEvent) -> None:
        """Open a turn, first writing what earlier turns left queued.

        Messages appended outside an agent call, such as a direct tool
        call, are written here as their own batch. A call that resumes a
        turn paused by an interrupt keeps that turn's messages buffered.
        A message whose write failed in an earlier turn is retried here,
        before the turn adds its own.

        The call that opens a turn owns it. A call that starts while the
        owner runs overlaps it, such as ``structured_output_async`` during
        ``invoke_async``, or a second call on an agent in
        ``UNSAFE_REENTRANT`` mode. Its messages join the owner's turn with
        a warning, and the owner's result decides that turn.

        A running turn whose owner cannot still be running is left over.
        A hook raised, so the owner's AfterInvocationEvent never reached
        ours, and the owner's caller got the exception. In Strands' default ``THROW``
        mode, a ``stream_async`` call that reaches this hook holds the
        invocation lock. A turn owned by another such call is then left
        over. Otherwise the owner's task has finished, or this call runs in
        that task with no message added since. This call takes over, and
        the left-over turn is handled as for any call that raised. See
        ``_discard_failed_turn_locked``. Unless the whole turn is kept, a
        warning reports how many messages were discarded.
        """
        self._check_open()
        task = asyncio.current_task()
        locks = _holds_invocation_lock(event)
        discarded: int | None = None
        with self._queue_lock:
            state = self._turn_state
            if state == "running" and self._left_over(task, locks):
                discarded = self._discard_failed_turn_locked(event.agent)
                self._owner = None
                self._owner_task = None
                # Only the owner's call is gone. A call that overlapped it,
                # such as structured_output_async, may still be running.
                self._open_calls = max(self._open_calls - 1, 0)
                state = self._turn_state
            if state == "idle":
                self._release_turn_locked()
                self._resume_from = None
            elif state == "paused":
                self._resume_from = len(self._turn)
        if discarded is not None:
            logger.warning(
                "Session %r: an agent call started before the previous call's "
                "turn ended. That call raised, so its %d message(s) were not "
                "written.",
                self.session_id,
                discarded,
            )
        await self._aflush_ready()
        with self._queue_lock:
            overlaps = self._turn_state == "running" and self._owner_running(task)
            if overlaps:
                self._open_calls += 1
            else:
                self._owner = event.invocation_state
                self._owner_task = task
                self._owner_appended = False
                self._owner_locks = locks
                self._open_calls += 1
            self._turn_state = "running"
            open_turn = len(self._turn)
        if overlaps:
            logger.warning(
                "Session %r: an agent call started while another call's turn "
                "is open with %d message(s). Its messages join that turn, and "
                "the first call's result decides whether the turn is written.",
                self.session_id,
                open_turn,
            )

    def _left_over(self, task: asyncio.Task[Any] | None, locks: bool) -> bool:
        """Whether the running turn is left over. Holds _queue_lock.

        ``locks`` says whether the new call holds the invocation lock. When
        the owner held it too, the owner has released it, so it has ended.
        """
        return (locks and self._owner_locks) or not self._owner_running(task)

    def _owner_running(self, task: asyncio.Task[Any] | None) -> bool:
        """Whether the turn's owner may still be running. Holds _queue_lock.

        A call in the owner's own task overlaps it only from inside it,
        such as from a tool, after the owner added its prompt.
        """
        owner_task = self._owner_task
        if owner_task is None or owner_task.done():
            return False
        return owner_task is not task or self._owner_appended

    async def _on_after_invocation(self, event: AfterInvocationEvent) -> None:
        """End the turn, and write it if it succeeded.

        A result with stop reason ``"interrupt"`` pauses the turn. Its
        messages stay buffered for the call that resumes it. No result
        means the agent call raised or a BeforeInvocationEvent hook
        cancelled it. Stop reason ``"cancelled"`` with the agent's cancel
        signal set means ``agent.cancel()`` or a ``cancel_signal`` stopped
        it. The turn's messages are then discarded with an INFO log, so a
        failed turn never stores a prompt without its reply. A call that
        fails while Strands still holds an interrupt pauses the turn
        instead. See ``_discard_failed_turn_locked``.

        Any other result means the turn succeeded, so its messages are
        written in order. That covers the ``limit_*`` stop reasons, since
        the caller got an answer. It also covers the ``stop`` tool, which
        cancels without setting the signal. Strands owns
        ``agent.messages``, so this leaves it alone.

        Strands fires this from its ``finally``, so it also runs when a
        caller abandons a stream and the generator is finalized after
        ``aclose()``. With nothing left to write there is nothing to lose,
        so a closed manager returns quietly. When messages would be lost,
        ``"raise"`` raises SessionException. ``"warn"`` logs one ERROR with
        their number, as for any other write that fails at close.

        Only the owner's call ends the turn. A call that overlapped it
        leaves the turn alone, so its failure discards nothing. When the
        last open call ends, it ends the turn whatever its
        ``invocation_state``. ``structured_output_async`` sends a new
        ``invocation_state`` with each event, so this is how its turn ends.
        """
        ended = self._end_turn(event.agent, event.result, event.invocation_state)
        if ended is None:
            logger.debug(
                "Session %r: an agent call that overlapped the open turn "
                "ended. The turn stays open for its first call.",
                self.session_id,
            )
            return
        discarded, turn = ended
        if discarded:
            logger.info(
                "Session %r: the agent call raised or was cancelled, so its "
                "turn's %d message(s) were not written.",
                self.session_id,
                discarded,
            )
        if self._closed:
            with self._queue_lock:
                unwritten = len(self._ready) + len(self._turn)
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
        await self._aflush_ready(ends_turn=turn)

    def _end_turn(
        self,
        agent: Agent | None,
        result: AgentResult | None,
        invocation_state: dict[str, Any] | None,
    ) -> tuple[int, int] | None:
        """Queue, keep, or discard the turn by its result.

        Returns how many messages were discarded, and the number of the
        turn that ended. Returns None when a call that overlapped the
        turn's owner ended, which leaves the turn alone. See
        ``_on_after_invocation``.
        """
        failed = result is None or (result.stop_reason == "cancelled" and _cancel_requested(agent))
        with self._queue_lock:
            self._open_calls = max(self._open_calls - 1, 0)
            if invocation_state is not self._owner and self._open_calls:
                return None
            self._owner = None
            self._owner_task = None
            turn = self._turns_ended
            discarded = 0
            if result is not None and result.stop_reason == "interrupt":
                self._turn_state = "paused"
            elif failed:
                discarded = self._discard_failed_turn_locked(agent) or 0
            else:
                self._turn_state = "idle"
                self._resume_from = None
                self._release_turn_locked()
            self._turns_ended += 1  # after the release, which numbers the messages
            return discarded, turn

    def _discard_failed_turn_locked(self, agent: Agent | None) -> int | None:
        """Discard a failed turn's messages and return how many. Holds _queue_lock.

        A call that fails while Strands still holds an interrupt leaves the
        turn paused, since the next call must resume it. A failed resume
        discards only what it added. A failed call that was not a resume
        keeps its whole turn, so the call that resumes it writes the whole
        turn. This returns None then. Any other failed turn is discarded
        whole, and the turn goes idle.
        """
        if _interrupt_active(agent):
            self._turn_state = "paused"
            if self._resume_from is None:
                return None
            discarded = len(self._turn) - self._resume_from
            del self._turn[self._resume_from :]
            return discarded
        discarded = len(self._turn)
        self._turn.clear()
        self._turn_state = "idle"
        self._resume_from = None
        return discarded

    def _turn_written(self, turn: int) -> None:
        """Mark ``turn``'s end-of-turn drain finished. Any thread."""
        with self._queue_lock:
            self._turns_written = max(self._turns_written, turn + 1)

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
                ),
                self._request_timeout,
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
        """Buffer the new message in the current turn. Does no I/O.

        The turn's messages wait until the turn ends. Guardrail redaction
        can then rewrite a message before it ever reaches the backend. NAMS
        has no endpoint to update or delete a message. A turn that fails
        stores none of its messages. See ``_on_after_invocation``.
        """
        self._check_open()
        latest = copy.deepcopy(message)
        with self._queue_lock:
            self._turn.append(latest)
            self._owner_appended = True

    def _release_turn(self) -> None:
        """Move the buffered messages to the ready queue."""
        with self._queue_lock:
            self._release_turn_locked()

    def _release_turn_locked(self) -> None:
        self._ready.extend(_Queued(message, self._turns_ended) for message in self._turn)
        self._turn.clear()

    def _release_idle_turn(self, *, closing: bool) -> None:
        """Move messages appended outside an agent call to the ready queue.

        A turn that is running or paused is left buffered, since only its
        end decides whether it is written. A close during a paused turn
        leaves that turn unwritten for good, so it logs a warning. A close
        during a running turn logs an ERROR and drops the turn. That turn's
        call may never end, because a BeforeInvocationEvent hook raised. If
        it does end, its after-invocation hook finds nothing left to report.
        """
        with self._queue_lock:
            state = self._turn_state
            if state == "idle":
                self._release_turn_locked()
                return
            unfinished = len(self._turn)
            if closing and state == "running":
                self._turn.clear()
                self._resume_from = None
        if not (closing and unfinished):
            return
        if state == "paused":
            logger.warning(
                "Closing session %r while its turn waits on an interrupt. "
                "The turn's %d message(s) are not written.",
                self.session_id,
                unfinished,
            )
        else:
            logger.error(
                "Closing session %r while an agent call's turn is still open. "
                "The turn's %d message(s) are not written.",
                self.session_id,
                unfinished,
            )

    def redact_latest_message(
        self, redact_message: StrandsMessage, agent: Agent, **kwargs: Any
    ) -> None:
        """Replace the latest message with redacted content.

        Normal path: the latest message is still buffered or queued, so it
        is rewritten there and the original never reaches the backend. A message
        stays queued while its write runs, so a redaction that lands then
        also rewrites the queue, and the drain applies it once the write
        finishes. Late path: the message is already stored. The Strands
        lifecycle redacts within the same invocation, so this path is
        defensive. See ``_aredact_latest``.
        """
        self._check_open()
        redacted = copy.deepcopy(redact_message)
        with self._queue_lock:
            if self._turn:
                self._turn[-1] = redacted
                return
            if self._ready:
                self._ready[-1].message = redacted
                return
        try:
            redacted_stored = self._bridge.run(
                self._aredact_latest(redacted), self._request_timeout
            )
        except Exception as e:
            raise SessionException(
                f"Failed to redact latest message for session {self.session_id!r}"
            ) from e
        if not redacted_stored:
            logger.warning(
                "redact_latest_message called for session %s, but its latest "
                "message was not stored, so there is nothing to redact.",
                self.session_id,
            )

    def _redaction_lock(self) -> asyncio.Lock:
        """The lock that lets one redaction at a time rewrite a stored message.

        Bridge loop only. The drain holds it from popping a message until
        any redaction that landed during the write is applied.
        """
        if self._redact_lock is None:
            self._redact_lock = asyncio.Lock()
        return self._redact_lock

    async def _aredact_latest(self, redacted: StrandsMessage) -> bool:
        """Redact the latest stored message. Runs on the bridge loop.

        Returns False when the latest message was not stored. The target is
        read under the redaction lock, so a redaction that waited rewrites
        the message the one before it stored, not a deleted one. See
        ``_aredact_stored`` for what each backend does.

        The rewrite is bounded by ``request_timeout``, so a hung backend
        cannot hold the lock that every later drain needs.
        """
        async with self._redaction_lock():
            with self._queue_lock:
                stored = self._last_persisted
            if stored is None:
                return False
            try:
                result = await asyncio.wait_for(
                    self._aredact_stored(stored, redacted), self._request_timeout
                )
            except BaseException:
                with self._queue_lock:
                    self._last_persisted = None  # its id may be deleted already
                raise
            with self._queue_lock:
                self._last_persisted = result
            return True

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
        # Dropped first, so a delete that commits and then fails or times
        # out still keeps the deleted text from a deferred extraction.
        self._awaiting_extraction.discard(stored.id)
        await self._client.short_term.delete_message(stored.id)
        text = _message_text(redacted) or "[REDACTED]"
        return await self._aadd_message(redacted.get("role", "user"), text)

    def sync_agent(self, agent: Agent, **kwargs: Any) -> None:
        """Agent state is not persisted (design decision: no Strands-specific
        nodes in the graph). Also fired on MessageAddedEvent by the base
        class, so it must not flush the buffer."""
        return None

    def _flush_buffer(self, *, closing: bool = False) -> None:
        """Persist every queued message, blocking until done.

        Messages buffered outside a turn are queued first. A turn that is
        running or paused stays buffered. See ``_release_idle_turn``.

        For ``close()`` and other sync callers. ``aclose()`` awaits
        ``_aflush_buffer`` instead. A failure is handled by
        ``on_persist_error`` (see ``_wait_failed``). ``closing`` is True
        only for the flush of ``close()`` itself.
        """
        self._release_idle_turn(closing=closing)
        future: concurrent.futures.Future[None] | None = None
        try:
            future = self._submit_drain()
            if future is not None:
                future.result(timeout=self._drain_timeout())
        except Exception as e:
            self._wait_failed(future, e, closing=closing)

    async def _aflush_buffer(self, *, closing: bool = False) -> None:
        """Like ``_flush_buffer``, without blocking the caller's loop."""
        self._release_idle_turn(closing=closing)
        await self._aflush_ready(closing=closing)

    async def _aflush_ready(self, *, closing: bool = False, ends_turn: int | None = None) -> None:
        """Persist the ready queue in order, leaving the current turn buffered.

        The writes run on the bridge and are shielded: cancelling the
        caller stops only the wait. The writes run to completion, and
        ``close()`` waits for them.

        A failure is handled by ``on_persist_error`` (see
        ``_wait_failed``).

        ``ends_turn`` is the number of the turn this flush ends. The turn
        counts as written when the drain finishes, not when the wait does.
        A wait that runs out leaves the drain going, and a message of this
        turn that fails late still gets its retry.
        """
        future: concurrent.futures.Future[None] | None = None
        try:
            future = self._submit_drain()
            if future is not None:
                await self._bridge.wait(future, self._drain_timeout(), shield=True)
        except Exception as e:
            self._wait_failed(future, e, closing=closing)
        finally:
            if ends_turn is not None:
                if future is None:
                    self._turn_written(ends_turn)
                else:
                    future.add_done_callback(lambda _: self._turn_written(ends_turn))

    def _drain_timeout(self) -> float:
        """How long a caller waits for a drain.

        Each step of a drain is bounded by ``request_timeout``, so a drain
        can take that long per step for each queued message. One message
        more covers a drain that is still finishing a popped message. A
        deferred extraction runs after the drain, so it is not a step.

        A redaction that lands while its message is being written adds one
        more step, and it is not counted. Strands redacts before the hook
        drains, so this is rare. If the wait runs out first, the drain goes
        on in the background, and the caller logs a warning.
        """
        extracts = self._extract_entities and not self._is_nams and not self._defer_extraction
        steps = 1 + int(extracts) + int(self._record_tool_calls)
        with self._queue_lock:
            queued = len(self._ready)
        return self._request_timeout * steps * (queued + 1)

    def _wait_failed(
        self,
        future: concurrent.futures.Future[None] | None,
        error: Exception,
        *,
        closing: bool,
    ) -> None:
        """Handle an error from a flush's wait for its drain.

        A drain that is still running means the wait ran out, not a write.
        The drain goes on. A drain that succeeded means it finished just
        after the wait ran out, so nothing failed. A drain that failed is
        reported with its own error, which may have come just after the
        wait ran out. Any other error is a failed flush.

        ``closing`` is True only for the wait of ``close()`` itself. Only
        that wait reports messages left unwritten at close.
        """
        if future is None or future.done():
            if future is not None and not future.cancelled():
                cause = future.exception()
                if cause is None:
                    return
                if isinstance(cause, Exception):
                    error = cause
            self._persist_failed(error, closing=closing)
            return
        with self._queue_lock:
            unwritten = len(self._ready)
        if closing:
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

    def _persist_failed(self, error: Exception, *, closing: bool) -> None:
        """Apply ``on_persist_error`` to a failed flush.

        A failure in the flush of ``close()`` leaves its messages unwritten
        for good, so it is logged as an ERROR in both modes. ``"raise"``
        then raises SessionException. ``"warn"`` logs a warning for a
        failure before close. A drop gets no warning, since the drain
        logged it as an error.

        Another flush that fails while a close runs leaves its message
        queued. The close retries it and reports the outcome, so this
        flush only logs a warning, in both modes.
        """
        dropped = isinstance(error, _MessageDropped)
        cause = error.error if isinstance(error, _MessageDropped) else error
        if self._closed and not closing and not dropped:
            logger.warning(
                "Failed to persist message for session %r. It stays queued, "
                "and the close that is running retries it: %s",
                self.session_id,
                _describe(cause),
            )
            return
        if dropped:
            detail = "a message was dropped after its write failed"
        elif closing:
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
        if not dropped and not closing:
            logger.warning(
                "Failed to persist message for session %r. It stays queued, "
                "and the next flush retries it: %s",
                self.session_id,
                _describe(cause),
            )

    def _submit_drain(self) -> concurrent.futures.Future[None] | None:
        """Start a drain of the ready queue on the bridge, or join the running one.

        A drain that has not yet found the queue empty writes every message
        queued before then, so a flush joins it instead of starting another.
        A close never joins. It starts its own drain, which runs once the
        running one ends and retries a message that drain failed on.
        Returns None when the queue is empty, unless the manager is closing
        while a drain still runs. The bridge is called outside the queue
        lock.
        """
        with self._queue_lock:
            run = self._drain_run
            running = run is not None and not run.done()
            if not self._ready and not (self._closed and running):
                return None
            if run is not None and running and not self._closed:
                return run
            run = self._drain_run = concurrent.futures.Future()
        try:
            submitted = self._bridge.submit(self._adrain(run))
        except BaseException as e:
            with self._queue_lock:
                if self._drain_run is run:
                    self._drain_run = None
            run.set_exception(e)  # a flush that joined in the meantime gets it too
            raise
        submitted.add_done_callback(lambda done: _copy_outcome(done, run))
        return run

    async def _adrain(self, run: concurrent.futures.Future[None]) -> None:
        """Write the ready queue in order. Runs on the bridge loop.

        ``run`` is the future flushes wait on. The drain stops being the
        one to join, under the queue lock, as soon as it finds the queue
        empty or fails. A flush after that starts a new drain.

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
        try:
            await self._adrain_locked(run)
        finally:
            with self._queue_lock:
                if self._drain_run is run:
                    self._drain_run = None

    async def _adrain_locked(self, run: concurrent.futures.Future[None]) -> None:
        """The body of ``_adrain``, one drain at a time."""
        if self._drain_lock is None:
            self._drain_lock = asyncio.Lock()
        dropped: _MessageDropped | None = None
        async with self._drain_lock:
            while True:
                with self._queue_lock:
                    if not self._ready:
                        if self._drain_run is run:
                            self._drain_run = None
                        break
                    entry = self._ready[0]
                try:
                    written, stored = await self._awrite(entry)
                except Exception as e:
                    stale = entry.turn < self._turns_written
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
                    async with self._redaction_lock():
                        self._pop_head()
                    continue
                async with self._redaction_lock():
                    latest = self._pop_head(stored)
                    if stored is not None and latest is not written:
                        if self._redacted_on_nams(stored, latest):
                            self._warn_first_write_may_remain(stored, written)
                        else:
                            stored = await self._aredact_written(stored, latest)
                if stored is not None:
                    if self._defer_extraction:
                        self._defer_extract(stored)
                    else:
                        await self._aextract(stored)
                if blocks := self._tool_uses(latest):
                    try:
                        await asyncio.wait_for(
                            self._arecord_tool_uses(blocks), self._request_timeout
                        )
                    except TimeoutError:
                        logger.warning(
                            "Tool calls of session %r are still being mirrored to "
                            "reasoning memory after %ss. The recording goes on in "
                            "the background.",
                            self.session_id,
                            self._request_timeout,
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

        The content returned is what the first attempt sent, since that
        attempt may have committed. A retry under the same id is then
        skipped, so a redaction made before the retry must still be applied
        to the stored original. If the first attempt never committed, the
        retry stores the redacted text. Bolt replies with the content sent
        whether or not the retry stored anything, so the drain rewrites the
        message there anyway. A NAMS retry is a new message, so the drain
        leaves it as is. The failed first attempt may still have stored the
        original there, so the drain warns. See ``_redacted_on_nams``.
        """
        task = entry.write
        if task is None or (task.done() and (task.cancelled() or task.exception() is not None)):
            message = entry.message
            if not (text := _message_text(message)):
                return message, None
            task = asyncio.ensure_future(self._aadd_message(message["role"], text, entry.id))
            entry.write = task
            if entry.written is None:
                entry.written = message
        written = cast("StrandsMessage", entry.written)
        return written, await asyncio.wait_for(asyncio.shield(task), self._request_timeout)

    def _pop_head(self, stored: StoredMessage | None = None) -> StrandsMessage:
        """Remove the head of the ready queue, count it, and return its content.

        The content is read under the lock, so it includes a redaction that
        landed while the write ran. ``stored`` becomes the target of a late
        redaction in the same step. So a redaction finds the message either
        in the queue or in ``_last_persisted``. A message that was dropped
        or has no text leaves None there, so a late redaction never
        rewrites the message before it.
        """
        with self._queue_lock:
            entry = self._ready.popleft()
            self._last_persisted = stored
        self._dequeued += 1
        return entry.message

    def _redacted_on_nams(self, stored: StoredMessage, latest: StrandsMessage) -> bool:
        """Whether a NAMS retry already stored the latest content of its message.

        NAMS assigns a new id to every write, so a retry stores what it
        sends and replies with it. Bolt skips a retry under the id of a
        message it already holds, yet replies with the content sent, so
        its reply cannot show what is stored.
        """
        return self._is_nams and stored.content == _message_text(latest)

    def _warn_first_write_may_remain(self, stored: StoredMessage, written: StrandsMessage) -> None:
        """Warn that a failed first write on NAMS may have kept the original text.

        The retry stored the redacted text as a new message. The first
        attempt failed, but it may have committed before its reply was
        lost. NAMS has no endpoint to delete a message, so that copy stays.
        """
        if _message_text(written) == stored.content:
            return  # the redaction left the text as it was
        logger.warning(
            "Message %s of session %r was redacted after its first write "
            "failed. That write may have stored the original text on NAMS "
            "before its reply was lost. NAMS has no endpoint to delete a "
            "message, so the original may stay.",
            stored.id,
            self.session_id,
        )

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

    def _defer_extract(self, stored: StoredMessage) -> None:
        """Start extracting a stored message in the background. Bridge loop only.

        Each extraction waits for the one before it on the bridge for this
        session id, so a session's extractions run in write order. That
        holds across managers that share a bridge. Does nothing when
        ``_aextract`` would do nothing, so no task starts on NAMS.
        """
        if not self._extract_entities or self._is_nams:
            return
        self._awaiting_extraction.add(stored.id)
        task = self._bridge.chain(
            f"extract:{self.session_id}",
            lambda previous: self._start_background(
                "extract", self._aextract_after(previous, stored)
            ),
        )
        self._extraction = task
        self._live_extractions.add(task)
        task.add_done_callback(self._live_extractions.discard)  # also runs for one never started

    async def _aextract_after(
        self, previous: asyncio.Task[None] | None, stored: StoredMessage
    ) -> None:
        """Wait for the ``previous`` extraction, then extract ``stored``.

        A late redaction may have deleted the message first. It then took
        the message out of ``_awaiting_extraction``. The check holds the
        redaction lock, so it sees every redaction that finished. The check
        takes the message out too, so the set holds only the backlog.
        The extraction runs without the lock, since the next drain needs
        it. A redaction that lands during the extraction leaves the
        entities in place, as it does when the drain extracts.
        """
        if previous is not None:
            await asyncio.wait({previous})
        async with self._redaction_lock():
            if stored.id not in self._awaiting_extraction:
                logger.debug(
                    "Skipped entity extraction for message %s of session %r. "
                    "A redaction deleted it first.",
                    stored.id,
                    self.session_id,
                )
                return
            self._awaiting_extraction.discard(stored.id)
        await self._aextract(stored)

    def _start_background(self, kind: str, coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        """Run ``coro`` as a task on the bridge loop and report it to the hook.

        The task and the hook both get ``"<kind>:<session_id>"`` as a name.
        The hook's done function runs once however the task ends, even
        when the task is cancelled before it starts. A task that ignores
        the bridge's shutdown cancel gets its done function at shutdown,
        with a WARNING. An error from the hook or the done function is
        logged, and the task runs anyway.
        """
        name = f"{kind}:{self.session_id}"
        task = asyncio.ensure_future(coro)
        task.set_name(name)
        if (hook := self._background_hook) is None:
            return task
        try:
            done = hook(name)
        except Exception as e:
            logger.warning("background_hook failed for task %r: %s", name, _describe(e))
            return task
        self._bridge.when_done(task, lambda: _report_done(name, done))
        return task

    async def _arecord_tool_uses(self, blocks: list[ToolUse]) -> None:
        """Mirror toolUse blocks into reasoning memory. Runs on the bridge loop.

        The recording runs as a shielded task, so a wait that runs out does
        not cancel it. A started trace always has its id kept, so the next
        message does not start a second trace. Every step gets its tool
        call. Each recording waits for the one before it, so steps keep
        their order. A failure is logged by the task and never raised.
        """
        previous = self._tool_record
        task = self._start_background("tool-calls", self._arecord_after(previous, blocks))
        self._tool_record = task
        self._live_tool_records.add(task)
        task.add_done_callback(self._live_tool_records.discard)  # also runs for one never started
        await asyncio.shield(task)

    async def _arecord_after(
        self, previous: asyncio.Task[None] | None, blocks: list[ToolUse]
    ) -> None:
        """Wait for the ``previous`` recording, then record ``blocks``.

        The error is logged here, since the waiter may have stopped waiting.
        A cancellation, which only closing the bridge causes, is logged too.
        """
        try:
            if previous is not None:
                await asyncio.wait({previous})
            await self._arecord_blocks(blocks)
        except asyncio.CancelledError:
            logger.warning(
                "Mirroring of tool calls of session %r to reasoning memory was "
                "cancelled, so some of them are not recorded.",
                self.session_id,
            )
            raise
        except Exception as e:
            logger.warning("Failed to mirror tool calls to reasoning memory: %s", _describe(e))

    async def _arecord_blocks(self, blocks: list[ToolUse]) -> None:
        """Record each block as a step and its tool call, starting the trace once."""
        trace_id = self._trace_id
        if trace_id is None:
            key = await self._aresolve_key()
            trace = await self._client.reasoning.start_trace(key, task="Strands agent session")
            trace_id = self._trace_id = trace.id
        for block in blocks:
            name = block.get("name") or "unknown"
            step = await self._client.reasoning.add_step(
                trace_id, thought=f"Tool use: {name}", action=name
            )
            await self._client.reasoning.record_tool_call(step.id, name, block.get("input") or {})

    def _tool_records_bound(self) -> float | None:
        """How long close waits for tool-call mirroring, the drain timeout.

        Returns None when no recording has started. ``_tool_record`` is read
        off the loop here. It only ever changes from None to a task, so None
        means there is nothing to wait for. The caller's wait on the bridge
        adds one ``request_timeout`` to cover scheduling.
        """
        if self._tool_record is None:
            return None
        return self._drain_timeout()

    async def _await_tool_records(self, timeout: float) -> None:
        """Wait up to ``timeout`` for tool-call mirroring. Runs on the bridge loop."""
        await self._await_chain(
            self._tool_record, self._live_tool_records, "tool-call recording", timeout
        )

    def _extractions_bound(self) -> float | None:
        """How long close waits for deferred extractions, the drain timeout.

        Returns None when there is nothing to wait for: no extraction has
        started, or the bridge is shared. Its owner waits for them when it
        closes. ``_extraction`` is read off the loop, as in
        ``_tool_records_bound``.
        """
        if not self._owns_bridge or self._extraction is None:
            return None
        return self._drain_timeout()

    async def _await_extractions(self, timeout: float) -> None:
        """Wait up to ``timeout`` for deferred extractions. Runs on the bridge loop."""
        await self._await_chain(
            self._extraction, self._live_extractions, "entity extraction", timeout
        )

    async def _await_chain(
        self,
        latest: asyncio.Task[None] | None,
        live: set[asyncio.Task[None]],
        what: str,
        timeout: float,
    ) -> None:
        """Wait up to ``timeout`` for a chain of background tasks to finish.

        Each task in the chain waits for the one before it, so the
        ``latest`` finishing means all have. The tasks still in ``live``
        after the wait are cancelled, and a warning names how many. They
        get ``_UNWIND_SECONDS`` to unwind, so they end before the client
        closes.

        A task that is already done is still waited on. Its done callbacks
        run in a later loop iteration. The wait's own callback runs after
        the one that drops the task from ``live``, so ``live`` is current
        when the wait returns.
        """
        if latest is not None:
            await asyncio.wait({latest}, timeout=timeout)
        if not (left := set(live)):
            return
        logger.warning(
            "Closing session %r while %d %s(s) are still running after %ss. They are cancelled.",
            self.session_id,
            len(left),
            what,
            timeout,
        )
        for task in left:
            task.cancel()
        await asyncio.wait(left, timeout=_UNWIND_SECONDS)

    def _close_wait(self, bound: float) -> float:
        """How long close waits on the bridge for a chain bounded by ``bound``.

        It adds the time cancelled tasks get to unwind, and one
        ``request_timeout`` to cover scheduling.
        """
        return bound + _UNWIND_SECONDS + self._request_timeout

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
                await self._aflush_buffer(closing=True)
            except BaseException as e:
                failure = e
            try:
                if (bound := self._tool_records_bound()) is not None:
                    await self._bridge.run_async(
                        self._await_tool_records(bound), self._close_wait(bound)
                    )
            except BaseException as e:
                failure = self._first_failure(failure, e, "wait for tool-call mirroring")
            try:
                if (bound := self._extractions_bound()) is not None:
                    await self._bridge.run_async(
                        self._await_extractions(bound), self._close_wait(bound)
                    )
            except BaseException as e:
                failure = self._first_failure(failure, e, "wait for entity extraction")
            try:
                if self._should_close_client and self._client.is_connected:
                    await self._bridge.run_async(self._client.close(), self._request_timeout)
            except BaseException as e:
                failure = self._first_failure(failure, e, "close the memory client")
            try:
                if self._owns_bridge:
                    await asyncio.to_thread(self._bridge.close)
            except BaseException as e:
                failure = self._first_failure(failure, e, "stop the bridge")
        finally:
            closing.set_result(None)
        if failure is not None:
            raise failure

    def close(self) -> None:
        """Flush, release the client (if owned/connected by us), stop the bridge.

        A manager given a shared ``bridge`` leaves the bridge and the client
        running.

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
                self._flush_buffer(closing=True)
            except BaseException as e:
                failure = e
            try:
                if (bound := self._tool_records_bound()) is not None:
                    self._bridge.run(self._await_tool_records(bound), self._close_wait(bound))
            except BaseException as e:
                failure = self._first_failure(failure, e, "wait for tool-call mirroring")
            try:
                if (bound := self._extractions_bound()) is not None:
                    self._bridge.run(self._await_extractions(bound), self._close_wait(bound))
            except BaseException as e:
                failure = self._first_failure(failure, e, "wait for entity extraction")
            try:
                if self._should_close_client and self._client.is_connected:
                    self._bridge.run(self._client.close(), self._request_timeout)
            except BaseException as e:
                failure = self._first_failure(failure, e, "close the memory client")
            try:
                if self._owns_bridge:
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

        An interrupt is the exception: a later BaseException that is not an
        Exception, such as KeyboardInterrupt or CancelledError, wins over an
        earlier Exception. The earlier one is logged instead, so Ctrl-C does
        not turn into a catchable error.
        """
        if failure is None:
            return error
        if isinstance(failure, Exception) and not isinstance(error, Exception):
            logger.error(
                "Closing session %r was interrupted while it tried to %s. "
                "An earlier step had failed: %s",
                self.session_id,
                step,
                _describe(failure),
                exc_info=failure,
            )
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
