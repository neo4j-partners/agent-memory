# DBX Branch Changes

## What changed

### Neo4j connection settings (uncommitted)

Added three new connection pool settings to `Neo4jConfig` and wired them into the driver in `Neo4jClient`:

- `max_connection_lifetime` (default 300s) — proactively closes and replaces pooled connections older than this limit, preventing use of connections that the server (e.g. Aura) may have already dropped.
- `liveness_check_timeout` (default 60s) — checks idle connections are still alive before reusing them.
- `keep_alive` (default True) — enables TCP keep-alive to prevent idle connection drops.

**Files:**
- `src/neo4j_agent_memory/config/settings.py`
- `src/neo4j_agent_memory/graph/client.py`

## Why

Cloud-hosted Neo4j instances (Aura, DBX) have server-side idle timeouts. Without these settings, the driver can hold stale connections in the pool and fail on the next query with a connection reset error. These defaults keep the pool healthy without requiring users to configure anything.

## Suggested commit

**Title:** add connection pool lifetime and liveness settings for cloud compatibility

**Message:**
```
add connection pool lifetime and liveness settings for cloud compatibility

Add max_connection_lifetime, liveness_check_timeout, and keep_alive
options to Neo4jConfig and pass them through to the driver. This prevents
stale connection errors when running against cloud-hosted Neo4j instances
that enforce server-side idle timeouts.
```
