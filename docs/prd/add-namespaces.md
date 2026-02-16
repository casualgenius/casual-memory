# PRD: Namespace Support in casual-memory

**Author:** Alex
**Status:** Draft
**Created:** 2026-02-16
**Library:** [casual-memory](https://github.com/casualgenius/casual-memory) ([PyPI](https://pypi.org/project/casual-memory/))

---

## 1. Overview

### 1.1 Problem Statement

The casual-memory library currently uses `user_id` as the sole scope identifier across all storage backends (Qdrant, PostgreSQL, Redis). This works when every memory belongs to a human user, but the AI Assistant platform now stores two fundamentally different categories of memory:

- **User memories** — facts extracted from conversations with a human user (preferences, history, context).
- **Agent memories** — facts extracted from an AI agent's autonomous interactions (opinions, insights, relationships formed on Moltbook).

These categories have different extraction prompts, different conflict resolution semantics, and different access patterns. The current workaround encodes the entity type into the `user_id` field using the format `__agent:<agentId>__`, which creates several problems:

1. **URL-unsafe** — Colons and double underscores cause encoding issues when used in REST path parameters, which the Memory Store Service is migrating to.
2. **Implicit typing** — The storage layer has no structured way to distinguish user memories from agent memories. Any filtering requires string parsing of the user_id.
3. **No isolation guarantee** — A query without proper filtering could accidentally return agent memories mixed with user memories or vice versa.
4. **Brittle convention** — The `__agent:X__` format is an undocumented convention that every consumer must know about and implement correctly.

### 1.2 Proposed Solution

Add a `namespace` field to the casual-memory library's storage protocols, models, and adapter implementations. Every storage operation is scoped to a `(namespace, entity_id)` pair rather than a bare `user_id`. This gives the platform a clean, structured way to isolate different categories of memory while keeping the library general-purpose for other consumers who may not need namespaces.

### 1.3 Goals

| Goal | Metric |
|------|--------|
| All storage operations accept a namespace parameter | 100% of protocol methods updated |
| Backward compatible for existing consumers | Existing code without namespace continues to work via default namespace |
| URL-safe identifiers | Namespace and entity IDs contain only lowercase alphanumeric characters, hyphens, and underscores |
| Storage isolation | Queries in one namespace never return results from another |
| Minimal migration effort | Existing Qdrant collections and PostgreSQL tables work with a schema addition, no data migration required for new deployments |

### 1.4 Non-Goals

- Namespace-level access control or permissions (out of scope for the library; handled by consuming services).
- Automatic migration tooling for existing deployments using the `__agent:X__` format (the AI Assistant platform will handle its own migration; the library just needs to support the new pattern).
- Namespace management APIs (create, delete, list namespaces). Namespaces are implicit — they exist when data is stored in them.

---

## 2. Design

### 2.1 Namespace as a Required Field with a Default

The namespace parameter is **required at the protocol level** but has a **default value of `"default"`** at the public API surface. This means:

- **Library consumers who don't need namespaces** can ignore the parameter entirely. All their data lands in the `"default"` namespace, and the library works exactly as it does today. No breaking change for them.
- **The AI Assistant platform** (and any other consumer that needs isolation) passes explicit namespaces: `"user"` for user memories, `"agent"` for agent memories. The consuming service is responsible for always passing the correct namespace.
- **Storage adapters always receive a namespace.** There is no ambiguity at the storage layer — every record has a namespace field, and every query filters by namespace. The default just means callers who don't care get a sensible value.

This avoids the problem of optional fields where forgetting to pass a namespace silently stores data in the wrong scope. If the platform's Agent Memory Service forgets to pass `namespace="agent"`, the data lands in `"default"` rather than contaminating the `"user"` namespace. The platform can then enforce explicit namespaces at its own service layer.

### 2.2 Naming: `user_id` Becomes `entity_id`

With namespaces, `user_id` is misleading — the identifier might refer to a user or an agent. The field should be renamed to `entity_id` to accurately reflect that it identifies whatever entity owns the memories within a given namespace.

However, this is a breaking change to the library's public API. To manage this:

- **Phase 1:** Add `entity_id` as an alias for `user_id` across all protocols and models. Both are accepted, but `entity_id` is preferred. Using `user_id` emits a deprecation warning.
- **Phase 2 (next major version):** Remove `user_id`, keep only `entity_id`.

The storage adapters should use `entity_id` internally from the start, mapping from `user_id` at the boundary if the caller uses the old name.

### 2.3 Validation Rules

Namespaces and entity IDs must be URL-safe and storage-safe:

- **Allowed characters:** Lowercase alphanumeric (`a-z`, `0-9`), hyphens (`-`), and underscores (`_`).
- **Length:** 1-100 characters.
- **Reserved namespaces:** `"default"` is the library's default namespace. No other namespaces are reserved.
- **Validation location:** The library validates namespace and entity_id format before passing to storage adapters. Invalid values raise a `ValueError` with a clear message.

---

## 3. Changes by Component

### 3.1 MemoryFact Model

The `MemoryFact` dataclass gains a `namespace` field:

```python
@dataclass
class MemoryFact:
    text: str
    type: str
    tags: list[str]
    importance: float
    confidence: float
    entity_id: str          # renamed from user_id
    namespace: str = "default"
    # ... existing fields unchanged
```

For backward compatibility, `user_id` is accepted as an alias during the deprecation period:

```python
@dataclass
class MemoryFact:
    # ... fields ...

    def __init__(self, ..., entity_id: str = "", user_id: str = "", namespace: str = "default", ...):
        if user_id and not entity_id:
            warnings.warn("user_id is deprecated, use entity_id", DeprecationWarning, stacklevel=2)
            entity_id = user_id
        self.entity_id = entity_id
        self.namespace = namespace
        # ...
```

### 3.2 Storage Protocols

All protocol methods gain a `namespace` parameter. Shown here for the vector store protocol — the same pattern applies to the conflict store and short-term store protocols.

**Current:**

```python
class VectorMemoryStore(Protocol):
    async def add_memory(self, memory: MemoryFact, embedding: list[float], user_id: str) -> str: ...
    async def query_similar(self, embedding: list[float], user_id: str, limit: int = 5) -> list[SimilarMemory]: ...
    async def delete_memory(self, memory_id: str, user_id: str) -> bool: ...
    async def clear_memories(self, user_id: str) -> int: ...
```

**Proposed:**

```python
class VectorMemoryStore(Protocol):
    async def add_memory(self, memory: MemoryFact, embedding: list[float], entity_id: str, namespace: str = "default") -> str: ...
    async def query_similar(self, embedding: list[float], entity_id: str, namespace: str = "default", limit: int = 5) -> list[SimilarMemory]: ...
    async def delete_memory(self, memory_id: str, entity_id: str, namespace: str = "default") -> bool: ...
    async def clear_memories(self, entity_id: str, namespace: str = "default") -> int: ...
```

The same pattern applies to:

- **`ConflictStore` protocol** — `get_pending`, `resolve`, `record_attempt`, `get_with_memories`, `get_count` all gain `namespace`.
- **`ShortTermStore` protocol** — `save`, `get`, `clear` all gain `namespace`.

During the deprecation period, `user_id` is accepted as a keyword argument alias for `entity_id` in all methods, with a deprecation warning.

### 3.3 Qdrant Adapter

**Payload changes:**

Every point stored in Qdrant gains a `namespace` field in its payload:

```python
payload = {
    "namespace": namespace,      # new
    "entity_id": entity_id,      # renamed from user_id
    "text": memory.text,
    "type": memory.type,
    "tags": memory.tags,
    # ... rest unchanged
}
```

**Query changes:**

All queries add a namespace filter condition:

```python
filter = models.Filter(must=[
    models.FieldCondition(
        key="namespace",
        match=models.MatchValue(value=namespace),
    ),
    models.FieldCondition(
        key="entity_id",
        match=models.MatchValue(value=entity_id),
    ),
])
```

**Index changes:**

A payload index should be created on `namespace` for efficient filtering:

```python
await client.create_payload_index(
    collection_name=collection_name,
    field_name="namespace",
    field_schema=models.PayloadSchemaType.KEYWORD,
)
```

**Backward compatibility for existing data:**

Existing points in Qdrant will not have a `namespace` field. The adapter should handle this gracefully:

- On **read**: Points without a `namespace` field are treated as belonging to the `"default"` namespace. The query filter should use a condition that matches either `namespace == "default"` or `namespace` field not present.
- On **write**: All new points always include `namespace`.
- A **migration utility** (optional, not in the library itself) can backfill existing points with `namespace: "default"`.

### 3.4 PostgreSQL Adapter (Conflict Store)

**Schema changes:**

Add a `namespace` column to the conflicts table:

```sql
ALTER TABLE conflicts ADD COLUMN namespace VARCHAR(100) NOT NULL DEFAULT 'default';
```

Update the compound index:

```sql
DROP INDEX IF EXISTS idx_conflicts_user_id;
CREATE INDEX idx_conflicts_namespace_entity ON conflicts(namespace, entity_id);
```

**Query changes:**

All queries include `namespace` in their WHERE clause:

```sql
SELECT * FROM conflicts
WHERE namespace = $1 AND entity_id = $2 AND status = 'pending';
```

**Backward compatibility:**

The `DEFAULT 'default'` on the column means existing rows automatically belong to the `"default"` namespace. No data migration required.

### 3.5 Redis Adapter (Short-Term Store)

**Key pattern change:**

Current: `memory:{user_id}:{session_id}`
Proposed: `memory:{namespace}:{entity_id}:{session_id}`

For the default namespace: `memory:default:alex:sess_abc123`
For the agent namespace: `memory:agent:dixie-flatline:heartbeat_123`

**Backward compatibility:**

Since Redis keys are just strings, old-format keys (`memory:alex:sess_123`) and new-format keys (`memory:default:alex:sess_123`) coexist without conflict. Old data remains accessible under the old keys until it expires via TTL. New writes always use the new format.

### 3.6 Classification Pipeline

The `MemoryClassificationPipeline` receives `MemoryFact` objects which will carry the `namespace` field. The pipeline itself doesn't need to change its logic — it classifies based on text content and similarity scores, not on namespace. However, the pipeline passes `MemoryFact` through to storage, so the namespace flows through naturally.

The one consideration is conflict resolution semantics. The platform's Memory Store Service currently applies different resolution logic for agent memories (treating opinion conflicts as evolution rather than errors). This logic lives in the consuming service, not in the library, so no changes needed in casual-memory's classification pipeline.

### 3.7 LLM Memory Extractor

The `LLMMemoryExtractor` produces `MemoryFact` objects from conversation messages. It currently sets `user_id` on extracted facts. This changes to setting `entity_id` and `namespace`:

```python
class LLMMemoryExtractor:
    def __init__(self, ..., default_namespace: str = "default"):
        self.default_namespace = default_namespace

    async def extract(self, messages, entity_id: str, namespace: str | None = None) -> list[MemoryFact]:
        ns = namespace or self.default_namespace
        # ... extraction logic unchanged ...
        # When creating MemoryFact objects:
        return [
            MemoryFact(
                text=fact_text,
                entity_id=entity_id,
                namespace=ns,
                # ...
            )
            for fact_text in extracted_facts
        ]
```

This allows the User Memory Service to instantiate an extractor with `default_namespace="user"` and the Agent Memory Service with `default_namespace="agent"`, avoiding the need to pass namespace on every call.

---

## 4. Public API Surface

### 4.1 High-Level API

For consumers using the high-level `MemoryManager` or equivalent orchestrator:

```python
from casual_memory import MemoryManager

manager = MemoryManager(
    vector_store=qdrant_adapter,
    conflict_store=postgres_adapter,
    short_term_store=redis_adapter,
    # ...
)

# Existing usage (unchanged, uses "default" namespace):
await manager.add(memory_fact, entity_id="alex")
results = await manager.query("what does alex like", entity_id="alex")

# New usage with explicit namespace:
await manager.add(memory_fact, entity_id="alex", namespace="user")
results = await manager.query("what does alex like", entity_id="alex", namespace="user")

await manager.add(agent_fact, entity_id="dixie", namespace="agent")
results = await manager.query("dixie's opinions", entity_id="dixie", namespace="agent")
```

### 4.2 Deprecation Warnings

During the transition period, using `user_id` anywhere in the API triggers a `DeprecationWarning`:

```
DeprecationWarning: 'user_id' is deprecated, use 'entity_id' instead. 
'user_id' will be removed in casual-memory 2.0.
```

This gives downstream consumers (including the AI Assistant platform services) time to migrate at their own pace.

---

## 5. Versioning and Release Strategy

### 5.1 Version Bump

This change adds a new parameter with a default value to all protocol methods and renames `user_id` to `entity_id` with a deprecation alias. This is a **minor version bump** (not major) because:

- All existing code continues to work without changes (default namespace, user_id alias).
- No existing behavior changes.
- Deprecation warnings are advisory, not errors.

### 5.2 Release Sequence

**v1.x.0 (this release):**

- Add `namespace` parameter (default `"default"`) to all protocols, adapters, and models.
- Add `entity_id` as primary field name with `user_id` as deprecated alias.
- Add validation for namespace and entity_id format.
- Update Qdrant, PostgreSQL, and Redis adapters.
- Update `LLMMemoryExtractor` to accept `namespace` and `default_namespace`.
- Backward-compatible read handling for Qdrant points without `namespace` field.
- All existing tests continue to pass without modification.
- New tests for namespace isolation and entity_id aliasing.

**v2.0.0 (future):**

- Remove `user_id` alias (breaking change).
- Remove backward-compatible read handling for data without `namespace` field.
- Require explicit `namespace` (remove default value) — to be decided based on adoption.

---

## 6. Testing Strategy

### 6.1 Unit Tests

| Component | Tests |
|-----------|-------|
| **MemoryFact** | Verify `entity_id` field works. Verify `user_id` alias sets `entity_id` with deprecation warning. Verify `namespace` defaults to `"default"`. |
| **Validation** | Reject invalid namespace formats (colons, spaces, uppercase, empty, >100 chars). Accept valid formats (lowercase, hyphens, underscores, digits). |
| **Qdrant adapter** | Verify namespace included in payload on write. Verify namespace filter applied on query. Verify reads handle missing namespace field (backward compat). |
| **PostgreSQL adapter** | Verify namespace column used in all queries. Verify default value applied to existing rows. |
| **Redis adapter** | Verify new key format includes namespace. Verify old-format keys are unaffected. |

### 6.2 Namespace Isolation Tests

| Test | Description |
|------|-------------|
| **Cross-namespace isolation** | Store a memory in namespace `"user"` for entity `"alex"`, store a different memory in namespace `"agent"` for entity `"alex"`. Query each namespace separately and verify no cross-contamination. |
| **Default namespace isolation** | Store a memory with no explicit namespace. Store a memory with `namespace="agent"`. Query with no explicit namespace and verify only the first memory is returned. |
| **Conflict isolation** | Create conflicting memories in different namespaces for the same entity_id. Verify conflicts are detected within namespace only, not across namespaces. |

### 6.3 Backward Compatibility Tests

| Test | Description |
|------|-------------|
| **No-namespace usage** | Run the full existing test suite without passing any namespace parameter. All tests should pass unchanged. |
| **user_id alias** | Call all API methods with `user_id` instead of `entity_id`. Verify they work correctly and emit deprecation warnings. |
| **Legacy Qdrant data** | Query a Qdrant collection containing points without a `namespace` field. Verify they are treated as `"default"` namespace. |

---

## 7. Decisions (Resolved)

| # | Question | Decision |
|---|----------|----------|
| 1 | Should `entity_id` rename happen in the same release as namespace support, or separately? | **Same release.** The changes are conceptually linked (namespace + entity_id together define the scope), and doing them together means one migration for consumers rather than two. |
| 2 | Should the library provide a migration script for backfilling `namespace` on existing Qdrant data? | **No.** The library provides backward-compatible reads (missing namespace = default). Migration scripts are deployment-specific and belong in the consuming service. Document the approach in the migration guide. |
| 3 | Should `namespace` default be removed in v2.0 (making it required everywhere)? | **Defer.** Evaluate after adoption whether most consumers use explicit namespaces or rely on the default. |