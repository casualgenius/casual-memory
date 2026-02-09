# Architecture Guide

This document provides a comprehensive overview of the casual-memory library architecture, design decisions, and key concepts.

---

## Table of Contents

1. [Overview](#overview)
2. [Core Components](#core-components)
3. [Classification Pipeline](#classification-pipeline)
4. [Intelligence Layer](#intelligence-layer)
5. [Storage Abstraction](#storage-abstraction)
6. [Memory Extraction](#memory-extraction)
7. [Design Patterns](#design-patterns)
8. [Performance Considerations](#performance-considerations)

---

## Overview

casual-memory is an intelligent semantic memory library built on three core principles:

1. **Protocol-based architecture** - Extensible without inheritance
2. **Composable components** - Mix and match classifiers, storage backends
3. **Graceful degradation** - Works even when optional dependencies fail

### Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│  Application Layer (Your Code)                          │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Service Layer                                          │
│  ├─ MemoryService  (Long-term semantic memory)         │
│  └─ ContextService (Short-term conversation context)   │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Classification Pipeline                                │
│  ├─ NLI Classifier (Fast pre-filter)                   │
│  ├─ Conflict Classifier (LLM-based)                    │
│  ├─ Duplicate Classifier (LLM-based)                   │
│  └─ Auto-Resolution Classifier (Confidence-based)      │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Intelligence Layer                                      │
│  ├─ NLI Pre-Filter (DeBERTa cross-encoder)             │
│  ├─ LLM Conflict Verifier (with heuristic fallback)    │
│  ├─ LLM Duplicate Detector (conservative fallback)     │
│  └─ Confidence Scorer (frequency + recency + spread)   │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Storage Layer (Protocol-based)                         │
│  ├─ Vector Storage (QdrantMemoryStore, InMemory)       │
│  ├─ Conflict Storage (SQLAlchemy, InMemory)            │
│  └─ Short-Term Storage (Redis, InMemory)               │
└─────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Classification Pipeline

The classification pipeline is the heart of casual-memory. It chains multiple classifiers to determine how to handle similar memory pairs.

**Key Characteristics:**
- Sequential execution (classifiers run in order)
- Early termination (first confident classification wins)
- Composable (add/remove classifiers)
- Protocol-based (no inheritance required)

**Classifier Interface:**

Individual classifiers implement `classify_pair()` for comparing one memory pair:

```python
async def classify_pair(
    self,
    new_memory: MemoryFact,
    similar_memory: SimilarMemory,
    check_type: CheckType = "primary",  # "primary" or "secondary"
    existing_result: Optional[SimilarityResult] = None,
) -> Optional[SimilarityResult]:
    """
    Classify a single memory pair.

    Returns:
        - SimilarityResult if classifier is confident
        - None to pass to next classifier
        - existing_result to pass through unchanged
    """
    ...
```

The pipeline calls `classify_pair()` for each classifier in sequence, passing results through the chain.

### 2. Classification Outcomes

**Similarity Outcomes** (for each similar memory):

- **conflict** - Contradictory memories needing manual resolution
  - Location conflicts ("I live in London" vs "I live in Paris")
  - Job conflicts ("I work as a teacher" vs "I work as a doctor")
  - Preference conflicts ("I like coffee" vs "I don't like coffee")

- **superseded** - Similar memory should be archived
  - New memory is a refinement (more specific)
  - New memory has higher confidence
  - Auto-resolved conflicts (new one clearly wins)

- **same** - Duplicate memory (update existing metadata)
  - Exact duplicates
  - Paraphrases of the same fact
  - Auto-resolved conflicts (old one clearly wins)

- **neutral** - Distinct memories that can coexist
  - Different facts ("I live in Bangkok" vs "I work in Bangkok")
  - Compatible preferences ("I like coffee" vs "I like tea")
  - Unrelated information

**Memory Outcomes** (overall action for new memory):

- **add** - Insert new memory to vector store
- **skip** - Update existing memory (increment mention_count)
- **conflict** - Create conflict record for user resolution

### 3. Memory Types

Four memory types supported:

```python
class MemoryFact:
    text: str              # Memory content (first-person perspective)
    type: str              # "fact", "preference", "goal", "event"
    tags: list[str]        # Semantic tags for filtering
    importance: float      # 0.0-1.0 (≥0.5 threshold for storage)
    source: str            # "user" or "assistant"
    valid_until: str | None  # Temporal validity (ISO timestamp)
```

### 4. Context Service

`ContextService` manages short-term conversation context — the recent messages that form the LLM's conversational window. It sits above the storage layer and handles business logic that storage backends don't need to know about.

**Key Responsibilities:**
- **Session isolation** — Messages keyed by `user_id:session_id` composite key
- **Message filtering** — `add()` drops system messages; only `user`, `assistant`, and `tool` messages are persisted
- **Safe boundary trimming** — `get()` ensures the returned window starts at a `user` message, never mid-tool-call sequence

**API:**

```python
from casual_memory import ContextService
from casual_memory.storage.short_term.memory import InMemoryShortTermStore

store = InMemoryShortTermStore(max_messages=100)
context = ContextService(short_term_store=store, short_term_limit=50)

# Add messages (accepts list[ChatMessage])
context.add("user1", "session1", messages)

# Get recent messages (safe boundary guaranteed)
messages = context.get("user1", "session1", limit=50)

# Clear session
context.clear("user1", "session1")
```

**Safe Boundary Trimming:**

When fetching the last N messages, the cut boundary can land mid-tool-call sequence:

```
... [user] [assistant w/ tool_calls] [tool_result] | ← cut → [tool_result] [user] ...
```

A `tool_result` without its preceding `assistant(tool_calls)` breaks LLM APIs. Since `add()` drops system messages, only `user`, `assistant`, and `tool` messages exist in the store — making `user` the only safe boundary. `get()` handles this by:

1. Over-fetching `limit + 10` messages from the store
2. Finding the ideal start position (`len - limit`)
3. If that position is a `user` message, return from there (exactly `limit` messages)
4. Otherwise search **backward** into the over-fetch buffer for the nearest `user` message — prefers returning slightly more than `limit` over losing messages
5. If no `user` message exists in the buffer, search **forward** from the ideal start, trimming messages until a `user` message is found (returns fewer than `limit`)
6. If no `user` message is found anywhere, return an **empty list** — this avoids silently returning a broken window that would cause LLM API errors

This logic lives in a shared utility (`trim_to_safe_boundary` in `storage/short_term/utils.py`) and is called only from `ContextService` — keeping storage backends as simple CRUD.

**Data Flow:**

```
ContextService.add(user_id, session_id, messages)
  1. → Filter out system messages
  2. → Wrap each ChatMessage in ShortTermMemory (with timestamp)
  3. → Store to ShortTermStore via composed key "user_id:session_id"

ContextService.get(user_id, session_id, limit)
  1. → Over-fetch (limit + 10) from ShortTermStore
  2. → trim_to_safe_boundary() ensures first message is user
  3. → Return trimmed message list
```

---

## Classification Pipeline

### Sequential Execution Flow

```python
Input:
  new_memory: MemoryFact          # New memory being added
  similar_memories: list[SimilarMemory]  # From vector search
  results: list[SimilarityResult]  # Empty initially

↓ NLI Classifier (Fast Filter, ~50-200ms)
  For each unclassified similar memory:
    - Calculate entailment/contradiction/neutral scores
    - High entailment (≥0.85) → same
    - High neutral (≥0.5) → neutral
    - Uncertain → Skip to next classifier

↓ Conflict Classifier (~500-2000ms)
  For each unclassified similar memory:
    - Call LLM conflict verifier
    - If contradiction detected → conflict (with category metadata)
    - If fallback triggered → Use heuristic patterns
    - No conflict → Skip to next classifier

↓ Duplicate Classifier (~500-2000ms)
  For each unclassified similar memory:
    - Call LLM duplicate detector
    - If same fact → same
    - If refinement → superseded
    - If distinct → neutral

↓ Auto-Resolution Classifier (instant)
  For each conflict result:
    - Calculate confidence ratio (new / existing)
    - Ratio ≥ 1.3 → superseded (keep_new)
    - Ratio ≤ 0.7 → same (keep_old)
    - Else → Keep as conflict

↓ Pipeline determines overall_outcome:
  - Any conflict result → "conflict"
  - Any same result → "skip"
  - Otherwise → "add"

Output: MemoryClassificationResult
  - overall_outcome: "add" | "skip" | "conflict"
  - similarity_results: list[SimilarityResult]
```

### Classifier Independence

Each classifier is independent and can be:
- Used standalone
- Removed from pipeline
- Replaced with custom implementation
- Configured independently

Example custom pipeline:
```python
# Fast pipeline (NLI only)
fast_pipeline = MemoryClassificationPipeline(classifiers=[
    NLIClassifier(nli_filter=nli_filter)
])

# Accuracy-focused (skip NLI, use only LLM)
accuracy_pipeline = MemoryClassificationPipeline(classifiers=[
    ConflictClassifier(llm_conflict_verifier=verifier),
    DuplicateClassifier(llm_duplicate_detector=detector)
])

# Strategy options: "single", "tiered", "all"
tiered_pipeline = MemoryClassificationPipeline(
    classifiers=[...],
    strategy="tiered",
    secondary_conflict_threshold=0.90,
    max_secondary_checks=3,
)
```

---

## Intelligence Layer

### NLI Pre-Filter

Uses DeBERTa-v3-base-mnli-fever-anli cross-encoder for fast semantic filtering.

**Model Details:**
- Accuracy: 92.38% (SNLI), 90.04% (MNLI)
- Speed: ~200ms CPU, ~50ms GPU
- Input: Two text statements
- Output: [contradiction, entailment, neutral] scores

**Caching Strategy:**
- LRU cache (1000 entries)
- Eviction: Removes oldest 200 when full
- Key: (text_a, text_b) tuple
- Reduces redundant model calls by 70-85%

**Lazy Loading:**
- Model loaded on first use
- Graceful handling if sentence-transformers unavailable
- ImportError raised with helpful message

### LLM Conflict Verifier

Detects contradictions using LLM with heuristic fallback.

**LLM-Based Detection:**
- System prompt defines contradiction rules
- User prompt: formatted memory pair
- Response: "YES" or "NO" (10 token limit)
- Temperature: 0.1 (deterministic)

**Heuristic Fallback (when LLM fails):**
- Requires similarity ≥ 0.90
- Negation patterns: "like" vs "don't like", "can" vs "can't"
- Location conflicts: similarity ≥ 0.92 + location keywords
- Job conflicts: similarity ≥ 0.92 + job keywords

**Metrics Tracked:**
- LLM call count
- Success/failure count
- Fallback count
- Success rate percentage

### LLM Duplicate Detector

Distinguishes duplicates/refinements from distinct facts.

**LLM-Based Detection:**
- System prompt with examples
- Response: "SAME" or "DISTINCT"
- Conservative interpretation

**Heuristic Fallback:**
- Similarity ≥ 0.95 → DUPLICATE
- Similarity < 0.95 → DISTINCT (conservative)

**Example Cases:**
```python
# SAME (duplicates/refinements)
"I live in London" vs "I live in Central London"
"I work as engineer" vs "I work as senior software engineer at Google"
"I like coffee" vs "I love coffee"

# DISTINCT (different facts)
"I live in Bangkok" vs "I work in Bangkok"
"I like coffee" vs "I like tea"
"I live in Paris" vs "I live in London" (contradiction)
```

### Confidence Scorer

Calculates memory confidence based on multiple factors.

**Base Confidence (mention frequency):**
- 1 mention: 0.50
- 2 mentions: 0.60
- 3 mentions: 0.70
- 4 mentions: 0.80
- 5+ mentions: 0.95

**Recency Factor:**
- Days since last mention
- Penalty starts after 30 days
- Formula: `max(0.0, 1.0 - (days_since - 30) / 365)`

**Spread Factor:**
- Mentions distributed over time (not all at once)
- Boost: `min(0.05, days_span / 365 * 0.1)`
- Max boost: 0.05

**Combined Formula:**
```python
confidence = min(
    MEMORY_MAX_CONFIDENCE,  # 0.95 cap
    base_confidence * recency_factor + spread_boost
)
```

---

## Storage Abstraction

### Protocol-Based Design

Storage backends implement runtime-checkable protocols (PEP 544).

**Benefits:**
- No inheritance required
- Duck typing (structural subtyping)
- Easy to implement custom backends
- Testable with simple mocks

**Protocols:**

```python
@runtime_checkable
class VectorMemoryStore(Protocol):
    """Vector storage for semantic search."""

    def add(self, vector: list[float], payload: dict[str, Any]) -> str:
        """Add memory vector and payload, return ID."""

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        min_score: float = 0.7,
        filters: Optional[Any] = None,
    ) -> list[Any]:
        """Semantic search for similar memories."""

    def find_similar_memories(
        self,
        embedding: list[float],
        user_id: Optional[str] = None,
        threshold: Optional[float] = None,
        limit: int = 5,
        exclude_archived: bool = True,
    ) -> list[tuple[Any, float]]:
        """Find similar memories for classification. Returns (memory_point, score) tuples."""

    def update_memory(self, memory_id: str, payload_updates: dict[str, Any]) -> bool:
        """Update memory metadata (mention_count, last_seen, etc.)."""

    def get_by_id(self, memory_id: str) -> Optional[Any]:
        """Retrieve a specific memory by ID."""

    def archive_memory(
        self, memory_id: str, superseded_by: Optional[str] = None
    ) -> bool:
        """Soft-delete memory (sets archived=True)."""

    def clear_user_memories(self, user_id: str) -> int:
        """Clear all memories for a user. Returns count deleted."""


@runtime_checkable
class ConflictStore(Protocol):
    """Storage for memory conflicts."""

    def add_conflict(self, conflict: MemoryConflict) -> str:
        """Store conflict and return ID."""

    def get_conflict(self, conflict_id: str) -> Optional[MemoryConflict]:
        """Retrieve conflict by ID."""

    def get_pending_conflicts(
        self, user_id: str, limit: Optional[int] = None
    ) -> list[MemoryConflict]:
        """List unresolved conflicts for a user."""

    def resolve_conflict(
        self, conflict_id: str, resolution: ConflictResolution
    ) -> bool:
        """Mark conflict as resolved."""

    def get_conflict_count(self, user_id: str, status: Optional[str] = None) -> int:
        """Count conflicts for a user."""

    def escalate_conflict(self, conflict_id: str) -> bool:
        """Escalate a conflict that couldn't be auto-resolved."""

    def clear_user_conflicts(self, user_id: str, status: Optional[str] = None) -> int:
        """Clear conflicts for a user. Returns count cleared."""


@runtime_checkable
class ShortTermStore(Protocol):
    """Storage for conversation history."""

    def add_messages(self, user_id: str, messages: list[ShortTermMemory]) -> int:
        """Add messages to history. Returns count added."""

    def get_recent_messages(
        self, user_id: str, limit: int = 20
    ) -> list[ShortTermMemory]:
        """Get recent messages for a user."""

    def clear_user_messages(self, user_id: str) -> int:
        """Clear all messages for user. Returns count deleted."""

    def get_message_count(self, user_id: str) -> int:
        """Get the number of messages stored for a user."""
```

### User Isolation

All storage operations scoped by `user_id`:

```python
# user_id is included in the memory payload
vector_store.add(vector, payload={"text": "I love hiking", "user_id": "alice", ...})
vector_store.add(vector, payload={"text": "I love gaming", "user_id": "bob", ...})

# find_similar_memories has explicit user_id parameter
results_alice = vector_store.find_similar_memories(embedding, user_id="alice")
results_bob = vector_store.find_similar_memories(embedding, user_id="bob")
# results_alice != results_bob - each user sees only their own memories

# search() uses filters for user isolation
results = vector_store.search(query_embedding, filters={"user_id": "alice"})
```

### Soft Delete Pattern

Memories are archived, not deleted:

```python
memory_fact = MemoryFact(
    id="mem_123",
    text="I live in London",
    archived=False,  # Active
    archived_at=None,
    superseded_by=None
)

# Archive when superseded
await vector_store.archive(
    memory_id="mem_123",
    user_id="user_1",
    superseded_by="mem_456"  # New memory ID
)

# After archiving:
# archived=True
# archived_at="2024-01-15T10:30:00Z"
# superseded_by="mem_456"

# Excluded from searches by default
results = await vector_store.search(
    query_text="location",
    user_id="user_1",
    exclude_archived=True  # Default
)
# mem_123 won't appear in results
```

---

## Memory Extraction

### LLMMemoryExtracter

```python
from casual_memory.extractors import LLMMemoryExtracter
from casual_memory.extractors.prompts import USER_MEMORY_PROMPT

extractor = LLMMemoryExtracter(
    llm_provider=llm_provider,
    prompt=USER_MEMORY_PROMPT,
)

memories = await extractor.extract(messages)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm_provider` | `LLMProvider` | (required) | casual-llm compatible provider |
| `prompt` | `str` | (required) | System prompt with `{today_natural}` and `{isonow}` placeholders |
| `extraction_model` | `type[BaseModel]` | `MemoryExtractionResponse` | Pydantic model defining the JSON schema for LLM output |

### Custom Extraction Models

By default, the extractor uses `MemoryExtractionResponse` which wraps `MemoryFactExtraction` — a strict schema with `Literal` types for the `type` field (`"fact"`, `"preference"`, `"event"`, `"goal"`, `"weather"`).

For applications that need custom memory types (e.g., `"insight"`, `"reflection"`, `"opinion"`), pass your own Pydantic model via the `extraction_model` parameter. The model must have a `memories` attribute that returns a list of objects compatible with `MemoryFact` (at minimum: `text`, `type`, `tags`, `importance`).

```python
from pydantic import BaseModel, Field

# 1. Define your custom memory extraction fields
class AgentMemory(BaseModel):
    text: str = Field(..., description="Memory text")
    type: str = Field(..., description="Custom types: 'insight', 'reflection', 'opinion'")
    tags: list[str] = Field(default_factory=list)
    importance: float = Field(..., ge=0.0, le=1.0)

# 2. Wrap in a response model with a 'memories' attribute
class AgentExtractionResponse(BaseModel):
    memories: list[AgentMemory] = Field(default_factory=list)

# 3. Pass to the extractor
extractor = LLMMemoryExtracter(
    llm_provider=provider,
    prompt=AGENT_MEMORY_PROMPT,  # Your custom prompt
    extraction_model=AgentExtractionResponse,
)

memories = await extractor.extract(messages)
# memories[0].type could be "insight", "reflection", etc.
```

This works because `MemoryFact.type` is a flexible `str` field at the storage layer, while `MemoryFactExtraction.type` (the default schema) is a strict `Literal` — giving you validation by default with the option to bypass it.

### Extraction Process

```
Input: Conversation messages
  [UserMessage("My name is Alex and I live in Bangkok"),
   AssistantMessage("Nice to meet you!")]

↓ LLM Memory Extraction
  System Prompt: Instructions for extraction (with date placeholders)
  Response Format: extraction_model JSON schema

↓ LLM Response (Structured JSON)
  {
    "memories": [
      {"text": "My name is Alex", "type": "fact", "tags": ["name"], "importance": 0.9},
      {"text": "I live in Bangkok", "type": "fact", "tags": ["location"], "importance": 0.8}
    ]
  }

↓ Filtering & Normalization
  - Validate against extraction_model
  - Normalize dates with date_normalizer
  - Filter importance ≥ 0.5
  - Convert to MemoryFact objects

Output: list[MemoryFact]
```

### Date Normalization

Temporal memories with natural language dates:

```python
"I'm traveling to Japan in 2 weeks"
→ valid_until = "2024-02-01T00:00:00Z"  # Calculated

"I lived in Paris until last year"
→ valid_until = "2023-12-31T23:59:59Z"
```

---

## Design Patterns

### 1. Protocol-Based Composition

Instead of inheritance, use structural subtyping:

```python
# ❌ Inheritance-based (rigid)
class MyClassifier(BaseClassifier):
    def classify(self, request):
        return super().classify(request)

# ✅ Protocol-based (flexible)
class MyClassifier:
    async def classify(self, request: ClassificationRequest):
        # Automatically implements MemoryClassifier protocol
        return request
```

### 2. Graceful Degradation

Optional dependencies with fallback:

```python
try:
    from sentence_transformers import CrossEncoder
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class NLIPreFilter:
    def __init__(self):
        self.model = None  # Lazy load

    def predict(self, text_a, text_b):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Install: pip install casual-memory[transformers]")

        if self.model is None:
            self.model = CrossEncoder("...")  # Load on first use

        return self.model.predict(...)
```

### 3. Retry Logic

Automatic retry for transient failures:

```python
async def _call_llm_with_retry(self, prompt: str, max_retries: int = 2):
    for attempt in range(max_retries + 1):
        try:
            return await self.llm_provider.chat(prompt)
        except Exception as e:
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                continue
            raise  # Final attempt failed
```

### 4. Metrics Collection

All intelligence components track metrics:

```python
metrics = {
    "nli_prediction_count": 150,
    "nli_cache_hits": 120,
    "nli_cache_misses": 30,
    "nli_cache_hit_rate_percent": 80.0,

    "conflict_verifier_llm_call_count": 50,
    "conflict_verifier_llm_success_count": 48,
    "conflict_verifier_llm_failure_count": 2,
    "conflict_verifier_fallback_count": 2,
    "conflict_verifier_llm_success_rate_percent": 96.0
}
```

---

## Performance Considerations

### 1. NLI Pre-Filter Effectiveness

Filters 70-85% of pairs before expensive LLM calls:

```
100 similar memory pairs
  ↓ NLI Classifier
  ├─ 60 pairs classified (entailment/neutral) ✅
  └─ 40 pairs uncertain → pass to Conflict Classifier
      ↓ Conflict Classifier (LLM)
      ├─ 20 conflicts detected
      └─ 20 pairs → pass to Duplicate Classifier
          ↓ Duplicate Classifier (LLM)
          └─ All 20 classified

Total LLM calls: 40 (instead of 100)
Savings: 60% reduction in LLM costs
```

### 2. Caching Strategy

NLI filter caching reduces redundant computation:

```python
# First call: 200ms (model inference)
label1, scores1 = nli_filter.predict("I live in London", "I live in Paris")

# Second call: <1ms (cache hit)
label2, scores2 = nli_filter.predict("I live in London", "I live in Paris")
```

### 3. Async/Await Design

All I/O operations are async for concurrency:

```python
# Sequential (slow)
for pair in memory_pairs:
    result = await classifier.classify_one(pair)

# Concurrent (fast)
tasks = [classifier.classify_one(pair) for pair in memory_pairs]
results = await asyncio.gather(*tasks)
```

### 4. Lazy Loading

Heavy dependencies loaded only when needed:

- NLI model: 438MB, loaded on first predict()
- Embedding models: Loaded on first encode()
- Database connections: Created on initialize()

### 5. Benchmarks

Typical performance on M1 Mac (CPU):

| Operation | Time | Notes |
|-----------|------|-------|
| NLI prediction | 200ms | Cold (first call) |
| NLI prediction | 50ms | Warm (cached) |
| LLM conflict check | 1.2s | qwen2.5:7b via Ollama |
| LLM duplicate check | 1.0s | qwen2.5:7b via Ollama |
| Full pipeline (5 pairs) | 3.5s | ~60% filtered by NLI |
| Qdrant vector search | 50ms | 10k memories, top 5 results |

---

## Summary

casual-memory achieves:

✅ **Modularity** - Protocol-based components, composable pipeline
✅ **Flexibility** - Swap classifiers, storage backends, LLM providers
✅ **Performance** - NLI pre-filtering, caching, async operations
✅ **Reliability** - Graceful degradation, retry logic, fallback heuristics
✅ **Testability** - Protocol mocks, in-memory backends, comprehensive tests

The architecture prioritizes **developer experience** while maintaining **production-ready reliability**.
