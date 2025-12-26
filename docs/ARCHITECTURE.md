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

**Design Pattern:**
```python
@runtime_checkable
class MemoryClassifier(Protocol):
    """Protocol for memory classifiers."""

    async def classify(
        self, request: ClassificationRequest
    ) -> ClassificationRequest:
        """Classify memory pairs and update results."""
        ...
```

### 2. Classification Outcomes

Three possible outcomes for each memory pair:

- **MERGE** - Memories should be combined
  - Duplicates (exact copies)
  - Refinements (general → specific)
  - Auto-resolved conflicts (one clearly supersedes the other)

- **CONFLICT** - Contradictory memories needing manual resolution
  - Location conflicts ("I live in London" vs "I live in Paris")
  - Job conflicts ("I work as a teacher" vs "I work as a doctor")
  - Preference conflicts ("I like coffee" vs "I don't like coffee")

- **ADD** - Distinct memories that should both be stored
  - Different facts ("I live in Bangkok" vs "I work in Bangkok")
  - Compatible preferences ("I like coffee" vs "I like tea")
  - Unrelated information

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

---

## Classification Pipeline

### Sequential Execution Flow

```python
Input: ClassificationRequest
  pairs: List[MemoryPair]      # Similar memories to classify
  results: List[ClassificationResult]  # Empty initially
  user_id: str

↓ NLI Classifier (Fast Filter, ~50-200ms)
  For each unclassified pair:
    - Calculate entailment/contradiction/neutral scores
    - High entailment (≥0.85) → MERGE
    - High neutral (≥0.5) → ADD
    - Uncertain → Skip to next classifier

↓ Conflict Classifier (~500-2000ms)
  For each unclassified pair:
    - Call LLM conflict verifier
    - If contradiction detected → CONFLICT (with metadata)
    - If fallback triggered → Use heuristic patterns
    - No conflict → Skip to next classifier

↓ Duplicate Classifier (~500-2000ms)
  For each unclassified pair:
    - Call LLM duplicate detector
    - If same fact/refinement → MERGE
    - If distinct → ADD

↓ Auto-Resolution Classifier (instant)
  For each CONFLICT result:
    - Calculate confidence ratio (new / existing)
    - Ratio ≥ 1.3 → MERGE (keep_new)
    - Ratio ≤ 0.7 → MERGE (keep_old)
    - Else → Keep as CONFLICT

↓ Default Handler (fallback)
  For each still unclassified pair:
    - Conservative default → ADD

Output: ClassificationRequest with populated results
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
fast_pipeline = ClassificationPipeline(classifiers=[
    NLIClassifier(nli_filter=nli_filter)
])

# Accuracy-focused (skip NLI, use only LLM)
accuracy_pipeline = ClassificationPipeline(classifiers=[
    ConflictClassifier(llm_conflict_verifier=verifier),
    DuplicateClassifier(llm_duplicate_detector=detector)
])
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
class VectorStore(Protocol):
    """Vector storage for semantic search."""

    async def add(self, memory: MemoryFact, user_id: str) -> str:
        """Add memory and return ID."""

    async def search(
        self, query_text: str, user_id: str, limit: int = 5
    ) -> list[MemoryFact]:
        """Semantic search for similar memories."""

    async def update(self, memory_id: str, memory: MemoryFact, user_id: str):
        """Update existing memory."""

    async def archive(
        self, memory_id: str, user_id: str, superseded_by: str | None
    ):
        """Soft-delete memory."""

@runtime_checkable
class ConflictStore(Protocol):
    """Storage for memory conflicts."""

    async def add(self, conflict: MemoryConflict, user_id: str) -> str:
        """Store conflict and return ID."""

    async def get(self, conflict_id: str, user_id: str) -> MemoryConflict | None:
        """Retrieve conflict by ID."""

    async def list_pending(self, user_id: str) -> list[MemoryConflict]:
        """List unresolved conflicts."""

    async def resolve(self, resolution: ConflictResolution, user_id: str):
        """Mark conflict as resolved."""

@runtime_checkable
class ShortTermStore(Protocol):
    """Storage for conversation history."""

    async def add(self, messages: list[ShortTermMemory], user_id: str):
        """Add messages to history."""

    async def get(self, user_id: str, limit: int = 20) -> list[ShortTermMemory]:
        """Get recent messages."""

    async def clear(self, user_id: str):
        """Clear all messages for user."""
```

### User Isolation

All storage operations scoped by `user_id`:

```python
# Different users have separate memory spaces
await vector_store.add(memory, user_id="alice")
await vector_store.add(memory, user_id="bob")

# Searches only return user's own memories
results_alice = await vector_store.search("hobby", user_id="alice")
results_bob = await vector_store.search("hobby", user_id="bob")
# results_alice != results_bob
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

### Two-Phase Extraction

1. **User-sourced memories** (importance × 1.0)
   - Extracted from user messages
   - First-person perspective: "My name is Alex"
   - Higher confidence (directly stated)

2. **Assistant-sourced memories** (importance × 0.6)
   - Extracted from assistant observations
   - Inferred from context
   - Lower confidence (observed, not stated)

### Extraction Process

```python
Input: Conversation messages
  [UserMessage("My name is Alex and I live in Bangkok"),
   AssistantMessage("Nice to meet you!")]

↓ LLM Memory Extraction
  System Prompt: Instructions for extraction
    - First-person perspective rules
    - Atomic fact splitting
    - Importance scoring (0.0-1.0)
    - Memory type classification

  User Prompt: Formatted conversation

↓ LLM Response (Structured JSON)
  [
    {
      "text": "My name is Alex",
      "type": "fact",
      "tags": ["name", "identity"],
      "importance": 0.9
    },
    {
      "text": "I live in Bangkok",
      "type": "fact",
      "tags": ["location", "residence"],
      "importance": 0.8
    }
  ]

↓ Filtering & Normalization
  - Filter importance ≥ 0.5
  - Normalize dates with date_normalizer
  - Apply source weighting (user=1.0x, assistant=0.6x)
  - Create MemoryFact objects

Output: List[MemoryFact]
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
