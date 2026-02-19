# Configuration Parameters

All tunable parameters for casual-memory in one place.

## Pipeline Strategy

| Parameter | Default | Location | Description |
|-----------|---------|----------|-------------|
| `strategy` | `"tiered"` | `MemoryClassificationPipeline` | Checking strategy: "single", "tiered", or "all" |
| `secondary_conflict_threshold` | `0.90` | `MemoryClassificationPipeline` | Min similarity for secondary checks in tiered mode |
| `max_secondary_checks` | `3` | `MemoryClassificationPipeline` | Max secondary memories to check in tiered mode |

### Strategy Options

- **`single`** (fastest): Check only highest-similarity memory
- **`tiered`** (default, balanced):
  - Primary check: Highest-similarity memory (full pipeline)
  - Secondary checks: Up to 3 additional memories ≥0.90 similarity (conflict-only)
- **`all`** (thorough): Check all similar memories with full pipeline

## NLI Thresholds

| Parameter | Default | Location | Description |
|-----------|---------|----------|-------------|
| `entailment_threshold` | `0.85` | `NLIClassifier` | Min entailment score for "same" outcome |
| `neutral_threshold` | `0.5` | `NLIClassifier` | Min neutral score for "neutral" outcome |

## Auto-Resolution

| Parameter | Default | Location | Description |
|-----------|---------|----------|-------------|
| `supersede_threshold` | `1.3` | `AutoResolutionClassifier` | Confidence ratio to auto-resolve to new memory |
| `keep_threshold` | `0.7` | `AutoResolutionClassifier` | Confidence ratio to auto-resolve to old memory |

### Tuning Auto-Resolution

```python
# More aggressive (fewer conflicts escalated to user)
auto_resolver = AutoResolutionClassifier(
    supersede_threshold=1.2,  # Lower = easier to supersede
    keep_threshold=0.8,       # Higher = easier to keep old
)

# More conservative (more conflicts escalated to user)
auto_resolver = AutoResolutionClassifier(
    supersede_threshold=1.5,  # Higher = harder to supersede
    keep_threshold=0.5,       # Lower = harder to keep old
)
```

## Memory Service

| Parameter | Default | Location | Description |
|-----------|---------|----------|-------------|
| `similarity_threshold` | `0.85` | `add_memory()` | Min cosine similarity to consider "similar" |
| `max_similar` | `5` | `add_memory()` | Max similar memories to retrieve and check |
| `top_k` | `5` | `query_memory()` | Number of results to return |
| `min_score` | `0.75` | `query_memory()` | Min similarity score for query results |

## Context Service

| Parameter | Default | Location | Description |
|-----------|---------|----------|-------------|
| `short_term_limit` | `50` | `ContextService` | Default max messages returned by `get()` |

## Namespace & Entity Isolation

| Parameter | Default | Location | Description |
|-----------|---------|----------|-------------|
| `namespace` | `"default"` | Models, service methods | Logical namespace for memory isolation (e.g., `"work"`, `"personal"`) |
| `entity_id` | `None` | Models, service methods | Entity identifier for multi-entity isolation (e.g., user ID) |

All memory operations (add, query, conflict tracking, short-term context) are scoped by `namespace` and `entity_id`. These fields appear on `MemoryFact`, `MemoryConflict`, and `MemoryQueryFilter` models, and are passed as parameters to `ContextService` methods.

> **Deprecation note**: The `user_id` parameter/field is deprecated across all models and services. Use `entity_id` instead. Passing `user_id` still works (with a `DeprecationWarning`).

## Duplicate Detection

| Parameter | Default | Location | Description |
|-----------|---------|----------|-------------|
| `refinement_length_ratio` | `1.2` | `DuplicateClassifier` | Length ratio to distinguish duplicate vs refinement |

## Confidence Scoring

| Parameter | Default | Location | Description |
|-----------|---------|----------|-------------|
| `recency_window_days` | `30` | Intelligence layer | Days before recency decay kicks in |
| `spread_window_days` | `90` | Intelligence layer | Days to measure mention spread |

### Confidence Calculation

Memory confidence is calculated based on:

1. **Mention frequency**: 1 mention = 0.5, 5+ mentions = 0.95
2. **Recency factor**: Decays after 30 days
3. **Spread factor**: Boost if mentioned over time (not just repeated once)

Formula:
```python
confidence = min(
    MEMORY_MAX_CONFIDENCE,  # 0.95 cap
    base_confidence * recency_factor + spread_boost
)
```

## Dependencies

### Core Dependencies
- **casual-llm**: LLM client/model abstraction (supports OpenAI, Anthropic, Ollama, etc.)
- **pydantic**: Data validation and serialization

### Optional Dependencies
- **sentence-transformers**: NLI classifier and E5 embeddings (90%+ accuracy, 50-200ms)
- **qdrant-client**: Production vector database
- **sqlalchemy** + **psycopg2-binary**: Database conflict stores
- **redis**: Production short-term memory store

Install extras:
```bash
uv sync --extra transformers    # For NLI
uv sync --extra qdrant          # For Qdrant
uv sync --extra postgres        # For PostgreSQL
uv sync --extra redis           # For Redis
uv sync --all-extras            # Everything
```

### CPU-only Installation

By default, PyTorch (pulled in by sentence-transformers) includes CUDA. For CPU-only:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install casual-memory[transformers]
```
