# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Overview

`casual-memory` is a Python library for intelligent semantic memory management with conflict detection, classification pipelines, and storage abstraction. It uses LLMs and NLI models to detect contradictory memories, classify duplicate vs. distinct facts, and automatically resolve conflicts based on confidence scoring.

## Quick Reference

| Topic | Link |
|-------|------|
| System Architecture | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Configuration | [docs/CONFIGURATION.md](docs/CONFIGURATION.md) |
| Data Models | [docs/DATA_MODELS.md](docs/DATA_MODELS.md) |
| Testing | [docs/TESTING.md](docs/TESTING.md) |
| Troubleshooting | [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) |
| Design Decisions | [docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md) |

## Development Commands

```bash
# Environment setup
uv sync --all-extras

# Testing
uv run pytest                                        # All tests
uv run pytest --cov=casual_memory --cov-report=html  # With coverage
uv run pytest tests/classifiers/test_pipeline.py -v  # Specific file

# Code quality
uv run black src/              # Format
uv run ruff check src/         # Lint
uv run mypy src/casual_memory/ # Type check
```

## Architecture

### Core Components

1. **Classification Pipeline** ([src/casual_memory/classifiers/](src/casual_memory/classifiers/))
   - `MemoryClassificationPipeline`: Orchestrates classifiers
   - `NLIClassifier`: Fast pre-filter (~50-200ms)
   - `ConflictClassifier`: LLM contradiction detection
   - `DuplicateClassifier`: Same/distinct fact detection
   - `AutoResolutionClassifier`: Confidence-based resolution

2. **Intelligence Layer** ([src/casual_memory/intelligence/](src/casual_memory/intelligence/))
   - `NLIPreFilter`: Sentence-transformers semantic filtering
   - `LLMConflictVerifier`, `LLMDuplicateDetector`: LLM verifiers

3. **Memory Service** ([src/casual_memory/memory_service.py](src/casual_memory/memory_service.py))
   - `add_memory()`: Find similar, classify, execute actions (scoped by `namespace` and `entity_id` on the `MemoryFact`)
   - `query_memory()`: Semantic search with filtering (pass `namespace` and `entity_id` via `MemoryQueryFilter`)

4. **Context Service** ([src/casual_memory/context_service.py](src/casual_memory/context_service.py))
   - `add()`: Store conversation messages (filters system messages). Accepts `entity_id`, `session_id`, `namespace`.
   - `get()`: Retrieve recent messages with safe boundary trimming. Accepts `entity_id`, `session_id`, `namespace`.
   - `clear()`: Clear session messages. Accepts `entity_id`, `session_id`, `namespace`.

5. **Storage** ([src/casual_memory/storage/](src/casual_memory/storage/))
   - Vector: `InMemoryVectorStore`, `QdrantMemoryStore` -- namespace-aware via payload `namespace`/`entity_id` fields
   - Conflict: `InMemoryConflictStore`, `SQLAlchemyConflictStore` -- indexed by `(namespace, entity_id)`
   - Short-term: `InMemoryShortTermStore`, `RedisShortTermStore` -- keyed by `(namespace, entity_id)`

6. **Extraction** ([src/casual_memory/extractors/](src/casual_memory/extractors/))
   - `LLMMemoryExtracter`: Extract facts from conversations

### Classification Flow

```
New Memory + Similar Memories
  ↓
NLI Classifier → same/neutral (fast)
  ↓
Conflict Classifier → conflict (LLM)
  ↓
Duplicate Classifier → same/superseded/neutral (LLM)
  ↓
Auto-Resolution → supersede by confidence
  ↓
Output: overall_outcome = "add" | "skip" | "conflict"
```

**Similarity Outcomes**: `conflict`, `superseded`, `same`, `neutral`
**Memory Outcomes**: `add`, `skip`, `conflict`

### Namespace Scoping

All memory operations are scoped by **namespace** and **entity_id**:

- **namespace** (default: `"default"`): Isolates data into logical groups (e.g., `"work"`, `"personal"`). All storage backends filter by namespace.
- **entity_id**: Identifies the entity a memory belongs to (e.g., a user ID). Required for conflict tracking, optional elsewhere.

These fields are set on `MemoryFact`, `MemoryConflict`, and `MemoryQueryFilter` models, and passed as parameters to `ContextService` methods.

> **Deprecation note**: The `user_id` parameter/field is deprecated across all models and services. Use `entity_id` instead. Passing `user_id` still works (with a `DeprecationWarning`) during the migration period.

### Data Flow

```
1. MemoryService.add_memory(new_memory)
   (new_memory.namespace + new_memory.entity_id scope the operation)
2. → Embed text, search similar memories (filtered by namespace + entity_id)
3. → Pipeline.classify(new_memory, similar_memories)
4. → ActionExecutor.execute(result)
   - "add": Insert to vector store (with namespace/entity_id in payload), archive superseded
   - "skip": Update existing (mention_count++)
   - "conflict": Create conflict record (scoped to namespace/entity_id)
5. → Return MemoryActionResult
```

## Public API

```python
# Core exports
from casual_memory import (
    MemoryFact,           # Core memory unit (has namespace, entity_id fields)
    MemoryFactExtraction, # Extraction model (LLM response format)
    MemoryBlock,          # MCP context block wrapping a list of MemoryFact
    MemoryConflict,       # Conflict tracking (has namespace, entity_id fields)
    ConflictResolution,   # Resolution decision
    ShortTermMemory,      # Conversation message
    MemoryQueryFilter,    # Query filtering (has namespace, entity_id fields)
    MemoryService,        # Long-term memory service
    ContextService,       # Short-term conversation context (accepts entity_id, namespace)
)

# Classification
from casual_memory.classifiers import (
    MemoryClassificationPipeline,
    SimilarMemory, SimilarityResult, MemoryClassificationResult,
    NLIClassifier, ConflictClassifier, DuplicateClassifier, AutoResolutionClassifier,
)

# Intelligence
from casual_memory.intelligence import NLIPreFilter, LLMConflictVerifier, LLMDuplicateDetector

# Extraction
from casual_memory.extractors import LLMMemoryExtracter

# Storage (use full module paths)
from casual_memory.storage.vector.memory import InMemoryVectorStore
from casual_memory.storage.vector.qdrant import QdrantMemoryStore
from casual_memory.storage.conflicts.memory import InMemoryConflictStore
from casual_memory.storage.conflicts.sqlalchemy import SQLAlchemyConflictStore

# Embeddings
from casual_memory.embeddings import E5Embedding, OpenAIEmbedding
```

## Key Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `strategy` | `"tiered"` | Pipeline strategy: "single", "tiered", "all" |
| `entailment_threshold` | `0.85` | NLI threshold for "same" |
| `supersede_threshold` | `1.3` | Confidence ratio to supersede |
| `keep_threshold` | `0.7` | Confidence ratio to keep old |
| `similarity_threshold` | `0.85` | Min similarity for "similar" |
| `namespace` | `"default"` | Namespace for memory isolation (on models and service methods) |
| `entity_id` | `None` | Entity identifier for multi-entity isolation |

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for all parameters.

## Project Structure

```
src/casual_memory/
├── classifiers/       # Classification pipeline
│   ├── pipeline.py    # MemoryClassificationPipeline
│   ├── models.py      # SimilarMemory, SimilarityResult, etc.
│   └── *_classifier.py
├── intelligence/      # NLI filter, LLM verifiers
├── extractors/        # LLMMemoryExtracter
├── storage/           # Protocol-based storage
│   ├── vector/        # Vector stores (namespace-aware)
│   ├── conflicts/     # Conflict stores (namespace-indexed)
│   └── short_term/    # Short-term stores (namespace-keyed)
├── embeddings/        # E5, OpenAI embeddings
├── execution/         # Action executor
├── utils/             # Shared utilities (validation, etc.)
├── models.py          # Core data models (MemoryFact, MemoryConflict, etc.)
├── memory_service.py  # Long-term memory service
└── context_service.py # Short-term context service
```

## Dependencies

- **Core**: pydantic, casual-llm
- **Optional**: sentence-transformers, qdrant-client, sqlalchemy, redis

```bash
uv sync --all-extras  # Install all
```

## Testing

- pytest + pytest-asyncio (`asyncio_mode = "auto"`)
- Mock LLM providers for classifier tests
- In-memory storage for unit tests
- SQLite `:memory:` for database tests

See [docs/TESTING.md](docs/TESTING.md) for details.

## Common Patterns

### Basic Pipeline Usage

```python
from casual_memory.classifiers import (
    MemoryClassificationPipeline, SimilarMemory,
    NLIClassifier, ConflictClassifier, DuplicateClassifier, AutoResolutionClassifier,
)
from casual_memory import MemoryFact

pipeline = MemoryClassificationPipeline(
    classifiers=[
        NLIClassifier(nli_filter=nli_filter),
        ConflictClassifier(llm_conflict_verifier=conflict_verifier),
        DuplicateClassifier(llm_duplicate_detector=duplicate_detector),
        AutoResolutionClassifier(),
    ],
    strategy="tiered",
)

# namespace and entity_id scope the memory for isolation
new_memory = MemoryFact(
    text="I live in Paris",
    type="fact",
    tags=["location"],
    importance=0.8,
    entity_id="user-123",
    namespace="default",  # optional, defaults to "default"
)
similar = [SimilarMemory(memory_id="...", memory=MemoryFact(...), similarity_score=0.91)]

result = await pipeline.classify(new_memory, similar)
print(result.overall_outcome)  # "add", "skip", or "conflict"
```

### Context Service Usage

```python
from casual_memory import ContextService
from casual_memory.storage.short_term.memory import InMemoryShortTermStore
from casual_llm.messages import UserMessage, AssistantMessage

store = InMemoryShortTermStore(max_messages=100)
context = ContextService(short_term_store=store, short_term_limit=50)

# Add messages with entity_id and namespace
# (system messages are filtered out automatically)
context.add("user-123", "session1", [
    UserMessage(content="What's the weather?"),
    AssistantMessage(content="It's sunny today!"),
], namespace="default")

# Get recent messages (trimmed to safe boundary -- never starts mid-tool-call)
messages = context.get("user-123", "session1", namespace="default")

# Clear a session
context.clear("user-123", "session1", namespace="default")

# Deprecated: user_id keyword still works but emits DeprecationWarning
context.add(user_id="user-123", session_id="session1", messages=[...])
```

### Memory Query with Namespace Filtering

```python
from casual_memory import MemoryQueryFilter

# Query memories scoped to a namespace and entity
filter = MemoryQueryFilter(
    entity_id="user-123",
    namespace="work",
    type=["fact", "preference"],
    min_importance=0.5,
)
results = await memory_service.query_memory("What city do I live in?", filter=filter)
```

### Memory Extraction

```python
from casual_memory.extractors import LLMMemoryExtracter
from casual_memory.extractors.prompts import USER_MEMORY_PROMPT

# Default: uses MemoryExtractionResponse (strict Literal types)
extractor = LLMMemoryExtracter(llm_provider=llm_provider, prompt=USER_MEMORY_PROMPT)
memories = await extractor.extract(messages)

# Custom: pass your own Pydantic model for custom memory types
extractor = LLMMemoryExtracter(
    llm_provider=provider,
    prompt=CUSTOM_PROMPT,
    extraction_model=CustomExtractionResponse,  # Must have 'memories' attribute
)
```

### SQLAlchemy Conflict Store

```python
from sqlalchemy import create_engine
from casual_memory.storage.conflicts.sqlalchemy import SQLAlchemyConflictStore

engine = create_engine("postgresql://user:pass@localhost/db")
store = SQLAlchemyConflictStore(engine)
store.create_tables()

# Query conflicts scoped to a namespace
pending = store.get_pending_conflicts(entity_id="user-123", namespace="default")
```

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for common issues.
