# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`casual-memory` is a Python library for intelligent semantic memory management with conflict detection, classification pipelines, and storage abstraction. It uses LLMs and NLI models to detect contradictory memories, classify duplicate vs. distinct facts, and automatically resolve conflicts based on confidence scoring.

## Quick Reference

- **Getting Started**: See [Complete Integration Example](#complete-integration-example)
- **Classification Pipeline**: See [Classification Pipeline Flow](#classification-pipeline-flow) and [Classification Strategies](#classification-strategies)
- **Storage Setup**: See [Storage Protocol System](#storage-protocol-system) and [Using SQLAlchemy for Conflicts](#using-sqlalchemy-for-conflicts)
- **Tuning Parameters**: See [Configuration Parameters](#configuration-parameters)
- **Troubleshooting**: See [Common Usage Scenarios and Troubleshooting](#common-usage-scenarios-and-troubleshooting)
- **Public API**: See [Public API / Entry Points](#public-api--entry-points)
- **Data Models**: See [Key Data Models](#key-data-models)
- **Design Decisions**: See [Key Design Patterns and Architectural Decisions](#key-design-patterns-and-architectural-decisions)

## Development Commands

### Environment Setup
```bash
# Install with all dependencies
uv sync --all-extras

# Install for development
uv sync --all-extras
```

### Testing
```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=casual_memory --cov-report=html

# Run specific test file
uv run pytest tests/classifiers/test_pipeline.py -v

# Run specific test function
uv run pytest tests/classifiers/test_pipeline.py::test_pipeline_sequential_execution -v

# Run tests for a specific module
uv run pytest tests/storage/ -v
uv run pytest tests/classifiers/ -v
```

### Code Quality
```bash
# Format code with black
uv run black src/

# Lint with ruff
uv run ruff check src/

# Type check with mypy
uv run mypy src/casual_memory/
```

## Architecture

### Core Components

The library is organized around six main subsystems:

1. **Classification Pipeline** ([src/casual_memory/classifiers/](src/casual_memory/classifiers/))
   - **Pipeline orchestrator**: Sequentially runs memory pairs through multiple classifiers
   - **NLI Classifier**: Fast pre-filter using sentence-transformers (~50-200ms)
   - **Conflict Classifier**: LLM-based contradiction detection with categorization
   - **Duplicate Classifier**: LLM-based same/distinct fact detection
   - **Auto-Resolution Classifier**: Post-processor that resolves conflicts using confidence ratios

2. **Intelligence Layer** ([src/casual_memory/intelligence/](src/casual_memory/intelligence/))
   - **NLI Pre-filter**: Uses sentence-transformers for fast semantic filtering
   - **LLM Verifiers**: Conflict and duplicate detection using structured LLM prompts
   - **Confidence Scoring**: Calculates memory confidence based on mention frequency, recency, and spread

3. **Memory Service** ([src/casual_memory/memory_service.py](src/casual_memory/memory_service.py))
   - **High-level orchestrator**: Combines all components into a simple API
   - **`add_memory()`**: Finds similar memories, classifies, executes actions
   - **`query_memory()`**: Semantic search with filtering
   - Main entry point for applications

4. **Memory Extraction** ([src/casual_memory/extractors/](src/casual_memory/extractors/))
   - **LLMMemoryExtracter**: Base implementation using LLM with structured prompts
   - Extracts facts, preferences, events, and goals from conversation messages
   - Returns list of MemoryFact objects with type, tags, and importance

5. **Embedding Abstraction** ([src/casual_memory/embeddings/](src/casual_memory/embeddings/))
   - **TextEmbedding protocol**: Abstract interface for text embedding providers
   - **E5Embedding**: E5 model family with "document:"/"query:" prefix handling
   - **OpenAIEmbedding**: OpenAI API embeddings
   - Supports batch operations for efficiency

6. **Storage Abstraction** ([src/casual_memory/storage/](src/casual_memory/storage/))
   - **Protocol-based**: All storage is defined via Python protocols
   - **Vector stores**: In-memory (testing), Qdrant (production)
   - **Conflict stores**: In-memory, SQLite, PostgreSQL, SQLAlchemy (unified)
   - **Short-term stores**: In-memory (deque), Redis (production)

### Classification Pipeline Flow

Memory pairs flow sequentially through classifiers:

1. **NLI Classifier** filters obvious cases:
   - High entailment (≥0.85) → same
   - High neutral (≥0.5) → neutral (distinct)
   - Uncertain → pass to next classifier

2. **Conflict Classifier** detects contradictions using LLM:
   - Categorizes conflicts (location, job, preference, temporal, factual)
   - Provides clarification hints for user resolution
   - Only confident contradictions → conflict

3. **Duplicate Classifier** distinguishes same vs. distinct facts:
   - Same fact → same
   - Refinement (>1.2x length) → superseded
   - Distinct facts → neutral

4. **Auto-Resolution Classifier** resolves conflicts by confidence:
   - High new confidence (ratio ≥1.3) → superseded (keep_new)
   - High old confidence (ratio ≤0.7) → same (keep_old)
   - Similar confidence → Keep as conflict

Each classifier only classifies what it's confident about. Unclassified pairs pass to the next classifier.

**Outcomes:**
- `conflict`: Contradictory memories requiring user resolution
- `superseded`: New memory replaces old (old gets archived)
- `same`: Duplicate memory (update mention count on existing)
- `neutral`: Distinct facts (add both)

**Overall Actions:**
- `add`: New memory added to vector store
- `skip`: Existing memory updated (mention_count, last_seen)
- `conflict`: Conflict record created, new memory not added

### Classification Strategies

The pipeline supports three checking strategies:

- **`single`** (fastest): Check only highest-similarity memory
- **`tiered`** (default, balanced):
  - Primary check: Highest-similarity memory (full pipeline)
  - Secondary checks: Up to 3 additional memories ≥0.90 similarity (conflict-only)
- **`all`** (thorough): Check all similar memories with full pipeline

Configure via: `MemoryClassificationPipeline(classifiers=[...], strategy="tiered")`

### Storage Protocol System

All storage implementations are protocol-based for maximum flexibility:

- **VectorMemoryStore**: Vector similarity search with filtering and archiving
- **ConflictStore**: Conflict tracking, resolution, and querying
- **ShortTermStore**: FIFO message queue for conversation history

Implementations can be swapped without code changes (e.g., SQLite dev → PostgreSQL prod).

### Confidence Scoring

Memory confidence is calculated based on:
- **Mention frequency**: 1 mention = 0.5, 5+ mentions = 0.95
- **Recency factor**: Decays after 30 days
- **Spread factor**: Boost if mentioned over time (not just repeated once)

Formula combines these factors to produce a score between 0.0 and 1.0.

### Complete Data Flow

```
1. User input → New MemoryFact created

2. MemoryService.add_memory():
   ├─ Embed new memory text → vector
   ├─ VectorStore.find_similar_memories(vector, threshold=0.85, limit=5)
   └─ Returns: List[SimilarMemory(memory_id, memory, similarity_score)]

3. Pipeline.classify(new_memory, similar_memories, strategy="tiered"):
   ├─ For each similar memory:
   │  ├─ NLI Classifier: Fast semantic check (50-200ms)
   │  │  └─ High confidence? → Return outcome, else pass to next
   │  ├─ Conflict Classifier: LLM contradiction detection (500-2000ms)
   │  │  └─ Conflict detected? → Return "conflict", else pass to next
   │  ├─ Duplicate Classifier: LLM same/distinct check (500-2000ms)
   │  │  └─ Always returns outcome (same/superseded/neutral)
   │  └─ Auto-Resolution: Confidence-based override (<1ms)
   │     └─ High ratio? → Override to superseded/same, else keep as-is
   └─ Returns: MemoryClassificationResult(overall_outcome, similarity_results)

4. ActionExecutor.execute(classification_result, embedding):
   ├─ If overall_outcome = "add":
   │  ├─ VectorStore.add(embedding, payload)
   │  └─ Archive any superseded memories
   ├─ If overall_outcome = "skip":
   │  └─ VectorStore.update_memory(increment mention_count, update last_seen)
   └─ If overall_outcome = "conflict":
      └─ ConflictStore.add_conflict(new_memory, existing_memory, category, hints)

5. Return MemoryActionResult:
   ├─ action: "added" | "updated" | "conflict"
   ├─ memory_id: ID of added/updated memory (or None)
   ├─ conflict_ids: List of conflict IDs created
   ├─ superseded_ids: List of archived memory IDs
   └─ metadata: Additional context
```

### Action Execution

Located in [src/casual_memory/execution/](src/casual_memory/execution/):

**MemoryActionExecutor** converts classification results to storage operations:

- **Add action**: Insert new memory to vector store, archive superseded memories
- **Skip action**: Update existing memory (mention_count++, last_seen=now)
- **Conflict action**: Create conflict record(s), don't add new memory

All operations are atomic and update related metadata (confidence scores, timestamps, etc.).

## Key Data Models

Located in [src/casual_memory/models.py](src/casual_memory/models.py):

### Core Models
- **MemoryFact**: Core memory unit (text, type, tags, importance, confidence, mention tracking)
- **MemoryConflict**: Tracks contradictory memories with category and clarification hints
- **ConflictResolution**: Resolution decision (keep_a, keep_b, merge, both_valid)
- **ShortTermMemory**: Conversation message (content, role, timestamp)
- **MemoryQueryFilter**: Query filtering by type, importance, user_id

Memory types: `fact`, `preference`, `event`, `goal`, `weather`

### Classification Models

Located in [src/casual_memory/classifiers/models.py](src/casual_memory/classifiers/models.py):

- **SimilarMemory**: Wrapper for similar memory (memory_id, memory, similarity_score)
- **SimilarityResult**: Individual classification outcome
  - `similar_memory`: The SimilarMemory being compared
  - `outcome`: "conflict" | "superseded" | "same" | "neutral"
  - `confidence`: 0.0-1.0 score
  - `classifier_name`: Which classifier made the decision
  - `metadata`: Additional context (NLI scores, conflict category, etc.)
- **MemoryClassificationResult**: Overall result for new memory
  - `new_memory`: The MemoryFact being added
  - `overall_outcome`: "add" | "skip" | "conflict"
  - `similarity_results`: List of SimilarityResult for each similar memory

### Action Models

Located in [src/casual_memory/execution/models.py](src/casual_memory/execution/models.py):

- **MemoryActionResult**: Result of executing a classification
  - `action`: "added" | "updated" | "conflict"
  - `memory_id`: ID of memory (added/updated) or None for conflicts
  - `conflict_ids`: List of conflict IDs created
  - `superseded_ids`: List of archived memory IDs
  - `metadata`: Additional context

## Important Implementation Details

### Classifier Design Principles

- Classifiers are **stateless** and operate on `ClassificationRequest` objects
- Each classifier examines `request.pairs` and appends results to `request.results`
- Classified pairs are removed from `request.pairs` for the next classifier
- Classifiers should only classify what they're **confident** about
- The pipeline provides graceful degradation (e.g., LLM unavailable → skip to next classifier)

### Storage Implementation Guidelines

When implementing storage backends:
- Follow the protocol exactly (method signatures, return types)
- Vector stores should support filtering by `user_id`, `exclude_archived`, etc.
- Conflict stores must handle status transitions (pending → resolved/escalated)
- Short-term stores must maintain FIFO order and enforce `max_messages` limit
- Use transactions where appropriate (SQLAlchemy stores use context managers)

### Testing Conventions

- Use pytest fixtures for setup/teardown
- Async tests use `pytest-asyncio` with `asyncio_mode = "auto"`
- Mock LLM providers for classifier tests to avoid external dependencies
- Use in-memory storage backends for unit tests
- Integration tests can use SQLite `:memory:` databases

## Dependencies

### Core Dependencies
- **casual-llm**: LLM provider abstraction (supports OpenAI, Anthropic, Ollama, etc.)
- **pydantic**: Data validation and serialization

### Optional Dependencies
- **sentence-transformers** + **torch**: NLI classifier (90%+ accuracy, 50-200ms)
- **qdrant-client**: Production vector database
- **sqlalchemy** + **psycopg2-binary**: Database conflict stores
- **redis**: Production short-term memory store
- **dateparser**: Temporal memory date normalization

Install extras: `uv sync --extra transformers` or `uv sync --all-extras`

## Common Patterns

### Building a Classification Pipeline

```python
from casual_memory.classifiers import (
    ClassificationPipeline,
    NLIClassifier,
    ConflictClassifier,
    DuplicateClassifier,
    AutoResolutionClassifier,
)

pipeline = ClassificationPipeline(classifiers=[
    NLIClassifier(nli_filter=nli_filter),
    ConflictClassifier(llm_conflict_verifier=conflict_verifier),
    DuplicateClassifier(llm_duplicate_detector=duplicate_detector),
    AutoResolutionClassifier(supersede_threshold=1.3, keep_threshold=0.7),
])

result = await pipeline.classify(request)
```

### Implementing Custom Storage

Implement the protocol methods:

```python
from casual_memory.storage import VectorMemoryStore

class MyVectorStore:
    def add(self, vector: List[float], payload: dict) -> str: ...
    def search(self, query_embedding: List[float], ...) -> List[...]: ...
    # ... implement all protocol methods
```

### Using SQLAlchemy for Conflicts

```python
from sqlalchemy import create_engine
from casual_memory.storage.conflicts.sqlalchemy import SQLAlchemyConflictStore

# Works with any SQLAlchemy-supported database
engine = create_engine("postgresql://user:pass@localhost/db")
# OR: create_engine("sqlite:///conflicts.db")
# OR: create_engine("mysql://user:pass@localhost/db")

store = SQLAlchemyConflictStore(engine)
store.create_tables()  # Or use Alembic migrations for production
```

### Complete Integration Example

```python
from casual_memory import MemoryService, MemoryFact, MemoryQueryFilter
from casual_memory.classifiers import (
    MemoryClassificationPipeline,
    NLIClassifier,
    ConflictClassifier,
    DuplicateClassifier,
    AutoResolutionClassifier,
)
from casual_memory.intelligence import (
    NLIPreFilter,
    LLMConflictVerifier,
    LLMDuplicateDetector,
)
from casual_memory.storage.vector import InMemoryVectorStore
from casual_memory.storage.conflicts import InMemoryConflictStore
from casual_memory.embeddings import E5Embedding
from casual_llm import LLMProvider

# 1. Initialize components
embedding = E5Embedding()  # or OpenAIEmbedding()
vector_store = InMemoryVectorStore()
conflict_store = InMemoryConflictStore()

# 2. Set up intelligence components
nli_filter = NLIPreFilter()
llm_provider = LLMProvider.from_env()  # from casual-llm
conflict_verifier = LLMConflictVerifier(llm_provider, model_name="gpt-4")
duplicate_detector = LLMDuplicateDetector(llm_provider, model_name="gpt-4")

# 3. Build classification pipeline
pipeline = MemoryClassificationPipeline(
    classifiers=[
        NLIClassifier(nli_filter=nli_filter),
        ConflictClassifier(llm_conflict_verifier=conflict_verifier),
        DuplicateClassifier(llm_duplicate_detector=duplicate_detector),
        AutoResolutionClassifier(supersede_threshold=1.3, keep_threshold=0.7),
    ],
    strategy="tiered",  # or "single" or "all"
)

# 4. Create memory service
service = MemoryService(
    vector_store=vector_store,
    conflict_store=conflict_store,
    pipeline=pipeline,
    embedding=embedding,
)

# 5. Add memories
memory1 = MemoryFact(
    text="I live in London",
    type="fact",
    tags=["location"],
    importance=0.8,
    user_id="user_123",
)
result1 = await service.add_memory(memory1)
print(f"Action: {result1.action}, ID: {result1.memory_id}")  # Action: added, ID: abc123

# Potential conflict
memory2 = MemoryFact(
    text="I live in Paris",
    type="fact",
    tags=["location"],
    importance=0.9,
    user_id="user_123",
)
result2 = await service.add_memory(memory2)
if result2.action == "conflict":
    print(f"Conflicts detected: {result2.conflict_ids}")

# 6. Query memories
filter = MemoryQueryFilter(type=["fact"], min_importance=0.7)
memories = await service.query_memory(
    query="where does the user live?",
    filter=filter,
    top_k=5,
    min_score=0.75,
)
for memory in memories:
    print(f"Memory: {memory.text} (confidence: {memory.confidence})")
```

## Public API / Entry Points

The main package exports the following in [src/casual_memory/__init__.py](src/casual_memory/__init__.py):

### Core Models
```python
from casual_memory import (
    MemoryFact,          # Core memory unit
    MemoryBlock,         # MCP-format wrapper
    MemoryConflict,      # Conflict tracking
    ConflictResolution,  # Resolution decision
    ShortTermMemory,     # Conversation message
    MemoryQueryFilter,   # Query filtering
    MemoryService,       # High-level service (main entry point)
)
```

### Submodules

Users can import from submodules for advanced usage:

```python
# Classification
from casual_memory.classifiers import (
    MemoryClassificationPipeline,
    NLIClassifier,
    ConflictClassifier,
    DuplicateClassifier,
    AutoResolutionClassifier,
)

# Intelligence
from casual_memory.intelligence import (
    NLIPreFilter,
    LLMConflictVerifier,
    LLMDuplicateDetector,
)

# Storage (implementations)
from casual_memory.storage.vector import InMemoryVectorStore, QdrantMemoryStore
from casual_memory.storage.conflicts import InMemoryConflictStore, SQLAlchemyConflictStore
from casual_memory.storage.short_term import InMemoryShortTermStore, RedisShortTermStore

# Embeddings
from casual_memory.embeddings import E5Embedding, OpenAIEmbedding

# Extraction
from casual_memory.extractors import LLMMemoryExtracter
```

## Configuration Parameters

All tunable parameters in one place:

| Parameter | Default | Location | Description |
|-----------|---------|----------|-------------|
| **Pipeline Strategy** |
| `strategy` | `"tiered"` | `MemoryClassificationPipeline` | Checking strategy: "single", "tiered", or "all" |
| `secondary_conflict_threshold` | `0.90` | `MemoryClassificationPipeline` | Min similarity for secondary checks in tiered mode |
| `max_secondary_checks` | `3` | `MemoryClassificationPipeline` | Max secondary memories to check in tiered mode |
| **NLI Thresholds** |
| `entailment_threshold` | `0.85` | `NLIClassifier` | Min entailment score for "same" outcome |
| `neutral_threshold` | `0.5` | `NLIClassifier` | Min neutral score for "neutral" outcome |
| **Auto-Resolution** |
| `supersede_threshold` | `1.3` | `AutoResolutionClassifier` | Confidence ratio to auto-resolve to new memory |
| `keep_threshold` | `0.7` | `AutoResolutionClassifier` | Confidence ratio to auto-resolve to old memory |
| **Memory Service** |
| `similarity_threshold` | `0.85` | `add_memory()` | Min cosine similarity to consider "similar" |
| `max_similar` | `5` | `add_memory()` | Max similar memories to retrieve and check |
| `top_k` | `5` | `query_memory()` | Number of results to return |
| `min_score` | `0.75` | `query_memory()` | Min similarity score for query results |
| **Duplicate Detection** |
| `refinement_length_ratio` | `1.2` | `DuplicateClassifier` | Length ratio to distinguish duplicate vs refinement |
| **Confidence Scoring** |
| `recency_window_days` | `30` | Intelligence layer | Days before recency decay kicks in |
| `spread_window_days` | `90` | Intelligence layer | Days to measure mention spread |

## Testing Structure

Tests are organized by module in [tests/](tests/):

```
tests/
├── classifiers/              # Classification pipeline tests
│   ├── test_pipeline.py      # Pipeline orchestration
│   ├── test_nli_classifier.py
│   ├── test_conflict_classifier.py
│   ├── test_duplicate_classifier.py
│   └── test_auto_resolution_classifier.py
├── intelligence/             # Intelligence component tests
│   ├── test_nli_filter.py
│   ├── test_llm_verifiers.py
│   └── test_confidence.py
├── extractors/               # Memory extraction tests
│   └── test_llm_extracter.py
├── storage/                  # Storage implementation tests
│   ├── test_vector_stores.py
│   ├── test_conflict_stores.py
│   └── test_short_term_stores.py
├── embeddings/               # Embedding adapter tests
│   ├── test_e5_embedding.py
│   └── test_openai_embedding.py
├── execution/                # Action execution tests
│   └── test_executor.py
├── integration/              # End-to-end tests
│   └── test_integration.py
├── utils/                    # Utility tests
│   └── test_utils.py
└── test_memory_service.py    # High-level service tests
```

**Testing Conventions:**
- pytest + pytest-asyncio (`asyncio_mode = "auto"`)
- Mock LLM providers to avoid external dependencies
- In-memory storage backends for unit tests
- SQLite `:memory:` databases for database integration tests
- Fixtures for common setup/teardown

## Project Structure Notes

- **[src/casual_memory/classifiers/models.py](src/casual_memory/classifiers/models.py)**: Classification data structures (MemoryPair, ClassificationResult, etc.)
- **[src/casual_memory/storage/protocols.py](src/casual_memory/storage/protocols.py)**: Protocol definitions (interfaces)
- **[examples/](examples/)**: Integration examples (e.g., with memory-store-service)
- **[tests/](tests/)**: Comprehensive test suite (108+ tests, high coverage)

## Performance Characteristics

- **NLI Classifier**: ~50-200ms (GPU: ~50ms, CPU: ~200ms)
- **LLM Classifiers**: ~500-2000ms depending on model and provider
- **Pipeline**: Sequential execution, total time ≈ sum of active classifiers
- **Vector Search**: In-memory cosine similarity is fast; Qdrant scales to millions of vectors

## Common Usage Scenarios and Troubleshooting

### Handling Conflicts

When a conflict is detected, you can retrieve and resolve it:

```python
# Get pending conflicts for a user
conflicts = await conflict_store.get_pending_conflicts(user_id="user_123")

for conflict in conflicts:
    print(f"Conflict ID: {conflict.id}")
    print(f"Category: {conflict.category}")  # location, job, preference, temporal, factual
    print(f"Memory A: {conflict.memory_a.text}")
    print(f"Memory B: {conflict.memory_b.text}")
    print(f"Hints: {conflict.clarification_hints}")

    # Resolve the conflict
    resolution = ConflictResolution(
        conflict_id=conflict.id,
        decision="keep_b",  # or "keep_a", "merge", "both_valid"
        resolved_memory=conflict.memory_b,  # the chosen memory
    )
    await conflict_store.resolve_conflict(resolution)
```

### Extracting Memories from Conversations

```python
from casual_memory.extractors import LLMMemoryExtracter
from casual_llm import LLMProvider

llm_provider = LLMProvider.from_env()
extracter = LLMMemoryExtracter(llm_provider, model_name="gpt-4")

# Extract from user message
user_message = "I love pizza and I work at Google in San Francisco"
memories = await extracter.extract(user_message, user_id="user_123")

for memory in memories:
    print(f"Type: {memory.type}, Text: {memory.text}, Tags: {memory.tags}")
    # Type: preference, Text: loves pizza, Tags: [food]
    # Type: fact, Text: works at Google, Tags: [job, company]
    # Type: fact, Text: works in San Francisco, Tags: [location, work]
```

### Tuning Classification Performance

```python
# For speed: Use single strategy (check only highest similarity)
pipeline = MemoryClassificationPipeline(
    classifiers=[...],
    strategy="single",  # Fastest
)

# For thoroughness: Use all strategy (check all similar memories)
pipeline = MemoryClassificationPipeline(
    classifiers=[...],
    strategy="all",  # Most thorough, slowest
)

# For balance: Use tiered strategy with custom thresholds
pipeline = MemoryClassificationPipeline(
    classifiers=[...],
    strategy="tiered",  # Default
    secondary_conflict_threshold=0.90,  # Only check very similar memories
    max_secondary_checks=3,  # Limit secondary checks
)
```

### Adjusting Auto-Resolution Sensitivity

```python
# More aggressive auto-resolution (fewer conflicts escalated to user)
auto_resolver = AutoResolutionClassifier(
    supersede_threshold=1.2,  # Lower threshold = easier to supersede
    keep_threshold=0.8,       # Higher threshold = easier to keep old
)

# More conservative (more conflicts escalated to user)
auto_resolver = AutoResolutionClassifier(
    supersede_threshold=1.5,  # Higher threshold = harder to supersede
    keep_threshold=0.5,       # Lower threshold = harder to keep old
)
```

### Troubleshooting

**Problem: NLI classifier not working**
- Check if `sentence-transformers` and `torch` are installed: `uv sync --extra transformers`
- Verify model downloads: First run downloads `cross-encoder/nli-deberta-v3-base` (~400MB)
- Check GPU availability: NLI is faster on GPU but works on CPU

**Problem: High memory usage**
- Use lazy loading (already default)
- Limit `max_similar` in `add_memory()` to reduce classification workload
- Use `strategy="single"` instead of `"tiered"` or `"all"`
- Archive old memories periodically

**Problem: Slow classification**
- Use NLI classifier first (fast pre-filter)
- Switch to smaller/faster LLM model (e.g., gpt-3.5-turbo instead of gpt-4)
- Use `strategy="single"` to check only highest-similarity memory
- Consider batching memory additions

**Problem: Too many false conflicts**
- Increase NLI thresholds: `entailment_threshold=0.90` (stricter)
- Adjust auto-resolution thresholds to resolve more automatically
- Use more powerful LLM model for conflict detection

**Problem: Missing conflicts**
- Lower similarity threshold: `similarity_threshold=0.75` (find more similar memories)
- Increase `max_similar` to check more memories
- Use `strategy="all"` to check all similar memories

## Key Design Patterns and Architectural Decisions

### 1. Protocol-Based Architecture
**Why:** Maximum flexibility without tight coupling
- All storage uses Python `Protocol` (structural subtyping, not inheritance)
- Enables swapping implementations without code changes
- Works with any vector database, SQL dialect, or cache backend
- Runtime type checking with `isinstance()` checks

### 2. Sequential Classifier Chain
**Why:** Chain-of-responsibility pattern with graceful fallback
- Each classifier can pass uncertain cases to next classifier
- Early stopping on high-confidence decisions (performance optimization)
- Graceful degradation when components fail (e.g., LLM unavailable)
- Classifiers are stateless and composable

### 3. Lazy Loading
**Why:** Faster startup, lower memory usage
- Models (NLI, embeddings) loaded only on first use
- Avoids loading sentence-transformers if not needed
- Reduces initial memory footprint
- Example: `NLIPreFilter._load_model()` called on first `predict()`

### 4. Soft Delete with Versioning
**Why:** Preserve history and enable recovery
- `archived=True` flag instead of deletion
- `superseded_by` field tracks replacement chain
- Enables temporal queries and rollback
- Supports audit trails and debugging

### 5. Multi-User Isolation by Default
**Why:** Built for production multi-tenant systems
- `user_id` field in all memory structures from the start
- All storage operations filter by `user_id`
- Prevents data leakage between users
- No retrofitting needed for multi-user support

### 6. Async/Await Throughout
**Why:** Non-blocking I/O for efficiency
- All LLM calls, embeddings, and storage operations are async
- Enables concurrent processing of multiple memories
- Better resource utilization in production
- Natural fit for web frameworks (FastAPI, etc.)

### 7. Metadata-Rich Results
**Why:** Auditability and debugging
- Every decision includes confidence score
- Results include classifier name that made the decision
- Metadata dict carries decision-specific context (NLI scores, conflict category)
- Enables metrics tracking and decision analysis
- Example: `SimilarityResult.metadata = {"nli_scores": [0.1, 0.8, 0.1], "label": "entailment"}`

### 8. Graceful Degradation with Optional Dependencies
**Why:** Core functionality works without all extras
- Core works without transformers/Qdrant/PostgreSQL
- Optional imports with try/except in `__init__.py`
- Fallback to heuristics when LLM unavailable
- Example: NLI classifier skipped if sentence-transformers not installed

### 9. Confidence-Based Decision Making
**Why:** Minimize user interruptions
- Auto-resolution when confidence ratio is clear (≥1.3 or ≤0.7)
- Only escalate to user when truly ambiguous
- Confidence scoring based on mention frequency, recency, and spread
- Reduces cognitive load on users

### 10. Pydantic Models for Validation
**Why:** Type safety and automatic validation
- All data models use Pydantic `BaseModel`
- Automatic validation on construction
- Serialization/deserialization for free
- OpenAPI schema generation for APIs
- Better IDE support and type hints

## Notes for Future Development

- Auto-resolution thresholds (1.3 and 0.7) are tunable via `AutoResolutionClassifier` constructor
- Classification pipeline supports metrics tracking via `get_metrics()`
- Memory archiving uses soft-delete pattern with `superseded_by` tracking
- Conflict stores support escalation for conflicts that couldn't be auto-resolved
- All timestamps use ISO format strings or Python datetime objects
- Consider implementing:
  - Memory clustering for large-scale systems
  - Incremental confidence updates (avoid recalculating on every mention)
  - Batch classification for multiple new memories at once
  - Caching layer for frequently accessed memories
  - Webhook notifications for conflict escalation
