# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`casual-memory` is a Python library for intelligent semantic memory management with conflict detection, classification pipelines, and storage abstraction. It uses LLMs and NLI models to detect contradictory memories, classify duplicate vs. distinct facts, and automatically resolve conflicts based on confidence scoring.

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

The library is organized around four main subsystems:

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

3. **Memory Extraction** ([src/casual_memory/extractors/](src/casual_memory/extractors/))
   - **UserMemoryExtracter**: Extracts memories from user messages
   - **AssistantMemoryExtracter**: Extracts memories observed by the assistant
   - Uses structured prompts with LLM to identify facts, preferences, events, and goals

4. **Storage Abstraction** ([src/casual_memory/storage/](src/casual_memory/storage/))
   - **Protocol-based**: All storage is defined via Python protocols
   - **Vector stores**: In-memory (testing), Qdrant (production)
   - **Conflict stores**: In-memory, SQLite, PostgreSQL, SQLAlchemy (unified)
   - **Short-term stores**: In-memory (deque), Redis (production)

### Classification Pipeline Flow

Memory pairs flow sequentially through classifiers:

1. **NLI Classifier** filters obvious cases:
   - High entailment (≥0.85) → MERGE
   - High neutral (≥0.5) → ADD
   - Uncertain → pass to next classifier

2. **Conflict Classifier** detects contradictions using LLM:
   - Categorizes conflicts (location, job, preference, temporal, factual)
   - Provides clarification hints for user resolution
   - Only confident contradictions → CONFLICT

3. **Duplicate Classifier** distinguishes same vs. distinct facts:
   - Same fact or refinement → MERGE
   - Distinct facts → ADD

4. **Auto-Resolution Classifier** resolves conflicts by confidence:
   - High new confidence (ratio ≥1.3) → MERGE (keep_new)
   - High old confidence (ratio ≤0.7) → MERGE (keep_old)
   - Similar confidence → Keep as CONFLICT

5. **Default Handler**: Unclassified pairs → ADD (conservative)

Each classifier only classifies what it's confident about. Classified pairs move from `request.pairs` to `request.results`. Unclassified pairs pass to the next classifier.

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

## Key Data Models

Located in [src/casual_memory/models.py](src/casual_memory/models.py):

- **MemoryFact**: Core memory unit (text, type, tags, importance, confidence, mention tracking)
- **MemoryConflict**: Tracks contradictory memories with category and clarification hints
- **ConflictResolution**: Resolution decision (keep_a, keep_b, merge, both_valid)
- **ShortTermMemory**: Conversation message (content, role, timestamp)

Memory types: `fact`, `preference`, `event`, `goal`, `weather`

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

## Notes for Future Development

- Auto-resolution thresholds (1.3 and 0.7) are tunable via `AutoResolutionClassifier` constructor
- Classification pipeline supports metrics tracking via `get_metrics()`
- Memory archiving uses soft-delete pattern with `superseded_by` tracking
- Conflict stores support escalation for conflicts that couldn't be auto-resolved
- All timestamps use ISO format strings or Python datetime objects
