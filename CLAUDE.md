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
   - `add_memory()`: Find similar, classify, execute actions
   - `query_memory()`: Semantic search with filtering

4. **Storage** ([src/casual_memory/storage/](src/casual_memory/storage/))
   - Vector: `InMemoryVectorStore`, `QdrantMemoryStore`
   - Conflict: `InMemoryConflictStore`, `SQLAlchemyConflictStore`
   - Short-term: `InMemoryShortTermStore`, `RedisShortTermStore`

5. **Extraction** ([src/casual_memory/extractors/](src/casual_memory/extractors/))
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

### Data Flow

```
1. MemoryService.add_memory(new_memory)
2. → Embed text, search similar memories
3. → Pipeline.classify(new_memory, similar_memories)
4. → ActionExecutor.execute(result)
   - "add": Insert to vector store, archive superseded
   - "skip": Update existing (mention_count++)
   - "conflict": Create conflict record
5. → Return MemoryActionResult
```

## Public API

```python
# Core exports
from casual_memory import (
    MemoryFact,           # Core memory unit
    MemoryFactExtraction, # Extraction model
    MemoryConflict,       # Conflict tracking
    ConflictResolution,   # Resolution decision
    ShortTermMemory,      # Conversation message
    MemoryQueryFilter,    # Query filtering
    MemoryService,        # Main entry point
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

# Storage
from casual_memory.storage.vector import InMemoryVectorStore, QdrantMemoryStore
from casual_memory.storage.conflicts import InMemoryConflictStore, SQLAlchemyConflictStore

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
│   ├── vector/        # Vector stores
│   ├── conflicts/     # Conflict stores
│   └── short_term/    # Short-term stores
├── embeddings/        # E5, OpenAI embeddings
├── execution/         # Action executor
├── models.py          # Core data models
└── memory_service.py  # High-level service
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

new_memory = MemoryFact(text="I live in Paris", type="fact", ...)
similar = [SimilarMemory(memory_id="...", memory=MemoryFact(...), similarity_score=0.91)]

result = await pipeline.classify(new_memory, similar)
print(result.overall_outcome)  # "add", "skip", or "conflict"
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
```

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for common issues.
