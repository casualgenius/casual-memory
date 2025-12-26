# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- PyPI package publishing
- Service migration examples
- Additional storage backend adapters (Pinecone, Weaviate)
- Streaming support for LLM responses
- Batch processing for memory classification

## [0.1.0] - 2025-01-XX

### Added

#### Core Features
- **Classification Pipeline** - Composable, protocol-based memory classification system
  - NLI Classifier with DeBERTa cross-encoder for fast pre-filtering
  - Conflict Classifier with LLM verification and heuristic fallback
  - Duplicate Classifier for distinguishing duplicates from distinct facts
  - Auto-Resolution Classifier for confidence-based conflict resolution
  - Sequential execution with early termination

#### Intelligence Layer
- **NLI Pre-Filter** - DeBERTa-v3-base-mnli-fever-anli cross-encoder
  - 92.38% accuracy on SNLI, 90.04% on MNLI
  - LRU caching with 1000-entry limit
  - Lazy loading for optional dependency
  - ~200ms CPU, ~50ms GPU performance

- **LLM Conflict Verifier** - High-accuracy contradiction detection
  - LLM-based detection with structured prompts
  - Heuristic fallback for negation, location, job conflicts
  - Configurable fallback threshold (similarity ≥ 0.90)
  - Metrics tracking (calls, success rate, fallbacks)

- **LLM Duplicate Detector** - Smart duplicate vs distinct classification
  - Distinguishes refinements from contradictions
  - Conservative fallback (similarity ≥ 0.95)
  - Handles paraphrases and intensity variations

- **Confidence Scoring** - Multi-factor memory confidence calculation
  - Mention frequency (1-5+ mentions: 0.5-0.95)
  - Recency factor (30-day decay)
  - Spread factor (temporal distribution bonus)

#### Memory Extraction
- **LLM Memory Extractor** - Extract structured memories from conversations
  - User-sourced extraction (importance × 1.0)
  - Assistant-sourced extraction (importance × 0.6)
  - First-person perspective normalization
  - Atomic fact splitting
  - Temporal memory support with date normalization
  - Four memory types: fact, preference, goal, event
  - Importance filtering (≥ 0.5 threshold)

#### Storage Abstraction
- **Protocol-based storage** - Runtime-checkable protocols (PEP 544)
  - VectorStore protocol for semantic search
  - ConflictStore protocol for conflict management
  - ShortTermStore protocol for conversation history

- **Optional adapters**
  - QdrantMemoryStore for vector storage
  - SQLAlchemyConflictStore for PostgreSQL conflicts
  - RedisShortTermStore for conversation caching
  - InMemory implementations for testing

- **Soft delete pattern** - Memory archiving with audit trail
  - `archived` flag with `archived_at` timestamp
  - `superseded_by` field for replacement tracking
  - Excluded from searches by default

#### Models
- `MemoryFact` - Core memory representation with metadata
- `MemoryConflict` - Conflict metadata with categorization
- `ConflictResolution` - Resolution decisions and actions
- `ShortTermMemory` - Conversation history messages
- `ClassificationRequest` - Pipeline input/output model
- `ClassificationResult` - Classification outcomes with metadata
- `MemoryPair` - Similar memory pair for classification

### Documentation
- Comprehensive README.md with installation, quickstart, benchmarks
- ARCHITECTURE.md with system design and performance analysis
- MIGRATION.md with step-by-step migration guide from memory services
- 5 working examples:
  - basic_classification.py - Pipeline usage
  - custom_classifier.py - Custom classifier implementation
  - memory_extraction.py - LLM-based extraction
  - conflict_detection_demo.py - Conflict detection with fallback
  - custom_storage_backend.py - Custom storage protocol

### Testing
- **76 unit tests** (51% code coverage)
  - 12 memory extraction tests
  - 23 confidence scoring tests
  - 12 NLI filter tests
  - 13 conflict verifier tests
  - 16 duplicate detector tests

- **17 integration tests** for optional backends
  - 5 Qdrant storage tests
  - 6 SQLAlchemy conflict storage tests
  - 6 Redis short-term storage tests
  - Auto-skip when services unavailable

- **pytest configuration** with async support, coverage reporting
- **Test fixtures** for mock LLM providers and storage backends

### CI/CD
- GitHub Actions workflow with test matrix (Python 3.10, 3.11, 3.12)
- Codecov integration for coverage reporting
- UV dependency caching for faster builds
- Linting with ruff (formatter + linter)
- Type checking with mypy

### Dependencies
- **Core**: pydantic, casual-llm (LLM provider abstraction)
- **Optional extras**:
  - `transformers` - sentence-transformers for NLI
  - `qdrant` - qdrant-client for vector storage
  - `postgres` - sqlalchemy, asyncpg for conflict storage
  - `redis` - redis-py for short-term storage
  - `dates` - dateparser for temporal normalization
  - `all` - All optional dependencies

### Development Tools
- UV for fast dependency management
- pytest with asyncio and coverage plugins
- ruff for linting and formatting
- mypy for static type checking

---

## Version History

- **0.1.0** (2025-01-XX) - Initial release with classification pipeline, intelligence layer, storage abstraction
- **Unreleased** - Future enhancements planned

---

## Migration Notes

### From memory-agent-service / memory-store-service

See [MIGRATION.md](docs/MIGRATION.md) for complete migration guide.

**Key Changes:**
- `provider.generate()` → `provider.chat()` (returns AssistantMessage)
- `MemoryExtractor(provider)` → `LLMMemoryExtractor(llm_provider=provider, source="user")`
- `ConflictDetector` → `LLMConflictVerifier`
- Environment config → ModelConfig objects
- Base URLs: Use `http://localhost:11434` (providers append paths)

---

## Links

- **Documentation**: [README.md](README.md) | [ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Examples**: [examples/](examples/)
- **Issue Tracker**: https://github.com/yourusername/casual-memory/issues
- **PyPI**: https://pypi.org/project/casual-memory/ (coming soon)
