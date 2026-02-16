# Changelog

## [0.3.0] - 2026-02-16

### Added

- **Namespace support** - All memory operations now scoped by `namespace` and `entity_id`
  - `namespace` field on `MemoryFact`, `MemoryConflict`, and `MemoryQueryFilter` (default: `"default"`)
  - All storage backends filter by namespace: vector stores via payload fields, conflict stores via indexed columns, short-term stores via composite keys
  - `ContextService` methods accept `namespace` parameter
  - Qdrant store builds namespace-aware filters for all queries and deletions

### Changed

- `user_id` deprecated across all models and services in favor of `entity_id` (backward-compatible with `DeprecationWarning`)

### Fixed

- Qdrant `find_similar_memories` now filters archived memories at query level instead of post-processing, respecting `limit` correctly
- Qdrant bulk delete operations use scroll pagination instead of a single 10k-point fetch
- Qdrant type filter validates input and coerces string to list
- Removed unnecessary `type: ignore` comments in Qdrant store with proper type alias

## [0.2.1] - 2026-02-09

### Added

- ContextService to use short-term storage for add/get/clear 

### Fixed

- Ensure short-term memory starts with a user message to avoid broken tool call/result pairs.

## [0.2.0] - 2026-02-06

### Added

- Support for custom json schema when using LLM Extractor

## [0.1.0] - 2026-02-03

Initial Release

### Added

#### Core Features
- **Classification Pipeline** - Composable, protocol-based memory classification system
  - NLI Classifier with DeBERTa cross-encoder for fast pre-filtering
  - Conflict Classifier with LLM verification and heuristic fallback
  - Duplicate Classifier for distinguishing duplicates from distinct facts
  - Auto-Resolution Classifier for confidence-based conflict resolution
  - Sequential execution with early termination

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
- `MemoryFactExtraction` - Extraction model for LLM parsing
- `MemoryConflict` - Conflict metadata with categorization
- `ConflictResolution` - Resolution decisions and actions
- `ShortTermMemory` - Conversation history messages
- `SimilarMemory` - Similar memory for classification
- `SimilarityResult` - Classification outcome for a memory pair
- `MemoryClassificationResult` - Overall classification result

### Documentation
- Comprehensive README.md with installation, quickstart, benchmarks
- ARCHITECTURE.md with system design and performance analysis
- Working examples

### Dependencies
- **Core**: pydantic, casual-llm (LLM provider abstraction)
- **Optional extras**:
  - `transformers` - sentence-transformers for NLI
  - `qdrant` - qdrant-client for vector storage
  - `postgres` - sqlalchemy, asyncpg for conflict storage
  - `redis` - redis-py for short-term storage
  - `all` - All optional dependencies

---

## Version History

- **0.3.0** (2026-02-16) - Namespace support, entity_id migration, Qdrant fixes
- **0.2.1** (2026-02-09) - ContextService for short-term storage
- **0.2.0** (2026-02-06) - Custom JSON schema for LLM Extractor
- **0.1.0** (2026-02-03) - Initial release with classification pipeline, intelligence layer, storage abstraction

---
