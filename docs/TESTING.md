# Testing Guide

This document describes the testing structure and conventions for casual-memory.

## Running Tests

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

## Test Structure

Tests are organized by module in [tests/](../tests/):

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

## Testing Conventions

### pytest Configuration

Tests use pytest with asyncio support (`asyncio_mode = "auto"` in pyproject.toml):

```python
# Async tests automatically work
async def test_my_async_function():
    result = await my_async_function()
    assert result == expected
```

### Mocking LLM Providers

Mock LLM providers for classifier tests to avoid external dependencies:

```python
class MockModel:
    async def chat(self, messages, **kwargs):
        # Return predictable responses for testing
        return AssistantMessage(content="YES")

def test_conflict_verifier():
    mock_model = MockModel()
    verifier = LLMConflictVerifier(mock_model)
    # ... test logic
```

### In-Memory Storage for Unit Tests

Use in-memory storage backends for fast, isolated tests:

```python
from casual_memory.storage.vector import InMemoryVectorStore
from casual_memory.storage.conflicts import InMemoryConflictStore

def test_memory_service():
    vector_store = InMemoryVectorStore()
    conflict_store = InMemoryConflictStore()
    # ... test logic
```

### Database Tests

Integration tests can use SQLite `:memory:` databases:

```python
from sqlalchemy import create_engine
from casual_memory.storage.conflicts.sqlalchemy import SQLAlchemyConflictStore

def test_sqlalchemy_store():
    engine = create_engine("sqlite:///:memory:")
    store = SQLAlchemyConflictStore(engine)
    store.create_tables()
    # ... test logic
```

### Fixtures

Common test fixtures are defined in `conftest.py`:

```python
@pytest.fixture
def memory_fact():
    return MemoryFact(
        text="I live in London",
        type="fact",
        tags=["location"],
        importance=0.8,
        entity_id="test-user",
    )

@pytest.fixture
def mock_model():
    return MockModel()
```

## Test Coverage

Current test statistics:
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

## Testing with Optional Dependencies

Tests for optional backends automatically skip when dependencies are unavailable:

```python
pytest.importorskip("qdrant_client")

def test_qdrant_store():
    # Only runs if qdrant-client is installed
    ...
```

For external service tests (Redis, PostgreSQL), use markers:

```python
@pytest.mark.redis
def test_redis_store():
    # Skip if Redis not available
    ...
```

## Code Quality

```bash
# Format code with black
uv run black src/

# Lint with ruff
uv run ruff check src/

# Type check with mypy
uv run mypy src/casual_memory/
```
