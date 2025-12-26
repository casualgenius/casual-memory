"""
Unit tests for MemoryService.

Tests the service layer with mocked dependencies to verify
correct orchestration of components.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from casual_memory.classifiers.models import (
    MemoryClassificationResult,
    SimilarityResult,
    SimilarMemory,
)
from casual_memory.memory_service import MemoryService
from casual_memory.models import MemoryFact, MemoryQueryFilter


@pytest.fixture
def mock_vector_store():
    """Mock vector store."""
    store = Mock()
    store.find_similar_memories = Mock(return_value=[])
    store.search = Mock(return_value=[])
    store.add = Mock(return_value="mem_123")
    store.update_memory = Mock(return_value=True)
    store.archive_memory = Mock(return_value=True)
    store.get_by_id = AsyncMock(return_value=None)
    return store


@pytest.fixture
def mock_conflict_store():
    """Mock conflict store."""
    store = Mock()
    store.add_conflict = Mock(return_value="conflict_123")
    return store


@pytest.fixture
def mock_pipeline():
    """Mock classification pipeline."""
    pipeline = Mock()
    pipeline.classify = AsyncMock()
    return pipeline


@pytest.fixture
def mock_embedding():
    """Mock embedding service."""
    embedding = Mock()
    embedding.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
    embedding.embed_document = AsyncMock(return_value=[0.1, 0.2, 0.3])
    return embedding


@pytest.fixture
def memory_service(mock_vector_store, mock_conflict_store, mock_pipeline, mock_embedding):
    """Create memory service with mocked dependencies."""
    return MemoryService(
        vector_store=mock_vector_store,
        conflict_store=mock_conflict_store,
        pipeline=mock_pipeline,
        embedding=mock_embedding,
    )


@pytest.mark.asyncio
async def test_add_memory_added_outcome(memory_service, mock_vector_store, mock_pipeline):
    """Test adding a new memory that doesn't conflict or duplicate."""
    new_memory = MemoryFact(
        text="I live in Paris",
        type="fact",
        tags=[],
        user_id="user_123",
    )

    # Mock no similar memories found
    mock_vector_store.find_similar_memories.return_value = []

    # Mock classification result: add
    mock_pipeline.classify.return_value = MemoryClassificationResult(
        new_memory=new_memory,
        overall_outcome="add",
        similarity_results=[],
    )

    # Execute
    result = await memory_service.add_memory(new_memory)

    # Verify
    assert result.action == "added"
    assert result.memory_id == "mem_123"
    assert result.conflict_ids == []
    assert result.superseded_ids == []

    # Verify interactions
    mock_vector_store.find_similar_memories.assert_called_once()
    mock_pipeline.classify.assert_called_once()
    mock_vector_store.add.assert_called_once()


@pytest.mark.asyncio
async def test_add_memory_updated_outcome(memory_service, mock_vector_store, mock_pipeline):
    """Test adding a duplicate memory that updates existing."""
    new_memory = MemoryFact(
        text="I live in Paris",
        type="fact",
        tags=[],
        user_id="user_123",
    )

    # Mock similar memory found
    similar_point = Mock()
    similar_point.id = "mem_100"
    similar_point.payload = {
        "text": "I live in Paris",
        "type": "fact",
        "tags": [],
        "importance": 0.8,
        "user_id": "user_123",
        "confidence": 0.8,
        "mention_count": 1,
    }

    mock_vector_store.find_similar_memories.return_value = [(similar_point, 0.95)]

    # Mock classification result: skip (duplicate)
    similar_memory = SimilarMemory(
        memory_id="mem_100",
        memory=MemoryFact(**similar_point.payload),
        similarity_score=0.95,
    )

    mock_pipeline.classify.return_value = MemoryClassificationResult(
        new_memory=new_memory,
        overall_outcome="skip",
        similarity_results=[
            SimilarityResult(
                similar_memory=similar_memory,
                outcome="same",
                confidence=0.9,
                classifier_name="nli",
                metadata={},
            )
        ],
    )

    # Execute
    result = await memory_service.add_memory(new_memory)

    # Verify
    assert result.action == "updated"
    assert result.memory_id == "mem_100"
    assert result.conflict_ids == []

    # Verify update was called
    mock_vector_store.update_memory.assert_called_once()


@pytest.mark.asyncio
async def test_add_memory_conflict_outcome(
    memory_service, mock_vector_store, mock_pipeline, mock_conflict_store
):
    """Test adding a conflicting memory."""
    new_memory = MemoryFact(
        text="I live in London",
        type="fact",
        tags=[],
        user_id="user_123",
    )

    # Mock conflicting memory
    similar_point = Mock()
    similar_point.id = "mem_100"
    similar_point.payload = {
        "text": "I live in Paris",
        "type": "fact",
        "tags": [],
        "importance": 0.8,
        "user_id": "user_123",
        "confidence": 0.8,
        "mention_count": 3,
    }

    mock_vector_store.find_similar_memories.return_value = [(similar_point, 0.92)]

    # Mock classification result: conflict
    similar_memory = SimilarMemory(
        memory_id="mem_100",
        memory=MemoryFact(**similar_point.payload),
        similarity_score=0.92,
    )

    mock_pipeline.classify.return_value = MemoryClassificationResult(
        new_memory=new_memory,
        overall_outcome="conflict",
        similarity_results=[
            SimilarityResult(
                similar_memory=similar_memory,
                outcome="conflict",
                confidence=0.9,
                classifier_name="conflict",
                metadata={
                    "category": "location",
                    "clarification_hint": "Where do you currently live?",
                },
            )
        ],
    )

    # Execute
    result = await memory_service.add_memory(new_memory)

    # Verify
    assert result.action == "conflict"
    assert len(result.conflict_ids) == 1
    assert result.conflict_ids[0] == "conflict_123"
    assert result.memory_id is None

    # Verify conflict was created
    mock_conflict_store.add_conflict.assert_called_once()


@pytest.mark.asyncio
async def test_add_memory_with_superseding(memory_service, mock_vector_store, mock_pipeline):
    """Test adding a memory that supersedes old memories."""
    new_memory = MemoryFact(
        text="I live in Central London near the Thames",
        type="fact",
        tags=[],
        user_id="user_123",
    )

    # Mock similar memory that will be superseded
    similar_point = Mock()
    similar_point.id = "mem_100"
    similar_point.payload = {
        "text": "I live in London",
        "type": "fact",
        "tags": [],
        "importance": 0.8,
        "user_id": "user_123",
        "confidence": 0.6,
        "mention_count": 1,
    }

    mock_vector_store.find_similar_memories.return_value = [(similar_point, 0.94)]

    # Mock classification result: add with superseding
    similar_memory = SimilarMemory(
        memory_id="mem_100",
        memory=MemoryFact(**similar_point.payload),
        similarity_score=0.94,
    )

    mock_pipeline.classify.return_value = MemoryClassificationResult(
        new_memory=new_memory,
        overall_outcome="add",
        similarity_results=[
            SimilarityResult(
                similar_memory=similar_memory,
                outcome="superseded",
                confidence=0.9,
                classifier_name="duplicate",
                metadata={},
            )
        ],
    )

    # Execute
    result = await memory_service.add_memory(new_memory)

    # Verify
    assert result.action == "added"
    assert result.memory_id == "mem_123"
    assert result.superseded_ids == ["mem_100"]

    # Verify archive was called
    mock_vector_store.archive_memory.assert_called_once_with(
        memory_id="mem_100",
        superseded_by="mem_123",
    )


@pytest.mark.asyncio
async def test_add_memory_custom_thresholds(memory_service, mock_vector_store, mock_pipeline):
    """Test add_memory with custom similarity threshold and max_similar."""
    new_memory = MemoryFact(
        text="Test memory",
        type="fact",
        tags=[],
        user_id="user_123",
    )

    mock_vector_store.find_similar_memories.return_value = []

    # Mock classification result
    mock_pipeline.classify.return_value = MemoryClassificationResult(
        new_memory=new_memory,
        overall_outcome="add",
        similarity_results=[],
    )

    # Execute with custom params
    await memory_service.add_memory(
        new_memory,
        similarity_threshold=0.9,
        max_similar=10,
    )

    # Verify custom params were used
    mock_vector_store.find_similar_memories.assert_called_once()
    call_kwargs = mock_vector_store.find_similar_memories.call_args.kwargs
    assert call_kwargs["threshold"] == 0.9
    assert call_kwargs["limit"] == 10


@pytest.mark.asyncio
async def test_add_memory_error_handling(memory_service, mock_embedding):
    """Test error handling when adding memory fails."""
    new_memory = MemoryFact(
        text="Test memory",
        type="fact",
        tags=[],
        user_id="user_123",
    )

    # Mock embedding failure
    mock_embedding.embed_document.side_effect = Exception("Embedding service unavailable")

    # Execute and expect exception
    with pytest.raises(Exception) as exc_info:
        await memory_service.add_memory(new_memory)

    assert "Embedding service unavailable" in str(exc_info.value)


@pytest.mark.asyncio
async def test_query_memory_basic(memory_service, mock_vector_store, mock_embedding):
    """Test basic memory querying."""
    # Mock search results
    result1 = Mock()
    result1.payload = Mock(
        text="I live in Paris",
        type="fact",
        tags=["location"],
        importance=0.8,
        source=None,
        valid_until=None,
        user_id="user_123",
        confidence=0.8,
        mention_count=2,
        first_seen="2024-01-01T00:00:00",
        last_seen="2024-01-02T00:00:00",
        archived=False,
        archived_at=None,
        superseded_by=None,
    )

    result2 = Mock()
    result2.payload = Mock(
        text="I work in London",
        type="fact",
        tags=["work"],
        importance=0.7,
        source=None,
        valid_until=None,
        user_id="user_123",
        confidence=0.7,
        mention_count=1,
        first_seen="2024-01-01T00:00:00",
        last_seen="2024-01-01T00:00:00",
        archived=False,
        archived_at=None,
        superseded_by=None,
    )

    mock_vector_store.search.return_value = [result1, result2]

    # Execute
    query_filter = MemoryQueryFilter(user_id="user_123")
    memories = await memory_service.query_memory(
        query="location information",
        filter=query_filter,
        top_k=5,
        min_score=0.75,
    )

    # Verify
    assert len(memories) == 2
    assert memories[0].text == "I live in Paris"
    assert memories[1].text == "I work in London"

    # Verify embedding was called
    mock_embedding.embed_query.assert_called_once_with("location information")

    # Verify search was called with correct params
    mock_vector_store.search.assert_called_once()
    call_kwargs = mock_vector_store.search.call_args.kwargs
    assert call_kwargs["top_k"] == 5
    assert call_kwargs["min_score"] == 0.75


@pytest.mark.asyncio
async def test_query_memory_filters_expired(memory_service, mock_vector_store):
    """Test that expired memories are filtered out."""
    # Mock result with expired memory
    expired_result = Mock()
    expired_result.payload = Mock(
        text="Old memory",
        type="fact",
        tags=[],
        importance=0.8,
        source=None,
        valid_until="2020-01-01T00:00:00",  # Expired
        user_id="user_123",
        confidence=0.8,
        mention_count=1,
        first_seen="2019-01-01T00:00:00",
        last_seen="2019-01-01T00:00:00",
        archived=False,
        archived_at=None,
        superseded_by=None,
    )

    valid_result = Mock()
    valid_result.payload = Mock(
        text="Valid memory",
        type="fact",
        tags=[],
        importance=0.8,
        source=None,
        valid_until=None,  # No expiration
        user_id="user_123",
        confidence=0.8,
        mention_count=1,
        first_seen="2024-01-01T00:00:00",
        last_seen="2024-01-01T00:00:00",
        archived=False,
        archived_at=None,
        superseded_by=None,
    )

    mock_vector_store.search.return_value = [expired_result, valid_result]

    # Execute
    query_filter = MemoryQueryFilter(user_id="user_123")
    memories = await memory_service.query_memory(
        query="test",
        filter=query_filter,
    )

    # Verify only valid memory returned
    assert len(memories) == 1
    assert memories[0].text == "Valid memory"


@pytest.mark.asyncio
async def test_query_memory_empty_results(memory_service, mock_vector_store):
    """Test querying when no memories match."""
    mock_vector_store.search.return_value = []

    # Execute
    query_filter = MemoryQueryFilter(user_id="user_123")
    memories = await memory_service.query_memory(
        query="nonexistent topic",
        filter=query_filter,
    )

    # Verify
    assert len(memories) == 0
