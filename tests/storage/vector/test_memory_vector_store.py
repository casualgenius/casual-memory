"""
Unit tests for in-memory vector storage.

Tests vector search, similarity matching, filtering, archiving, and memory management.
"""

from datetime import datetime

import pytest

from casual_memory.storage.vector.memory import InMemoryVectorStore


@pytest.fixture
def vector_store():
    """Create a fresh in-memory vector store."""
    return InMemoryVectorStore()


@pytest.fixture
def sample_vectors():
    """Sample vectors for testing."""
    return {
        "vec1": [1.0, 0.0, 0.0],  # Orthogonal to vec2
        "vec2": [0.0, 1.0, 0.0],  # Orthogonal to vec1
        "vec3": [0.9, 0.1, 0.0],  # Similar to vec1
        "vec4": [0.1, 0.9, 0.0],  # Similar to vec2
    }


@pytest.fixture
def sample_payload():
    """Sample memory payload."""
    return {
        "text": "I live in London",
        "type": "fact",
        "tags": ["location"],
        "importance": 0.8,
        "timestamp": datetime.now().isoformat(),
        "user_id": "user123",
        "confidence": 0.7,
        "mention_count": 1,
    }


def test_add_memory(vector_store, sample_vectors, sample_payload):
    """Test adding a memory to the store."""
    memory_id = vector_store.add(sample_vectors["vec1"], sample_payload)

    assert memory_id is not None
    assert isinstance(memory_id, str)


def test_get_by_id(vector_store, sample_vectors, sample_payload):
    """Test retrieving a memory by ID."""
    memory_id = vector_store.add(sample_vectors["vec1"], sample_payload)

    memory = vector_store.get_by_id(memory_id)

    assert memory is not None
    assert memory.id == memory_id
    assert memory.payload.text == "I live in London"
    assert memory.vector == sample_vectors["vec1"]


def test_get_by_id_nonexistent(vector_store):
    """Test retrieving a nonexistent memory."""
    memory = vector_store.get_by_id("nonexistent_id")

    assert memory is None


def test_search_basic(vector_store, sample_vectors, sample_payload):
    """Test basic vector search."""
    # Add a memory
    vector_store.add(sample_vectors["vec1"], sample_payload)

    # Search with similar vector
    results = vector_store.search(
        query_embedding=sample_vectors["vec3"],  # Similar to vec1
        top_k=5,
        min_score=0.8,
    )

    assert len(results) == 1
    assert results[0].payload.text == "I live in London"


def test_search_with_filtering(vector_store, sample_vectors):
    """Test search with user_id filtering."""
    # Add memories for different users
    payload1 = {
        "text": "Memory 1",
        "type": "fact",
        "tags": [],
        "importance": 0.7,
        "timestamp": datetime.now().isoformat(),
        "user_id": "user1",
        "confidence": 0.5,
        "mention_count": 1,
    }

    payload2 = {
        "text": "Memory 2",
        "type": "fact",
        "tags": [],
        "importance": 0.7,
        "timestamp": datetime.now().isoformat(),
        "user_id": "user2",
        "confidence": 0.5,
        "mention_count": 1,
    }

    vector_store.add(sample_vectors["vec1"], payload1)
    vector_store.add(sample_vectors["vec1"], payload2)

    # Search with user_id filter
    results = vector_store.search(
        query_embedding=sample_vectors["vec1"],
        top_k=5,
        min_score=0.5,
        filters={"user_id": "user1"},
    )

    assert len(results) == 1
    assert results[0].payload.user_id == "user1"


def test_search_top_k_limit(vector_store, sample_vectors, sample_payload):
    """Test that search respects top_k limit."""
    # Add multiple memories
    for i in range(5):
        payload = sample_payload.copy()
        payload["text"] = f"Memory {i}"
        vector_store.add(sample_vectors["vec1"], payload)

    # Search with top_k=3
    results = vector_store.search(
        query_embedding=sample_vectors["vec1"],
        top_k=3,
        min_score=0.5,
    )

    assert len(results) == 3


def test_search_min_score_threshold(vector_store, sample_vectors, sample_payload):
    """Test that search respects min_score threshold."""
    # Add memory with vec1
    vector_store.add(sample_vectors["vec1"], sample_payload)

    # Search with orthogonal vector (low similarity)
    results = vector_store.search(
        query_embedding=sample_vectors["vec2"],
        top_k=5,
        min_score=0.5,  # High threshold
    )

    # Should not find anything due to low similarity
    assert len(results) == 0


def test_find_similar_memories(vector_store, sample_vectors, sample_payload):
    """Test finding similar memories."""
    memory_id = vector_store.add(sample_vectors["vec1"], sample_payload)

    # Search with similar vector
    results = vector_store.find_similar_memories(
        embedding=sample_vectors["vec3"],  # Similar to vec1
        threshold=0.8,
        limit=5,
    )

    assert len(results) == 1
    memory_point, score = results[0]
    assert memory_point.id == memory_id
    assert score >= 0.8


def test_find_similar_with_user_filter(vector_store, sample_vectors):
    """Test finding similar memories with user_id filter."""
    payload1 = {
        "text": "Memory 1",
        "type": "fact",
        "tags": [],
        "importance": 0.7,
        "timestamp": datetime.now().isoformat(),
        "user_id": "user1",
        "confidence": 0.5,
        "mention_count": 1,
    }

    payload2 = {
        "text": "Memory 2",
        "type": "fact",
        "tags": [],
        "importance": 0.7,
        "timestamp": datetime.now().isoformat(),
        "user_id": "user2",
        "confidence": 0.5,
        "mention_count": 1,
    }

    vector_store.add(sample_vectors["vec1"], payload1)
    vector_store.add(sample_vectors["vec1"], payload2)

    # Find similar with user_id filter
    results = vector_store.find_similar_memories(
        embedding=sample_vectors["vec1"],
        user_id="user1",
        threshold=0.5,
    )

    assert len(results) == 1
    assert results[0][0].payload.user_id == "user1"


def test_update_memory(vector_store, sample_vectors, sample_payload):
    """Test updating a memory's payload."""
    memory_id = vector_store.add(sample_vectors["vec1"], sample_payload)

    # Update memory
    success = vector_store.update_memory(
        memory_id,
        {"mention_count": 5, "confidence": 0.9},
    )

    assert success is True

    # Verify update
    memory = vector_store.get_by_id(memory_id)
    assert memory.payload.mention_count == 5
    assert memory.payload.confidence == 0.9


def test_update_nonexistent_memory(vector_store):
    """Test updating a nonexistent memory."""
    success = vector_store.update_memory(
        "nonexistent_id",
        {"mention_count": 5},
    )

    assert success is False


def test_archive_memory(vector_store, sample_vectors, sample_payload):
    """Test archiving a memory."""
    memory_id = vector_store.add(sample_vectors["vec1"], sample_payload)

    # Archive memory
    success = vector_store.archive_memory(memory_id)

    assert success is True

    # Verify archival
    memory = vector_store.get_by_id(memory_id)
    assert memory.payload.archived is True
    assert memory.payload.archived_at is not None


def test_archive_with_superseded_by(vector_store, sample_vectors, sample_payload):
    """Test archiving a memory with superseded_by."""
    memory_id1 = vector_store.add(sample_vectors["vec1"], sample_payload)
    memory_id2 = vector_store.add(sample_vectors["vec1"], sample_payload)

    # Archive with superseded_by
    success = vector_store.archive_memory(memory_id1, superseded_by=memory_id2)

    assert success is True

    # Verify superseded_by
    memory = vector_store.get_by_id(memory_id1)
    assert memory.payload.superseded_by == memory_id2


def test_archived_excluded_from_search(vector_store, sample_vectors, sample_payload):
    """Test that archived memories are excluded from search."""
    memory_id = vector_store.add(sample_vectors["vec1"], sample_payload)

    # Archive memory
    vector_store.archive_memory(memory_id)

    # Search should not find archived memory
    results = vector_store.search(
        query_embedding=sample_vectors["vec1"],
        top_k=5,
        min_score=0.5,
    )

    assert len(results) == 0


def test_archived_excluded_from_find_similar(vector_store, sample_vectors, sample_payload):
    """Test that archived memories can be excluded from find_similar."""
    memory_id = vector_store.add(sample_vectors["vec1"], sample_payload)

    # Archive memory
    vector_store.archive_memory(memory_id)

    # Find similar with exclude_archived=True
    results = vector_store.find_similar_memories(
        embedding=sample_vectors["vec1"],
        threshold=0.5,
        exclude_archived=True,
    )

    assert len(results) == 0


def test_archived_included_when_not_excluded(vector_store, sample_vectors, sample_payload):
    """Test that archived memories can be included if not excluded."""
    memory_id = vector_store.add(sample_vectors["vec1"], sample_payload)

    # Archive memory
    vector_store.archive_memory(memory_id)

    # Find similar with exclude_archived=False
    results = vector_store.find_similar_memories(
        embedding=sample_vectors["vec1"],
        threshold=0.5,
        exclude_archived=False,
    )

    assert len(results) == 1


def test_clear_user_memories(vector_store, sample_vectors):
    """Test clearing all memories for a user."""
    # Add memories for different users
    payload1 = {
        "text": "Memory 1",
        "type": "fact",
        "tags": [],
        "importance": 0.7,
        "timestamp": datetime.now().isoformat(),
        "user_id": "user1",
        "confidence": 0.5,
        "mention_count": 1,
    }

    payload2 = {
        "text": "Memory 2",
        "type": "fact",
        "tags": [],
        "importance": 0.7,
        "timestamp": datetime.now().isoformat(),
        "user_id": "user2",
        "confidence": 0.5,
        "mention_count": 1,
    }

    vector_store.add(sample_vectors["vec1"], payload1)
    vector_store.add(sample_vectors["vec1"], payload1)
    vector_store.add(sample_vectors["vec1"], payload2)

    # Clear user1's memories
    count = vector_store.clear_user_memories("user1")

    assert count == 2

    # Verify user1's memories are gone
    results = vector_store.search(
        query_embedding=sample_vectors["vec1"],
        top_k=10,
        min_score=0.5,
        filters={"user_id": "user1"},
    )

    assert len(results) == 0

    # Verify user2's memory still exists
    results = vector_store.search(
        query_embedding=sample_vectors["vec1"],
        top_k=10,
        min_score=0.5,
        filters={"user_id": "user2"},
    )

    assert len(results) == 1


def test_clear_all_memories(vector_store, sample_vectors, sample_payload):
    """Test clearing all memories."""
    # Add multiple memories
    for i in range(5):
        vector_store.add(sample_vectors["vec1"], sample_payload)

    # Clear all
    vector_store.clear()

    # Verify all cleared
    results = vector_store.search(
        query_embedding=sample_vectors["vec1"],
        top_k=10,
        min_score=0.5,
    )

    assert len(results) == 0


def test_cosine_similarity_calculation(vector_store):
    """Test cosine similarity calculation."""
    # Identical vectors should have similarity of 1.0
    vec1 = [1.0, 0.0, 0.0]
    similarity = vector_store._cosine_similarity(vec1, vec1)
    assert abs(similarity - 1.0) < 0.001

    # Orthogonal vectors should have similarity of 0.0
    vec2 = [0.0, 1.0, 0.0]
    similarity = vector_store._cosine_similarity(vec1, vec2)
    assert abs(similarity - 0.0) < 0.001

    # Opposite vectors should have similarity of -1.0
    vec3 = [-1.0, 0.0, 0.0]
    similarity = vector_store._cosine_similarity(vec1, vec3)
    assert abs(similarity - (-1.0)) < 0.001


def test_filter_by_type(vector_store, sample_vectors):
    """Test filtering by memory type."""
    payload_fact = {
        "text": "Fact memory",
        "type": "fact",
        "tags": [],
        "importance": 0.7,
        "timestamp": datetime.now().isoformat(),
        "user_id": "user1",
        "confidence": 0.5,
        "mention_count": 1,
    }

    payload_pref = {
        "text": "Preference memory",
        "type": "preference",
        "tags": [],
        "importance": 0.7,
        "timestamp": datetime.now().isoformat(),
        "user_id": "user1",
        "confidence": 0.5,
        "mention_count": 1,
    }

    vector_store.add(sample_vectors["vec1"], payload_fact)
    vector_store.add(sample_vectors["vec1"], payload_pref)

    # Search for facts only
    results = vector_store.search(
        query_embedding=sample_vectors["vec1"],
        top_k=10,
        min_score=0.5,
        filters={"type": ["fact"]},
    )

    assert len(results) == 1
    assert results[0].payload.type == "fact"


def test_filter_by_min_importance(vector_store, sample_vectors):
    """Test filtering by minimum importance."""
    payload_high = {
        "text": "High importance",
        "type": "fact",
        "tags": [],
        "importance": 0.9,
        "timestamp": datetime.now().isoformat(),
        "user_id": "user1",
        "confidence": 0.5,
        "mention_count": 1,
    }

    payload_low = {
        "text": "Low importance",
        "type": "fact",
        "tags": [],
        "importance": 0.3,
        "timestamp": datetime.now().isoformat(),
        "user_id": "user1",
        "confidence": 0.5,
        "mention_count": 1,
    }

    vector_store.add(sample_vectors["vec1"], payload_high)
    vector_store.add(sample_vectors["vec1"], payload_low)

    # Search with min_importance filter
    results = vector_store.search(
        query_embedding=sample_vectors["vec1"],
        top_k=10,
        min_score=0.5,
        filters={"min_importance": 0.7},
    )

    assert len(results) == 1
    assert results[0].payload.importance >= 0.7
