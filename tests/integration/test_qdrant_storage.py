"""Integration tests for Qdrant vector storage backend."""

from uuid import uuid4

import pytest


@pytest.mark.integration
def test_qdrant_add_and_search(skip_if_no_qdrant):
    """Test adding memories and searching with Qdrant."""
    pytest.importorskip("qdrant_client")

    from casual_memory.storage.vector.qdrant import QdrantMemoryStore

    host = skip_if_no_qdrant  # Fixture returns the host

    # Create storage instance with unique collection
    storage = QdrantMemoryStore(
        collection_name=f"test_collection_{uuid4().hex[:8]}", host=host, port=6333
    )

    try:
        # Create test payload (simulating MemoryFact structure)
        payload = {
            "text": "I work as a software engineer at Google",
            "type": "fact",
            "tags": ["job", "career"],
            "importance": 0.9,
            "source": "user",
            "entity_id": "test_user",
            "namespace": "default",
            "archived": False,
            "timestamp": "2024-01-01T10:00:00",
        }

        # Create a fake embedding vector (768 dimensions for e5 model)
        vector = [0.1] * 768

        # Add memory
        memory_id = storage.add(vector=vector, payload=payload)
        assert memory_id is not None

        # Search for similar memory using the same vector
        results = storage.search(
            query_embedding=vector,
            top_k=5,
            min_score=0.5,
            filters={"entity_id": "test_user", "type": None, "min_importance": None},
        )

        # Should find the memory we just added
        assert len(results) > 0
        assert results[0].payload.text == payload["text"]
        assert results[0].payload.type == "fact"

    finally:
        # Cleanup: delete collection
        try:
            storage.client.delete_collection(storage.collection_name)
        except Exception:
            pass


@pytest.mark.integration
def test_qdrant_update_memory(skip_if_no_qdrant):
    """Test updating memories in Qdrant."""
    pytest.importorskip("qdrant_client")

    from casual_memory.storage.vector.qdrant import QdrantMemoryStore

    host = skip_if_no_qdrant

    storage = QdrantMemoryStore(
        collection_name=f"test_collection_{uuid4().hex[:8]}", host=host, port=6333
    )

    try:
        # Add initial memory
        payload = {
            "text": "I live in London",
            "type": "fact",
            "tags": ["location"],
            "importance": 0.8,
            "source": "user",
            "entity_id": "test_user",
            "namespace": "default",
            "archived": False,
            "timestamp": "2024-01-01T10:00:00",
        }
        vector = [0.2] * 768

        memory_id = storage.add(vector=vector, payload=payload)

        # Update memory metadata
        storage.update_memory(
            memory_id=memory_id,
            payload_updates={
                "text": "I live in Central London",
                "tags": ["location", "residence"],
                "importance": 0.9,
            },
        )

        # Retrieve updated memory
        result = storage.get_memory_by_id(memory_id)

        assert result is not None
        assert result.payload.text == "I live in Central London"
        assert "residence" in result.payload.tags

    finally:
        # Cleanup
        try:
            storage.client.delete_collection(storage.collection_name)
        except Exception:
            pass


@pytest.mark.integration
def test_qdrant_archive_memory(skip_if_no_qdrant):
    """Test archiving memories in Qdrant."""
    pytest.importorskip("qdrant_client")

    from casual_memory.storage.vector.qdrant import QdrantMemoryStore

    host = skip_if_no_qdrant

    storage = QdrantMemoryStore(
        collection_name=f"test_collection_{uuid4().hex[:8]}", host=host, port=6333
    )

    try:
        # Add memory
        payload = {
            "text": "I work as a teacher",
            "type": "fact",
            "tags": ["job"],
            "importance": 0.8,
            "source": "user",
            "entity_id": "test_user",
            "namespace": "default",
            "archived": False,
            "timestamp": "2024-01-01T10:00:00",
        }
        vector = [0.3] * 768

        memory_id = storage.add(vector=vector, payload=payload)

        # Archive memory
        storage.archive_memory(memory_id=memory_id, superseded_by="new_memory_id")

        # Search excluding archived (using find_similar_memories which has exclude_archived)
        results = storage.find_similar_memories(
            embedding=vector,
            entity_id="test_user",
            threshold=0.5,
            limit=5,
            exclude_archived=True,
        )

        # Should not find archived memory
        assert all(point.id != memory_id for point, _ in results)

    finally:
        # Cleanup
        try:
            storage.client.delete_collection(storage.collection_name)
        except Exception:
            pass


@pytest.mark.integration
def test_qdrant_user_isolation(skip_if_no_qdrant):
    """Test that memories are isolated by entity_id."""
    pytest.importorskip("qdrant_client")

    from casual_memory.storage.vector.qdrant import QdrantMemoryStore

    host = skip_if_no_qdrant

    storage = QdrantMemoryStore(
        collection_name=f"test_collection_{uuid4().hex[:8]}", host=host, port=6333
    )

    try:
        # Use the same vector so similarity is guaranteed
        vector = [0.4] * 768

        # Add memory for user1
        payload1 = {
            "text": "User 1's secret hobby is painting",
            "type": "fact",
            "tags": ["hobby"],
            "importance": 0.7,
            "source": "user",
            "entity_id": "user_1",
            "namespace": "default",
            "archived": False,
            "timestamp": "2024-01-01T10:00:00",
        }
        storage.add(vector=vector, payload=payload1)

        # Add memory for user2
        payload2 = {
            "text": "User 2's secret hobby is gardening",
            "type": "fact",
            "tags": ["hobby"],
            "importance": 0.7,
            "source": "user",
            "entity_id": "user_2",
            "namespace": "default",
            "archived": False,
            "timestamp": "2024-01-01T10:00:00",
        }
        storage.add(vector=vector, payload=payload2)

        # Search as user1 using find_similar_memories
        results_user1 = storage.find_similar_memories(
            embedding=vector, entity_id="user_1", threshold=0.5, limit=5
        )

        # Search as user2
        results_user2 = storage.find_similar_memories(
            embedding=vector, entity_id="user_2", threshold=0.5, limit=5
        )

        # Each user should only see their own memories
        assert len(results_user1) > 0
        assert all("painting" in point.payload.text.lower() for point, _ in results_user1)

        assert len(results_user2) > 0
        assert all("gardening" in point.payload.text.lower() for point, _ in results_user2)

    finally:
        # Cleanup
        try:
            storage.client.delete_collection(storage.collection_name)
        except Exception:
            pass
