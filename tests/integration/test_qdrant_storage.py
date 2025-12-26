"""Integration tests for Qdrant vector storage backend."""

from uuid import uuid4

import pytest

from casual_memory.models import MemoryFact
from casual_memory.storage.vector.qdrant import QdrantMemoryStore


@pytest.mark.integration
@pytest.mark.asyncio
async def test_qdrant_add_and_search(skip_if_no_qdrant):
    """Test adding memories and searching with Qdrant."""
    pytest.importorskip("qdrant_client")

    # Create storage instance
    storage = QdrantMemoryStore(
        collection_name=f"test_collection_{uuid4().hex[:8]}", host="localhost", port=6333
    )

    # Wait for initialization
    await storage.initialize()

    try:
        # Create test memory
        memory = MemoryFact(
            text="I work as a software engineer at Google",
            type="fact",
            tags=["job", "career"],
            importance=0.9,
            source="user",
        )

        # Add memory
        memory_id = await storage.add(memory, user_id="test_user")
        assert memory_id is not None

        # Search for similar memory
        results = await storage.search(
            query_text="software engineering job", user_id="test_user", limit=5
        )

        # Should find the memory we just added
        assert len(results) > 0
        assert results[0].text == memory.text
        assert results[0].type == "fact"

    finally:
        # Cleanup: delete collection
        try:
            await storage._ensure_collection()
            storage.client.delete_collection(storage.collection_name)
        except Exception:
            pass


@pytest.mark.integration
@pytest.mark.asyncio
async def test_qdrant_update_memory(skip_if_no_qdrant):
    """Test updating memories in Qdrant."""
    pytest.importorskip("qdrant_client")

    storage = QdrantVectorStorage(
        collection_name=f"test_collection_{uuid4().hex[:8]}", host="localhost", port=6333
    )

    await storage.initialize()

    try:
        # Add initial memory
        memory = MemoryFact(
            text="I live in London", type="fact", tags=["location"], importance=0.8, source="user"
        )

        memory_id = await storage.add(memory, user_id="test_user")

        # Update memory
        updated_memory = MemoryFact(
            text="I live in Central London",
            type="fact",
            tags=["location", "residence"],
            importance=0.9,
            source="user",
        )

        await storage.update(memory_id, updated_memory, user_id="test_user")

        # Search for updated memory
        results = await storage.search(query_text="where do you live", user_id="test_user", limit=1)

        assert len(results) > 0
        assert results[0].text == "I live in Central London"
        assert "residence" in results[0].tags

    finally:
        # Cleanup
        try:
            storage.client.delete_collection(storage.collection_name)
        except Exception:
            pass


@pytest.mark.integration
@pytest.mark.asyncio
async def test_qdrant_archive_memory(skip_if_no_qdrant):
    """Test archiving memories in Qdrant."""
    pytest.importorskip("qdrant_client")

    storage = QdrantVectorStorage(
        collection_name=f"test_collection_{uuid4().hex[:8]}", host="localhost", port=6333
    )

    await storage.initialize()

    try:
        # Add memory
        memory = MemoryFact(
            text="I work as a teacher", type="fact", tags=["job"], importance=0.8, source="user"
        )

        memory_id = await storage.add(memory, user_id="test_user")

        # Archive memory
        await storage.archive(memory_id, user_id="test_user", superseded_by="new_memory_id")

        # Search excluding archived (default behavior)
        results = await storage.search(
            query_text="what is your job", user_id="test_user", limit=5, exclude_archived=True
        )

        # Should not find archived memory
        assert all(r.id != memory_id for r in results)

    finally:
        # Cleanup
        try:
            storage.client.delete_collection(storage.collection_name)
        except Exception:
            pass


@pytest.mark.integration
@pytest.mark.asyncio
async def test_qdrant_user_isolation(skip_if_no_qdrant):
    """Test that memories are isolated by user_id."""
    pytest.importorskip("qdrant_client")

    storage = QdrantVectorStorage(
        collection_name=f"test_collection_{uuid4().hex[:8]}", host="localhost", port=6333
    )

    await storage.initialize()

    try:
        # Add memory for user1
        memory1 = MemoryFact(
            text="User 1's secret hobby is painting",
            type="fact",
            tags=["hobby"],
            importance=0.7,
            source="user",
        )
        await storage.add(memory1, user_id="user_1")

        # Add memory for user2
        memory2 = MemoryFact(
            text="User 2's secret hobby is gardening",
            type="fact",
            tags=["hobby"],
            importance=0.7,
            source="user",
        )
        await storage.add(memory2, user_id="user_2")

        # Search as user1
        results_user1 = await storage.search(query_text="hobby", user_id="user_1", limit=5)

        # Search as user2
        results_user2 = await storage.search(query_text="hobby", user_id="user_2", limit=5)

        # Each user should only see their own memories
        assert len(results_user1) > 0
        assert all("painting" in r.text.lower() for r in results_user1)

        assert len(results_user2) > 0
        assert all("gardening" in r.text.lower() for r in results_user2)

    finally:
        # Cleanup
        try:
            storage.client.delete_collection(storage.collection_name)
        except Exception:
            pass
