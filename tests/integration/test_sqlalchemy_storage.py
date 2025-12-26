"""Integration tests for SQLAlchemy conflict storage backend."""


import pytest

from casual_memory.models import MemoryConflict
from casual_memory.storage.conflicts.sqlalchemy import SQLAlchemyConflictStore


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sqlalchemy_add_and_get_conflict(skip_if_no_postgres):
    """Test adding and retrieving conflicts with SQLAlchemy."""
    pytest.importorskip("sqlalchemy")

    # Create storage instance with test database
    storage = SQLAlchemyConflictStore(
        connection_string="postgresql://postgres:postgres@localhost:5432/test_conflicts"
    )

    await storage.initialize()

    try:
        # Create test conflict
        conflict = MemoryConflict(
            memory_a_id="memory_a",
            memory_b_id="memory_b",
            memory_a_text="I live in London",
            memory_b_text="I live in Paris",
            category="location",
            hint="User mentioned different cities for residence",
            importance=0.9,
        )

        # Add conflict
        conflict_id = await storage.add(conflict, user_id="test_user")
        assert conflict_id is not None

        # Get conflict
        retrieved = await storage.get(conflict_id, user_id="test_user")
        assert retrieved is not None
        assert retrieved.memory_a_id == "memory_a"
        assert retrieved.memory_b_id == "memory_b"
        assert retrieved.category == "location"
        assert retrieved.status == "pending"

    finally:
        # Cleanup: drop test table
        try:
            await storage._drop_tables()
        except Exception:
            pass


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sqlalchemy_list_pending_conflicts(skip_if_no_postgres):
    """Test listing pending conflicts."""
    pytest.importorskip("sqlalchemy")

    storage = SQLAlchemyConflictStorage(
        connection_string="postgresql://postgres:postgres@localhost:5432/test_conflicts"
    )

    await storage.initialize()

    try:
        # Add multiple conflicts
        conflicts = [
            MemoryConflict(
                memory_a_id=f"memory_a_{i}",
                memory_b_id=f"memory_b_{i}",
                memory_a_text=f"Statement A {i}",
                memory_b_text=f"Statement B {i}",
                category="test",
                hint="Test conflict",
                importance=0.7,
            )
            for i in range(3)
        ]

        for conflict in conflicts:
            await storage.add(conflict, user_id="test_user")

        # List pending conflicts
        pending = await storage.list_pending(user_id="test_user")

        assert len(pending) == 3
        assert all(c.status == "pending" for c in pending)

    finally:
        try:
            await storage._drop_tables()
        except Exception:
            pass


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sqlalchemy_resolve_conflict(skip_if_no_postgres):
    """Test resolving conflicts."""
    pytest.importorskip("sqlalchemy")

    storage = SQLAlchemyConflictStorage(
        connection_string="postgresql://postgres:postgres@localhost:5432/test_conflicts"
    )

    await storage.initialize()

    try:
        # Add conflict
        conflict = MemoryConflict(
            memory_a_id="memory_a",
            memory_b_id="memory_b",
            memory_a_text="I work as a teacher",
            memory_b_text="I work as a doctor",
            category="job",
            hint="Different occupations mentioned",
            importance=0.9,
        )

        conflict_id = await storage.add(conflict, user_id="test_user")

        # Resolve conflict
        from casual_memory.models import ConflictResolution

        resolution = ConflictResolution(
            conflict_id=conflict_id, decision="keep_a", reason="Teacher is the current role"
        )

        await storage.resolve(resolution, user_id="test_user")

        # Verify resolution
        resolved = await storage.get(conflict_id, user_id="test_user")
        assert resolved.status == "resolved"
        assert resolved.resolution_decision == "keep_a"
        assert resolved.resolved_at is not None

    finally:
        try:
            await storage._drop_tables()
        except Exception:
            pass


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sqlalchemy_user_isolation(skip_if_no_postgres):
    """Test that conflicts are isolated by user_id."""
    pytest.importorskip("sqlalchemy")

    storage = SQLAlchemyConflictStorage(
        connection_string="postgresql://postgres:postgres@localhost:5432/test_conflicts"
    )

    await storage.initialize()

    try:
        # Add conflict for user1
        conflict1 = MemoryConflict(
            memory_a_id="user1_a",
            memory_b_id="user1_b",
            memory_a_text="User 1 conflict",
            memory_b_text="User 1 conflict",
            category="test",
            hint="Test",
            importance=0.7,
        )
        await storage.add(conflict1, user_id="user_1")

        # Add conflict for user2
        conflict2 = MemoryConflict(
            memory_a_id="user2_a",
            memory_b_id="user2_b",
            memory_a_text="User 2 conflict",
            memory_b_text="User 2 conflict",
            category="test",
            hint="Test",
            importance=0.7,
        )
        await storage.add(conflict2, user_id="user_2")

        # List conflicts for each user
        user1_conflicts = await storage.list_pending(user_id="user_1")
        user2_conflicts = await storage.list_pending(user_id="user_2")

        # Each user should only see their own conflicts
        assert len(user1_conflicts) == 1
        assert user1_conflicts[0].memory_a_id == "user1_a"

        assert len(user2_conflicts) == 1
        assert user2_conflicts[0].memory_a_id == "user2_a"

    finally:
        try:
            await storage._drop_tables()
        except Exception:
            pass


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sqlalchemy_count_conflicts(skip_if_no_postgres):
    """Test counting conflicts by status."""
    pytest.importorskip("sqlalchemy")

    storage = SQLAlchemyConflictStorage(
        connection_string="postgresql://postgres:postgres@localhost:5432/test_conflicts"
    )

    await storage.initialize()

    try:
        # Add pending conflicts
        for i in range(3):
            conflict = MemoryConflict(
                memory_a_id=f"memory_a_{i}",
                memory_b_id=f"memory_b_{i}",
                memory_a_text="Statement A",
                memory_b_text="Statement B",
                category="test",
                hint="Test",
                importance=0.7,
            )
            await storage.add(conflict, user_id="test_user")

        # Count conflicts
        count = await storage.count(user_id="test_user", status="pending")
        assert count == 3

    finally:
        try:
            await storage._drop_tables()
        except Exception:
            pass
