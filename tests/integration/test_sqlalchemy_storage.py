"""Integration tests for SQLAlchemy conflict storage backend."""

import pytest
from sqlalchemy import create_engine

from casual_memory.models import ConflictResolution, MemoryConflict
from casual_memory.storage.conflicts.sqlalchemy import SQLAlchemyConflictStore


@pytest.mark.integration
def test_sqlalchemy_add_and_get_conflict(skip_if_no_postgres):
    """Test adding and retrieving conflicts with SQLAlchemy."""
    pytest.importorskip("sqlalchemy")

    host = skip_if_no_postgres  # Fixture returns the host

    # Create engine and storage instance
    engine = create_engine(f"postgresql://postgres:postgres@{host}:5432/test_conflicts")
    storage = SQLAlchemyConflictStore(engine=engine)
    storage.create_tables()

    try:
        # Create test conflict with all required fields
        conflict = MemoryConflict(
            user_id="test_user",
            memory_a_id="memory_a",
            memory_b_id="memory_b",
            category="location",
            similarity_score=0.92,
            avg_importance=0.85,
            clarification_hint="User mentioned different cities for residence",
        )

        # Add conflict
        conflict_id = storage.add_conflict(conflict)
        assert conflict_id is not None

        # Get conflict
        retrieved = storage.get_conflict(conflict_id)
        assert retrieved is not None
        assert retrieved.memory_a_id == "memory_a"
        assert retrieved.memory_b_id == "memory_b"
        assert retrieved.category == "location"
        assert retrieved.status == "pending"

    finally:
        # Cleanup: clear user conflicts
        try:
            storage.clear_user_conflicts("test_user")
        except Exception:
            pass


@pytest.mark.integration
def test_sqlalchemy_list_pending_conflicts(skip_if_no_postgres):
    """Test listing pending conflicts."""
    pytest.importorskip("sqlalchemy")

    host = skip_if_no_postgres

    engine = create_engine(f"postgresql://postgres:postgres@{host}:5432/test_conflicts")
    storage = SQLAlchemyConflictStore(engine=engine)
    storage.create_tables()

    try:
        # Add multiple conflicts
        for i in range(3):
            conflict = MemoryConflict(
                user_id="test_user",
                memory_a_id=f"memory_a_{i}",
                memory_b_id=f"memory_b_{i}",
                category="test",
                similarity_score=0.88,
                avg_importance=0.7,
                clarification_hint="Test conflict",
            )
            storage.add_conflict(conflict)

        # List pending conflicts
        pending = storage.get_pending_conflicts(user_id="test_user")

        assert len(pending) == 3
        assert all(c.status == "pending" for c in pending)

    finally:
        try:
            storage.clear_user_conflicts("test_user")
        except Exception:
            pass


@pytest.mark.integration
def test_sqlalchemy_resolve_conflict(skip_if_no_postgres):
    """Test resolving conflicts."""
    pytest.importorskip("sqlalchemy")

    host = skip_if_no_postgres

    engine = create_engine(f"postgresql://postgres:postgres@{host}:5432/test_conflicts")
    storage = SQLAlchemyConflictStore(engine=engine)
    storage.create_tables()

    try:
        # Add conflict
        conflict = MemoryConflict(
            user_id="test_user",
            memory_a_id="memory_a",
            memory_b_id="memory_b",
            category="job",
            similarity_score=0.91,
            avg_importance=0.9,
            clarification_hint="Different occupations mentioned",
        )

        conflict_id = storage.add_conflict(conflict)

        # Resolve conflict
        resolution = ConflictResolution(
            conflict_id=conflict_id,
            decision="keep_a",
            resolution_type="manual",
            resolved_by="test_user",
        )

        storage.resolve_conflict(conflict_id, resolution)

        # Verify resolution
        resolved = storage.get_conflict(conflict_id)
        assert resolved is not None
        assert resolved.status == "resolved"
        assert resolved.winning_memory_id == "memory_a"
        assert resolved.resolved_at is not None

    finally:
        try:
            storage.clear_user_conflicts("test_user")
        except Exception:
            pass


@pytest.mark.integration
def test_sqlalchemy_user_isolation(skip_if_no_postgres):
    """Test that conflicts are isolated by user_id."""
    pytest.importorskip("sqlalchemy")

    host = skip_if_no_postgres

    engine = create_engine(f"postgresql://postgres:postgres@{host}:5432/test_conflicts")
    storage = SQLAlchemyConflictStore(engine=engine)
    storage.create_tables()

    try:
        # Add conflict for user1
        conflict1 = MemoryConflict(
            user_id="user_1",
            memory_a_id="user1_a",
            memory_b_id="user1_b",
            category="test",
            similarity_score=0.85,
            avg_importance=0.7,
            clarification_hint="Test",
        )
        storage.add_conflict(conflict1)

        # Add conflict for user2
        conflict2 = MemoryConflict(
            user_id="user_2",
            memory_a_id="user2_a",
            memory_b_id="user2_b",
            category="test",
            similarity_score=0.85,
            avg_importance=0.7,
            clarification_hint="Test",
        )
        storage.add_conflict(conflict2)

        # List conflicts for each user
        user1_conflicts = storage.get_pending_conflicts(user_id="user_1")
        user2_conflicts = storage.get_pending_conflicts(user_id="user_2")

        # Each user should only see their own conflicts
        assert len(user1_conflicts) == 1
        assert user1_conflicts[0].memory_a_id == "user1_a"

        assert len(user2_conflicts) == 1
        assert user2_conflicts[0].memory_a_id == "user2_a"

    finally:
        try:
            storage.clear_user_conflicts("user_1")
            storage.clear_user_conflicts("user_2")
        except Exception:
            pass


@pytest.mark.integration
def test_sqlalchemy_count_conflicts(skip_if_no_postgres):
    """Test counting conflicts by status."""
    pytest.importorskip("sqlalchemy")

    host = skip_if_no_postgres

    engine = create_engine(f"postgresql://postgres:postgres@{host}:5432/test_conflicts")
    storage = SQLAlchemyConflictStore(engine=engine)
    storage.create_tables()

    try:
        # Add pending conflicts
        for i in range(3):
            conflict = MemoryConflict(
                user_id="test_user",
                memory_a_id=f"memory_a_{i}",
                memory_b_id=f"memory_b_{i}",
                category="test",
                similarity_score=0.88,
                avg_importance=0.7,
                clarification_hint="Test",
            )
            storage.add_conflict(conflict)

        # Count conflicts
        count = storage.get_conflict_count(user_id="test_user", status="pending")
        assert count == 3

    finally:
        try:
            storage.clear_user_conflicts("test_user")
        except Exception:
            pass
