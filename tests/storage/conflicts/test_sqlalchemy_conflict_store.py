"""
Unit tests for SQLAlchemy conflict storage.

Tests persistence, transactions, and database operations across
different database backends (SQLite, PostgreSQL, etc.).
"""

import pytest
from sqlalchemy import create_engine

from casual_memory.models import ConflictResolution, MemoryConflict
from casual_memory.storage.conflicts.sqlalchemy import (
    SQLAlchemyConflictStore,
)


@pytest.fixture
def conflict_store():
    """Create a fresh SQLAlchemy conflict store with in-memory SQLite."""
    engine = create_engine("sqlite:///:memory:")
    store = SQLAlchemyConflictStore(engine)
    store.create_tables()
    return store


@pytest.fixture
def sample_conflict():
    """Sample conflict for testing."""
    return MemoryConflict(
        id="conflict_123",
        user_id="user1",
        memory_a_id="mem_a",
        memory_b_id="mem_b",
        category="location",
        similarity_score=0.91,
        status="pending",
        avg_importance=0.8,
        clarification_hint="Which location is correct?",
        metadata={"similarity": 0.91},
    )


def test_create_tables(conflict_store):
    """Test that database schema is created."""
    # Should not raise any errors
    assert conflict_store is not None
    # Verify tables exist by querying
    count = conflict_store.get_conflict_count("any_user")
    assert count == 0


def test_add_and_retrieve_conflict(conflict_store, sample_conflict):
    """Test adding and retrieving a conflict."""
    conflict_id = conflict_store.add_conflict(sample_conflict)

    assert conflict_id == "conflict_123"

    # Retrieve
    conflict = conflict_store.get_conflict("conflict_123")

    assert conflict is not None
    assert conflict.id == "conflict_123"
    assert conflict.user_id == "user1"
    assert conflict.category == "location"
    assert conflict.similarity_score == 0.91
    assert conflict.metadata["similarity"] == 0.91


def test_get_nonexistent_conflict(conflict_store):
    """Test retrieving a nonexistent conflict."""
    conflict = conflict_store.get_conflict("nonexistent")

    assert conflict is None


def test_get_pending_conflicts_sorted(conflict_store):
    """Test getting pending conflicts sorted by importance."""
    # Add conflicts with different importance scores
    for i in range(5):
        conflict = MemoryConflict(
            id=f"c{i}",
            user_id="user1",
            memory_a_id=f"mem_a{i}",
            memory_b_id=f"mem_b{i}",
            category="fact",
            status="pending",
            avg_importance=0.5 + i * 0.1,
            similarity_score=0.85,
            clarification_hint="Please clarify",
        )
        conflict_store.add_conflict(conflict)

    # Get pending conflicts
    pending = conflict_store.get_pending_conflicts("user1")

    assert len(pending) == 5
    # Should be sorted by importance (high to low)
    for i in range(len(pending) - 1):
        assert pending[i].avg_importance >= pending[i + 1].avg_importance


def test_get_pending_conflicts_with_limit(conflict_store):
    """Test getting pending conflicts with limit."""
    # Add multiple conflicts
    for i in range(10):
        conflict = MemoryConflict(
            id=f"c{i}",
            user_id="user1",
            memory_a_id=f"mem_a{i}",
            memory_b_id=f"mem_b{i}",
            category="fact",
            status="pending",
            avg_importance=0.7,
            similarity_score=0.85,
            clarification_hint="Please clarify",
        )
        conflict_store.add_conflict(conflict)

    # Get with limit
    pending = conflict_store.get_pending_conflicts("user1", limit=5)

    assert len(pending) == 5


def test_resolve_conflict_keep_a(conflict_store, sample_conflict):
    """Test resolving a conflict by keeping memory A."""
    conflict_store.add_conflict(sample_conflict)

    resolution = ConflictResolution(
        conflict_id="conflict_123",
        decision="keep_a",
        resolution_type="manual",
        resolved_by="user",
        notes="User chose memory A",
    )

    success = conflict_store.resolve_conflict("conflict_123", resolution)

    assert success is True

    # Verify resolution
    conflict = conflict_store.get_conflict("conflict_123")
    assert conflict.status == "resolved"
    assert conflict.winning_memory_id == "mem_a"
    assert conflict.resolution_type == "manual"
    assert conflict.resolved_at is not None
    assert conflict.metadata["resolution_decision"] == "keep_a"
    assert conflict.metadata["resolved_by"] == "user"


def test_resolve_conflict_keep_b(conflict_store, sample_conflict):
    """Test resolving a conflict by keeping memory B."""
    conflict_store.add_conflict(sample_conflict)

    resolution = ConflictResolution(
        conflict_id="conflict_123",
        decision="keep_b",
        resolution_type="automated",
        resolved_by="system",
    )

    success = conflict_store.resolve_conflict("conflict_123", resolution)

    assert success is True

    # Verify winning memory
    conflict = conflict_store.get_conflict("conflict_123")
    assert conflict.winning_memory_id == "mem_b"


def test_resolve_conflict_merge(conflict_store, sample_conflict):
    """Test resolving a conflict by merging."""
    conflict_store.add_conflict(sample_conflict)

    resolution = ConflictResolution(
        conflict_id="conflict_123",
        decision="merge",
        resolution_type="automated",
        resolved_by="system",
    )

    success = conflict_store.resolve_conflict("conflict_123", resolution)

    assert success is True

    # Verify resolution
    conflict = conflict_store.get_conflict("conflict_123")
    assert conflict.status == "resolved"
    assert conflict.winning_memory_id is None  # No single winner for merge


def test_resolve_nonexistent_conflict(conflict_store):
    """Test resolving a nonexistent conflict."""
    resolution = ConflictResolution(
        conflict_id="nonexistent",
        decision="keep_a",
        resolution_type="manual",
        resolved_by="user",
    )

    success = conflict_store.resolve_conflict("nonexistent", resolution)

    assert success is False


def test_get_conflict_count(conflict_store):
    """Test counting conflicts."""
    # Add conflicts with different statuses
    for i, status in enumerate(["pending", "pending", "resolved", "escalated", "pending"]):
        conflict = MemoryConflict(
            id=f"c{i}",
            user_id="user1",
            memory_a_id=f"mem_a{i}",
            memory_b_id=f"mem_b{i}",
            category="fact",
            status=status,
            avg_importance=0.7,
            similarity_score=0.85,
            clarification_hint="Please clarify",
        )
        conflict_store.add_conflict(conflict)

    # Count all
    total = conflict_store.get_conflict_count("user1")
    assert total == 5

    # Count by status
    pending = conflict_store.get_conflict_count("user1", status="pending")
    assert pending == 3

    resolved = conflict_store.get_conflict_count("user1", status="resolved")
    assert resolved == 1


def test_escalate_conflict(conflict_store, sample_conflict):
    """Test escalating a conflict."""
    conflict_store.add_conflict(sample_conflict)

    success = conflict_store.escalate_conflict("conflict_123")

    assert success is True

    # Verify escalation
    conflict = conflict_store.get_conflict("conflict_123")
    assert conflict.status == "escalated"
    assert conflict.resolution_attempts == 1

    # Escalate again
    conflict_store.escalate_conflict("conflict_123")
    conflict = conflict_store.get_conflict("conflict_123")
    assert conflict.resolution_attempts == 2


def test_escalate_nonexistent_conflict(conflict_store):
    """Test escalating a nonexistent conflict."""
    success = conflict_store.escalate_conflict("nonexistent")

    assert success is False


def test_clear_user_conflicts_all(conflict_store):
    """Test clearing all conflicts for a user."""
    # Add conflicts for different users
    for i in range(3):
        conflict = MemoryConflict(
            id=f"c1_{i}",
            user_id="user1",
            memory_a_id=f"mem_a{i}",
            memory_b_id=f"mem_b{i}",
            category="fact",
            status="pending",
            avg_importance=0.7,
            similarity_score=0.85,
            clarification_hint="Please clarify",
        )
        conflict_store.add_conflict(conflict)

    conflict_user2 = MemoryConflict(
        id="c2_1",
        user_id="user2",
        memory_a_id="mem_a",
        memory_b_id="mem_b",
        category="fact",
        status="pending",
        avg_importance=0.7,
        similarity_score=0.85,
        clarification_hint="Please clarify",
    )
    conflict_store.add_conflict(conflict_user2)

    # Clear user1's conflicts
    count = conflict_store.clear_user_conflicts("user1")

    assert count == 3

    # Verify user1 has no conflicts
    assert conflict_store.get_conflict_count("user1") == 0

    # Verify user2 still has conflicts
    assert conflict_store.get_conflict_count("user2") == 1


def test_clear_user_conflicts_by_status(conflict_store):
    """Test clearing conflicts by status."""
    # Add conflicts with different statuses
    pending = MemoryConflict(
        id="c1",
        user_id="user1",
        memory_a_id="mem_a",
        memory_b_id="mem_b",
        category="fact",
        status="pending",
        avg_importance=0.7,
        similarity_score=0.85,
        clarification_hint="Please clarify",
    )

    resolved = MemoryConflict(
        id="c2",
        user_id="user1",
        memory_a_id="mem_c",
        memory_b_id="mem_d",
        category="fact",
        status="resolved",
        avg_importance=0.7,
        similarity_score=0.85,
        clarification_hint="Please clarify",
    )

    conflict_store.add_conflict(pending)
    conflict_store.add_conflict(resolved)

    # Clear only pending conflicts
    count = conflict_store.clear_user_conflicts("user1", status="pending")

    assert count == 1

    # Verify only resolved conflict remains
    assert conflict_store.get_conflict_count("user1", status="pending") == 0
    assert conflict_store.get_conflict_count("user1", status="resolved") == 1


def test_user_isolation(conflict_store):
    """Test that conflicts are isolated per user."""
    conflict_user1 = MemoryConflict(
        id="c1",
        user_id="user1",
        memory_a_id="mem_a",
        memory_b_id="mem_b",
        category="fact",
        status="pending",
        avg_importance=0.7,
        similarity_score=0.85,
        clarification_hint="Please clarify",
    )

    conflict_user2 = MemoryConflict(
        id="c2",
        user_id="user2",
        memory_a_id="mem_c",
        memory_b_id="mem_d",
        category="fact",
        status="pending",
        avg_importance=0.7,
        similarity_score=0.85,
        clarification_hint="Please clarify",
    )

    conflict_store.add_conflict(conflict_user1)
    conflict_store.add_conflict(conflict_user2)

    # Each user should only see their own conflicts
    user1_conflicts = conflict_store.get_pending_conflicts("user1")
    user2_conflicts = conflict_store.get_pending_conflicts("user2")

    assert len(user1_conflicts) == 1
    assert len(user2_conflicts) == 1
    assert user1_conflicts[0].id == "c1"
    assert user2_conflicts[0].id == "c2"


def test_metadata_json_serialization(conflict_store):
    """Test that complex metadata is properly serialized/deserialized."""
    conflict = MemoryConflict(
        id="c1",
        user_id="user1",
        memory_a_id="mem_a",
        memory_b_id="mem_b",
        category="fact",
        status="pending",
        avg_importance=0.7,
        metadata={
            "similarity": 0.91,
            "nested": {"key": "value"},
            "list": [1, 2, 3],
        },
        similarity_score=0.85,
        clarification_hint="Please clarify",
    )

    conflict_store.add_conflict(conflict)

    # Retrieve and verify metadata
    retrieved = conflict_store.get_conflict("c1")

    assert retrieved.metadata["similarity"] == 0.91
    assert retrieved.metadata["nested"]["key"] == "value"
    assert retrieved.metadata["list"] == [1, 2, 3]


def test_transaction_rollback_on_error(conflict_store):
    """Test that transactions rollback on error."""
    # Add a conflict
    conflict = MemoryConflict(
        id="c1",
        user_id="user1",
        memory_a_id="mem_a",
        memory_b_id="mem_b",
        category="fact",
        status="pending",
        avg_importance=0.7,
        similarity_score=0.85,
        clarification_hint="Please clarify",
    )
    conflict_store.add_conflict(conflict)

    # Verify it was added
    assert conflict_store.get_conflict_count("user1") == 1

    # Try to add duplicate (should fail due to primary key constraint)
    try:
        conflict_store.add_conflict(conflict)
    except Exception:
        pass  # Expected to fail

    # Verify count is still 1 (transaction rolled back)
    assert conflict_store.get_conflict_count("user1") == 1
