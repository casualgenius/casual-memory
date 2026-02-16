"""
Unit tests for namespace isolation in conflict stores.

Tests that conflicts are properly isolated by namespace across both
the InMemoryConflictStore and SQLAlchemyConflictStore implementations.
"""

import warnings

import pytest
from sqlalchemy import create_engine

from casual_memory.models import ConflictResolution, MemoryConflict
from casual_memory.storage.conflicts.memory import InMemoryConflictStore
from casual_memory.storage.conflicts.sqlalchemy import SQLAlchemyConflictStore


def _make_conflict(
    conflict_id: str,
    entity_id: str,
    namespace: str = "default",
    status: str = "pending",
    avg_importance: float = 0.7,
) -> MemoryConflict:
    """Helper to create a MemoryConflict with required fields."""
    return MemoryConflict(
        id=conflict_id,
        entity_id=entity_id,
        namespace=namespace,
        memory_a_id=f"mem_a_{conflict_id}",
        memory_b_id=f"mem_b_{conflict_id}",
        category="fact",
        status=status,
        avg_importance=avg_importance,
        similarity_score=0.85,
        clarification_hint="Please clarify",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def memory_store():
    """Create a fresh InMemoryConflictStore."""
    return InMemoryConflictStore()


@pytest.fixture
def sqlalchemy_store():
    """Create a fresh SQLAlchemyConflictStore with in-memory SQLite."""
    engine = create_engine("sqlite:///:memory:")
    store = SQLAlchemyConflictStore(engine)
    store.create_tables()
    return store


# Parametrize tests to run against both store implementations.
@pytest.fixture(params=["memory", "sqlalchemy"])
def conflict_store(request, memory_store, sqlalchemy_store):
    """Parametrized fixture yielding each conflict store implementation."""
    if request.param == "memory":
        return memory_store
    return sqlalchemy_store


# ---------------------------------------------------------------------------
# Namespace isolation tests
# ---------------------------------------------------------------------------


class TestNamespaceIsolation:
    """Tests verifying that conflicts in different namespaces are isolated."""

    def test_add_conflict_with_namespace(self, conflict_store):
        """Test that adding a conflict with a namespace stores it correctly."""
        conflict = _make_conflict("c1", entity_id="user1", namespace="work")

        conflict_id = conflict_store.add_conflict(conflict)

        assert conflict_id == "c1"

        # Should be retrievable by ID regardless of namespace
        retrieved = conflict_store.get_conflict("c1")
        assert retrieved is not None
        assert retrieved.namespace == "work"
        assert retrieved.entity_id == "user1"

    def test_get_pending_conflicts_namespace_isolation(self, conflict_store):
        """Test that get_pending_conflicts only returns conflicts from the specified namespace."""
        # Same entity, different namespaces
        conflict_store.add_conflict(_make_conflict("c1", "user1", namespace="work"))
        conflict_store.add_conflict(_make_conflict("c2", "user1", namespace="personal"))
        conflict_store.add_conflict(_make_conflict("c3", "user1", namespace="work"))

        work_conflicts = conflict_store.get_pending_conflicts("user1", namespace="work")
        personal_conflicts = conflict_store.get_pending_conflicts("user1", namespace="personal")

        assert len(work_conflicts) == 2
        assert len(personal_conflicts) == 1
        assert {c.id for c in work_conflicts} == {"c1", "c3"}
        assert personal_conflicts[0].id == "c2"

    def test_get_pending_conflicts_default_namespace(self, conflict_store):
        """Test that default namespace is used when not specified."""
        conflict_store.add_conflict(_make_conflict("c1", "user1", namespace="default"))
        conflict_store.add_conflict(_make_conflict("c2", "user1", namespace="other"))

        # Should use default namespace when not specified
        default_conflicts = conflict_store.get_pending_conflicts("user1")

        assert len(default_conflicts) == 1
        assert default_conflicts[0].id == "c1"

    def test_get_conflict_count_namespace_isolation(self, conflict_store):
        """Test that get_conflict_count respects namespace boundaries."""
        conflict_store.add_conflict(_make_conflict("c1", "user1", namespace="ns1"))
        conflict_store.add_conflict(_make_conflict("c2", "user1", namespace="ns1"))
        conflict_store.add_conflict(_make_conflict("c3", "user1", namespace="ns2"))

        count_ns1 = conflict_store.get_conflict_count("user1", namespace="ns1")
        count_ns2 = conflict_store.get_conflict_count("user1", namespace="ns2")
        count_default = conflict_store.get_conflict_count("user1")  # default namespace

        assert count_ns1 == 2
        assert count_ns2 == 1
        assert count_default == 0  # no conflicts in default namespace

    def test_get_conflict_count_with_status_and_namespace(self, conflict_store):
        """Test that get_conflict_count filters by both namespace and status."""
        conflict_store.add_conflict(
            _make_conflict("c1", "user1", namespace="ns1", status="pending")
        )
        conflict_store.add_conflict(
            _make_conflict("c2", "user1", namespace="ns1", status="resolved")
        )
        conflict_store.add_conflict(
            _make_conflict("c3", "user1", namespace="ns2", status="pending")
        )

        pending_ns1 = conflict_store.get_conflict_count("user1", namespace="ns1", status="pending")
        resolved_ns1 = conflict_store.get_conflict_count(
            "user1", namespace="ns1", status="resolved"
        )
        pending_ns2 = conflict_store.get_conflict_count("user1", namespace="ns2", status="pending")

        assert pending_ns1 == 1
        assert resolved_ns1 == 1
        assert pending_ns2 == 1

    def test_clear_conflicts_namespace_isolation(self, conflict_store):
        """Test that clear_conflicts only affects the specified namespace."""
        conflict_store.add_conflict(_make_conflict("c1", "user1", namespace="ns1"))
        conflict_store.add_conflict(_make_conflict("c2", "user1", namespace="ns1"))
        conflict_store.add_conflict(_make_conflict("c3", "user1", namespace="ns2"))

        cleared = conflict_store.clear_conflicts("user1", namespace="ns1")

        assert cleared == 2
        assert conflict_store.get_conflict_count("user1", namespace="ns1") == 0
        assert conflict_store.get_conflict_count("user1", namespace="ns2") == 1

    def test_clear_conflicts_with_status_and_namespace(self, conflict_store):
        """Test clearing conflicts filtered by both namespace and status."""
        conflict_store.add_conflict(
            _make_conflict("c1", "user1", namespace="ns1", status="pending")
        )
        conflict_store.add_conflict(
            _make_conflict("c2", "user1", namespace="ns1", status="resolved")
        )
        conflict_store.add_conflict(
            _make_conflict("c3", "user1", namespace="ns2", status="pending")
        )

        cleared = conflict_store.clear_conflicts("user1", namespace="ns1", status="pending")

        assert cleared == 1
        # Resolved in ns1 should remain
        assert conflict_store.get_conflict_count("user1", namespace="ns1") == 1
        # ns2 should be unaffected
        assert conflict_store.get_conflict_count("user1", namespace="ns2") == 1

    def test_same_entity_different_namespaces_fully_isolated(self, conflict_store):
        """Test complete isolation: same entity_id across namespaces should never interfere."""
        # Add, query, clear all in separate namespaces
        conflict_store.add_conflict(_make_conflict("c1", "user1", namespace="alpha"))
        conflict_store.add_conflict(_make_conflict("c2", "user1", namespace="beta"))

        # Pending counts are isolated
        assert conflict_store.get_pending_conflicts("user1", namespace="alpha") != []
        assert conflict_store.get_pending_conflicts("user1", namespace="beta") != []

        # Clearing one namespace does not affect the other
        conflict_store.clear_conflicts("user1", namespace="alpha")

        assert conflict_store.get_conflict_count("user1", namespace="alpha") == 0
        assert conflict_store.get_conflict_count("user1", namespace="beta") == 1

    def test_different_entities_same_namespace_isolated(self, conflict_store):
        """Test that different entities within the same namespace are isolated."""
        conflict_store.add_conflict(_make_conflict("c1", "user1", namespace="shared"))
        conflict_store.add_conflict(_make_conflict("c2", "user2", namespace="shared"))

        user1_conflicts = conflict_store.get_pending_conflicts("user1", namespace="shared")
        user2_conflicts = conflict_store.get_pending_conflicts("user2", namespace="shared")

        assert len(user1_conflicts) == 1
        assert len(user2_conflicts) == 1
        assert user1_conflicts[0].id == "c1"
        assert user2_conflicts[0].id == "c2"

    def test_namespace_preserved_in_roundtrip(self, conflict_store):
        """Test that namespace is correctly preserved through add/retrieve cycle."""
        conflict = _make_conflict("c1", "user1", namespace="custom-ns")
        conflict_store.add_conflict(conflict)

        retrieved = conflict_store.get_conflict("c1")
        assert retrieved is not None
        assert retrieved.namespace == "custom-ns"
        assert retrieved.entity_id == "user1"


class TestDeprecatedClearUserConflicts:
    """Tests for the deprecated clear_user_conflicts method."""

    def test_clear_user_conflicts_deprecated_warning(self, conflict_store):
        """Test that clear_user_conflicts raises a deprecation warning."""
        conflict_store.add_conflict(_make_conflict("c1", "user1", namespace="default"))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            conflict_store.clear_user_conflicts("user1")
            assert any(issubclass(x.category, DeprecationWarning) for x in w)
            assert any("clear_user_conflicts" in str(x.message) for x in w)

    def test_clear_user_conflicts_clears_all_namespaces_memory(self, memory_store):
        """Test that deprecated clear_user_conflicts clears across all namespaces (in-memory)."""
        memory_store.add_conflict(_make_conflict("c1", "user1", namespace="ns1"))
        memory_store.add_conflict(_make_conflict("c2", "user1", namespace="ns2"))
        memory_store.add_conflict(_make_conflict("c3", "user2", namespace="ns1"))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            cleared = memory_store.clear_user_conflicts("user1")

        assert cleared == 2
        # user2 should be unaffected
        assert memory_store.get_conflict_count("user2", namespace="ns1") == 1

    def test_clear_user_conflicts_clears_all_namespaces_sqlalchemy(self, sqlalchemy_store):
        """Test that deprecated clear_user_conflicts clears across all namespaces (SQLAlchemy)."""
        sqlalchemy_store.add_conflict(_make_conflict("c1", "user1", namespace="ns1"))
        sqlalchemy_store.add_conflict(_make_conflict("c2", "user1", namespace="ns2"))
        sqlalchemy_store.add_conflict(_make_conflict("c3", "user2", namespace="ns1"))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            cleared = sqlalchemy_store.clear_user_conflicts("user1")

        assert cleared == 2
        # user2 should be unaffected
        assert sqlalchemy_store.get_conflict_count("user2", namespace="ns1") == 1


class TestNamespaceWithResolution:
    """Tests verifying namespace behavior with conflict resolution operations."""

    def test_resolve_conflict_does_not_affect_other_namespaces(self, conflict_store):
        """Test that resolving a conflict in one namespace doesn't affect others."""
        conflict_store.add_conflict(_make_conflict("c1", "user1", namespace="ns1"))
        conflict_store.add_conflict(_make_conflict("c2", "user1", namespace="ns2"))

        resolution = ConflictResolution(
            conflict_id="c1",
            decision="keep_a",
            resolution_type="manual",
            resolved_by="user",
        )
        conflict_store.resolve_conflict("c1", resolution)

        # c1 is resolved, c2 in ns2 should remain pending
        ns1_pending = conflict_store.get_pending_conflicts("user1", namespace="ns1")
        ns2_pending = conflict_store.get_pending_conflicts("user1", namespace="ns2")

        assert len(ns1_pending) == 0
        assert len(ns2_pending) == 1

    def test_escalate_conflict_does_not_affect_other_namespaces(self, conflict_store):
        """Test that escalating a conflict in one namespace doesn't affect others."""
        conflict_store.add_conflict(_make_conflict("c1", "user1", namespace="ns1"))
        conflict_store.add_conflict(_make_conflict("c2", "user1", namespace="ns2"))

        conflict_store.escalate_conflict("c1")

        # c1 should be escalated, c2 should remain pending
        c1 = conflict_store.get_conflict("c1")
        c2 = conflict_store.get_conflict("c2")

        assert c1.status == "escalated"
        assert c2.status == "pending"

    def test_get_pending_conflicts_with_limit_respects_namespace(self, conflict_store):
        """Test that limit works correctly within namespace boundaries."""
        for i in range(5):
            conflict_store.add_conflict(
                _make_conflict(
                    f"c_ns1_{i}",
                    "user1",
                    namespace="ns1",
                    avg_importance=0.5 + i * 0.1,
                )
            )
        for i in range(3):
            conflict_store.add_conflict(_make_conflict(f"c_ns2_{i}", "user1", namespace="ns2"))

        ns1_limited = conflict_store.get_pending_conflicts("user1", namespace="ns1", limit=2)
        ns2_all = conflict_store.get_pending_conflicts("user1", namespace="ns2")

        assert len(ns1_limited) == 2
        assert len(ns2_all) == 3
        # Verify sorted by importance (high to low) within ns1
        assert ns1_limited[0].avg_importance >= ns1_limited[1].avg_importance
