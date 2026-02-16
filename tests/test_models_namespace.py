"""
Tests for namespace and entity_id support in core models.

Tests MemoryFact, MemoryConflict, and MemoryQueryFilter for:
- namespace field with default value
- entity_id as primary field
- user_id as deprecated alias (construction and property access)
- validation of namespace and entity_id
- backward compatibility with existing code using user_id
"""

import warnings

import pytest

from casual_memory.models import MemoryConflict, MemoryFact, MemoryQueryFilter


class TestMemoryFactNamespace:
    """Tests for namespace field on MemoryFact."""

    def test_default_namespace(self):
        fact = MemoryFact(text="test", type="fact", tags=[], importance=0.5)
        assert fact.namespace == "default"

    def test_custom_namespace(self):
        fact = MemoryFact(text="test", type="fact", tags=[], importance=0.5, namespace="work")
        assert fact.namespace == "work"

    def test_namespace_validation_rejects_uppercase(self):
        with pytest.raises(ValueError):
            MemoryFact(text="test", type="fact", tags=[], importance=0.5, namespace="Work")

    def test_namespace_validation_rejects_spaces(self):
        with pytest.raises(ValueError):
            MemoryFact(text="test", type="fact", tags=[], importance=0.5, namespace="my space")

    def test_namespace_validation_rejects_double_underscores(self):
        with pytest.raises(ValueError):
            MemoryFact(text="test", type="fact", tags=[], importance=0.5, namespace="my__ns")

    def test_namespace_validation_accepts_hyphens_and_underscores(self):
        fact = MemoryFact(text="test", type="fact", tags=[], importance=0.5, namespace="my-work_ns")
        assert fact.namespace == "my-work_ns"


class TestMemoryFactEntityId:
    """Tests for entity_id / user_id handling on MemoryFact."""

    def test_entity_id_as_primary(self):
        """entity_id should work without any warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fact = MemoryFact(
                text="test", type="fact", tags=[], importance=0.5, entity_id="user123"
            )
            # No deprecation warnings during construction
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 0
        assert fact.entity_id == "user123"

    def test_user_id_sets_entity_id_with_warning(self):
        """Using user_id= in constructor should set entity_id and emit warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fact = MemoryFact(text="test", type="fact", tags=[], importance=0.5, user_id="user123")
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 1
            assert "deprecated" in str(dep_warnings[0].message).lower()
        assert fact.entity_id == "user123"

    def test_user_id_property_returns_entity_id_with_warning(self):
        """Reading .user_id property should return entity_id with deprecation warning."""
        fact = MemoryFact(text="test", type="fact", tags=[], importance=0.5, entity_id="user123")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            value = fact.user_id
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 1
            assert "deprecated" in str(dep_warnings[0].message).lower()
        assert value == "user123"

    def test_both_entity_id_and_user_id_raises_error(self):
        """Providing both entity_id and user_id should raise an error."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            MemoryFact(
                text="test",
                type="fact",
                tags=[],
                importance=0.5,
                entity_id="abc",
                user_id="def",
            )

    def test_entity_id_none_by_default(self):
        """entity_id should be None when not provided."""
        fact = MemoryFact(text="test", type="fact", tags=[], importance=0.5)
        assert fact.entity_id is None

    def test_entity_id_validation(self):
        """entity_id should be validated when provided."""
        with pytest.raises(ValueError):
            MemoryFact(text="test", type="fact", tags=[], importance=0.5, entity_id="BAD VALUE")

    def test_entity_id_none_skips_validation(self):
        """None entity_id should skip validation."""
        fact = MemoryFact(text="test", type="fact", tags=[], importance=0.5, entity_id=None)
        assert fact.entity_id is None

    def test_model_dump_includes_user_id_for_compat(self):
        """model_dump() should include 'user_id' key for backward compatibility."""
        fact = MemoryFact(text="test", type="fact", tags=[], importance=0.5, entity_id="user123")
        dumped = fact.model_dump()
        assert dumped["entity_id"] == "user123"
        assert dumped["user_id"] == "user123"

    def test_model_dump_user_id_none_when_no_entity(self):
        """model_dump() user_id should be None when entity_id is None."""
        fact = MemoryFact(text="test", type="fact", tags=[], importance=0.5)
        dumped = fact.model_dump()
        assert dumped["entity_id"] is None
        assert dumped["user_id"] is None


class TestMemoryConflictNamespace:
    """Tests for namespace field on MemoryConflict."""

    def _make_conflict(self, **kwargs):
        """Helper to create a MemoryConflict with required fields."""
        defaults = {
            "entity_id": "user1",
            "memory_a_id": "mem-a",
            "memory_b_id": "mem-b",
            "category": "location",
            "similarity_score": 0.9,
            "avg_importance": 0.8,
        }
        defaults.update(kwargs)
        return MemoryConflict(**defaults)

    def test_default_namespace(self):
        conflict = self._make_conflict()
        assert conflict.namespace == "default"

    def test_custom_namespace(self):
        conflict = self._make_conflict(namespace="work")
        assert conflict.namespace == "work"

    def test_namespace_validation(self):
        with pytest.raises(ValueError):
            self._make_conflict(namespace="Bad Namespace")


class TestMemoryConflictEntityId:
    """Tests for entity_id / user_id handling on MemoryConflict."""

    def _make_conflict(self, **kwargs):
        defaults = {
            "memory_a_id": "mem-a",
            "memory_b_id": "mem-b",
            "category": "location",
            "similarity_score": 0.9,
            "avg_importance": 0.8,
        }
        defaults.update(kwargs)
        return MemoryConflict(**defaults)

    def test_entity_id_as_primary(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            conflict = self._make_conflict(entity_id="user1")
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 0
        assert conflict.entity_id == "user1"

    def test_user_id_sets_entity_id_with_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            conflict = self._make_conflict(user_id="user1")
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 1
        assert conflict.entity_id == "user1"

    def test_user_id_property_returns_entity_id(self):
        conflict = self._make_conflict(entity_id="user1")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            value = conflict.user_id
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 1
        assert value == "user1"

    def test_both_raises_error(self):
        with pytest.raises(ValueError, match="Cannot specify both"):
            self._make_conflict(entity_id="abc", user_id="def")

    def test_entity_id_required(self):
        """entity_id (or user_id) must be provided for MemoryConflict."""
        with pytest.raises(ValueError):
            self._make_conflict()  # No entity_id or user_id

    def test_entity_id_validation(self):
        with pytest.raises(ValueError):
            self._make_conflict(entity_id="BAD VALUE")

    def test_model_dump_includes_user_id(self):
        conflict = self._make_conflict(entity_id="user1")
        dumped = conflict.model_dump()
        assert dumped["entity_id"] == "user1"
        assert dumped["user_id"] == "user1"


class TestMemoryQueryFilterNamespace:
    """Tests for namespace and entity_id on MemoryQueryFilter."""

    def test_default_namespace_is_none(self):
        """MemoryQueryFilter namespace should be None by default (no filtering)."""
        f = MemoryQueryFilter()
        assert f.namespace is None

    def test_custom_namespace(self):
        f = MemoryQueryFilter(namespace="work")
        assert f.namespace == "work"

    def test_entity_id_as_primary(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            f = MemoryQueryFilter(entity_id="user123")
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 0
        assert f.entity_id == "user123"

    def test_user_id_sets_entity_id_with_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            f = MemoryQueryFilter(user_id="user123")
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 1
        assert f.entity_id == "user123"

    def test_user_id_property_returns_entity_id(self):
        f = MemoryQueryFilter(entity_id="user123")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            value = f.user_id
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 1
        assert value == "user123"

    def test_both_raises_error(self):
        with pytest.raises(ValueError, match="Cannot specify both"):
            MemoryQueryFilter(entity_id="abc", user_id="def")

    def test_model_dump_includes_user_id_for_compat(self):
        """model_dump() should include 'user_id' for storage layer backward compat."""
        f = MemoryQueryFilter(entity_id="user123")
        dumped = f.model_dump()
        assert dumped["entity_id"] == "user123"
        assert dumped["user_id"] == "user123"

    def test_model_dump_user_id_none_when_no_entity(self):
        f = MemoryQueryFilter()
        dumped = f.model_dump()
        assert dumped["entity_id"] is None
        assert dumped["user_id"] is None


class TestBackwardCompatibility:
    """Tests that existing code patterns still work."""

    def test_memory_fact_user_id_construction(self):
        """MemoryFact(user_id="abc") should work (with deprecation warning)."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            fact = MemoryFact(
                text="I live in Paris",
                type="fact",
                tags=["location"],
                importance=0.8,
                user_id="user_123",
            )
        assert fact.entity_id == "user_123"
        assert fact.namespace == "default"

    def test_memory_conflict_user_id_construction(self):
        """MemoryConflict(user_id="abc") should work (with deprecation warning)."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            conflict = MemoryConflict(
                user_id="user1",
                memory_a_id="mem-a",
                memory_b_id="mem-b",
                category="location",
                similarity_score=0.9,
                avg_importance=0.8,
            )
        assert conflict.entity_id == "user1"
        assert conflict.namespace == "default"

    def test_memory_query_filter_user_id_construction(self):
        """MemoryQueryFilter(user_id="abc") should work (with deprecation warning)."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            f = MemoryQueryFilter(user_id="user_123")
        assert f.entity_id == "user_123"

    def test_memory_fact_model_dump_roundtrip(self):
        """A MemoryFact dumped and reconstructed should preserve entity_id."""
        original = MemoryFact(text="test", type="fact", tags=[], importance=0.5, entity_id="user1")
        dumped = original.model_dump()
        # Remove user_id from dump to avoid ambiguity when reconstructing
        dumped.pop("user_id", None)
        reconstructed = MemoryFact(**dumped)
        assert reconstructed.entity_id == "user1"
        assert reconstructed.namespace == "default"
