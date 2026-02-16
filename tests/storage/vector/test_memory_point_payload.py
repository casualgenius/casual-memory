"""
Tests for MemoryPointPayload namespace and entity_id support.

Tests that MemoryPointPayload (the single canonical source in storage/vector/models.py):
- Has namespace field with default value
- Has entity_id as primary field
- Handles user_id as deprecated alias (construction and property access)
- Validates namespace and entity_id
- Provides backward-compatible model_dump with user_id key
- Roundtrips cleanly through model_dump/reconstruct
"""

import warnings

import pytest

from casual_memory.storage.vector.models import MemoryPoint, MemoryPointPayload


class TestMemoryPointPayloadNamespace:
    """Tests for namespace field on MemoryPointPayload."""

    def _make_payload(self, **kwargs):
        defaults = {"text": "I live in London", "timestamp": "2025-01-01T00:00:00"}
        defaults.update(kwargs)
        return MemoryPointPayload(**defaults)

    def test_default_namespace(self):
        payload = self._make_payload()
        assert payload.namespace == "default"

    def test_custom_namespace(self):
        payload = self._make_payload(namespace="work")
        assert payload.namespace == "work"

    def test_namespace_validation_rejects_uppercase(self):
        with pytest.raises(ValueError):
            self._make_payload(namespace="Work")

    def test_namespace_validation_rejects_spaces(self):
        with pytest.raises(ValueError):
            self._make_payload(namespace="my space")

    def test_namespace_validation_rejects_double_underscores(self):
        with pytest.raises(ValueError):
            self._make_payload(namespace="my__ns")

    def test_namespace_validation_accepts_hyphens_and_underscores(self):
        payload = self._make_payload(namespace="my-work_ns")
        assert payload.namespace == "my-work_ns"

    def test_namespace_in_model_dump(self):
        payload = self._make_payload(namespace="work")
        dumped = payload.model_dump()
        assert dumped["namespace"] == "work"


class TestMemoryPointPayloadEntityId:
    """Tests for entity_id / user_id handling on MemoryPointPayload."""

    def _make_payload(self, **kwargs):
        defaults = {"text": "I live in London", "timestamp": "2025-01-01T00:00:00"}
        defaults.update(kwargs)
        return MemoryPointPayload(**defaults)

    def test_entity_id_as_primary(self):
        """entity_id should work without any warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            payload = self._make_payload(entity_id="user123")
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 0
        assert payload.entity_id == "user123"

    def test_entity_id_none_by_default(self):
        payload = self._make_payload()
        assert payload.entity_id is None

    def test_entity_id_validation(self):
        with pytest.raises(ValueError):
            self._make_payload(entity_id="BAD VALUE")

    def test_entity_id_none_skips_validation(self):
        payload = self._make_payload(entity_id=None)
        assert payload.entity_id is None

    def test_user_id_sets_entity_id_with_warning(self):
        """Using user_id= in constructor should set entity_id and emit warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            payload = self._make_payload(user_id="user123")
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 1
            assert "deprecated" in str(dep_warnings[0].message).lower()
        assert payload.entity_id == "user123"

    def test_user_id_property_returns_entity_id_with_warning(self):
        """Reading .user_id property should return entity_id with deprecation warning."""
        payload = self._make_payload(entity_id="user123")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            value = payload.user_id
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 1
            assert "deprecated" in str(dep_warnings[0].message).lower()
        assert value == "user123"

    def test_both_entity_id_and_user_id_same_value_no_error(self):
        """If both have the same value, no error should occur."""
        payload = self._make_payload(entity_id="abc", user_id="abc")
        assert payload.entity_id == "abc"

    def test_both_entity_id_and_user_id_different_raises(self):
        """Different non-None values should raise."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            self._make_payload(entity_id="abc", user_id="def")

    def test_user_id_none_entity_id_set_no_error(self):
        """user_id=None with entity_id set should resolve to entity_id."""
        payload = self._make_payload(entity_id="abc", user_id=None)
        assert payload.entity_id == "abc"


class TestMemoryPointPayloadModelDump:
    """Tests for model_dump backward compatibility."""

    def _make_payload(self, **kwargs):
        defaults = {"text": "I live in London", "timestamp": "2025-01-01T00:00:00"}
        defaults.update(kwargs)
        return MemoryPointPayload(**defaults)

    def test_model_dump_includes_user_id_for_compat(self):
        payload = self._make_payload(entity_id="user123")
        dumped = payload.model_dump()
        assert dumped["entity_id"] == "user123"
        assert dumped["user_id"] == "user123"

    def test_model_dump_user_id_none_when_no_entity(self):
        payload = self._make_payload()
        dumped = payload.model_dump()
        assert dumped["entity_id"] is None
        assert dumped["user_id"] is None

    def test_model_dump_exclude_none_omits_user_id_when_none(self):
        payload = self._make_payload()
        dumped = payload.model_dump(exclude_none=True)
        assert "user_id" not in dumped
        assert "entity_id" not in dumped

    def test_model_dump_exclude_none_includes_user_id_when_set(self):
        payload = self._make_payload(entity_id="user1")
        dumped = payload.model_dump(exclude_none=True)
        assert dumped["user_id"] == "user1"
        assert dumped["entity_id"] == "user1"

    def test_model_dump_includes_namespace(self):
        payload = self._make_payload(namespace="work", entity_id="user1")
        dumped = payload.model_dump()
        assert dumped["namespace"] == "work"
        assert dumped["entity_id"] == "user1"
        assert dumped["user_id"] == "user1"


class TestMemoryPointPayloadRoundtrip:
    """Tests that model_dump/reconstruct roundtrips work."""

    def _make_payload(self, **kwargs):
        defaults = {"text": "I live in London", "timestamp": "2025-01-01T00:00:00"}
        defaults.update(kwargs)
        return MemoryPointPayload(**defaults)

    def test_roundtrip_with_entity_id(self):
        """A payload dumped and reconstructed should preserve entity_id."""
        original = self._make_payload(entity_id="user1", namespace="work")
        dumped = original.model_dump()
        # Both user_id and entity_id present with same value
        assert "user_id" in dumped
        assert "entity_id" in dumped
        reconstructed = MemoryPointPayload(**dumped)
        assert reconstructed.entity_id == "user1"
        assert reconstructed.namespace == "work"

    def test_roundtrip_with_none_entity_id(self):
        """Roundtrip when entity_id is None should work."""
        original = self._make_payload()
        dumped = original.model_dump()
        assert dumped["entity_id"] is None
        assert dumped["user_id"] is None
        reconstructed = MemoryPointPayload(**dumped)
        assert reconstructed.entity_id is None
        assert reconstructed.namespace == "default"

    def test_roundtrip_preserves_all_fields(self):
        """All fields should survive a roundtrip."""
        original = self._make_payload(
            entity_id="user1",
            namespace="work",
            tags=["location"],
            importance=0.8,
            type="fact",
            source="user",
            confidence=0.9,
            mention_count=3,
            first_seen="2025-01-01T00:00:00",
            last_seen="2025-06-01T00:00:00",
        )
        dumped = original.model_dump()
        reconstructed = MemoryPointPayload(**dumped)
        assert reconstructed.entity_id == "user1"
        assert reconstructed.namespace == "work"
        assert reconstructed.tags == ["location"]
        assert reconstructed.importance == 0.8
        assert reconstructed.type == "fact"
        assert reconstructed.source == "user"
        assert reconstructed.confidence == 0.9
        assert reconstructed.mention_count == 3


class TestMemoryPointPayloadConsolidation:
    """Tests that only one MemoryPointPayload exists (consolidation verification)."""

    def test_no_memory_point_payload_in_models(self):
        """MemoryPointPayload should NOT be importable from casual_memory.models."""
        import casual_memory.models as models_module

        assert not hasattr(models_module, "MemoryPointPayload"), (
            "MemoryPointPayload should be removed from casual_memory.models"
        )

    def test_no_memory_point_in_models(self):
        """MemoryPoint should NOT be importable from casual_memory.models."""
        import casual_memory.models as models_module

        assert not hasattr(models_module, "MemoryPoint"), (
            "MemoryPoint should be removed from casual_memory.models"
        )

    def test_memory_point_payload_in_vector_models(self):
        """MemoryPointPayload should be importable from storage.vector.models."""
        from casual_memory.storage.vector.models import MemoryPointPayload as MPP

        assert MPP is not None

    def test_memory_point_in_vector_models(self):
        """MemoryPoint should be importable from storage.vector.models."""
        from casual_memory.storage.vector.models import MemoryPoint as MP

        assert MP is not None


class TestMemoryPointPayloadLegacyUserIdPayloads:
    """Tests that existing payloads with user_id (from storage) are handled correctly."""

    def test_legacy_payload_with_user_id(self):
        """Payloads from storage that have user_id should be converted to entity_id."""
        # Simulates what happens when reading from vector store
        legacy_data = {
            "text": "I live in London",
            "timestamp": "2025-01-01T00:00:00",
            "user_id": "user123",
            "type": "fact",
            "tags": ["location"],
            "importance": 0.8,
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            payload = MemoryPointPayload(**legacy_data)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 1
        assert payload.entity_id == "user123"
        assert payload.namespace == "default"

    def test_legacy_payload_without_namespace(self):
        """Payloads from storage without namespace should get default."""
        legacy_data = {
            "text": "I live in London",
            "timestamp": "2025-01-01T00:00:00",
            "user_id": "user123",
        }
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            payload = MemoryPointPayload(**legacy_data)
        assert payload.namespace == "default"

    def test_memory_point_wraps_payload(self):
        """MemoryPoint should correctly wrap a MemoryPointPayload with new fields."""
        payload = MemoryPointPayload(
            text="test",
            timestamp="2025-01-01T00:00:00",
            entity_id="user1",
            namespace="work",
        )
        point = MemoryPoint(id="test-id", vector=[1.0, 0.0], payload=payload)
        assert point.payload.entity_id == "user1"
        assert point.payload.namespace == "work"
