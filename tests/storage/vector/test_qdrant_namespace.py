"""
Unit tests for Qdrant vector storage namespace support.

Tests namespace filtering logic, entity_id filtering, backward compatibility
with old data, and the clear_memories / clear_user_memories methods.
All tests use a mocked Qdrant client (no running Qdrant instance required).
"""

from datetime import datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from qdrant_client.models import (
    FieldCondition,
    Filter,
    IsNullCondition,
    MatchValue,
    PayloadField,
)

from casual_memory.storage.vector.qdrant import (
    QdrantMemoryStore,
    _build_entity_id_filter,
    _build_namespace_filter,
)

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_payload(
    text: str = "test memory",
    entity_id: str = "user1",
    namespace: str = "default",
    **extra: Any,
) -> dict[str, Any]:
    """Create a minimal payload dict."""
    payload: dict[str, Any] = {
        "text": text,
        "type": "fact",
        "tags": [],
        "importance": 0.7,
        "timestamp": datetime.now().isoformat(),
        "entity_id": entity_id,
        "namespace": namespace,
        "confidence": 0.5,
        "mention_count": 1,
        "archived": False,
    }
    payload.update(extra)
    return payload


def _make_scored_point(
    point_id: str = "point-1",
    score: float = 0.95,
    payload: dict[str, Any] | None = None,
    vector: list[float] | None = None,
) -> SimpleNamespace:
    """Create a fake scored point (mimics qdrant_client response)."""
    if payload is None:
        payload = _make_payload()
    if vector is None:
        vector = [0.1, 0.2, 0.3]
    return SimpleNamespace(id=point_id, score=score, payload=payload, vector=vector)


def _make_record(
    point_id: str = "point-1",
    payload: dict[str, Any] | None = None,
) -> SimpleNamespace:
    """Create a fake scroll record (no score, no vector)."""
    if payload is None:
        payload = _make_payload()
    return SimpleNamespace(id=point_id, payload=payload)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client() -> MagicMock:
    """Create a mock QdrantClient."""
    client = MagicMock()
    client.collection_exists.return_value = True  # Skip collection creation
    return client


@pytest.fixture
def store(mock_client: MagicMock) -> QdrantMemoryStore:
    """Create a QdrantMemoryStore with a mocked client."""
    with patch("casual_memory.storage.vector.qdrant.QdrantClient", return_value=mock_client):
        s = QdrantMemoryStore(host="localhost", port=6333, collection_name="test")
    return s


# ---------------------------------------------------------------------------
# Tests: _build_namespace_filter helper
# ---------------------------------------------------------------------------


class TestBuildNamespaceFilter:
    """Tests for the _build_namespace_filter helper function."""

    def test_default_namespace_uses_should_with_null(self) -> None:
        """Default namespace filter uses OR (should) with null check."""
        result = _build_namespace_filter("default")
        assert result is not None
        assert result.should is not None
        assert len(result.should) == 2

        # First condition: namespace == "default"
        first = result.should[0]
        assert isinstance(first, FieldCondition)
        assert first.key == "namespace"
        assert isinstance(first.match, MatchValue)
        assert first.match.value == "default"

        # Second condition: namespace is null
        second = result.should[1]
        assert isinstance(second, IsNullCondition)
        assert second.is_null == PayloadField(key="namespace")

    def test_non_default_namespace_uses_strict_match(self) -> None:
        """Non-default namespace filter uses strict must match."""
        result = _build_namespace_filter("work")
        assert result is not None
        assert result.must is not None
        assert len(result.must) == 1
        cond = result.must[0]
        assert isinstance(cond, FieldCondition)
        assert cond.key == "namespace"
        assert isinstance(cond.match, MatchValue)
        assert cond.match.value == "work"


class TestBuildEntityIdFilter:
    """Tests for the _build_entity_id_filter helper function."""

    def test_entity_id_filter_matches_both_fields(self) -> None:
        """Entity filter uses OR to match entity_id or user_id."""
        result = _build_entity_id_filter("user1")
        assert result.should is not None
        assert len(result.should) == 2

        keys = {c.key for c in result.should}
        assert keys == {"entity_id", "user_id"}

        for cond in result.should:
            assert isinstance(cond.match, MatchValue)
            assert cond.match.value == "user1"


# ---------------------------------------------------------------------------
# Tests: find_similar_memories
# ---------------------------------------------------------------------------


class TestFindSimilarMemories:
    """Tests for find_similar_memories with namespace and entity_id."""

    def test_filters_by_entity_id_and_default_namespace(
        self, store: QdrantMemoryStore, mock_client: MagicMock
    ) -> None:
        """find_similar_memories includes entity_id and namespace filters."""
        mock_client.query_points.return_value = SimpleNamespace(points=[])

        store.find_similar_memories(
            embedding=[0.1, 0.2, 0.3],
            entity_id="user1",
            namespace="default",
            threshold=0.8,
        )

        call_args = mock_client.query_points.call_args
        query_filter = call_args.kwargs["query_filter"]
        assert query_filter is not None
        assert query_filter.must is not None
        # Should have entity_id filter + namespace filter
        assert len(query_filter.must) == 2

    def test_filters_by_non_default_namespace(
        self, store: QdrantMemoryStore, mock_client: MagicMock
    ) -> None:
        """Non-default namespace uses strict match."""
        mock_client.query_points.return_value = SimpleNamespace(points=[])

        store.find_similar_memories(
            embedding=[0.1, 0.2, 0.3],
            entity_id="user1",
            namespace="work",
            threshold=0.8,
        )

        call_args = mock_client.query_points.call_args
        query_filter = call_args.kwargs["query_filter"]
        assert query_filter is not None
        # Both entity_id filter and namespace filter
        assert len(query_filter.must) == 2
        # The namespace sub-filter should be a strict must
        ns_filter = query_filter.must[1]
        assert isinstance(ns_filter, Filter)
        assert ns_filter.must is not None
        assert len(ns_filter.must) == 1
        assert ns_filter.must[0].key == "namespace"
        assert ns_filter.must[0].match.value == "work"

    def test_default_namespace_backward_compat(
        self, store: QdrantMemoryStore, mock_client: MagicMock
    ) -> None:
        """Default namespace filter should allow null namespace fields."""
        mock_client.query_points.return_value = SimpleNamespace(points=[])

        store.find_similar_memories(
            embedding=[0.1, 0.2, 0.3],
            entity_id="user1",
        )

        call_args = mock_client.query_points.call_args
        query_filter = call_args.kwargs["query_filter"]
        # The namespace filter (second must condition) should use should (OR)
        ns_filter = query_filter.must[1]
        assert isinstance(ns_filter, Filter)
        assert ns_filter.should is not None
        assert len(ns_filter.should) == 2

    def test_no_entity_id_only_namespace(
        self, store: QdrantMemoryStore, mock_client: MagicMock
    ) -> None:
        """When entity_id is None, only namespace filter is applied."""
        mock_client.query_points.return_value = SimpleNamespace(points=[])

        store.find_similar_memories(
            embedding=[0.1, 0.2, 0.3],
            namespace="work",
            threshold=0.8,
        )

        call_args = mock_client.query_points.call_args
        query_filter = call_args.kwargs["query_filter"]
        assert query_filter is not None
        # Only the namespace filter
        assert len(query_filter.must) == 1

    def test_deprecated_user_id_kwarg(
        self, store: QdrantMemoryStore, mock_client: MagicMock
    ) -> None:
        """Deprecated user_id kwarg maps to entity_id with warning."""
        mock_client.query_points.return_value = SimpleNamespace(points=[])

        with pytest.warns(DeprecationWarning, match="user_id.*deprecated"):
            store.find_similar_memories(
                embedding=[0.1, 0.2, 0.3],
                user_id="user1",
                threshold=0.8,
            )

        call_args = mock_client.query_points.call_args
        query_filter = call_args.kwargs["query_filter"]
        assert query_filter is not None
        # Should still have entity_id filter + namespace filter
        assert len(query_filter.must) == 2

    def test_results_returned_correctly(
        self, store: QdrantMemoryStore, mock_client: MagicMock
    ) -> None:
        """Results are converted to (MemoryPoint, score) tuples."""
        payload = _make_payload(text="I live in Paris", entity_id="user1", namespace="default")
        point = _make_scored_point(point_id="p1", score=0.92, payload=payload)
        mock_client.query_points.return_value = SimpleNamespace(points=[point])

        results = store.find_similar_memories(
            embedding=[0.1, 0.2, 0.3],
            entity_id="user1",
            threshold=0.8,
        )

        assert len(results) == 1
        memory_point, score = results[0]
        assert memory_point.id == "p1"
        assert memory_point.payload.text == "I live in Paris"
        assert memory_point.payload.entity_id == "user1"
        assert memory_point.payload.namespace == "default"
        assert score == 0.92

    def test_archived_excluded_by_default(
        self, store: QdrantMemoryStore, mock_client: MagicMock
    ) -> None:
        """Archived memories are excluded in post-processing by default."""
        payload = _make_payload(archived=True)
        point = _make_scored_point(payload=payload)
        mock_client.query_points.return_value = SimpleNamespace(points=[point])

        results = store.find_similar_memories(
            embedding=[0.1, 0.2, 0.3],
            entity_id="user1",
            threshold=0.5,
        )

        assert len(results) == 0

    def test_archived_included_when_not_excluded(
        self, store: QdrantMemoryStore, mock_client: MagicMock
    ) -> None:
        """Archived memories included when exclude_archived=False."""
        payload = _make_payload(archived=True)
        point = _make_scored_point(payload=payload)
        mock_client.query_points.return_value = SimpleNamespace(points=[point])

        results = store.find_similar_memories(
            embedding=[0.1, 0.2, 0.3],
            entity_id="user1",
            threshold=0.5,
            exclude_archived=False,
        )

        assert len(results) == 1

    def test_old_payload_with_user_id_only(
        self, store: QdrantMemoryStore, mock_client: MagicMock
    ) -> None:
        """Old payloads with user_id (no entity_id) are handled via MemoryPointPayload migration."""
        old_payload = {
            "text": "Old memory",
            "type": "fact",
            "tags": [],
            "importance": 0.7,
            "timestamp": datetime.now().isoformat(),
            "user_id": "user1",
            "confidence": 0.5,
            "mention_count": 1,
            "archived": False,
        }
        point = _make_scored_point(payload=old_payload)
        mock_client.query_points.return_value = SimpleNamespace(points=[point])

        results = store.find_similar_memories(
            embedding=[0.1, 0.2, 0.3],
            entity_id="user1",
            threshold=0.5,
        )

        assert len(results) == 1
        # MemoryPointPayload model_validator migrates user_id -> entity_id
        assert results[0][0].payload.entity_id == "user1"
        # Namespace defaults to "default"
        assert results[0][0].payload.namespace == "default"


# ---------------------------------------------------------------------------
# Tests: search
# ---------------------------------------------------------------------------


class TestSearch:
    """Tests for search() with namespace and entity_id filter keys."""

    def test_search_with_entity_id_filter(
        self, store: QdrantMemoryStore, mock_client: MagicMock
    ) -> None:
        """search() handles entity_id filter key."""
        mock_client.query_points.return_value = SimpleNamespace(points=[])

        store.search(
            query_embedding=[0.1, 0.2],
            filters={"entity_id": "user1", "type": None, "min_importance": None},
        )

        call_args = mock_client.query_points.call_args
        query_filter = call_args.kwargs["query_filter"]
        assert query_filter is not None
        assert len(query_filter.must) == 1
        # The entity_id filter should be a Filter with should conditions
        entity_filter = query_filter.must[0]
        assert isinstance(entity_filter, Filter)
        assert entity_filter.should is not None

    def test_search_with_namespace_filter(
        self, store: QdrantMemoryStore, mock_client: MagicMock
    ) -> None:
        """search() handles namespace filter key."""
        mock_client.query_points.return_value = SimpleNamespace(points=[])

        store.search(
            query_embedding=[0.1, 0.2],
            filters={
                "entity_id": "user1",
                "namespace": "work",
                "type": None,
                "min_importance": None,
            },
        )

        call_args = mock_client.query_points.call_args
        query_filter = call_args.kwargs["query_filter"]
        assert query_filter is not None
        # entity_id + namespace
        assert len(query_filter.must) == 2

    def test_search_with_deprecated_user_id_filter(
        self, store: QdrantMemoryStore, mock_client: MagicMock
    ) -> None:
        """search() handles deprecated user_id filter key with warning."""
        mock_client.query_points.return_value = SimpleNamespace(points=[])

        with pytest.warns(DeprecationWarning, match="user_id.*deprecated"):
            store.search(
                query_embedding=[0.1, 0.2],
                filters={"user_id": "user1", "type": None, "min_importance": None},
            )

        call_args = mock_client.query_points.call_args
        query_filter = call_args.kwargs["query_filter"]
        assert query_filter is not None

    def test_search_entity_id_takes_precedence_over_user_id(
        self, store: QdrantMemoryStore, mock_client: MagicMock
    ) -> None:
        """When both entity_id and user_id filters present, entity_id wins (no warning)."""
        mock_client.query_points.return_value = SimpleNamespace(points=[])

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            store.search(
                query_embedding=[0.1, 0.2],
                filters={
                    "entity_id": "user1",
                    "user_id": "user1",
                    "type": None,
                    "min_importance": None,
                    "namespace": "default",
                },
            )

    def test_search_with_default_namespace_uses_or_filter(
        self, store: QdrantMemoryStore, mock_client: MagicMock
    ) -> None:
        """Default namespace uses OR filter with null check."""
        mock_client.query_points.return_value = SimpleNamespace(points=[])

        store.search(
            query_embedding=[0.1, 0.2],
            filters={
                "entity_id": "user1",
                "namespace": "default",
                "type": None,
                "min_importance": None,
            },
        )

        call_args = mock_client.query_points.call_args
        query_filter = call_args.kwargs["query_filter"]
        # Find the namespace sub-filter
        ns_filter = query_filter.must[1]
        assert isinstance(ns_filter, Filter)
        assert ns_filter.should is not None
        assert len(ns_filter.should) == 2

    def test_search_no_filters(self, store: QdrantMemoryStore, mock_client: MagicMock) -> None:
        """search() with no filters passes None query_filter."""
        mock_client.query_points.return_value = SimpleNamespace(points=[])

        store.search(query_embedding=[0.1, 0.2])

        call_args = mock_client.query_points.call_args
        assert call_args.kwargs["query_filter"] is None

    def test_search_returns_results(self, store: QdrantMemoryStore, mock_client: MagicMock) -> None:
        """search() returns MemoryPoint list from query results."""
        payload = _make_payload(text="Found memory")
        point = _make_scored_point(payload=payload, score=0.95)
        mock_client.query_points.return_value = SimpleNamespace(points=[point])

        results = store.search(query_embedding=[0.1, 0.2], min_score=0.5)

        assert len(results) == 1
        assert results[0].payload.text == "Found memory"


# ---------------------------------------------------------------------------
# Tests: clear_memories
# ---------------------------------------------------------------------------


class TestClearMemories:
    """Tests for clear_memories with namespace support."""

    def test_clear_memories_filters_by_entity_and_namespace(
        self, store: QdrantMemoryStore, mock_client: MagicMock
    ) -> None:
        """clear_memories builds filter with entity_id and namespace."""
        mock_client.scroll.return_value = (
            [_make_record("p1"), _make_record("p2")],
            None,
        )

        count = store.clear_memories("user1", namespace="default")

        assert count == 2
        mock_client.delete.assert_called_once()

        # Verify the scroll filter was built correctly
        scroll_call = mock_client.scroll.call_args
        scroll_filter = scroll_call.kwargs["scroll_filter"]
        assert scroll_filter.must is not None
        assert len(scroll_filter.must) == 2

    def test_clear_memories_non_default_namespace(
        self, store: QdrantMemoryStore, mock_client: MagicMock
    ) -> None:
        """clear_memories with non-default namespace uses strict filter."""
        mock_client.scroll.return_value = ([_make_record("p1")], None)

        count = store.clear_memories("user1", namespace="work")

        assert count == 1
        scroll_call = mock_client.scroll.call_args
        scroll_filter = scroll_call.kwargs["scroll_filter"]
        # Second must condition is the namespace filter (strict)
        ns_filter = scroll_filter.must[1]
        assert isinstance(ns_filter, Filter)
        assert ns_filter.must is not None
        assert ns_filter.must[0].match.value == "work"

    def test_clear_memories_no_points_found(
        self, store: QdrantMemoryStore, mock_client: MagicMock
    ) -> None:
        """clear_memories returns 0 and does not call delete when no points found."""
        mock_client.scroll.return_value = ([], None)

        count = store.clear_memories("user1")

        assert count == 0
        mock_client.delete.assert_not_called()

    def test_clear_memories_error_propagated(
        self, store: QdrantMemoryStore, mock_client: MagicMock
    ) -> None:
        """clear_memories propagates exceptions."""
        mock_client.scroll.side_effect = RuntimeError("Connection failed")

        with pytest.raises(RuntimeError, match="Connection failed"):
            store.clear_memories("user1")


# ---------------------------------------------------------------------------
# Tests: clear_user_memories (deprecated)
# ---------------------------------------------------------------------------


class TestClearUserMemories:
    """Tests for deprecated clear_user_memories."""

    def test_emits_deprecation_warning(
        self, store: QdrantMemoryStore, mock_client: MagicMock
    ) -> None:
        """clear_user_memories emits a deprecation warning."""
        mock_client.scroll.return_value = ([], None)

        with pytest.warns(DeprecationWarning, match="clear_user_memories.*deprecated"):
            store.clear_user_memories("user1")

    def test_uses_entity_id_filter(self, store: QdrantMemoryStore, mock_client: MagicMock) -> None:
        """clear_user_memories uses _build_entity_id_filter for backward compat."""
        mock_client.scroll.return_value = ([_make_record("p1")], None)

        with pytest.warns(DeprecationWarning):
            count = store.clear_user_memories("user1")

        assert count == 1
        scroll_call = mock_client.scroll.call_args
        scroll_filter = scroll_call.kwargs["scroll_filter"]
        # Should have entity_id filter (no namespace filter)
        assert scroll_filter.must is not None
        entity_filter = scroll_filter.must[0]
        assert isinstance(entity_filter, Filter)
        assert entity_filter.should is not None  # OR filter for entity_id/user_id


# ---------------------------------------------------------------------------
# Tests: add
# ---------------------------------------------------------------------------


class TestAdd:
    """Tests for add() method."""

    def test_add_includes_namespace_and_entity_id_in_payload(
        self, store: QdrantMemoryStore, mock_client: MagicMock
    ) -> None:
        """add() passes the payload dict (with namespace/entity_id) to Qdrant."""
        payload = _make_payload(
            text="I live in Paris",
            entity_id="user1",
            namespace="work",
        )

        store.add(vector=[0.1, 0.2], payload=payload)

        call_args = mock_client.upsert.call_args
        points = call_args.kwargs["points"]
        assert len(points) == 1
        stored_payload = points[0].payload
        assert stored_payload["namespace"] == "work"
        assert stored_payload["entity_id"] == "user1"
