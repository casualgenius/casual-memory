import logging
import uuid
import warnings
from typing import Any, TypeGuard

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HasIdCondition,
    HasVectorCondition,
    IsEmptyCondition,
    IsNullCondition,
    MatchAny,
    MatchValue,
    NestedCondition,
    PayloadField,
    PointStruct,
    Range,
    VectorParams,
)

from casual_memory.storage.vector.models import MemoryPoint, MemoryPointPayload

# Union of all condition types accepted by Filter.must / must_not / should
_Condition = (
    FieldCondition
    | IsEmptyCondition
    | IsNullCondition
    | HasIdCondition
    | HasVectorCondition
    | NestedCondition
    | Filter
)

logger = logging.getLogger(__name__)

vector_dimension = 768


def _is_flat_vector(v: Any) -> TypeGuard[list[float]]:
    """Type guard to verify vector is a flat list of floats (not nested)."""
    return isinstance(v, list) and len(v) > 0 and not isinstance(v[0], list)


def _build_namespace_filter(namespace: str) -> Filter:
    """Build a Qdrant filter condition for namespace.

    For the ``"default"`` namespace we use an OR filter that matches points
    with ``namespace == "default"`` **or** points where the ``namespace``
    field is absent/null.  This provides backward compatibility with data
    stored before namespace support was added (those points have no
    ``namespace`` payload field and are logically part of the default
    namespace).

    For any other namespace value the filter is a strict equality match --
    only points whose ``namespace`` field is exactly that value will be
    returned.
    """
    if namespace == "default":
        return Filter(
            should=[
                FieldCondition(key="namespace", match=MatchValue(value="default")),
                IsNullCondition(is_null=PayloadField(key="namespace")),
            ]
        )
    return Filter(must=[FieldCondition(key="namespace", match=MatchValue(value=namespace))])


def _build_entity_id_filter(entity_id: str) -> Filter:
    """Build a Qdrant filter that matches an entity across both field names.

    Old data may have ``user_id`` while new data uses ``entity_id``.  We
    use an OR (``should``) so that either field matching is sufficient.
    """
    return Filter(
        should=[
            FieldCondition(key="entity_id", match=MatchValue(value=entity_id)),
            FieldCondition(key="user_id", match=MatchValue(value=entity_id)),
        ]
    )


class QdrantMemoryStore:
    def __init__(
        self, host: str = "localhost", port: int = 6333, collection_name: str = "memories"
    ):
        """
        Initialize Qdrant memory store.

        Args:
            host: Qdrant host (default: localhost)
            port: Qdrant port (default: 6333)
            collection_name: Collection name (default: memories)
        """
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self._init_collection()

    def _init_collection(self) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_dimension, distance=Distance.COSINE),
            )

    def clear(self) -> None:
        """Clear ALL memories from the collection (dangerous!)"""
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_dimension, distance=Distance.COSINE),
        )

    def _scroll_and_delete(self, scroll_filter: Filter, label: str) -> int:
        """Paginate through scroll results and delete all matching points.

        Args:
            scroll_filter: Qdrant filter to select points
            label: Human-readable label for log messages

        Returns:
            Total number of points deleted
        """
        page_size = 10000
        total_deleted = 0
        offset = None  # Qdrant uses point ID as cursor

        while True:
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=page_size,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )

            points, next_offset = scroll_result
            point_ids = [point.id for point in points]

            if point_ids:
                self.client.delete(collection_name=self.collection_name, points_selector=point_ids)
                total_deleted += len(point_ids)

            if next_offset is None or len(points) < page_size:
                break
            offset = next_offset

        return total_deleted

    def clear_memories(self, entity_id: str, namespace: str = "default") -> int:
        """Clear all memories for a specific entity within a namespace.

        Args:
            entity_id: The ID of the entity whose memories to clear
            namespace: Namespace to clear within (default: "default")

        Returns:
            Number of memories deleted
        """
        try:
            must_filters: list[_Condition] = [
                _build_entity_id_filter(entity_id),
                _build_namespace_filter(namespace),
            ]
            scroll_filter = Filter(must=must_filters)
            label = f"entity_id={entity_id}, namespace={namespace}"

            count = self._scroll_and_delete(scroll_filter, label)

            if count > 0:
                logger.info(f"Cleared {count} memories for {label}")
            else:
                logger.info(f"No memories found for {label}")

            return count
        except Exception as e:
            logger.error(
                f"Failed to clear memories for entity_id={entity_id}, namespace={namespace}: {e}"
            )
            raise

    def clear_user_memories(self, user_id: str) -> int:
        """Deprecated: Use clear_memories() instead.

        Clear all memories for a specific user across all namespaces.

        Args:
            user_id: The ID of the user whose memories to clear

        Returns:
            Number of memories deleted
        """
        warnings.warn(
            "clear_user_memories() is deprecated, use clear_memories() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            scroll_filter = Filter(must=[_build_entity_id_filter(user_id)])
            label = f"user_id={user_id} (deprecated)"

            count = self._scroll_and_delete(scroll_filter, label)

            if count > 0:
                logger.info(f"Cleared {count} memories for {label}")
            else:
                logger.info(f"No memories found for user_id={user_id}")

            return count
        except Exception as e:
            logger.error(f"Failed to clear memories for user_id={user_id}: {e}")
            raise

    def add(self, vector: list[float], payload: dict[str, Any]) -> str:
        """
        Add a memory to the Qdrant collection.

        The ``payload`` dict should be obtained from
        ``MemoryPointPayload.model_dump()`` and will include ``namespace``
        (default ``"default"``) and ``entity_id`` fields alongside the
        backward-compatible ``user_id`` key.

        Args:
            vector: The embedding vector
            payload: Dictionary of memory fields (from MemoryPointPayload.model_dump()).
                     Should include ``namespace`` and ``entity_id`` fields.

        Returns:
            The generated memory ID
        """
        id = str(uuid.uuid4())
        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=id, vector=vector, payload=payload)],
        )
        logger.debug(f"Inserted memory {id}: '{payload.get('text', '')[:50]}...'")
        return id

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        min_score: float = 0.7,
        filters: Any | None = None,
    ) -> list[MemoryPoint]:
        """Search for memories by vector similarity with optional filters.

        Supported filter keys (dict):
            - ``entity_id``: Match by entity ID
            - ``namespace``: Match by namespace (with backward-compat for default)
            - ``user_id``: Deprecated, maps to entity_id
            - ``type``: Match by memory type (list of str)
            - ``min_importance``: Minimum importance threshold
        """
        qdrant_filter = None
        if filters:
            must_conditions: list[_Condition] = []

            # Resolve entity_id / user_id
            entity_id_value: str | None = filters.get("entity_id")
            user_id_value: str | None = filters.get("user_id")

            if entity_id_value is not None:
                must_conditions.append(_build_entity_id_filter(entity_id_value))
            elif user_id_value is not None:
                # Deprecated path - only if entity_id is absent
                warnings.warn(
                    "Filter key 'user_id' is deprecated, use 'entity_id' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                must_conditions.append(_build_entity_id_filter(user_id_value))

            # Handle namespace filter
            namespace_value: str | None = filters.get("namespace")
            if namespace_value is not None:
                must_conditions.append(_build_namespace_filter(namespace_value))

            # Handle type filter (list of types)
            type_value = filters.get("type")
            if type_value is not None:
                if isinstance(type_value, str):
                    type_value = [type_value]
                elif not isinstance(type_value, list):
                    raise TypeError(
                        f"Filter 'type' must be a list of strings or a string, got {type(type_value).__name__}"
                    )
                must_conditions.append(FieldCondition(key="type", match=MatchAny(any=type_value)))

            # Handle min_importance filter
            min_importance_value = filters.get("min_importance")
            if min_importance_value is not None:
                must_conditions.append(
                    FieldCondition(key="importance", range=Range(gte=min_importance_value))
                )

            qdrant_filter = Filter(must=must_conditions) if must_conditions else None

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            query_filter=qdrant_filter,
            score_threshold=min_score,
            with_vectors=True,
            with_payload=True,
        )

        results = []
        hits = response.points
        logger.debug(f"{len(hits)} hits found")
        for hit in hits:
            if hit.payload is None or not _is_flat_vector(hit.vector):
                continue
            logger.debug(f"Score: {hit.score}, Memory: '{hit.payload.get('text', '')[:50]}...'")
            memory = MemoryPoint(
                id=str(hit.id), vector=hit.vector, payload=MemoryPointPayload(**hit.payload)
            )
            results.append(memory)

        return results

    def find_similar_memories(
        self,
        embedding: list[float],
        entity_id: str | None = None,
        namespace: str = "default",
        threshold: float | None = None,
        limit: int = 5,
        exclude_archived: bool = True,
        **kwargs: Any,
    ) -> list[tuple[MemoryPoint, float]]:
        """
        Find similar memories based on vector similarity.

        Args:
            embedding: The embedding vector to search for
            entity_id: Filter by entity ID (for multi-entity isolation)
            namespace: Namespace for memory isolation (default: "default")
            threshold: Similarity threshold (0.0-1.0). Defaults to 0.85
            limit: Maximum number of results to return
            exclude_archived: Whether to exclude archived memories (default: True)
            **kwargs: Accepts deprecated 'user_id' keyword (mapped to entity_id)

        Returns:
            List of tuples containing (MemoryPoint, similarity_score)
        """
        # Handle deprecated user_id keyword argument
        if "user_id" in kwargs:
            warnings.warn(
                "find_similar_memories(user_id=...) is deprecated, use entity_id instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if entity_id is None:
                entity_id = kwargs["user_id"]

        if threshold is None:
            threshold = 0.85  # Default similarity threshold

        # Build filter conditions
        must_conditions: list[_Condition] = []
        must_not_conditions: list[_Condition] = []

        if entity_id:
            must_conditions.append(_build_entity_id_filter(entity_id))

        # Add namespace filter
        must_conditions.append(_build_namespace_filter(namespace))

        # Exclude archived at query level so limit is respected
        if exclude_archived:
            must_not_conditions.append(FieldCondition(key="archived", match=MatchValue(value=True)))

        qdrant_filter = Filter(
            must=must_conditions or None,
            must_not=must_not_conditions or None,
        )

        # Perform similarity search
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            limit=limit,
            query_filter=qdrant_filter,
            score_threshold=threshold,
            with_vectors=True,
            with_payload=True,
        )

        results = []
        for hit in response.points:
            if hit.payload is None or not _is_flat_vector(hit.vector):
                continue
            memory_point = MemoryPoint(
                id=str(hit.id), vector=hit.vector, payload=MemoryPointPayload(**hit.payload)
            )

            results.append((memory_point, hit.score))
            logger.debug(
                f"Similar memory found: score={hit.score:.3f}, "
                f"text='{memory_point.payload.text[:50]}...'"
            )

        logger.info(
            f"Found {len(results)} similar memories "
            f"(threshold={threshold}, entity_id={entity_id}, namespace={namespace})"
        )
        return results

    def update_memory(self, memory_id: str, payload_updates: dict[str, Any]) -> bool:
        """
        Update specific fields in a memory's payload.

        Args:
            memory_id: The ID of the memory to update
            payload_updates: Dictionary of fields to update

        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.set_payload(
                collection_name=self.collection_name, payload=payload_updates, points=[memory_id]
            )
            logger.debug(f"Updated memory {memory_id}: {payload_updates}")
            return True
        except Exception as e:
            logger.error(f"Failed to update memory {memory_id}: {e}")
            return False

    def get_memory_by_id(self, memory_id: str) -> MemoryPoint | None:
        """
        Retrieve a specific memory by its ID.

        Args:
            memory_id: The ID of the memory to retrieve

        Returns:
            MemoryPoint if found, None otherwise
        """
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[memory_id],
                with_vectors=True,
                with_payload=True,
            )

            if result and len(result) > 0:
                point = result[0]
                return MemoryPoint(
                    id=str(point.id),
                    vector=point.vector,  # type: ignore[arg-type]
                    payload=MemoryPointPayload(**point.payload),  # type: ignore[arg-type]
                )
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve memory {memory_id}: {e}")
            return None

    def archive_memory(self, memory_id: str, superseded_by: str | None = None) -> bool:
        """
        Archive a memory by marking it as archived.

        Args:
            memory_id: The ID of the memory to archive
            superseded_by: Optional ID of the memory that supersedes this one

        Returns:
            True if successful, False otherwise
        """
        from datetime import datetime

        try:
            # Verify memory exists first
            memory = self.get_memory_by_id(memory_id)
            if not memory:
                logger.warning(f"Cannot archive memory {memory_id}: not found")
                return False

            # Prepare update payload
            updates: dict[str, Any] = {
                "archived": True,
                "archived_at": datetime.now().isoformat(),
            }

            if superseded_by:
                updates["superseded_by"] = superseded_by

            # Update the memory
            success = self.update_memory(memory_id, updates)

            if success:
                logger.info(
                    f"Archived memory {memory_id}"
                    f"{f' (superseded by {superseded_by})' if superseded_by else ''}"
                )

            return success

        except Exception as e:
            logger.error(f"Failed to archive memory {memory_id}: {e}")
            return False
