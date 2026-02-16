"""
In-memory vector storage implementation.

Provides a simple in-memory store for vector embeddings and similarity search,
suitable for testing and development. For production, use Qdrant implementation.
"""

import logging
import uuid
import warnings
from datetime import datetime
from typing import Any

from casual_memory.storage.vector.models import MemoryPoint, MemoryPointPayload

logger = logging.getLogger(__name__)


class InMemoryVectorStore:
    """
    In-memory implementation of the VectorMemoryStore protocol.

    Stores vectors and payloads in dictionaries with cosine similarity search.
    Data is lost on restart.
    """

    def __init__(self) -> None:
        # Store memory points by ID
        self._memories: dict[str, dict[str, Any]] = {}  # id -> {vector, payload}

        logger.info("InMemoryVectorStore initialized")

    def add(self, vector: list[float], payload: dict[str, Any]) -> str:
        """Add a memory to the store."""
        memory_id = str(uuid.uuid4())

        self._memories[memory_id] = {
            "vector": vector,
            "payload": payload,
        }

        logger.debug(f"Inserted memory {memory_id}: '{payload.get('text', '')[:50]}...'")
        return memory_id

    @staticmethod
    def _resolve_entity_id(payload: dict[str, Any]) -> str | None:
        """Resolve entity_id from a raw payload dict.

        Checks 'entity_id' first, falls back to 'user_id' for backward
        compatibility with payloads that have not been migrated.
        """
        entity_id: str | None = payload.get("entity_id")
        if entity_id is not None:
            return entity_id
        result: str | None = payload.get("user_id")
        return result

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same length")

        dot_product: float = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1: float = sum(a * a for a in vec1) ** 0.5
        magnitude2: float = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return float(dot_product / (magnitude1 * magnitude2))

    def _matches_filters(self, payload: dict[str, Any], filters: Any | None) -> bool:
        """Check if a payload matches the given filters.

        Supported filter keys:
            - 'entity_id': Match by entity ID
            - 'namespace': Match by namespace
            - 'user_id': Deprecated, maps to entity_id
            - 'type': Match by memory type (str or list[str])
            - 'min_importance': Minimum importance threshold
        """
        if filters is None:
            return True

        # Simple filter support (assumes dict with exact match criteria)
        if not isinstance(filters, dict):
            return True

        for key, value in filters.items():
            if value is None:
                continue

            if key == "entity_id":
                if self._resolve_entity_id(payload) != value:
                    return False
            elif key == "user_id":
                # Skip user_id if entity_id is already present (e.g. from
                # MemoryQueryFilter.model_dump() which emits both keys).
                if "entity_id" in filters and filters["entity_id"] is not None:
                    continue
                # Deprecated: map user_id filter to entity_id lookup
                warnings.warn(
                    "Filter key 'user_id' is deprecated, use 'entity_id' instead.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                if self._resolve_entity_id(payload) != value:
                    return False
            elif key == "namespace":
                if payload.get("namespace", "default") != value:
                    return False
            elif key == "type":
                # Handle list of types
                if isinstance(value, list):
                    if payload.get("type") not in value:
                        return False
                else:
                    if payload.get("type") != value:
                        return False
            elif key == "min_importance":
                if payload.get("importance", 0) < value:
                    return False

        return True

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        min_score: float = 0.7,
        filters: Any | None = None,
    ) -> list[MemoryPoint]:
        """Search for memories by vector similarity."""
        results = []

        for memory_id, memory_data in self._memories.items():

            vector = memory_data["vector"]
            payload = memory_data["payload"]

            # Skip archived memories (unless explicitly requested)
            if payload.get("archived", False):
                continue

            # Check filters
            if not self._matches_filters(payload, filters):
                continue

            # Calculate similarity
            score = self._cosine_similarity(query_embedding, vector)

            if score >= min_score:
                memory_point = MemoryPoint(
                    id=memory_id,
                    vector=vector,
                    payload=MemoryPointPayload(**payload),
                )
                results.append((memory_point, score))

        # Sort by score (highest first) and limit to top_k
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_k]

        logger.debug(f"{len(results)} results found (min_score={min_score})")

        return [memory_point for memory_point, score in results]

    def find_similar_memories(
        self,
        embedding: list[float],
        entity_id: str | None = None,
        namespace: str = "default",
        threshold: float | None = None,
        limit: int = 5,
        exclude_archived: bool = True,
        **kwargs: Any,
    ) -> list[tuple[Any, float]]:
        """Find similar memories based on vector similarity.

        Args:
            embedding: The embedding vector to search for
            entity_id: Filter by entity ID (for multi-entity isolation)
            namespace: Namespace for memory isolation (default: "default")
            threshold: Similarity threshold (0.0-1.0)
            limit: Maximum number of results to return
            exclude_archived: Whether to exclude archived memories
            **kwargs: Accepts deprecated 'user_id' keyword (mapped to entity_id)
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
            threshold = 0.85

        results = []

        for memory_id, memory_data in self._memories.items():
            vector = memory_data["vector"]
            payload = memory_data["payload"]

            # Filter by namespace
            if payload.get("namespace", "default") != namespace:
                continue

            # Filter by entity_id (with backward compat for user_id in raw payload)
            if entity_id and self._resolve_entity_id(payload) != entity_id:
                continue

            # Skip archived memories if requested
            if exclude_archived and payload.get("archived", False):
                continue

            # Calculate similarity
            score = self._cosine_similarity(embedding, vector)

            if score >= threshold:
                memory_point = MemoryPoint(
                    id=memory_id,
                    vector=vector,
                    payload=MemoryPointPayload(**payload),
                )
                results.append((memory_point, score))

        # Sort by score (highest first) and limit
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:limit]

        logger.info(
            f"Found {len(results)} similar memories "
            f"(threshold={threshold}, entity_id={entity_id}, namespace={namespace})"
        )

        return results

    def update_memory(self, memory_id: str, payload_updates: dict[str, Any]) -> bool:
        """Update specific fields in a memory's payload."""
        if memory_id not in self._memories:
            logger.error(f"Memory {memory_id} not found")
            return False

        # Update payload fields
        self._memories[memory_id]["payload"].update(payload_updates)

        logger.debug(f"Updated memory {memory_id}: {payload_updates}")
        return True

    def get_by_id(self, memory_id: str) -> MemoryPoint | None:
        """Retrieve a specific memory by its ID."""
        if memory_id not in self._memories:
            return None

        memory_data = self._memories[memory_id]

        return MemoryPoint(
            id=memory_id,
            vector=memory_data["vector"],
            payload=MemoryPointPayload(**memory_data["payload"]),
        )

    def archive_memory(self, memory_id: str, superseded_by: str | None = None) -> bool:
        """Archive a memory by marking it as archived."""
        if memory_id not in self._memories:
            logger.warning(f"Cannot archive memory {memory_id}: not found")
            return False

        # Update payload
        updates: dict[str, Any] = {
            "archived": True,
            "archived_at": datetime.now().isoformat(),
        }

        if superseded_by:
            updates["superseded_by"] = superseded_by

        success = self.update_memory(memory_id, updates)

        if success:
            logger.info(
                f"Archived memory {memory_id}"
                f"{f' (superseded by {superseded_by})' if superseded_by else ''}"
            )

        return success

    def clear_memories(self, entity_id: str, namespace: str = "default") -> int:
        """Clear all memories for a specific entity within a namespace.

        Args:
            entity_id: The ID of the entity whose memories to clear
            namespace: Namespace to clear within (default: "default")

        Returns:
            Number of memories deleted
        """
        memory_ids_to_delete = [
            memory_id
            for memory_id, memory_data in self._memories.items()
            if self._resolve_entity_id(memory_data["payload"]) == entity_id
            and memory_data["payload"].get("namespace", "default") == namespace
        ]

        count = len(memory_ids_to_delete)

        for memory_id in memory_ids_to_delete:
            del self._memories[memory_id]

        logger.info(f"Cleared {count} memories for entity_id={entity_id}, namespace={namespace}")

        return count

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
        memory_ids_to_delete = [
            memory_id
            for memory_id, memory_data in self._memories.items()
            if self._resolve_entity_id(memory_data["payload"]) == user_id
        ]

        count = len(memory_ids_to_delete)

        for memory_id in memory_ids_to_delete:
            del self._memories[memory_id]

        logger.info(f"Cleared {count} memories for user_id={user_id} (deprecated)")

        return count

    def clear(self) -> None:
        """Clear ALL memories from the store."""
        count = len(self._memories)
        self._memories.clear()
        logger.info(f"Cleared all memories ({count} total)")
