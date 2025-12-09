"""
In-memory vector storage implementation.

Provides a simple in-memory store for vector embeddings and similarity search,
suitable for testing and development. For production, use Qdrant implementation.
"""

import logging
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
from casual_memory.storage.vector.models import MemoryPoint, MemoryPointPayload

logger = logging.getLogger(__name__)


class InMemoryVectorStore:
    """
    In-memory implementation of the VectorMemoryStore protocol.

    Stores vectors and payloads in dictionaries with cosine similarity search.
    Data is lost on restart.
    """

    def __init__(self):
        # Store memory points by ID
        self._memories: Dict[str, Dict[str, Any]] = {}  # id -> {vector, payload}

        logger.info("InMemoryVectorStore initialized")

    def add(self, vector: List[float], payload: dict) -> str:
        """Add a memory to the store."""
        memory_id = str(uuid.uuid4())

        self._memories[memory_id] = {
            "vector": vector,
            "payload": payload,
        }

        logger.debug(f"Inserted memory {memory_id}: '{payload.get('text', '')[:50]}...'")
        return memory_id

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same length")

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def _matches_filters(self, payload: dict, filters: Optional[Any]) -> bool:
        """Check if a payload matches the given filters."""
        if filters is None:
            return True

        # Simple filter support (assumes dict with exact match criteria)
        if not isinstance(filters, dict):
            return True

        for key, value in filters.items():
            if key == "user_id":
                if payload.get("user_id") != value:
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
        query_embedding: List[float],
        top_k: int = 5,
        min_score: float = 0.7,
        filters: Optional[Any] = None,
    ) -> List[MemoryPoint]:
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
        embedding: List[float],
        user_id: Optional[str] = None,
        threshold: Optional[float] = None,
        limit: int = 5,
        exclude_archived: bool = True,
    ) -> List[tuple[Any, float]]:
        """Find similar memories based on vector similarity."""
        if threshold is None:
            threshold = 0.85

        results = []

        for memory_id, memory_data in self._memories.items():
            vector = memory_data["vector"]
            payload = memory_data["payload"]

            # Filter by user_id
            if user_id and payload.get("user_id") != user_id:
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
            f"(threshold={threshold}, user_id={user_id})"
        )

        return results

    def update_memory(self, memory_id: str, payload_updates: dict) -> bool:
        """Update specific fields in a memory's payload."""
        if memory_id not in self._memories:
            logger.error(f"Memory {memory_id} not found")
            return False

        # Update payload fields
        self._memories[memory_id]["payload"].update(payload_updates)

        logger.debug(f"Updated memory {memory_id}: {payload_updates}")
        return True

    def get_by_id(self, memory_id: str) -> Optional[MemoryPoint]:
        """Retrieve a specific memory by its ID."""
        if memory_id not in self._memories:
            return None

        memory_data = self._memories[memory_id]

        return MemoryPoint(
            id=memory_id,
            vector=memory_data["vector"],
            payload=MemoryPointPayload(**memory_data["payload"]),
        )

    def archive_memory(
        self, memory_id: str, superseded_by: Optional[str] = None
    ) -> bool:
        """Archive a memory by marking it as archived."""
        if memory_id not in self._memories:
            logger.warning(f"Cannot archive memory {memory_id}: not found")
            return False

        # Update payload
        updates = {
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

    def clear_user_memories(self, user_id: str) -> int:
        """Clear all memories for a specific user."""
        memory_ids_to_delete = [
            memory_id
            for memory_id, memory_data in self._memories.items()
            if memory_data["payload"].get("user_id") == user_id
        ]

        count = len(memory_ids_to_delete)

        for memory_id in memory_ids_to_delete:
            del self._memories[memory_id]

        logger.info(f"Cleared {count} memories for user_id={user_id}")

        return count

    def clear(self):
        """Clear ALL memories from the store."""
        count = len(self._memories)
        self._memories.clear()
        logger.info(f"Cleared all memories ({count} total)")
