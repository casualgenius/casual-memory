import logging
import uuid
from typing import Any, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

from casual_memory.storage.vector.models import MemoryPoint, MemoryPointPayload

logger = logging.getLogger(__name__)

vector_dimension = 768


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

    def _init_collection(self):
        if not self.client.collection_exists(self.collection_name):
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "size": vector_dimension,
                    "distance": "Cosine",
                },  # Adjust size as needed
            )

    def clear(self):
        """Clear ALL memories from the collection (dangerous!)"""
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_dimension, distance=Distance.COSINE),
        )

    def clear_user_memories(self, user_id: str) -> int:
        """
        Clear all memories for a specific user.

        Args:
            user_id: The ID of the user whose memories to clear

        Returns:
            Number of memories deleted
        """
        try:
            # First, count how many memories we're about to delete
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
                ),
                limit=10000,  # Large enough to get all user memories in one call
                with_payload=False,
                with_vectors=False,
            )

            points_to_delete = [point.id for point in scroll_result[0]]
            count = len(points_to_delete)

            if count > 0:
                # Delete all matching points
                self.client.delete(
                    collection_name=self.collection_name, points_selector=points_to_delete
                )
                logger.info(f"Cleared {count} memories for user_id={user_id}")
            else:
                logger.info(f"No memories found for user_id={user_id}")

            return count
        except Exception as e:
            logger.error(f"Failed to clear memories for user_id={user_id}: {e}")
            raise

    def add(self, vector: List[float], payload: dict):
        """
        Add a memory to the Qdrant collection.

        Args:
            vector: The embedding vector
            payload: Dictionary of memory fields (from MemoryPointPayload.model_dump())

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
        query_embedding: List[float],
        top_k: int = 5,
        min_score: float = 0.7,
        filters: Optional[Any] = None,
    ) -> List[MemoryPoint]:
        qdrant_filter = None
        if filters:
            conditions = []

            # Handle user_id filter
            if filters.user_id is not None:
                conditions.append(
                    FieldCondition(key="user_id", match=MatchValue(value=filters.user_id))
                )

            # Handle type filter (list of types)
            if filters.type is not None:
                conditions.append(FieldCondition(key="type", match=MatchAny(any=filters.type)))

            # Handle min_importance filter
            if filters.min_importance is not None:
                conditions.append(
                    FieldCondition(key="importance", range=Range(gte=filters.min_importance))
                )

            qdrant_filter = Filter(must=conditions) if conditions else None

        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=qdrant_filter,
            with_vectors=True,
            with_payload=True,
        )

        results = []
        logger.debug(f"{len(hits)} hits found")
        for hit in hits:
            if hit.score >= min_score:
                logger.debug(f"Score: {hit.score}, Memory: '{hit.payload.get('text', '')[:50]}...'")
                memory = MemoryPoint(
                    id=str(hit.id), vector=hit.vector, payload=MemoryPointPayload(**hit.payload)
                )
                results.append(memory)
            else:
                logger.debug(
                    f"Skipping Due to Low Score: {hit.score}, Memory: '{hit.payload.get('text', '')[:50]}...'"
                )

        return results

    def find_similar_memories(
        self,
        embedding: List[float],
        user_id: Optional[str] = None,
        threshold: Optional[float] = None,
        limit: int = 5,
        exclude_archived: bool = True,
    ) -> List[tuple[MemoryPoint, float]]:
        """
        Find similar memories based on vector similarity.

        Args:
            embedding: The embedding vector to search for
            user_id: Filter by user ID (for multi-user isolation)
            threshold: Similarity threshold (0.0-1.0). Defaults to config.MEMORY_SIMILARITY_THRESHOLD
            limit: Maximum number of results to return
            exclude_archived: Whether to exclude archived memories (default: True)

        Returns:
            List of tuples containing (MemoryPoint, similarity_score)
        """
        if threshold is None:
            threshold = config.MEMORY_SIMILARITY_THRESHOLD

        # Build filter conditions
        conditions = []

        if user_id:
            conditions.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))

        # Note: We don't filter by archived here because:
        # 1. The field might not exist in older records
        # 2. We handle archived filtering in post-processing if needed
        # For now, we rely on archived field being set to False by default in MemoryPointPayload

        qdrant_filter = Filter(must=conditions) if conditions else None

        # Perform similarity search
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=limit,
            query_filter=qdrant_filter,
            score_threshold=threshold,
            with_vectors=True,
            with_payload=True,
        )

        # Convert results and filter archived in post-processing
        results = []
        for hit in hits:
            memory_point = MemoryPoint(
                id=str(hit.id), vector=hit.vector, payload=MemoryPointPayload(**hit.payload)
            )

            # Skip archived memories if requested
            if exclude_archived and memory_point.payload.archived:
                logger.debug(f"Skipping archived memory: {hit.payload.get('text', '')[:50]}")
                continue

            results.append((memory_point, hit.score))
            logger.debug(
                f"Similar memory found: score={hit.score:.3f}, "
                f"text='{hit.payload.get('text', '')[:50]}...'"
            )

        logger.info(
            f"Found {len(results)} similar memories " f"(threshold={threshold}, user_id={user_id})"
        )
        return results

    def update_memory(self, memory_id: str, payload_updates: dict) -> bool:
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

    def get_memory_by_id(self, memory_id: str) -> Optional[MemoryPoint]:
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
                    vector=point.vector,
                    payload=MemoryPointPayload(**point.payload),
                )
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve memory {memory_id}: {e}")
            return None

    def archive_memory(self, memory_id: str, superseded_by: Optional[str] = None) -> bool:
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
            updates = {"archived": True, "archived_at": datetime.now().isoformat()}

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
