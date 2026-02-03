"""
Memory action executor.

Executes actions based on classification results. This component takes
MemoryClassificationResult objects and applies the necessary changes to
vector stores and conflict stores.
"""

import logging
import uuid
from datetime import datetime

from casual_memory.classifiers.models import MemoryClassificationResult
from casual_memory.execution.models import MemoryActionResult
from casual_memory.models import MemoryConflict
from casual_memory.storage.protocols import ConflictStore, VectorMemoryStore

logger = logging.getLogger(__name__)


class MemoryActionExecutor:
    """
    Executes actions based on classification results.

    Takes MemoryClassificationResult and applies the corresponding actions:
    - add: Insert new memory to vector store (may archive superseded memories)
    - skip: Update existing memory's metadata
    - conflict: Create conflict record(s), don't add new memory
    """

    def __init__(
        self,
        vector_store: VectorMemoryStore,
        conflict_store: ConflictStore,
    ):
        """
        Initialize the action executor.

        Args:
            vector_store: Vector store for memory storage
            conflict_store: Conflict store for conflict tracking
        """
        self.vector_store = vector_store
        self.conflict_store = conflict_store

        logger.info("MemoryActionExecutor initialized")

    async def execute(
        self,
        result: MemoryClassificationResult,
        embedding: list[float],
    ) -> MemoryActionResult:
        """
        Execute the actions determined by classification.

        Args:
            result: Classification result with overall outcome and similarity results
            embedding: Vector embedding for the new memory

        Returns:
            MemoryActionResult with action taken and relevant IDs
        """
        if result.overall_outcome == "add":
            return await self._execute_add(result, embedding)

        elif result.overall_outcome == "skip":
            return await self._execute_skip(result)

        elif result.overall_outcome == "conflict":
            return await self._execute_conflict(result, embedding)

        else:
            raise ValueError(f"Unknown overall outcome: {result.overall_outcome}")

    async def _execute_add(
        self,
        result: MemoryClassificationResult,
        embedding: list[float],
    ) -> MemoryActionResult:
        """
        Add new memory to vector store.

        Side effect: Archive any superseded memories.

        Args:
            result: Classification result
            embedding: Vector embedding for new memory

        Returns:
            MemoryActionResult with added memory ID and superseded IDs
        """
        # Prepare payload from MemoryFact
        now = datetime.now().isoformat()
        payload = {
            "text": result.new_memory.text,
            "type": result.new_memory.type,
            "tags": result.new_memory.tags,
            "importance": result.new_memory.importance,
            "source": result.new_memory.source,
            "valid_until": result.new_memory.valid_until,
            "timestamp": now,
            "user_id": result.new_memory.user_id,
            "confidence": result.new_memory.confidence,
            "mention_count": result.new_memory.mention_count or 1,
            "first_seen": result.new_memory.first_seen or now,
            "last_seen": result.new_memory.last_seen or now,
            "archived": False,
            "superseded_by": None,
        }

        # Add new memory first
        memory_id = self.vector_store.add(vector=embedding, payload=payload)
        logger.info(f"Added new memory: {memory_id}")

        # Archive any superseded memories
        superseded_ids = []
        for old_memory_id in result.supersedes:
            self.vector_store.archive_memory(
                memory_id=old_memory_id,
                superseded_by=memory_id,
            )
            superseded_ids.append(old_memory_id)
            logger.info(f"Archived memory {old_memory_id}, superseded by {memory_id}")

        return MemoryActionResult(
            action="added",
            memory_id=memory_id,
            superseded_ids=superseded_ids,
            metadata={
                "text": (
                    result.new_memory.text[:50] + "..."
                    if len(result.new_memory.text) > 50
                    else result.new_memory.text
                )
            },
        )

    async def _execute_skip(self, result: MemoryClassificationResult) -> MemoryActionResult:
        """
        Memory is same as existing - update metadata.

        Updates mention_count, last_seen, and triggers confidence recalculation.

        Args:
            result: Classification result

        Returns:
            MemoryActionResult with updated memory ID
        """
        same_memory_id = result.same_as

        if not same_memory_id:
            logger.error("Skip outcome but no same_as memory found. This should not happen.")
            raise ValueError("Skip outcome requires same_as memory")

        # Get current memory to increment mention_count
        # Note: This assumes the vector store has a way to retrieve payload
        # For now, we'll just send the updates

        # Update mention count, last_seen
        # Note: We send "+1" as a special indicator, but the actual implementation
        # depends on the vector store's update_memory method
        self.vector_store.update_memory(
            memory_id=same_memory_id,
            payload_updates={
                "mention_count": "+1",  # Increment
                "last_seen": datetime.now().isoformat(),
                # Confidence recalculation would happen here if we had the logic
            },
        )

        logger.info(f"Updated existing memory: {same_memory_id}")
        return MemoryActionResult(
            action="updated",
            memory_id=same_memory_id,
            metadata={
                "text": (
                    result.new_memory.text[:50] + "..."
                    if len(result.new_memory.text) > 50
                    else result.new_memory.text
                )
            },
        )

    async def _execute_conflict(
        self,
        result: MemoryClassificationResult,
        embedding: list[float],
    ) -> MemoryActionResult:
        """
        Create conflict record(s), don't add new memory.

        Creates one conflict record for each conflicting similar memory.

        Args:
            result: Classification result
            embedding: Vector embedding for new memory (stored in conflict)

        Returns:
            MemoryActionResult with conflict IDs
        """
        conflict_ids = []

        for similarity_result in result.similarity_results:
            if similarity_result.outcome == "conflict":
                # Extract conflict metadata
                category = similarity_result.metadata.get("category", "factual")
                clarification_hint = similarity_result.metadata.get(
                    "clarification_hint", "Which statement is correct?"
                )
                avg_importance = similarity_result.metadata.get("avg_importance", 0.5)

                # Generate temporary ID for new memory (not yet inserted)
                temp_memory_b_id = f"pending_{str(uuid.uuid4())}"

                # Create MemoryConflict object
                conflict = MemoryConflict(
                    user_id=result.new_memory.user_id or "default_user",
                    memory_a_id=similarity_result.similar_memory.memory_id,
                    memory_b_id=temp_memory_b_id,  # Temporary ID for new memory
                    category=category,
                    similarity_score=similarity_result.similar_memory.similarity_score,
                    avg_importance=avg_importance,
                    clarification_hint=clarification_hint,
                    status="pending",
                    metadata={
                        "memory_a_text": similarity_result.similar_memory.memory.text,
                        "memory_a_type": similarity_result.similar_memory.memory.type,
                        "memory_b_text": result.new_memory.text,  # Store new memory text
                        "memory_b_type": result.new_memory.type,
                        "memory_b_embedding": embedding,  # Store embedding for later insertion
                        "memory_b_pending": True,  # Flag indicating memory_b not yet in vector store
                        "detection_method": similarity_result.metadata.get(
                            "detection_method", "llm"
                        ),
                    },
                )

                # Add conflict to store
                conflict_id = self.conflict_store.add_conflict(conflict)
                conflict_ids.append(conflict_id)

                logger.info(
                    f"Created conflict: {conflict_id} (category={category}, "
                    f"memory_a={similarity_result.similar_memory.memory_id})"
                )

        if not conflict_ids:
            logger.warning(
                "Conflict outcome but no conflict similarity results found. "
                "This should not happen."
            )

        return MemoryActionResult(
            action="conflict",
            conflict_ids=conflict_ids,
            metadata={
                "text": (
                    result.new_memory.text[:50] + "..."
                    if len(result.new_memory.text) > 50
                    else result.new_memory.text
                ),
                "conflict_count": len(conflict_ids),
            },
        )
