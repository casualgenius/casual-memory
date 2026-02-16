import logging
from datetime import datetime, timezone

from casual_memory.classifiers import MemoryClassificationPipeline
from casual_memory.classifiers.models import SimilarMemory
from casual_memory.embeddings import TextEmbedding
from casual_memory.execution import MemoryActionExecutor, MemoryActionResult
from casual_memory.models import MemoryFact, MemoryQueryFilter
from casual_memory.storage import ConflictStore, VectorMemoryStore

logger = logging.getLogger(__name__)


class MemoryService:
    """Service for managing long-term semantic memories.

    Orchestrates the full memory lifecycle: embedding, similarity search,
    classification, and action execution. All operations are scoped by the
    ``namespace`` and ``entity_id`` fields on the ``MemoryFact`` being added
    or the ``MemoryQueryFilter`` being queried.
    """

    def __init__(
        self,
        vector_store: VectorMemoryStore,
        conflict_store: ConflictStore,
        pipeline: MemoryClassificationPipeline,
        embedding: TextEmbedding,
    ):
        self.vector_store = vector_store
        self.pipeline = pipeline
        self.embedding = embedding
        self.action_executor = MemoryActionExecutor(vector_store, conflict_store)

    async def add_memory(
        self, new_memory: MemoryFact, similarity_threshold: float = 0.85, max_similar: int = 5
    ) -> MemoryActionResult:
        """Add a memory, classifying it against existing similar memories.

        The operation is scoped by ``new_memory.namespace`` and
        ``new_memory.entity_id``. Only memories in the same namespace and
        belonging to the same entity are considered as potential duplicates
        or conflicts.

        Args:
            new_memory: The memory to add. Must have ``namespace`` (default
                ``"default"``) and optionally ``entity_id`` set for proper
                isolation.
            similarity_threshold: Minimum cosine similarity to consider a
                memory as "similar" (default: 0.85).
            max_similar: Maximum number of similar memories to retrieve for
                classification (default: 5).

        Returns:
            MemoryActionResult describing the action taken (added, updated,
            or conflict created).
        """
        try:
            # Get similar memories
            query_vector = await self.embedding.embed_document(new_memory.text)
            similar_results = self.vector_store.find_similar_memories(
                embedding=query_vector,
                entity_id=new_memory.entity_id,
                namespace=new_memory.namespace,
                threshold=similarity_threshold,
                limit=max_similar,
                exclude_archived=True,
            )
            similar_memories = [
                SimilarMemory(
                    memory_id=point.id,
                    memory=MemoryFact(
                        **(
                            point.payload.model_dump()
                            if hasattr(point.payload, "model_dump")
                            else point.payload
                        )
                    ),
                    similarity_score=score,
                )
                for point, score in similar_results
            ]

            # Classify the memory
            classification_result = await self.pipeline.classify(new_memory, similar_memories)

            # Perform actions
            vector = await self.embedding.embed_document(new_memory.text)
            result = await self.action_executor.execute(classification_result, vector)

            # Log structured result
            logger.info(
                f"Memory action: {result.action}, "
                f"memory_id={result.memory_id}, "
                f"conflicts={len(result.conflict_ids)}, "
                f"superseded={len(result.superseded_ids)}"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            raise

    async def query_memory(
        self,
        query: str,
        filter: MemoryQueryFilter,
        top_k: int = 5,
        min_score: float = 0.75,
    ) -> list[MemoryFact]:
        """Query memories by semantic similarity with optional filtering.

        Results are scoped by the ``namespace`` and ``entity_id`` fields on
        the provided ``MemoryQueryFilter``.

        Args:
            query: Natural language query string to search for.
            filter: Filter criteria including ``namespace``, ``entity_id``,
                ``type``, and ``min_importance``. Pass ``entity_id`` (not the
                deprecated ``user_id``) for entity scoping.
            top_k: Maximum number of results to return (default: 5).
            min_score: Minimum similarity score threshold (default: 0.75).

        Returns:
            List of matching ``MemoryFact`` objects, excluding expired memories.
        """
        query_vector = await self.embedding.embed_query(query)

        results = self.vector_store.search(
            query_embedding=query_vector,
            top_k=top_k,
            min_score=min_score,
            filters=filter.model_dump(),
        )

        memories: list[MemoryFact] = []
        now = datetime.now(timezone.utc)

        for result in results:
            # Filter out expired memories
            if result.payload.valid_until:
                try:
                    valid_until = datetime.fromisoformat(result.payload.valid_until)
                    # Normalize naive datetimes to UTC for comparison
                    if valid_until.tzinfo is None:
                        valid_until = valid_until.replace(tzinfo=timezone.utc)
                    if valid_until < now:
                        logger.debug(f"Skipping expired memory: {result.payload.text}")
                        continue
                except ValueError:
                    logger.warning(f"Invalid valid_until format: {result.payload.valid_until}")

            memory = MemoryFact(
                text=result.payload.text,
                type=result.payload.type,
                tags=result.payload.tags,
                importance=result.payload.importance,
                source=result.payload.source,
                valid_until=result.payload.valid_until,
                # Namespace and entity identification
                namespace=result.payload.namespace,
                entity_id=result.payload.entity_id,
                # Intelligence fields
                confidence=result.payload.confidence,
                mention_count=result.payload.mention_count,
                first_seen=result.payload.first_seen,
                last_seen=result.payload.last_seen,
                archived=result.payload.archived,
                archived_at=result.payload.archived_at,
                superseded_by=result.payload.superseded_by,
            )
            memories.append(memory)

        logger.info(f"{len(memories)} memories found")
        return memories
