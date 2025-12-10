from casual_memory.storage import VectorMemoryStore, ConflictStore
from casual_memory.classifiers import MemoryClassificationPipeline
from casual_memory.classifiers.models import SimilarMemory
from casual_memory.models import MemoryFact, MemoryQueryFilter
from casual_memory.execution import MemoryActionExecutor, MemoryActionResult
from casual_memory.embeddings import TextEmbedding
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MemoryService:
    def __init__(
        self, 
        vector_store: VectorMemoryStore, 
        conflict_store: ConflictStore, 
        pipeline: MemoryClassificationPipeline, 
        embedding: TextEmbedding
    ):
        self.vector_store = vector_store
        self.pipeline = pipeline
        self.embedding = embedding
        self.action_executor = MemoryActionExecutor(vector_store, conflict_store)


    async def add_memory(
        self,
        new_memory: MemoryFact,
        similarity_threshold: float = 0.85,
        max_similar: int = 5
    ) -> MemoryActionResult:
        try:
            # Get similar memories
            query_vector = await self.embedding.embed_document(new_memory.text)
            similar_results = self.vector_store.find_similar_memories(
                embedding=query_vector,
                user_id=new_memory.user_id,
                threshold=similarity_threshold,
                limit=max_similar,
                exclude_archived=True
            )
            similar_memories = [
                SimilarMemory(
                    memory_id=point.id,
                    memory=MemoryFact(**point.payload.model_dump() if hasattr(point.payload, 'model_dump') else point.payload),
                    similarity_score=score
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
        query_vector = await self.embedding.embed_query(query)

        results = self.vector_store.search(
            query_embedding=query_vector,
            top_k=top_k,
            min_score=min_score,
            filters=filter.model_dump()
        )
    
        memories: list[MemoryFact] = []
        now = datetime.now()

        for result in results:
            # Filter out expired memories
            if result.payload.valid_until:
                try:
                    valid_until = datetime.fromisoformat(result.payload.valid_until)
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
                # Intelligence fields
                user_id=result.payload.user_id,
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
