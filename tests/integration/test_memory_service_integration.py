"""
Integration tests for MemoryService.

Tests the complete memory service flow with:
- Real ClassificationPipeline orchestrator
- Real NLIClassifier, ConflictClassifier, DuplicateClassifier, AutoResolutionClassifier
- Real InMemoryVectorStore and InMemoryConflictStore
- Real ActionExecutor

Only the LLM and NLI model backends are mocked to avoid:
- External API calls (costly, slow, non-deterministic)
- Loading heavy transformer models (slow, memory-intensive)

This provides true integration testing of the classification pipeline
while remaining fast and deterministic.
"""


import pytest

from casual_memory.classifiers import (
    AutoResolutionClassifier,
    ConflictClassifier,
    DuplicateClassifier,
    MemoryClassificationPipeline,
    NLIClassifier,
)
from casual_memory.memory_service import MemoryService
from casual_memory.models import MemoryFact, MemoryQueryFilter
from casual_memory.storage.conflicts.memory import InMemoryConflictStore
from casual_memory.storage.vector.memory import InMemoryVectorStore


class MockEmbedding:
    """Mock embedding service that returns deterministic embeddings."""

    def __init__(self):
        # Simple hash-based embeddings for testing
        self.cache = {}

    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for query."""
        return self._generate_embedding(text)

    async def embed_document(self, text: str) -> list[float]:
        """Generate embedding for document."""
        return self._generate_embedding(text)

    def _generate_embedding(self, text: str) -> list[float]:
        """Generate deterministic embedding based on text content."""
        if text in self.cache:
            return self.cache[text]

        # Simple hash-based embedding (128-dim for testing)
        # Uses keyword weighting to boost similarity for semantic queries
        words = text.lower().split()
        embedding = [0.0] * 128

        # Boost important keywords (locations, jobs, preferences)
        important_keywords = {
            "live",
            "work",
            "like",
            "prefer",
            "paris",
            "london",
            "teacher",
            "doctor",
            "engineer",
            "coffee",
            "pizza",
        }

        for i, word in enumerate(words):
            hash_val = hash(word)
            # Boost weight for important keywords
            weight = 2.0 if word in important_keywords else 1.0
            position_weight = 1.0 / (i + 1)

            for j in range(128):
                embedding[j] += ((hash_val >> j) & 1) * weight * position_weight

        # Normalize
        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        self.cache[text] = embedding
        return embedding


class MockNLIFilter:
    """
    Mock NLI filter that simulates sentence-transformers NLI model behavior.

    Returns realistic entailment/neutral/contradiction predictions based on
    text similarity and semantic relationships without loading a real model.
    """

    def __init__(self):
        self.prediction_count = 0

    def predict(self, premise: str, hypothesis: str):
        """Predict NLI relationship with realistic logic."""
        self.prediction_count += 1

        premise_lower = premise.lower()
        hypothesis_lower = hypothesis.lower()

        # High entailment: texts are very similar or one contains the other
        if premise_lower == hypothesis_lower:
            return "entailment", [0.05, 0.95, 0.0]  # Very high entailment

        if premise_lower in hypothesis_lower or hypothesis_lower in premise_lower:
            return "entailment", [0.1, 0.85, 0.05]  # High entailment

        # Check for contradictions (opposite locations, jobs, preferences)
        contradiction_patterns = [
            ("live in", "live in"),
            ("work as", "work as"),
            ("work in", "work in"),
            ("like", "hate"),
            ("prefer", "dislike"),
        ]

        for pattern1, pattern2 in contradiction_patterns:
            if pattern1 in premise_lower and pattern2 in hypothesis_lower:
                # Extract the objects being compared (skip articles like "a", "an", "the")
                premise_words = premise_lower.split(pattern1)[-1].strip().split()
                hypo_words = hypothesis_lower.split(pattern2)[-1].strip().split()

                # Skip articles and get the actual noun
                articles = {"a", "an", "the"}
                premise_obj = next(
                    (w.rstrip(".,;") for w in premise_words if w not in articles), ""
                )
                hypo_obj = next((w.rstrip(".,;") for w in hypo_words if w not in articles), "")

                if premise_obj and hypo_obj and premise_obj != hypo_obj:
                    # Return [contradiction, entailment, neutral] scores
                    return "contradiction", [0.9, 0.05, 0.05]  # High contradiction

        # Check for semantic overlap (share keywords)
        premise_words = set(premise_lower.split())
        hypo_words = set(hypothesis_lower.split())
        overlap = len(premise_words & hypo_words) / max(len(premise_words), len(hypo_words))

        if overlap > 0.5:
            # Moderate entailment
            return "entailment", [0.15, 0.7, 0.15]
        elif overlap > 0.3:
            # Weak neutral - [contradiction, entailment, neutral]
            return "neutral", [0.2, 0.3, 0.5]
        else:
            # Strong neutral (unrelated) - [contradiction, entailment, neutral]
            return "neutral", [0.1, 0.2, 0.7]

    def get_metrics(self):
        return {"nli_prediction_count": self.prediction_count}


class MockConflictVerifier:
    """
    Mock LLM-based conflict verifier that simulates real conflict detection.

    Returns proper conflict metadata including category and clarification hints
    that match the structure of the real LLM-based verifier.
    """

    def __init__(self):
        self.llm_call_count = 0

    async def verify_conflict(self, memory_a, memory_b, similarity_score):
        """
        Check if memories conflict and return detailed metadata.

        Returns:
            tuple: (is_conflict, metadata_dict) where metadata includes:
                - category: str (location, job, preference, temporal, factual)
                - clarification_hint: str
                - detection_method: str
                - avg_importance: float
        """
        self.llm_call_count += 1

        a_text = memory_a.text.lower()
        b_text = memory_b.text.lower()

        # Detect location conflicts
        if "live in" in a_text and "live in" in b_text:
            # Extract location, skipping articles
            a_words = a_text.split("live in")[-1].strip().split()
            b_words = b_text.split("live in")[-1].strip().split()
            articles = {"a", "an", "the"}
            a_loc = next((w.rstrip(".,;") for w in a_words if w not in articles), "")
            b_loc = next((w.rstrip(".,;") for w in b_words if w not in articles), "")
            if a_loc and b_loc and a_loc != b_loc:
                return True, {
                    "category": "location",
                    "clarification_hint": f"Do you live in {a_loc} or {b_loc}?",
                    "detection_method": "llm",
                    "avg_importance": (memory_a.importance + memory_b.importance) / 2,
                }

        # Detect job conflicts
        if "work as" in a_text and "work as" in b_text:
            # Extract job title (take text before location markers like "at", "in")
            a_job_text = a_text.split("work as")[-1].strip()
            b_job_text = b_text.split("work as")[-1].strip()

            # Stop at location markers
            for marker in [" at ", " in ", ",", ".", ";"]:
                if marker in a_job_text:
                    a_job_text = a_job_text.split(marker)[0]
                if marker in b_job_text:
                    b_job_text = b_job_text.split(marker)[0]

            # Get words, filter articles, take the last word (usually the actual job)
            articles = {"a", "an", "the"}
            a_words = [w for w in a_job_text.split() if w not in articles]
            b_words = [w for w in b_job_text.split() if w not in articles]
            a_job = a_words[-1].rstrip(".,;") if a_words else ""
            b_job = b_words[-1].rstrip(".,;") if b_words else ""

            # Only conflict if jobs are different AND not related (one doesn't contain the other)
            if a_job and b_job and a_job != b_job and a_job not in b_job and b_job not in a_job:
                return True, {
                    "category": "job",
                    "clarification_hint": f"Do you work as {a_job} or {b_job}?",
                    "detection_method": "llm",
                    "avg_importance": (memory_a.importance + memory_b.importance) / 2,
                }

        # Detect work location conflicts
        if "work in" in a_text and "work in" in b_text:
            # Extract location, skipping articles
            a_words = a_text.split("work in")[-1].strip().split()
            b_words = b_text.split("work in")[-1].strip().split()
            articles = {"a", "an", "the"}
            a_loc = next((w.rstrip(".,;") for w in a_words if w not in articles), "")
            b_loc = next((w.rstrip(".,;") for w in b_words if w not in articles), "")
            if a_loc and b_loc and a_loc != b_loc:
                return True, {
                    "category": "job_location",
                    "clarification_hint": f"Do you work in {a_loc} or {b_loc}?",
                    "detection_method": "llm",
                    "avg_importance": (memory_a.importance + memory_b.importance) / 2,
                }

        # Detect preference conflicts (like vs hate)
        if ("like" in a_text and "hate" in b_text) or ("hate" in a_text and "like" in b_text):
            return True, {
                "category": "preference",
                "clarification_hint": "Which preference is correct?",
                "detection_method": "llm",
                "avg_importance": (memory_a.importance + memory_b.importance) / 2,
            }

        return False, {}

    def get_metrics(self):
        return {"conflict_verifier_llm_call_count": self.llm_call_count}


class MockDuplicateDetector:
    """
    Mock LLM-based duplicate detector that simulates real duplicate/refinement detection.

    Distinguishes between:
    - Exact duplicates (same fact stated again)
    - Refinements (one memory is more detailed version of the other)
    - Distinct facts (different information, even if related)
    """

    def __init__(self):
        self.llm_call_count = 0

    async def is_duplicate_or_refinement(self, memory_a, memory_b, similarity_score):
        """
        Check if memories are duplicates or refinements.

        Returns:
            tuple: (is_same_or_refinement, decision) where decision is:
                - "same": Exact duplicate
                - "refinement": One is more detailed version of the other
                - "distinct": Different facts (return False)
        """
        self.llm_call_count += 1

        a_text = memory_a.text.lower()
        b_text = memory_b.text.lower()

        # Exact duplicates
        if a_text == b_text:
            return True, "same"

        # Refinement: one text contains the other AND adds meaningful detail
        if a_text in b_text:
            # b_text is more detailed
            added_words = b_text.replace(a_text, "").strip().split()
            if len(added_words) >= 2:  # Meaningful addition
                return True, "refinement"
            else:
                return True, "same"  # Just minor variation

        if b_text in a_text:
            # a_text is more detailed
            added_words = a_text.replace(b_text, "").strip().split()
            if len(added_words) >= 2:  # Meaningful addition
                return True, "refinement"
            else:
                return True, "same"  # Just minor variation

        # Check for paraphrases (high word overlap, similar structure)
        a_words = set(a_text.split())
        b_words = set(b_text.split())
        overlap = len(a_words & b_words) / max(len(a_words), len(b_words))

        if overlap > 0.8:
            # Very high overlap = likely same fact, slightly reworded
            return True, "same"

        # Check if they're talking about the same specific topic with different details
        # Example: "I work as an engineer" vs "I work as a senior engineer at Google"
        topic_patterns = [
            ("live in", "live in"),
            ("work as", "work as"),
            ("work in", "work in"),
            ("like", "like"),
            ("prefer", "prefer"),
        ]

        for pattern_a, pattern_b in topic_patterns:
            if pattern_a in a_text and pattern_b in b_text:
                # Same topic, check if it's the same object (skip articles)
                a_words = a_text.split(pattern_a)[-1].strip().split()
                b_words = b_text.split(pattern_b)[-1].strip().split()
                articles = {"a", "an", "the"}
                a_obj = next((w.rstrip(".,;") for w in a_words if w not in articles), "")
                b_obj = next((w.rstrip(".,;") for w in b_words if w not in articles), "")

                if a_obj and b_obj and (a_obj == b_obj or a_obj in b_obj or b_obj in a_obj):
                    # Same core fact, check for refinement
                    if len(b_text) > len(a_text) + 10:  # Significant detail added
                        return True, "refinement"
                    elif len(a_text) > len(b_text) + 10:
                        return True, "refinement"
                    else:
                        return True, "same"

        # Not a duplicate or refinement - distinct facts
        return False, "distinct"

    def get_metrics(self):
        return {"duplicate_detector_llm_call_count": self.llm_call_count}


@pytest.fixture
def vector_store():
    """Create REAL in-memory vector store for integration testing."""
    return InMemoryVectorStore()


@pytest.fixture
def conflict_store():
    """Create REAL in-memory conflict store for integration testing."""
    return InMemoryConflictStore()


@pytest.fixture
def embedding():
    """
    Create mock embedding service (external dependency).

    Uses deterministic hash-based embeddings to avoid loading
    heavy transformer models while remaining repeatable.
    """
    return MockEmbedding()


@pytest.fixture
def pipeline():
    """
    Create REAL classification pipeline with REAL classifier classes.

    Only the LLM/NLI backends are mocked. This fixture creates:
    - Real ClassificationPipeline orchestrator
    - Real NLIClassifier (with mocked NLI model backend)
    - Real ConflictClassifier (with mocked LLM backend)
    - Real DuplicateClassifier (with mocked LLM backend)
    - Real AutoResolutionClassifier (no external dependencies)

    This provides true integration testing of the classification logic,
    pipeline orchestration, and classifier chaining.
    """
    return MemoryClassificationPipeline(
        classifiers=[
            NLIClassifier(nli_filter=MockNLIFilter()),
            ConflictClassifier(llm_conflict_verifier=MockConflictVerifier()),
            DuplicateClassifier(llm_duplicate_detector=MockDuplicateDetector()),
            AutoResolutionClassifier(),
        ],
        strategy="tiered",
    )


@pytest.fixture
def memory_service(vector_store, conflict_store, pipeline, embedding):
    """
    Create REAL MemoryService for integration testing.

    This is the complete, production MemoryService with:
    - Real ActionExecutor executing storage operations
    - Real ClassificationPipeline orchestrating classifiers
    - Real NLI, Conflict, Duplicate, and AutoResolution classifiers
    - Real InMemoryVectorStore and InMemoryConflictStore

    Only external dependencies (LLM, NLI model, embedding model) are mocked.
    """
    return MemoryService(
        vector_store=vector_store,
        conflict_store=conflict_store,
        pipeline=pipeline,
        embedding=embedding,
    )


@pytest.mark.asyncio
async def test_add_first_memory(memory_service, vector_store):
    """Test adding the first memory (no similar memories)."""
    memory = MemoryFact(
        text="I live in Paris",
        type="fact",
        tags=[],
        user_id="user_123",
        importance=0.8,
    )

    result = await memory_service.add_memory(memory)

    # Should be added
    assert result.action == "added"
    assert result.memory_id is not None
    assert len(result.conflict_ids) == 0
    assert len(result.superseded_ids) == 0

    # Verify it's in vector store
    query_embedding = await memory_service.embedding.embed_query("Who lives in Paris?")
    stored = vector_store.find_similar_memories(
        embedding=query_embedding,
        user_id="user_123",
        threshold=0.7,
        limit=5,
    )
    assert len(stored) == 1


@pytest.mark.asyncio
async def test_add_duplicate_memory(memory_service):
    """Test adding a duplicate memory updates existing."""
    # Add first memory
    memory1 = MemoryFact(
        text="I live in Paris",
        type="fact",
        tags=[],
        user_id="user_123",
        importance=0.8,
    )
    result1 = await memory_service.add_memory(memory1)
    assert result1.action == "added"
    memory_id = result1.memory_id

    # Add duplicate (paraphrase)
    memory2 = MemoryFact(
        text="I live in Paris",
        type="fact",
        tags=[],
        user_id="user_123",
        importance=0.8,
    )
    result2 = await memory_service.add_memory(memory2)

    # Should update existing
    assert result2.action == "updated"
    assert result2.memory_id == memory_id


@pytest.mark.asyncio
async def test_add_refinement_supersedes_old(memory_service, vector_store):
    """Test adding a refinement supersedes the old memory."""
    # Add basic memory
    memory1 = MemoryFact(
        text="I work as an engineer",
        type="fact",
        tags=[],
        user_id="user_123",
        importance=0.7,
        confidence=0.6,
    )
    result1 = await memory_service.add_memory(memory1)
    assert result1.action == "added"
    old_id = result1.memory_id

    # Add refined version
    memory2 = MemoryFact(
        text="I work as a senior software engineer at Google",
        type="fact",
        tags=[],
        user_id="user_123",
        importance=0.7,
        confidence=0.8,
    )
    result2 = await memory_service.add_memory(memory2)

    # Should add new and supersede old
    assert result2.action == "added"
    assert result2.memory_id != old_id
    assert old_id in result2.superseded_ids

    # Verify old memory is archived
    old_memory = vector_store.get_by_id(old_id)
    assert old_memory is not None
    assert old_memory.payload.archived is True
    assert old_memory.payload.superseded_by == result2.memory_id


@pytest.mark.asyncio
async def test_add_conflicting_memory_creates_conflict(
    memory_service, conflict_store, vector_store
):
    """Test adding a conflicting memory creates a conflict record."""
    # Add first memory
    memory1 = MemoryFact(
        text="I live in London",
        type="fact",
        tags=[],
        user_id="user_123",
        importance=0.8,
        confidence=0.7,
    )
    result1 = await memory_service.add_memory(memory1)
    assert result1.action == "added"

    # Add conflicting memory
    memory2 = MemoryFact(
        text="I live in Paris",
        type="fact",
        tags=[],
        user_id="user_123",
        importance=0.8,
        confidence=0.7,
    )
    result2 = await memory_service.add_memory(memory2)

    # Should create conflict
    assert result2.action == "conflict"
    assert len(result2.conflict_ids) == 1
    assert result2.memory_id is None

    # Verify conflict is stored
    conflicts = conflict_store.get_pending_conflicts(user_id="user_123")
    assert len(conflicts) == 1
    assert conflicts[0].category == "location"


@pytest.mark.asyncio
async def test_auto_resolution_high_new_confidence(memory_service):
    """Test auto-resolution when new memory has much higher confidence."""
    # Add memory with low confidence
    memory1 = MemoryFact(
        text="I work as a teacher",
        type="fact",
        tags=[],
        user_id="user_123",
        importance=0.7,
        confidence=0.5,  # Low confidence
        mention_count=1,
    )
    result1 = await memory_service.add_memory(memory1)
    assert result1.action == "added"
    old_id = result1.memory_id

    # Add conflicting memory with high confidence
    memory2 = MemoryFact(
        text="I work as a doctor",
        type="fact",
        tags=[],
        user_id="user_123",
        importance=0.7,
        confidence=0.9,  # High confidence (ratio = 1.8)
        mention_count=5,
    )
    result2 = await memory_service.add_memory(memory2)

    # Should auto-resolve by superseding old memory
    assert result2.action == "added"
    assert old_id in result2.superseded_ids
    assert len(result2.conflict_ids) == 0


@pytest.mark.asyncio
async def test_auto_resolution_high_old_confidence(memory_service):
    """Test auto-resolution when old memory has much higher confidence."""
    # Add memory with high confidence
    memory1 = MemoryFact(
        text="I work as a teacher",
        type="fact",
        tags=[],
        user_id="user_123",
        importance=0.7,
        confidence=0.9,  # High confidence
        mention_count=10,
    )
    result1 = await memory_service.add_memory(memory1)
    assert result1.action == "added"
    memory_id = result1.memory_id

    # Add conflicting memory with low confidence
    memory2 = MemoryFact(
        text="I work as a doctor",
        type="fact",
        tags=[],
        user_id="user_123",
        importance=0.7,
        confidence=0.4,  # Low confidence (ratio = 0.44)
        mention_count=1,
    )
    result2 = await memory_service.add_memory(memory2)

    # Should auto-resolve by keeping old memory
    assert result2.action == "updated"
    assert result2.memory_id == memory_id


@pytest.mark.asyncio
async def test_query_memory_returns_relevant_results(memory_service):
    """Test querying memories returns relevant results."""
    # Add some memories
    memories = [
        MemoryFact(
            text="I live in Paris", type="fact", tags=[], user_id="user_123", importance=0.8
        ),
        MemoryFact(
            text="I work in London", type="fact", tags=[], user_id="user_123", importance=0.7
        ),
        MemoryFact(
            text="I like pizza", type="preference", tags=[], user_id="user_123", importance=0.5
        ),
    ]

    for memory in memories:
        await memory_service.add_memory(memory)

    # Query for location-related memories
    query_filter = MemoryQueryFilter(user_id="user_123")
    results = await memory_service.query_memory(
        query="Where do I live and work",
        filter=query_filter,
        top_k=5,
        min_score=0.5,
    )

    # Should return location-related memories
    # Note: With mock hash-based embeddings, exact results may vary,
    # but we should get at least some location-related memories
    assert len(results) >= 1
    texts = [r.text for r in results]
    # At least one location should be present
    has_location = any("Paris" in text or "London" in text for text in texts)
    assert has_location, f"Expected location-related results but got: {texts}"


@pytest.mark.asyncio
async def test_multiple_users_isolation(memory_service):
    """Test that memories are isolated per user."""
    # Add memory for user 1
    memory1 = MemoryFact(
        text="I live in Paris",
        type="fact",
        tags=[],
        user_id="user_1",
        importance=0.8,
    )
    await memory_service.add_memory(memory1)

    # Add memory for user 2
    memory2 = MemoryFact(
        text="I live in London",
        type="fact",
        tags=[],
        user_id="user_2",
        importance=0.8,
    )
    await memory_service.add_memory(memory2)

    # Query for user 1
    filter1 = MemoryQueryFilter(user_id="user_1")
    print("Fetch results 1")
    results1 = await memory_service.query_memory(
        query="Where do I live?",
        filter=filter1,
        top_k=5,
    )

    # Query for user 2
    filter2 = MemoryQueryFilter(user_id="user_2")
    print("Fetch results 2")
    results2 = await memory_service.query_memory(
        query="Where do I live?",
        filter=filter2,
        top_k=5,
    )

    # Each user should only see their own memories
    print("debug")
    print(results1)
    assert len(results1) >= 1
    assert all(r.user_id == "user_1" for r in results1)
    assert any("Paris" in r.text for r in results1)

    assert len(results2) >= 1
    assert all(r.user_id == "user_2" for r in results2)
    assert any("London" in r.text for r in results2)


@pytest.mark.asyncio
async def test_complex_scenario_multiple_memories(memory_service, vector_store, conflict_store):
    """Test a complex scenario with multiple memory operations."""
    user_id = "user_123"

    # 1. Add initial facts
    await memory_service.add_memory(
        MemoryFact(text="I live in London", type="fact", tags=[], user_id=user_id, confidence=0.8)
    )
    await memory_service.add_memory(
        MemoryFact(
            text="I work as a teacher", type="fact", tags=[], user_id=user_id, confidence=0.7
        )
    )
    await memory_service.add_memory(
        MemoryFact(
            text="I like coffee", type="preference", tags=[], user_id=user_id, confidence=0.6
        )
    )

    # 2. Add refinement (should supersede)
    result = await memory_service.add_memory(
        MemoryFact(
            text="I work as a senior teacher at Cambridge",
            type="fact",
            tags=[],
            user_id=user_id,
            confidence=0.9,
        )
    )
    assert result.action == "added"
    assert len(result.superseded_ids) > 0

    # 3. Add duplicate (should update)
    result = await memory_service.add_memory(
        MemoryFact(
            text="I like coffee", type="preference", tags=[], user_id=user_id, confidence=0.7
        )
    )
    assert result.action == "updated"

    # 4. Add conflict (should create conflict record)
    result = await memory_service.add_memory(
        MemoryFact(text="I live in Paris", type="fact", tags=[], user_id=user_id, confidence=0.8)
    )
    assert result.action == "conflict"
    assert len(result.conflict_ids) > 0

    # Verify final state
    query_filter = MemoryQueryFilter(user_id=user_id)
    all_memories = await memory_service.query_memory(
        query="facts about me",
        filter=query_filter,
        top_k=10,
        min_score=0.0,
    )

    # Should have memories (excluding archived ones)
    assert len(all_memories) >= 2

    # Should have conflict record
    conflicts = conflict_store.get_pending_conflicts(user_id=user_id)
    assert len(conflicts) == 1
