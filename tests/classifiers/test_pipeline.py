"""
Unit tests for Memory-Centric Classification Pipeline.

Tests the pipeline orchestration logic including:
- Sequential classifier execution with pass-through
- Tiered strategy (single/tiered/all)
- Overall outcome derivation
- Early stopping logic
- Metrics aggregation
"""

from unittest.mock import AsyncMock, Mock

import pytest

from casual_memory.classifiers.models import (
    SimilarityResult,
    SimilarMemory,
)
from casual_memory.classifiers.pipeline import MemoryClassificationPipeline
from casual_memory.models import MemoryFact


@pytest.fixture
def new_memory():
    """Create a new memory for testing."""
    return MemoryFact(
        text="I live in London",
        type="fact",
        tags=[],
        importance=0.8,
        confidence=0.7,
        user_id="user123",
    )


@pytest.fixture
def similar_memories():
    """Create sample similar memories for testing."""
    mem1 = SimilarMemory(
        memory_id="mem_1",
        memory=MemoryFact(
            text="I live in Paris",
            type="fact",
            tags=[],
            importance=0.8,
            confidence=0.6,
            user_id="user123",
        ),
        similarity_score=0.95,
    )

    mem2 = SimilarMemory(
        memory_id="mem_2",
        memory=MemoryFact(
            text="I work in London",
            type="fact",
            tags=[],
            importance=0.7,
            confidence=0.5,
            user_id="user123",
        ),
        similarity_score=0.92,
    )

    mem3 = SimilarMemory(
        memory_id="mem_3",
        memory=MemoryFact(
            text="I visited London",
            type="fact",
            tags=[],
            importance=0.5,
            confidence=0.4,
            user_id="user123",
        ),
        similarity_score=0.87,
    )

    mem4 = SimilarMemory(
        memory_id="mem_4",
        memory=MemoryFact(
            text="London is great",
            type="preference",
            tags=[],
            importance=0.4,
            confidence=0.3,
            user_id="user123",
        ),
        similarity_score=0.82,
    )

    return [mem1, mem2, mem3, mem4]


def create_mock_classifier(name: str, behavior_fn):
    """
    Create a mock classifier.

    Args:
        name: Classifier name
        behavior_fn: Function(new_memory, similar_memory, check_type, existing_result) -> Optional[SimilarityResult]
    """
    mock = Mock()
    mock.__class__.__name__ = name
    mock.classify_pair = AsyncMock(side_effect=behavior_fn)
    return mock


@pytest.mark.asyncio
async def test_pipeline_pass_through_chain(new_memory, similar_memories):
    """Test that classifiers pass results through the chain."""

    def classifier1_behavior(new_mem, similar_mem, check_type, existing_result):
        # First classifier: classifies as "conflict"
        if existing_result is None:
            return SimilarityResult(
                similar_memory=similar_mem,
                outcome="conflict",
                confidence=0.9,
                classifier_name="classifier1",
                metadata={},
            )
        return existing_result

    def classifier2_behavior(new_mem, similar_mem, check_type, existing_result):
        # Second classifier: overrides conflict to superseded (like AutoResolution)
        if existing_result and existing_result.outcome == "conflict":
            return SimilarityResult(
                similar_memory=similar_mem,
                outcome="superseded",
                confidence=0.9,
                classifier_name="classifier2",
                metadata={"auto_resolved": True},
            )
        return existing_result

    classifier1 = create_mock_classifier("Classifier1", classifier1_behavior)
    classifier2 = create_mock_classifier("Classifier2", classifier2_behavior)

    pipeline = MemoryClassificationPipeline(
        classifiers=[classifier1, classifier2], strategy="single"
    )

    result = await pipeline.classify(new_memory, similar_memories)

    # Should have one result (single strategy checks only highest)
    assert len(result.similarity_results) == 1

    # Result should be superseded (overridden by classifier2)
    assert result.similarity_results[0].outcome == "superseded"
    assert result.similarity_results[0].classifier_name == "classifier2"
    assert result.similarity_results[0].metadata["auto_resolved"] is True

    # Overall outcome should be "add" (superseded → add new memory)
    assert result.overall_outcome == "add"


@pytest.mark.asyncio
async def test_pipeline_strategy_single(new_memory, similar_memories):
    """Test single strategy - only checks highest-scoring memory."""

    def classifier_behavior(new_mem, similar_mem, check_type, existing_result):
        if existing_result is None:
            return SimilarityResult(
                similar_memory=similar_mem,
                outcome="neutral",
                confidence=0.8,
                classifier_name="classifier",
                metadata={"check_type": check_type},
            )
        return existing_result

    classifier = create_mock_classifier("Classifier", classifier_behavior)
    pipeline = MemoryClassificationPipeline(classifiers=[classifier], strategy="single")

    result = await pipeline.classify(new_memory, similar_memories)

    # Should only check first memory
    assert len(result.similarity_results) == 1
    assert result.similarity_results[0].similar_memory.memory_id == "mem_1"
    assert result.similarity_results[0].metadata["check_type"] == "primary"


@pytest.mark.asyncio
async def test_pipeline_strategy_tiered(new_memory, similar_memories):
    """Test tiered strategy - checks highest + high-scoring secondaries."""

    check_types_seen = []

    def classifier_behavior(new_mem, similar_mem, check_type, existing_result):
        if existing_result is None:
            check_types_seen.append((similar_mem.memory_id, check_type))
            return SimilarityResult(
                similar_memory=similar_mem,
                outcome="neutral",
                confidence=0.8,
                classifier_name="classifier",
                metadata={"check_type": check_type},
            )
        return existing_result

    classifier = create_mock_classifier("Classifier", classifier_behavior)
    pipeline = MemoryClassificationPipeline(
        classifiers=[classifier],
        strategy="tiered",
        secondary_conflict_threshold=0.90,
        max_secondary_checks=2,
    )

    _result = await pipeline.classify(new_memory, similar_memories)

    # Should check mem_1 (0.95) as primary, mem_2 (0.92) as secondary
    # mem_3 (0.87) and mem_4 (0.82) are below threshold
    assert len(check_types_seen) == 2
    assert check_types_seen[0] == ("mem_1", "primary")
    assert check_types_seen[1] == ("mem_2", "secondary")


@pytest.mark.asyncio
async def test_pipeline_strategy_all(new_memory, similar_memories):
    """Test all strategy - checks all similar memories."""

    def classifier_behavior(new_mem, similar_mem, check_type, existing_result):
        if existing_result is None:
            return SimilarityResult(
                similar_memory=similar_mem,
                outcome="neutral",
                confidence=0.8,
                classifier_name="classifier",
                metadata={},
            )
        return existing_result

    classifier = create_mock_classifier("Classifier", classifier_behavior)
    pipeline = MemoryClassificationPipeline(classifiers=[classifier], strategy="all")

    result = await pipeline.classify(new_memory, similar_memories)

    # Should check all 4 memories
    assert len(result.similarity_results) == 4


@pytest.mark.asyncio
async def test_pipeline_early_stopping_on_conflict(new_memory, similar_memories):
    """Test that pipeline stops early when finding a conflict."""

    memories_checked = []

    def classifier_behavior(new_mem, similar_mem, check_type, existing_result):
        if existing_result is None:
            memories_checked.append(similar_mem.memory_id)
            # First memory: conflict
            if similar_mem.memory_id == "mem_1":
                return SimilarityResult(
                    similar_memory=similar_mem,
                    outcome="conflict",
                    confidence=0.9,
                    classifier_name="classifier",
                    metadata={},
                )
            # Others: neutral
            return SimilarityResult(
                similar_memory=similar_mem,
                outcome="neutral",
                confidence=0.8,
                classifier_name="classifier",
                metadata={},
            )
        return existing_result

    classifier = create_mock_classifier("Classifier", classifier_behavior)
    pipeline = MemoryClassificationPipeline(classifiers=[classifier], strategy="all")

    result = await pipeline.classify(new_memory, similar_memories)

    # Should stop after first memory (conflict triggers early stopping)
    assert len(memories_checked) == 1
    assert memories_checked[0] == "mem_1"
    assert result.overall_outcome == "conflict"


@pytest.mark.asyncio
async def test_pipeline_early_stopping_on_same(new_memory, similar_memories):
    """Test that pipeline stops early when finding same memory."""

    memories_checked = []

    def classifier_behavior(new_mem, similar_mem, check_type, existing_result):
        if existing_result is None:
            memories_checked.append(similar_mem.memory_id)
            # First memory: same
            if similar_mem.memory_id == "mem_1":
                return SimilarityResult(
                    similar_memory=similar_mem,
                    outcome="same",
                    confidence=0.9,
                    classifier_name="classifier",
                    metadata={},
                )
            return SimilarityResult(
                similar_memory=similar_mem,
                outcome="neutral",
                confidence=0.8,
                classifier_name="classifier",
                metadata={},
            )
        return existing_result

    classifier = create_mock_classifier("Classifier", classifier_behavior)
    pipeline = MemoryClassificationPipeline(classifiers=[classifier], strategy="all")

    result = await pipeline.classify(new_memory, similar_memories)

    # Should stop after first memory (same triggers early stopping)
    assert len(memories_checked) == 1
    assert result.overall_outcome == "skip"


@pytest.mark.asyncio
async def test_pipeline_overall_outcome_derivation(new_memory, similar_memories):
    """Test that overall outcome is correctly derived from similarity results."""

    # Test 1: conflict outcome
    def conflict_behavior(new_mem, similar_mem, check_type, existing_result):
        if existing_result is None and similar_mem.memory_id == "mem_1":
            return SimilarityResult(
                similar_memory=similar_mem,
                outcome="conflict",
                confidence=0.9,
                classifier_name="classifier",
                metadata={},
            )
        return None

    classifier = create_mock_classifier("Classifier", conflict_behavior)
    pipeline = MemoryClassificationPipeline(classifiers=[classifier], strategy="single")
    result = await pipeline.classify(new_memory, similar_memories[:1])
    assert result.overall_outcome == "conflict"

    # Test 2: skip outcome (same)
    def same_behavior(new_mem, similar_mem, check_type, existing_result):
        if existing_result is None:
            return SimilarityResult(
                similar_memory=similar_mem,
                outcome="same",
                confidence=0.9,
                classifier_name="classifier",
                metadata={},
            )
        return None

    classifier = create_mock_classifier("Classifier", same_behavior)
    pipeline = MemoryClassificationPipeline(classifiers=[classifier], strategy="single")
    result = await pipeline.classify(new_memory, similar_memories[:1])
    assert result.overall_outcome == "skip"

    # Test 3: add outcome (superseded)
    def superseded_behavior(new_mem, similar_mem, check_type, existing_result):
        if existing_result is None:
            return SimilarityResult(
                similar_memory=similar_mem,
                outcome="superseded",
                confidence=0.9,
                classifier_name="classifier",
                metadata={},
            )
        return None

    classifier = create_mock_classifier("Classifier", superseded_behavior)
    pipeline = MemoryClassificationPipeline(classifiers=[classifier], strategy="single")
    result = await pipeline.classify(new_memory, similar_memories[:1])
    assert result.overall_outcome == "add"

    # Test 4: add outcome (neutral)
    def neutral_behavior(new_mem, similar_mem, check_type, existing_result):
        if existing_result is None:
            return SimilarityResult(
                similar_memory=similar_mem,
                outcome="neutral",
                confidence=0.8,
                classifier_name="classifier",
                metadata={},
            )
        return None

    classifier = create_mock_classifier("Classifier", neutral_behavior)
    pipeline = MemoryClassificationPipeline(classifiers=[classifier], strategy="single")
    result = await pipeline.classify(new_memory, similar_memories[:1])
    assert result.overall_outcome == "add"


@pytest.mark.asyncio
async def test_pipeline_no_similar_memories(new_memory):
    """Test pipeline with no similar memories."""

    def classifier_behavior(new_mem, similar_mem, check_type, existing_result):
        # Should never be called
        raise AssertionError("Classifier should not be called with no similar memories")

    classifier = create_mock_classifier("Classifier", classifier_behavior)
    pipeline = MemoryClassificationPipeline(classifiers=[classifier], strategy="all")

    result = await pipeline.classify(new_memory, [])

    # Should return add with no similarity results
    assert result.overall_outcome == "add"
    assert len(result.similarity_results) == 0


@pytest.mark.asyncio
async def test_pipeline_default_neutral_result(new_memory, similar_memories):
    """Test that pipeline returns default neutral when no classifier classifies."""

    def pass_through_behavior(new_mem, similar_mem, check_type, existing_result):
        # All classifiers return None
        return None

    classifier1 = create_mock_classifier("Classifier1", pass_through_behavior)
    classifier2 = create_mock_classifier("Classifier2", pass_through_behavior)

    pipeline = MemoryClassificationPipeline(
        classifiers=[classifier1, classifier2], strategy="single"
    )

    result = await pipeline.classify(new_memory, similar_memories)

    # Should get default neutral result
    assert len(result.similarity_results) == 1
    assert result.similarity_results[0].outcome == "neutral"
    assert result.similarity_results[0].classifier_name == "default"
    assert result.similarity_results[0].confidence == 0.5
    assert result.similarity_results[0].metadata["reason"] == "no_classifier_confident"


@pytest.mark.asyncio
async def test_pipeline_derived_properties(new_memory, similar_memories):
    """Test derived properties on MemoryClassificationResult.

    Note: Pipeline has early stopping - stops when it finds conflict or same.
    So we order outcomes to test all properties:
    - mem_1: superseded (doesn't stop)
    - mem_2: neutral (doesn't stop)
    - mem_3: same (stops here - triggers overall outcome = skip)
    - mem_4: never reached due to early stopping
    """

    def classifier_behavior(new_mem, similar_mem, check_type, existing_result):
        if existing_result is None:
            # Order matters: superseded, neutral, then same (which stops)
            outcomes = {
                "mem_1": "superseded",
                "mem_2": "neutral",
                "mem_3": "same",
                "mem_4": "conflict",  # Never reached
            }
            return SimilarityResult(
                similar_memory=similar_mem,
                outcome=outcomes[similar_mem.memory_id],
                confidence=0.9,
                classifier_name="classifier",
                metadata={},
            )
        return existing_result

    classifier = create_mock_classifier("Classifier", classifier_behavior)
    pipeline = MemoryClassificationPipeline(classifiers=[classifier], strategy="all")

    result = await pipeline.classify(new_memory, similar_memories)

    # Test supersedes property (mem_1 was superseded)
    assert result.supersedes == ["mem_1"]

    # Test same_as property (mem_3 was same, triggers early stop)
    assert result.same_as == "mem_3"

    # Test conflicts_with property (mem_4 never reached due to early stop)
    assert result.conflicts_with == []

    # Overall outcome should be skip (due to same)
    assert result.overall_outcome == "skip"


def test_pipeline_get_metrics():
    """Test that pipeline returns correct metrics."""

    classifier1 = Mock()
    classifier1.__class__.__name__ = "Classifier1"

    classifier2 = Mock()
    classifier2.__class__.__name__ = "Classifier2"

    pipeline = MemoryClassificationPipeline(
        classifiers=[classifier1, classifier2],
        strategy="tiered",
        secondary_conflict_threshold=0.92,
        max_secondary_checks=5,
    )

    metrics = pipeline.get_metrics()

    assert metrics["strategy"] == "tiered"
    assert metrics["secondary_conflict_threshold"] == 0.92
    assert metrics["max_secondary_checks"] == 5
    assert metrics["classifier_count"] == 2
    assert metrics["classifiers"] == ["Classifier1", "Classifier2"]


@pytest.mark.asyncio
async def test_pipeline_classifier_order(new_memory, similar_memories):
    """Test that classifiers execute in the correct order."""
    execution_order = []

    def create_tracking_classifier(name):
        def behavior(new_mem, similar_mem, check_type, existing_result):
            execution_order.append(name)
            return existing_result

        return create_mock_classifier(name, behavior)

    classifier_a = create_tracking_classifier("a")
    classifier_b = create_tracking_classifier("b")
    classifier_c = create_tracking_classifier("c")

    pipeline = MemoryClassificationPipeline(
        classifiers=[classifier_a, classifier_b, classifier_c], strategy="single"
    )

    await pipeline.classify(new_memory, similar_memories)

    # Should execute in order
    assert execution_order == ["a", "b", "c"]
