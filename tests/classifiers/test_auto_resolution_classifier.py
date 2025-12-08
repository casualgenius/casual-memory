"""
Unit tests for Auto-resolution classifier.

Tests the confidence-based auto-resolution logic including:
- High confidence ratio → MERGE (keep_new)
- Low confidence ratio → MERGE (keep_old)
- Mid confidence ratio → Keep as CONFLICT
- Error handling
- Post-processing of existing results
"""

import pytest
from casual_memory.models import MemoryFact, MemoryConflict
from casual_memory.classifiers.models import (
    MemoryPair,
    ClassificationRequest,
    ClassificationResult,
)
from casual_memory.classifiers.auto_resolution_classifier import AutoResolutionClassifier


@pytest.fixture
def conflict_result_high_new_confidence():
    """Create a CONFLICT result where new memory has much higher confidence."""
    existing = MemoryFact(
        text="I live in London",
        type="fact",
        tags=["location"],
        importance=0.8,
        user_id="user123",
        confidence=0.5,  # Lower confidence
        mention_count=2,
    )

    new = MemoryFact(
        text="I live in Paris",
        type="fact",
        tags=["location"],
        importance=0.8,
        user_id="user123",
        confidence=0.8,  # Higher confidence (ratio = 1.6)
        mention_count=5,
    )

    pair = MemoryPair(
        existing_memory=existing,
        new_memory=new,
        similarity_score=0.91,
        existing_memory_id="mem_123",
    )

    conflict = MemoryConflict(
        user_id="user123",
        memory_a_id="mem_123",
        memory_b_id="pending",
        category="location",
        similarity_score=0.91,
        avg_importance=0.8,
        clarification_hint="Where do you currently live?",
    )

    return ClassificationResult(
        pair=pair,
        classification="CONFLICT",
        classifier_name="conflict",
        metadata={"conflict": conflict},
    )


@pytest.fixture
def conflict_result_high_old_confidence():
    """Create a CONFLICT result where old memory has much higher confidence."""
    existing = MemoryFact(
        text="I live in London",
        type="fact",
        tags=["location"],
        importance=0.8,
        user_id="user123",
        confidence=0.8,  # Higher confidence
        mention_count=10,
    )

    new = MemoryFact(
        text="I live in Paris",
        type="fact",
        tags=["location"],
        importance=0.8,
        user_id="user123",
        confidence=0.4,  # Lower confidence (ratio = 0.5)
        mention_count=1,
    )

    pair = MemoryPair(
        existing_memory=existing,
        new_memory=new,
        similarity_score=0.91,
        existing_memory_id="mem_456",
    )

    conflict = MemoryConflict(
        user_id="user123",
        memory_a_id="mem_456",
        memory_b_id="pending",
        category="location",
        similarity_score=0.91,
        avg_importance=0.8,
        clarification_hint="Where do you currently live?",
    )

    return ClassificationResult(
        pair=pair,
        classification="CONFLICT",
        classifier_name="conflict",
        metadata={"conflict": conflict},
    )


@pytest.fixture
def conflict_result_similar_confidence():
    """Create a CONFLICT result where both memories have similar confidence."""
    existing = MemoryFact(
        text="I live in London",
        type="fact",
        tags=["location"],
        importance=0.8,
        user_id="user123",
        confidence=0.6,
        mention_count=4,
    )

    new = MemoryFact(
        text="I live in Paris",
        type="fact",
        tags=["location"],
        importance=0.8,
        user_id="user123",
        confidence=0.7,  # Similar confidence (ratio = 1.16)
        mention_count=5,
    )

    pair = MemoryPair(
        existing_memory=existing,
        new_memory=new,
        similarity_score=0.91,
        existing_memory_id="mem_789",
    )

    conflict = MemoryConflict(
        user_id="user123",
        memory_a_id="mem_789",
        memory_b_id="pending",
        category="location",
        similarity_score=0.91,
        avg_importance=0.8,
        clarification_hint="Where do you currently live?",
    )

    return ClassificationResult(
        pair=pair,
        classification="CONFLICT",
        classifier_name="conflict",
        metadata={"conflict": conflict},
    )


@pytest.mark.asyncio
async def test_auto_resolution_keeps_new_high_confidence(conflict_result_high_new_confidence):
    """Test that high new confidence ratio results in keep_new resolution."""
    classifier = AutoResolutionClassifier(
        supersede_threshold=1.3,
        keep_threshold=0.7,
    )

    request = ClassificationRequest(
        pairs=[],
        results=[conflict_result_high_new_confidence],
        user_id="user123",
    )

    result = await classifier.classify(request)

    # Should be reclassified to MERGE with keep_new decision
    assert len(result.results) == 1
    assert result.results[0].classification == "MERGE"
    assert result.results[0].metadata["auto_resolved"] is True
    assert result.results[0].metadata["resolution_decision"] == "keep_new"
    assert result.results[0].metadata["confidence_ratio"] == 1.6


@pytest.mark.asyncio
async def test_auto_resolution_keeps_old_high_confidence(conflict_result_high_old_confidence):
    """Test that low confidence ratio results in keep_old resolution."""
    classifier = AutoResolutionClassifier(
        supersede_threshold=1.3,
        keep_threshold=0.7,
    )

    request = ClassificationRequest(
        pairs=[],
        results=[conflict_result_high_old_confidence],
        user_id="user123",
    )

    result = await classifier.classify(request)

    # Should be reclassified to MERGE with keep_old decision
    assert len(result.results) == 1
    assert result.results[0].classification == "MERGE"
    assert result.results[0].metadata["auto_resolved"] is True
    assert result.results[0].metadata["resolution_decision"] == "keep_old"
    assert result.results[0].metadata["confidence_ratio"] == 0.5


@pytest.mark.asyncio
async def test_auto_resolution_keeps_conflict_similar_confidence(
    conflict_result_similar_confidence,
):
    """Test that similar confidence keeps conflict for manual resolution."""
    classifier = AutoResolutionClassifier(
        supersede_threshold=1.3,
        keep_threshold=0.7,
    )

    request = ClassificationRequest(
        pairs=[],
        results=[conflict_result_similar_confidence],
        user_id="user123",
    )

    result = await classifier.classify(request)

    # Should remain as CONFLICT
    assert len(result.results) == 1
    assert result.results[0].classification == "CONFLICT"
    assert result.results[0].metadata.get("auto_resolved", False) is False
    assert "confidence_ratio" in result.results[0].metadata
    assert result.results[0].metadata["confidence_ratio"] == pytest.approx(1.166, rel=0.01)


@pytest.mark.asyncio
async def test_auto_resolution_zero_old_confidence():
    """Test handling of zero old confidence (cannot calculate ratio)."""
    existing = MemoryFact(
        text="I live in London",
        type="fact",
        tags=[],
        importance=0.8,
        user_id="user123",
        confidence=0.0,  # Zero confidence
        mention_count=1,
    )

    new = MemoryFact(
        text="I live in Paris",
        type="fact",
        tags=[],
        importance=0.8,
        user_id="user123",
        confidence=0.8,
        mention_count=1,
    )

    pair = MemoryPair(
        existing_memory=existing,
        new_memory=new,
        similarity_score=0.91,
        existing_memory_id="mem_000",
    )

    conflict = MemoryConflict(
        user_id="user123",
        memory_a_id="mem_000",
        memory_b_id="pending",
        category="location",
        similarity_score=0.91,
        avg_importance=0.8,
        clarification_hint="Where do you currently live?",
    )

    conflict_result = ClassificationResult(
        pair=pair,
        classification="CONFLICT",
        classifier_name="conflict",
        metadata={"conflict": conflict},
    )

    classifier = AutoResolutionClassifier(
        supersede_threshold=1.3,
        keep_threshold=0.7,
    )

    request = ClassificationRequest(
        pairs=[],
        results=[conflict_result],
        user_id="user123",
    )

    result = await classifier.classify(request)

    # Should remain as CONFLICT (cannot calculate ratio)
    assert result.results[0].classification == "CONFLICT"


@pytest.mark.asyncio
async def test_auto_resolution_non_conflict_results_unchanged():
    """Test that non-CONFLICT results are not modified."""
    # Create MERGE and ADD results
    merge_result = ClassificationResult(
        pair=MemoryPair(
            existing_memory=MemoryFact(
                text="I live in London",
                type="fact",
                tags=[],
                importance=0.8,
                user_id="user123",
            ),
            new_memory=MemoryFact(
                text="I live in Central London",
                type="fact",
                tags=[],
                importance=0.8,
                user_id="user123",
            ),
            similarity_score=0.92,
            existing_memory_id="mem_1",
        ),
        classification="MERGE",
        classifier_name="nli",
    )

    add_result = ClassificationResult(
        pair=MemoryPair(
            existing_memory=MemoryFact(
                text="I like coffee",
                type="preference",
                tags=[],
                importance=0.5,
                user_id="user123",
            ),
            new_memory=MemoryFact(
                text="I like tea",
                type="preference",
                tags=[],
                importance=0.5,
                user_id="user123",
            ),
            similarity_score=0.87,
            existing_memory_id="mem_2",
        ),
        classification="ADD",
        classifier_name="nli",
    )

    classifier = AutoResolutionClassifier()

    request = ClassificationRequest(
        pairs=[],
        results=[merge_result, add_result],
        user_id="user123",
    )

    result = await classifier.classify(request)

    # Should remain unchanged
    assert len(result.results) == 2
    assert result.results[0].classification == "MERGE"
    assert result.results[1].classification == "ADD"


@pytest.mark.asyncio
async def test_auto_resolution_mixed_results(
    conflict_result_high_new_confidence,
    conflict_result_similar_confidence,
):
    """Test processing of mixed CONFLICT and non-CONFLICT results."""
    merge_result = ClassificationResult(
        pair=MemoryPair(
            existing_memory=MemoryFact(
                text="I like pizza",
                type="preference",
                tags=[],
                importance=0.5,
                user_id="user123",
            ),
            new_memory=MemoryFact(
                text="I really enjoy pizza",
                type="preference",
                tags=[],
                importance=0.5,
                user_id="user123",
            ),
            similarity_score=0.94,
            existing_memory_id="mem_999",
        ),
        classification="MERGE",
        classifier_name="nli",
    )

    classifier = AutoResolutionClassifier(
        supersede_threshold=1.3,
        keep_threshold=0.7,
    )

    request = ClassificationRequest(
        pairs=[],
        results=[
            merge_result,
            conflict_result_high_new_confidence,
            conflict_result_similar_confidence,
        ],
        user_id="user123",
    )

    result = await classifier.classify(request)

    # Should process all results
    assert len(result.results) == 3

    # MERGE should be unchanged
    assert result.results[0].classification == "MERGE"

    # High confidence conflict should be resolved
    assert result.results[1].classification == "MERGE"
    assert result.results[1].metadata["resolution_decision"] == "keep_new"

    # Similar confidence conflict should remain CONFLICT
    assert result.results[2].classification == "CONFLICT"


@pytest.mark.asyncio
async def test_auto_resolution_error_handling():
    """Test that errors during resolution are handled gracefully."""
    # Create result with invalid data that will cause error
    invalid_result = ClassificationResult(
        pair=MemoryPair(
            existing_memory=None,  # Invalid - will cause error
            new_memory=None,
            similarity_score=0.9,
            existing_memory_id="mem_error",
        ),
        classification="CONFLICT",
        classifier_name="conflict",
        metadata={},
    )

    classifier = AutoResolutionClassifier()

    request = ClassificationRequest(
        pairs=[],
        results=[invalid_result],
        user_id="user123",
    )

    result = await classifier.classify(request)

    # Should keep result despite error
    assert len(result.results) == 1
    assert result.results[0].classification == "CONFLICT"


def test_auto_resolution_get_metrics():
    """Test that metrics return configuration values."""
    classifier = AutoResolutionClassifier(
        supersede_threshold=1.5,
        keep_threshold=0.6,
    )

    metrics = classifier.get_metrics()

    assert metrics["auto_resolution_supersede_threshold"] == 1.5
    assert metrics["auto_resolution_keep_threshold"] == 0.6
