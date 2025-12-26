"""
Unit tests for Auto-resolution classifier.

Tests the confidence-based auto-resolution logic including:
- High confidence ratio → superseded (keep_new)
- Low confidence ratio → same (keep_old)
- Mid confidence ratio → Keep as conflict
- Pass-through behavior
- Error handling
"""

import pytest

from casual_memory.classifiers.auto_resolution_classifier import AutoResolutionClassifier
from casual_memory.classifiers.models import SimilarityResult, SimilarMemory
from casual_memory.models import MemoryFact


@pytest.fixture
def new_memory_high_confidence():
    """Create a new memory with high confidence."""
    return MemoryFact(
        text="I live in Paris",
        type="fact",
        tags=["location"],
        importance=0.8,
        confidence=0.8,  # High confidence
        user_id="user123",
    )


@pytest.fixture
def new_memory_low_confidence():
    """Create a new memory with low confidence."""
    return MemoryFact(
        text="I live in Paris",
        type="fact",
        tags=["location"],
        importance=0.8,
        confidence=0.4,  # Low confidence
        user_id="user123",
    )


@pytest.fixture
def similar_memory_low_confidence():
    """Create a similar memory with low confidence."""
    return SimilarMemory(
        memory_id="mem_123",
        memory=MemoryFact(
            text="I live in London",
            type="fact",
            tags=["location"],
            importance=0.8,
            confidence=0.5,  # Low confidence
            user_id="user123",
        ),
        similarity_score=0.91,
    )


@pytest.fixture
def similar_memory_high_confidence():
    """Create a similar memory with high confidence."""
    return SimilarMemory(
        memory_id="mem_456",
        memory=MemoryFact(
            text="I live in London",
            type="fact",
            tags=["location"],
            importance=0.8,
            confidence=0.8,  # High confidence
            user_id="user123",
        ),
        similarity_score=0.91,
    )


@pytest.fixture
def conflict_result():
    """Create a conflict result."""

    def _create(similar_memory):
        return SimilarityResult(
            similar_memory=similar_memory,
            outcome="conflict",
            confidence=0.9,
            classifier_name="conflict",
            metadata={"category": "location"},
        )

    return _create


@pytest.mark.asyncio
async def test_auto_resolve_high_new_confidence(
    new_memory_high_confidence, similar_memory_low_confidence, conflict_result
):
    """Test that high new confidence ratio auto-resolves to superseded."""
    classifier = AutoResolutionClassifier(supersede_threshold=1.3, keep_threshold=0.7)

    existing_conflict = conflict_result(similar_memory_low_confidence)

    # Ratio = 0.8 / 0.5 = 1.6 (≥ 1.3) → superseded
    result = await classifier.classify_pair(
        new_memory_high_confidence,
        similar_memory_low_confidence,
        check_type="primary",
        existing_result=existing_conflict,
    )

    assert result.outcome == "superseded"
    assert result.classifier_name == "auto_resolution"
    assert result.metadata["auto_resolved"] is True
    assert result.metadata["resolution_decision"] == "keep_new"
    assert result.metadata["confidence_ratio"] == 1.6


@pytest.mark.asyncio
async def test_auto_resolve_high_old_confidence(
    new_memory_low_confidence, similar_memory_high_confidence, conflict_result
):
    """Test that high old confidence ratio auto-resolves to same."""
    classifier = AutoResolutionClassifier(supersede_threshold=1.3, keep_threshold=0.7)

    existing_conflict = conflict_result(similar_memory_high_confidence)

    # Ratio = 0.4 / 0.8 = 0.5 (≤ 0.7) → same
    result = await classifier.classify_pair(
        new_memory_low_confidence,
        similar_memory_high_confidence,
        check_type="primary",
        existing_result=existing_conflict,
    )

    assert result.outcome == "same"
    assert result.classifier_name == "auto_resolution"
    assert result.metadata["auto_resolved"] is True
    assert result.metadata["resolution_decision"] == "keep_old"
    assert result.metadata["confidence_ratio"] == 0.5


@pytest.mark.asyncio
async def test_auto_resolve_mid_confidence_keeps_conflict(
    new_memory_high_confidence, similar_memory_high_confidence, conflict_result
):
    """Test that mid-range confidence ratio keeps conflict."""
    classifier = AutoResolutionClassifier(supersede_threshold=1.3, keep_threshold=0.7)

    existing_conflict = conflict_result(similar_memory_high_confidence)

    # Ratio = 0.8 / 0.8 = 1.0 (between 0.7 and 1.3) → keep conflict
    result = await classifier.classify_pair(
        new_memory_high_confidence,
        similar_memory_high_confidence,
        check_type="primary",
        existing_result=existing_conflict,
    )

    assert result.outcome == "conflict"
    assert result.classifier_name == "conflict"  # Original classifier
    assert result.metadata["auto_resolved"] is False
    assert result.metadata["confidence_ratio"] == 1.0


@pytest.mark.asyncio
async def test_pass_through_non_conflict(new_memory_high_confidence, similar_memory_low_confidence):
    """Test that non-conflict results pass through unchanged."""
    classifier = AutoResolutionClassifier()

    # Test with superseded result
    superseded_result = SimilarityResult(
        similar_memory=similar_memory_low_confidence,
        outcome="superseded",
        confidence=0.9,
        classifier_name="duplicate",
        metadata={},
    )

    result = await classifier.classify_pair(
        new_memory_high_confidence,
        similar_memory_low_confidence,
        check_type="primary",
        existing_result=superseded_result,
    )

    assert result == superseded_result
    assert result.outcome == "superseded"


@pytest.mark.asyncio
async def test_pass_through_none_result(new_memory_high_confidence, similar_memory_low_confidence):
    """Test that None result passes through."""
    classifier = AutoResolutionClassifier()

    result = await classifier.classify_pair(
        new_memory_high_confidence,
        similar_memory_low_confidence,
        check_type="primary",
        existing_result=None,
    )

    assert result is None


@pytest.mark.asyncio
async def test_zero_confidence_handling(new_memory_high_confidence, conflict_result):
    """Test handling when old confidence is zero."""
    classifier = AutoResolutionClassifier()

    # Create similar memory with zero confidence
    similar_memory_zero = SimilarMemory(
        memory_id="mem_789",
        memory=MemoryFact(
            text="I live in London",
            type="fact",
            tags=[],
            importance=0.5,
            confidence=0.0,  # Zero confidence
            user_id="user123",
        ),
        similarity_score=0.90,
    )

    existing_conflict = conflict_result(similar_memory_zero)

    result = await classifier.classify_pair(
        new_memory_high_confidence,
        similar_memory_zero,
        check_type="primary",
        existing_result=existing_conflict,
    )

    # Should keep as conflict (cannot calculate ratio)
    assert result.outcome == "conflict"


@pytest.mark.asyncio
async def test_custom_thresholds(
    new_memory_high_confidence, similar_memory_low_confidence, conflict_result
):
    """Test that custom thresholds are respected."""
    # Set more lenient thresholds
    classifier = AutoResolutionClassifier(supersede_threshold=1.5, keep_threshold=0.5)

    existing_conflict = conflict_result(similar_memory_low_confidence)

    # Ratio = 0.8 / 0.5 = 1.6 (≥ 1.5) → superseded
    result = await classifier.classify_pair(
        new_memory_high_confidence,
        similar_memory_low_confidence,
        check_type="primary",
        existing_result=existing_conflict,
    )

    assert result.outcome == "superseded"
    assert result.metadata["confidence_ratio"] == 1.6


def test_get_metrics():
    """Test that classifier returns correct metrics."""
    classifier = AutoResolutionClassifier(supersede_threshold=1.5, keep_threshold=0.6)

    metrics = classifier.get_metrics()

    assert metrics["auto_resolution_supersede_threshold"] == 1.5
    assert metrics["auto_resolution_keep_threshold"] == 0.6
