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
    classifier = AutoResolutionClassifier(
        base_supersede_threshold=1.3,
        base_keep_threshold=0.7,
        use_importance_weighting=False,
    )

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
    classifier = AutoResolutionClassifier(
        base_supersede_threshold=1.3,
        base_keep_threshold=0.7,
        use_importance_weighting=False,
    )

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
    classifier = AutoResolutionClassifier(
        base_supersede_threshold=1.3,
        base_keep_threshold=0.7,
        use_importance_weighting=False,
    )

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
    classifier = AutoResolutionClassifier(
        base_supersede_threshold=1.5,
        base_keep_threshold=0.5,
        use_importance_weighting=False,
    )

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
    classifier = AutoResolutionClassifier(
        base_supersede_threshold=1.5,
        base_keep_threshold=0.6,
        use_importance_weighting=False,
    )

    metrics = classifier.get_metrics()

    assert metrics["auto_resolution_base_supersede_threshold"] == 1.5
    assert metrics["auto_resolution_base_keep_threshold"] == 0.6
    assert metrics["auto_resolution_importance_weighting"] is False


# ============================================================================
# Importance-Weighted Auto-Resolution Tests
# ============================================================================


def test_importance_factor_calculation():
    """Test importance factor calculation for threshold scaling."""
    classifier = AutoResolutionClassifier(use_importance_weighting=True)

    # At importance 0.6 (baseline): factor = 1.0
    assert classifier._calculate_importance_factor(0.6) == 1.0

    # At importance 0.7: factor ≈ 1.4
    # normalized = (0.7 - 0.6) / 0.2 = 0.5
    # factor = 2^0.5 ≈ 1.414
    factor_07 = classifier._calculate_importance_factor(0.7)
    assert 1.4 < factor_07 < 1.5

    # At importance 0.8: factor = 2.0 (already at cap)
    # normalized = (0.8 - 0.6) / 0.2 = 1.0
    # factor = 2^1.0 = 2.0
    assert classifier._calculate_importance_factor(0.8) == 2.0

    # At importance 1.0: factor = 2.0 (capped)
    assert classifier._calculate_importance_factor(1.0) == 2.0

    # At importance 0.5: factor ≈ 0.71
    # normalized = (0.5 - 0.6) / 0.2 = -0.5
    # factor = 2^-0.5 ≈ 0.707
    factor_05 = classifier._calculate_importance_factor(0.5)
    assert 0.7 < factor_05 < 0.72

    # Above 1.0 should still cap at 2.0
    assert classifier._calculate_importance_factor(1.2) == 2.0


def test_adaptive_thresholds_high_importance():
    """Test that high importance memories have stricter thresholds."""
    classifier = AutoResolutionClassifier(
        base_supersede_threshold=1.3,
        base_keep_threshold=0.7,
        use_importance_weighting=True,
    )

    # High importance memories (1.0)
    high_imp_new = MemoryFact(
        text="I am allergic to peanuts",
        type="fact",
        tags=["allergy"],
        importance=1.0,
        confidence=0.8,
        user_id="user123",
    )
    high_imp_old = MemoryFact(
        text="I can eat peanuts",
        type="fact",
        tags=["allergy"],
        importance=1.0,
        confidence=0.5,
        user_id="user123",
    )

    supersede, keep = classifier._get_adaptive_thresholds(high_imp_new, high_imp_old)

    # At importance 1.0, factor = 2.0
    # supersede_threshold = 1.3 * 2.0 = 2.6
    # keep_threshold = 0.7 / 2.0 = 0.35
    assert supersede == 2.6
    assert keep == 0.35


def test_adaptive_thresholds_low_importance():
    """Test that low importance memories use base thresholds."""
    classifier = AutoResolutionClassifier(
        base_supersede_threshold=1.3,
        base_keep_threshold=0.7,
        use_importance_weighting=True,
    )

    # Low importance memories (0.6)
    low_imp_new = MemoryFact(
        text="I prefer tea",
        type="preference",
        tags=["beverage"],
        importance=0.6,
        confidence=0.8,
        user_id="user123",
    )
    low_imp_old = MemoryFact(
        text="I prefer coffee",
        type="preference",
        tags=["beverage"],
        importance=0.6,
        confidence=0.5,
        user_id="user123",
    )

    supersede, keep = classifier._get_adaptive_thresholds(low_imp_new, low_imp_old)

    # At importance 0.6, factor = 1.0 (base thresholds)
    assert supersede == 1.3
    assert keep == 0.7


@pytest.mark.asyncio
async def test_importance_weighted_prevents_auto_resolve_high_importance(conflict_result):
    """Test that high importance conflicts DON'T auto-resolve with marginal confidence differences."""
    classifier = AutoResolutionClassifier(
        base_supersede_threshold=1.3,
        base_keep_threshold=0.7,
        use_importance_weighting=True,
    )

    # High importance memories
    new_memory = MemoryFact(
        text="I am allergic to peanuts",
        type="fact",
        tags=["allergy"],
        importance=1.0,  # Critical importance
        confidence=0.8,
        user_id="user123",
    )

    similar_memory = SimilarMemory(
        memory_id="mem_123",
        memory=MemoryFact(
            text="I can eat peanuts",
            type="fact",
            tags=["allergy"],
            importance=1.0,  # Critical importance
            confidence=0.5,
            user_id="user123",
        ),
        similarity_score=0.91,
    )

    existing_conflict = conflict_result(similar_memory)

    # Ratio = 0.8 / 0.5 = 1.6
    # At importance 1.0: supersede_threshold = 2.6, keep_threshold = 0.35
    # 1.6 < 2.6, so it should NOT auto-resolve (stays as conflict)
    result = await classifier.classify_pair(
        new_memory,
        similar_memory,
        check_type="primary",
        existing_result=existing_conflict,
    )

    assert result.outcome == "conflict"  # NOT auto-resolved
    assert result.metadata["auto_resolved"] is False
    assert result.metadata["confidence_ratio"] == 1.6
    assert result.metadata["supersede_threshold"] == 2.6
    assert result.metadata["keep_threshold"] == 0.35


@pytest.mark.asyncio
async def test_importance_weighted_allows_auto_resolve_low_importance(conflict_result):
    """Test that low importance conflicts DO auto-resolve with same confidence ratios."""
    classifier = AutoResolutionClassifier(
        base_supersede_threshold=1.3,
        base_keep_threshold=0.7,
        use_importance_weighting=True,
    )

    # Low importance memories
    new_memory = MemoryFact(
        text="I prefer tea",
        type="preference",
        tags=["beverage"],
        importance=0.6,  # Low importance
        confidence=0.8,
        user_id="user123",
    )

    similar_memory = SimilarMemory(
        memory_id="mem_456",
        memory=MemoryFact(
            text="I prefer coffee",
            type="preference",
            tags=["beverage"],
            importance=0.6,  # Low importance
            confidence=0.5,
            user_id="user123",
        ),
        similarity_score=0.91,
    )

    existing_conflict = conflict_result(similar_memory)

    # Ratio = 0.8 / 0.5 = 1.6
    # At importance 0.6: supersede_threshold = 1.3, keep_threshold = 0.7
    # 1.6 >= 1.3, so it SHOULD auto-resolve to superseded
    result = await classifier.classify_pair(
        new_memory,
        similar_memory,
        check_type="primary",
        existing_result=existing_conflict,
    )

    assert result.outcome == "superseded"  # Auto-resolved!
    assert result.metadata["auto_resolved"] is True
    assert result.metadata["resolution_decision"] == "keep_new"
    assert result.metadata["confidence_ratio"] == 1.6
    assert result.metadata["supersede_threshold"] == 1.3
    assert result.metadata["keep_threshold"] == 0.7


@pytest.mark.asyncio
async def test_importance_weighted_very_strong_evidence_resolves_high_importance(conflict_result):
    """Test that high importance conflicts CAN auto-resolve with very strong confidence differences."""
    classifier = AutoResolutionClassifier(
        base_supersede_threshold=1.3,
        base_keep_threshold=0.7,
        use_importance_weighting=True,
    )

    # High importance memories
    new_memory = MemoryFact(
        text="I am allergic to peanuts",
        type="fact",
        tags=["allergy"],
        importance=1.0,  # Critical importance
        confidence=0.9,  # Very high confidence
        user_id="user123",
    )

    similar_memory = SimilarMemory(
        memory_id="mem_789",
        memory=MemoryFact(
            text="I can eat peanuts",
            type="fact",
            tags=["allergy"],
            importance=1.0,  # Critical importance
            confidence=0.3,  # Very low confidence
            user_id="user123",
        ),
        similarity_score=0.91,
    )

    existing_conflict = conflict_result(similar_memory)

    # Ratio = 0.9 / 0.3 = 3.0
    # At importance 1.0: supersede_threshold = 2.6
    # 3.0 >= 2.6, so it SHOULD auto-resolve even for critical info
    result = await classifier.classify_pair(
        new_memory,
        similar_memory,
        check_type="primary",
        existing_result=existing_conflict,
    )

    assert result.outcome == "superseded"  # Auto-resolved with strong evidence
    assert result.metadata["auto_resolved"] is True
    assert result.metadata["resolution_decision"] == "keep_new"
    assert result.metadata["confidence_ratio"] == 3.0
    assert result.metadata["supersede_threshold"] == 2.6


@pytest.mark.asyncio
async def test_importance_weighted_medium_importance(conflict_result):
    """Test adaptive thresholds for medium importance memories."""
    classifier = AutoResolutionClassifier(
        base_supersede_threshold=1.3,
        base_keep_threshold=0.7,
        use_importance_weighting=True,
    )

    # Medium importance memories (0.8)
    new_memory = MemoryFact(
        text="I live in Paris",
        type="fact",
        tags=["location"],
        importance=0.8,  # Medium importance
        confidence=0.8,
        user_id="user123",
    )

    similar_memory = SimilarMemory(
        memory_id="mem_999",
        memory=MemoryFact(
            text="I live in London",
            type="fact",
            tags=["location"],
            importance=0.8,  # Medium importance
            confidence=0.5,
            user_id="user123",
        ),
        similarity_score=0.91,
    )

    existing_conflict = conflict_result(similar_memory)

    # Ratio = 0.8 / 0.5 = 1.6
    # At importance 0.8: factor = 2.0
    # supersede_threshold = 1.3 * 2.0 = 2.6
    # keep_threshold = 0.7 / 2.0 = 0.35
    # 1.6 < 2.6, so should NOT auto-resolve
    result = await classifier.classify_pair(
        new_memory,
        similar_memory,
        check_type="primary",
        existing_result=existing_conflict,
    )

    assert result.outcome == "conflict"  # Not quite enough evidence
    assert result.metadata["auto_resolved"] is False
    assert result.metadata["confidence_ratio"] == 1.6
    # Check that thresholds were adjusted to 2x stricter
    assert result.metadata["supersede_threshold"] == 2.6
    assert result.metadata["keep_threshold"] == 0.35
