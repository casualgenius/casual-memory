"""
Unit tests for Duplicate classifier with pass-through architecture.

Tests the LLM-based duplicate/refinement detection logic including:
- Duplicate/refinement detection → superseded (if longer) or same (if similar length)
- Distinct facts → neutral
- Pass-through of existing results
- Secondary check optimization
- Error handling with fallback to neutral
"""

import pytest
from unittest.mock import Mock, AsyncMock
from casual_memory.models import MemoryFact
from casual_memory.classifiers.models import SimilarMemory, SimilarityResult
from casual_memory.classifiers.duplicate_classifier import DuplicateClassifier


@pytest.fixture
def mock_llm_detector():
    """Create a mock LLM duplicate detector."""
    mock = Mock()
    mock.is_duplicate_or_refinement = AsyncMock()
    mock.get_metrics.return_value = {
        "duplicate_detector_llm_call_count": 0,
        "duplicate_detector_llm_success_count": 0,
    }
    return mock


@pytest.fixture
def new_memory_refinement():
    """Create a new memory that's a refinement (longer)."""
    return MemoryFact(
        text="I live in Central London near the Thames",
        type="fact",
        tags=["location"],
        importance=0.8,
        user_id="user123",
        confidence=0.5,
        mention_count=1,
    )


@pytest.fixture
def new_memory_duplicate():
    """Create a new memory that's a duplicate (similar length)."""
    return MemoryFact(
        text="I live in London",
        type="fact",
        tags=["location"],
        importance=0.8,
        user_id="user123",
        confidence=0.5,
        mention_count=1,
    )


@pytest.fixture
def new_memory_distinct():
    """Create a new memory that's distinct."""
    return MemoryFact(
        text="I work in Bangkok",
        type="fact",
        tags=["work", "location"],
        importance=0.8,
        user_id="user123",
        confidence=0.5,
        mention_count=1,
    )


@pytest.fixture
def similar_memory():
    """Create a similar memory."""
    return SimilarMemory(
        memory_id="mem_123",
        memory=MemoryFact(
            text="I live in London",
            type="fact",
            tags=["location"],
            importance=0.8,
            user_id="user123",
            confidence=0.7,
            mention_count=3,
        ),
        similarity_score=0.92,
    )


@pytest.mark.asyncio
async def test_duplicate_classifier_detects_refinement_superseded(
    mock_llm_detector, new_memory_refinement, similar_memory
):
    """Test that refinements (longer new memory) result in superseded."""
    # Mock LLM to detect duplicate/refinement
    mock_llm_detector.is_duplicate_or_refinement.return_value = True

    classifier = DuplicateClassifier(llm_duplicate_detector=mock_llm_detector)

    result = await classifier.classify_pair(
        new_memory_refinement,
        similar_memory,
        check_type="primary",
        existing_result=None,
    )

    # Should classify as superseded (new is 20%+ longer)
    assert result is not None
    assert result.outcome == "superseded"
    assert result.classifier_name == "duplicate"
    assert result.confidence == 0.9
    assert result.metadata["duplicate_type"] == "refinement"
    assert result.metadata["length_ratio"] > 1.2


@pytest.mark.asyncio
async def test_duplicate_classifier_detects_duplicate_same(
    mock_llm_detector, new_memory_duplicate, similar_memory
):
    """Test that duplicates (similar length) result in same."""
    # Mock LLM to detect duplicate
    mock_llm_detector.is_duplicate_or_refinement.return_value = True

    classifier = DuplicateClassifier(llm_duplicate_detector=mock_llm_detector)

    result = await classifier.classify_pair(
        new_memory_duplicate,
        similar_memory,
        check_type="primary",
        existing_result=None,
    )

    # Should classify as same (similar length)
    assert result is not None
    assert result.outcome == "same"
    assert result.classifier_name == "duplicate"
    assert result.confidence == 0.9
    assert result.metadata["duplicate_type"] == "duplicate"
    assert result.metadata["length_ratio"] <= 1.2


@pytest.mark.asyncio
async def test_duplicate_classifier_detects_distinct(
    mock_llm_detector, new_memory_distinct, similar_memory
):
    """Test that distinct facts result in neutral classification."""
    # Mock LLM to detect distinct facts
    mock_llm_detector.is_duplicate_or_refinement.return_value = False

    classifier = DuplicateClassifier(llm_duplicate_detector=mock_llm_detector)

    result = await classifier.classify_pair(
        new_memory_distinct,
        similar_memory,
        check_type="primary",
        existing_result=None,
    )

    # Should classify as neutral
    assert result is not None
    assert result.outcome == "neutral"
    assert result.classifier_name == "duplicate"
    assert result.confidence == 0.8
    assert result.metadata["duplicate_type"] == "distinct_facts"


@pytest.mark.asyncio
async def test_duplicate_classifier_pass_through_existing_result(
    mock_llm_detector, new_memory_duplicate, similar_memory
):
    """Test that existing results are passed through unchanged."""
    # Create an existing result from a previous classifier
    existing_result = SimilarityResult(
        similar_memory=similar_memory,
        outcome="conflict",
        confidence=0.9,
        classifier_name="conflict",
        metadata={"detection_method": "llm"},
    )

    classifier = DuplicateClassifier(llm_duplicate_detector=mock_llm_detector)

    result = await classifier.classify_pair(
        new_memory_duplicate,
        similar_memory,
        check_type="primary",
        existing_result=existing_result,
    )

    # Should pass through the existing result
    assert result == existing_result
    assert result.outcome == "conflict"
    assert result.classifier_name == "conflict"

    # LLM should not be called
    mock_llm_detector.is_duplicate_or_refinement.assert_not_called()


@pytest.mark.asyncio
async def test_duplicate_classifier_skip_secondary_check(
    mock_llm_detector, new_memory_duplicate, similar_memory
):
    """Test that duplicate detection is skipped on secondary checks (expensive LLM call)."""
    # Mock LLM (shouldn't be called)
    mock_llm_detector.is_duplicate_or_refinement.return_value = True

    classifier = DuplicateClassifier(llm_duplicate_detector=mock_llm_detector)

    result = await classifier.classify_pair(
        new_memory_duplicate,
        similar_memory,
        check_type="secondary",
        existing_result=None,
    )

    # Should pass through on secondary checks
    assert result is None

    # LLM should not be called for secondary checks
    mock_llm_detector.is_duplicate_or_refinement.assert_not_called()


@pytest.mark.asyncio
async def test_duplicate_classifier_error_fallback_neutral(
    mock_llm_detector, new_memory_duplicate, similar_memory
):
    """Test that errors result in neutral classification (conservative fallback)."""
    # Mock LLM to raise exception
    mock_llm_detector.is_duplicate_or_refinement.side_effect = Exception(
        "LLM call failed"
    )

    classifier = DuplicateClassifier(llm_duplicate_detector=mock_llm_detector)

    result = await classifier.classify_pair(
        new_memory_duplicate,
        similar_memory,
        check_type="primary",
        existing_result=None,
    )

    # Should default to neutral on error (conservative)
    assert result is not None
    assert result.outcome == "neutral"
    assert result.confidence == 0.5
    assert result.metadata["duplicate_type"] == "error_fallback"
    assert "error" in result.metadata


@pytest.mark.asyncio
async def test_duplicate_classifier_paraphrase_detection():
    """Test detection of paraphrases."""
    mock_llm_detector = Mock()
    mock_llm_detector.is_duplicate_or_refinement = AsyncMock(return_value=True)
    mock_llm_detector.get_metrics.return_value = {}

    classifier = DuplicateClassifier(llm_duplicate_detector=mock_llm_detector)

    new_pref = MemoryFact(
        text="I love to read books",
        type="preference",
        tags=[],
        importance=0.6,
        user_id="user123",
    )
    similar_pref = SimilarMemory(
        memory_id="mem_789",
        memory=MemoryFact(
            text="I enjoy reading books",
            type="preference",
            tags=[],
            importance=0.6,
            user_id="user123",
        ),
        similarity_score=0.94,
    )

    result = await classifier.classify_pair(new_pref, similar_pref, "primary", None)

    # Paraphrase should be same (similar length)
    assert result.outcome == "same"


@pytest.mark.asyncio
async def test_duplicate_classifier_refinement_detection():
    """Test detection of refinements (more specific versions)."""
    mock_llm_detector = Mock()
    mock_llm_detector.is_duplicate_or_refinement = AsyncMock(return_value=True)
    mock_llm_detector.get_metrics.return_value = {}

    classifier = DuplicateClassifier(llm_duplicate_detector=mock_llm_detector)

    new_job = MemoryFact(
        text="I work as a senior software engineer at Google in Mountain View",
        type="fact",
        tags=[],
        importance=0.7,
        user_id="user123",
    )
    similar_job = SimilarMemory(
        memory_id="mem_999",
        memory=MemoryFact(
            text="I work as an engineer",
            type="fact",
            tags=[],
            importance=0.7,
            user_id="user123",
        ),
        similarity_score=0.89,
    )

    result = await classifier.classify_pair(new_job, similar_job, "primary", None)

    # Refinement should be superseded (new is much longer)
    assert result.outcome == "superseded"
    assert result.metadata["duplicate_type"] == "refinement"


@pytest.mark.asyncio
async def test_duplicate_classifier_multiple_distinct_facts():
    """Test detection of multiple distinct facts about same topic."""
    mock_llm_detector = Mock()
    mock_llm_detector.is_duplicate_or_refinement = AsyncMock(return_value=False)
    mock_llm_detector.get_metrics.return_value = {}

    classifier = DuplicateClassifier(llm_duplicate_detector=mock_llm_detector)

    # Different skills - both should be kept
    new_skill = MemoryFact(
        text="I can speak Spanish",
        type="fact",
        tags=[],
        importance=0.6,
        user_id="user123",
    )
    similar_skill = SimilarMemory(
        memory_id="mem_111",
        memory=MemoryFact(
            text="I can speak French",
            type="fact",
            tags=[],
            importance=0.6,
            user_id="user123",
        ),
        similarity_score=0.86,
    )

    result = await classifier.classify_pair(new_skill, similar_skill, "primary", None)

    # Distinct facts should be neutral
    assert result.outcome == "neutral"
    assert result.metadata["duplicate_type"] == "distinct_facts"


def test_duplicate_classifier_get_metrics(mock_llm_detector):
    """Test that metrics are retrieved from detector."""
    mock_llm_detector.get_metrics.return_value = {
        "duplicate_detector_llm_call_count": 8,
        "duplicate_detector_llm_success_count": 7,
        "duplicate_detector_heuristic_fallback_count": 1,
    }

    classifier = DuplicateClassifier(llm_duplicate_detector=mock_llm_detector)
    metrics = classifier.get_metrics()

    assert metrics["duplicate_detector_llm_call_count"] == 8
    assert metrics["duplicate_detector_llm_success_count"] == 7
    assert metrics["duplicate_detector_heuristic_fallback_count"] == 1
