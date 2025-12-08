"""
Unit tests for Duplicate classifier.

Tests the LLM-based duplicate/refinement detection logic including:
- Duplicate/refinement detection → MERGE classification
- Distinct facts → ADD classification
- Error handling with fallback to ADD
- Always classifies (never passes)
"""

import pytest
from unittest.mock import Mock, AsyncMock
from casual_memory.models import MemoryFact
from casual_memory.classifiers.models import MemoryPair, ClassificationRequest
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
def duplicate_memory_pair():
    """Create a pair of duplicate/refinement memories."""
    existing = MemoryFact(
        text="I live in London",
        type="fact",
        tags=["location"],
        importance=0.8,
        user_id="user123",
        confidence=0.7,
        mention_count=3,
    )

    new = MemoryFact(
        text="I live in Central London",
        type="fact",
        tags=["location"],
        importance=0.8,
        user_id="user123",
        confidence=0.5,
        mention_count=1,
    )

    return MemoryPair(
        existing_memory=existing,
        new_memory=new,
        similarity_score=0.92,
        existing_memory_id="mem_123",
    )


@pytest.fixture
def distinct_memory_pair():
    """Create a pair of distinct memories."""
    existing = MemoryFact(
        text="I live in Bangkok",
        type="fact",
        tags=["location"],
        importance=0.8,
        user_id="user123",
        confidence=0.7,
        mention_count=3,
    )

    new = MemoryFact(
        text="I work in Bangkok",
        type="fact",
        tags=["work", "location"],
        importance=0.8,
        user_id="user123",
        confidence=0.5,
        mention_count=1,
    )

    return MemoryPair(
        existing_memory=existing,
        new_memory=new,
        similarity_score=0.87,
        existing_memory_id="mem_456",
    )


@pytest.mark.asyncio
async def test_duplicate_classifier_detects_duplicate(mock_llm_detector, duplicate_memory_pair):
    """Test that duplicates/refinements result in MERGE classification."""
    # Mock LLM to detect duplicate
    mock_llm_detector.is_duplicate_or_refinement.return_value = True

    classifier = DuplicateClassifier(llm_duplicate_detector=mock_llm_detector)

    request = ClassificationRequest(
        pairs=[duplicate_memory_pair],
        results=[],
        user_id="user123",
    )

    result = await classifier.classify(request)

    # Should classify as MERGE
    assert len(result.results) == 1
    assert len(result.pairs) == 0  # All pairs should be classified
    assert result.results[0].classification == "MERGE"
    assert result.results[0].classifier_name == "duplicate"
    assert result.results[0].metadata["duplicate_type"] == "duplicate_or_refinement"


@pytest.mark.asyncio
async def test_duplicate_classifier_detects_distinct(mock_llm_detector, distinct_memory_pair):
    """Test that distinct facts result in ADD classification."""
    # Mock LLM to detect distinct facts
    mock_llm_detector.is_duplicate_or_refinement.return_value = False

    classifier = DuplicateClassifier(llm_duplicate_detector=mock_llm_detector)

    request = ClassificationRequest(
        pairs=[distinct_memory_pair],
        results=[],
        user_id="user123",
    )

    result = await classifier.classify(request)

    # Should classify as ADD
    assert len(result.results) == 1
    assert len(result.pairs) == 0  # All pairs should be classified
    assert result.results[0].classification == "ADD"
    assert result.results[0].classifier_name == "duplicate"
    assert result.results[0].metadata["duplicate_type"] == "distinct_facts"


@pytest.mark.asyncio
async def test_duplicate_classifier_error_fallback_add(mock_llm_detector, duplicate_memory_pair):
    """Test that errors result in ADD classification (conservative fallback)."""
    # Mock LLM to raise exception
    mock_llm_detector.is_duplicate_or_refinement.side_effect = Exception("LLM call failed")

    classifier = DuplicateClassifier(llm_duplicate_detector=mock_llm_detector)

    request = ClassificationRequest(
        pairs=[duplicate_memory_pair],
        results=[],
        user_id="user123",
    )

    result = await classifier.classify(request)

    # Should default to ADD on error (conservative)
    assert len(result.results) == 1
    assert len(result.pairs) == 0
    assert result.results[0].classification == "ADD"
    assert result.results[0].metadata["duplicate_type"] == "error_fallback"
    assert "error" in result.results[0].metadata


@pytest.mark.asyncio
async def test_duplicate_classifier_always_classifies(mock_llm_detector):
    """Test that duplicate classifier always classifies (never passes)."""
    # Create multiple pairs
    pair1 = MemoryPair(
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
    )

    pair2 = MemoryPair(
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
    )

    # Mock LLM to return different results
    mock_llm_detector.is_duplicate_or_refinement.side_effect = [True, False]

    classifier = DuplicateClassifier(llm_duplicate_detector=mock_llm_detector)

    request = ClassificationRequest(
        pairs=[pair1, pair2],
        results=[],
        user_id="user123",
    )

    result = await classifier.classify(request)

    # Should classify ALL pairs (never pass)
    assert len(result.results) == 2
    assert len(result.pairs) == 0

    assert result.results[0].classification == "MERGE"
    assert result.results[1].classification == "ADD"


@pytest.mark.asyncio
async def test_duplicate_classifier_paraphrase_detection(mock_llm_detector):
    """Test detection of paraphrases."""
    paraphrase_pair = MemoryPair(
        existing_memory=MemoryFact(
            text="I enjoy reading books",
            type="preference",
            tags=[],
            importance=0.6,
            user_id="user123",
        ),
        new_memory=MemoryFact(
            text="I love to read books",
            type="preference",
            tags=[],
            importance=0.6,
            user_id="user123",
        ),
        similarity_score=0.94,
        existing_memory_id="mem_789",
    )

    mock_llm_detector.is_duplicate_or_refinement.return_value = True

    classifier = DuplicateClassifier(llm_duplicate_detector=mock_llm_detector)

    request = ClassificationRequest(
        pairs=[paraphrase_pair],
        results=[],
        user_id="user123",
    )

    result = await classifier.classify(request)

    # Paraphrase should be merged
    assert result.results[0].classification == "MERGE"


@pytest.mark.asyncio
async def test_duplicate_classifier_refinement_detection(mock_llm_detector):
    """Test detection of refinements (more specific versions)."""
    refinement_pair = MemoryPair(
        existing_memory=MemoryFact(
            text="I work as an engineer",
            type="fact",
            tags=[],
            importance=0.7,
            user_id="user123",
        ),
        new_memory=MemoryFact(
            text="I work as a senior software engineer at Google",
            type="fact",
            tags=[],
            importance=0.7,
            user_id="user123",
        ),
        similarity_score=0.89,
        existing_memory_id="mem_999",
    )

    mock_llm_detector.is_duplicate_or_refinement.return_value = True

    classifier = DuplicateClassifier(llm_duplicate_detector=mock_llm_detector)

    request = ClassificationRequest(
        pairs=[refinement_pair],
        results=[],
        user_id="user123",
    )

    result = await classifier.classify(request)

    # Refinement should be merged
    assert result.results[0].classification == "MERGE"


@pytest.mark.asyncio
async def test_duplicate_classifier_multiple_distinct_facts(mock_llm_detector):
    """Test detection of multiple distinct facts about same topic."""
    # Different skills - both should be kept
    skill1_pair = MemoryPair(
        existing_memory=MemoryFact(
            text="I can speak French",
            type="fact",
            tags=[],
            importance=0.6,
            user_id="user123",
        ),
        new_memory=MemoryFact(
            text="I can speak Spanish",
            type="fact",
            tags=[],
            importance=0.6,
            user_id="user123",
        ),
        similarity_score=0.86,
        existing_memory_id="mem_111",
    )

    mock_llm_detector.is_duplicate_or_refinement.return_value = False

    classifier = DuplicateClassifier(llm_duplicate_detector=mock_llm_detector)

    request = ClassificationRequest(
        pairs=[skill1_pair],
        results=[],
        user_id="user123",
    )

    result = await classifier.classify(request)

    # Distinct facts should be added
    assert result.results[0].classification == "ADD"


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
