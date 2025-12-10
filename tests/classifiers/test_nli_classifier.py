"""
Unit tests for NLI classifier with pass-through architecture.

Tests the NLI-based classification logic including:
- High entailment → same outcome
- High neutral + low entailment → neutral outcome
- Uncertain cases → pass through (return None)
- Pass-through of existing results
- Error handling
"""

import pytest
from unittest.mock import Mock
from casual_memory.models import MemoryFact
from casual_memory.classifiers.models import SimilarMemory, SimilarityResult
from casual_memory.classifiers.nli_classifier import NLIClassifier


@pytest.fixture
def mock_nli_filter():
    """Create a mock NLI filter."""
    mock = Mock()
    mock.get_metrics.return_value = {
        "nli_prediction_count": 0,
        "nli_cache_hits": 0,
    }
    return mock


@pytest.fixture
def new_memory():
    """Create a new memory for testing."""
    return MemoryFact(
        text="I live in Central London",
        type="fact",
        tags=["location"],
        importance=0.8,
        user_id="user123",
        confidence=0.5,
        mention_count=1,
    )


@pytest.fixture
def similar_memory():
    """Create a similar memory for testing."""
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
async def test_nli_classifier_high_entailment_same(
    mock_nli_filter, new_memory, similar_memory
):
    """Test that high entailment scores result in same classification."""
    # Mock NLI to return high entailment score
    mock_nli_filter.predict.return_value = (
        "entailment",
        [0.1, 0.95, 0.05],  # [contradiction, entailment, neutral]
    )

    classifier = NLIClassifier(
        nli_filter=mock_nli_filter,
        entailment_threshold=0.85,
        neutral_threshold=0.5,
    )

    result = await classifier.classify_pair(
        new_memory, similar_memory, check_type="primary", existing_result=None
    )

    # Should classify as same
    assert result is not None
    assert result.outcome == "same"
    assert result.classifier_name == "nli"
    assert result.confidence == 0.95
    assert "nli_scores" in result.metadata


@pytest.mark.asyncio
async def test_nli_classifier_high_neutral_neutral(
    mock_nli_filter, new_memory, similar_memory
):
    """Test that high neutral + low entailment results in neutral classification."""
    # Mock NLI to return high neutral, low entailment
    mock_nli_filter.predict.return_value = (
        "neutral",
        [0.1, 0.2, 0.7],  # [contradiction, entailment, neutral]
    )

    classifier = NLIClassifier(
        nli_filter=mock_nli_filter,
        entailment_threshold=0.85,
        neutral_threshold=0.5,
    )

    result = await classifier.classify_pair(
        new_memory, similar_memory, check_type="primary", existing_result=None
    )

    # Should classify as neutral
    assert result is not None
    assert result.outcome == "neutral"
    assert result.classifier_name == "nli"
    assert result.confidence == 0.7


@pytest.mark.asyncio
async def test_nli_classifier_uncertain_pass(
    mock_nli_filter, new_memory, similar_memory
):
    """Test that uncertain cases are passed to next classifier (return None)."""
    # Mock NLI to return uncertain scores (no clear winner)
    mock_nli_filter.predict.return_value = (
        "contradiction",
        [0.6, 0.3, 0.1],  # [contradiction, entailment, neutral]
    )

    classifier = NLIClassifier(
        nli_filter=mock_nli_filter,
        entailment_threshold=0.85,
        neutral_threshold=0.5,
    )

    result = await classifier.classify_pair(
        new_memory, similar_memory, check_type="primary", existing_result=None
    )

    # Should pass to next classifier (return None)
    assert result is None


@pytest.mark.asyncio
async def test_nli_classifier_neutral_but_high_entailment_pass(
    mock_nli_filter, new_memory, similar_memory
):
    """Test that high neutral but also high entailment is passed (ambiguous)."""
    # Mock NLI to return high neutral BUT also higher entailment
    mock_nli_filter.predict.return_value = (
        "entailment",
        [0.1, 0.5, 0.4],  # [contradiction, entailment, neutral]
    )

    classifier = NLIClassifier(
        nli_filter=mock_nli_filter,
        entailment_threshold=0.85,
        neutral_threshold=0.5,
    )

    result = await classifier.classify_pair(
        new_memory, similar_memory, check_type="primary", existing_result=None
    )

    # Entailment is 0.5, below threshold of 0.85, so should pass
    # Neutral is 0.4, below threshold of 0.5, so doesn't meet neutral criteria
    assert result is None


@pytest.mark.asyncio
async def test_nli_classifier_error_handling(
    mock_nli_filter, new_memory, similar_memory
):
    """Test that NLI errors are handled gracefully."""
    # Mock NLI to raise an exception
    mock_nli_filter.predict.side_effect = Exception("NLI model failed")

    classifier = NLIClassifier(
        nli_filter=mock_nli_filter,
        entailment_threshold=0.85,
        neutral_threshold=0.5,
    )

    result = await classifier.classify_pair(
        new_memory, similar_memory, check_type="primary", existing_result=None
    )

    # Should pass to next classifier on error (return None)
    assert result is None


@pytest.mark.asyncio
async def test_nli_classifier_pass_through_existing_result(
    mock_nli_filter, new_memory, similar_memory
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

    classifier = NLIClassifier(nli_filter=mock_nli_filter)

    result = await classifier.classify_pair(
        new_memory, similar_memory, check_type="primary", existing_result=existing_result
    )

    # Should pass through the existing result
    assert result == existing_result
    assert result.outcome == "conflict"
    assert result.classifier_name == "conflict"

    # NLI should not be called
    mock_nli_filter.predict.assert_not_called()


@pytest.mark.asyncio
async def test_nli_classifier_multiple_scenarios():
    """Test various entailment/neutral combinations."""
    mock_nli_filter = Mock()

    classifier = NLIClassifier(
        nli_filter=mock_nli_filter,
        entailment_threshold=0.85,
        neutral_threshold=0.5,
    )

    new_mem = MemoryFact(
        text="Test memory",
        type="fact",
        tags=[],
        importance=0.7,
        user_id="user123",
    )

    similar_mem = SimilarMemory(
        memory_id="mem_1",
        memory=MemoryFact(
            text="Similar memory",
            type="fact",
            tags=[],
            importance=0.7,
            user_id="user123",
        ),
        similarity_score=0.90,
    )

    # Test 1: Very high entailment → same
    mock_nli_filter.predict.return_value = ("entailment", [0.02, 0.96, 0.02])
    result = await classifier.classify_pair(new_mem, similar_mem, "primary", None)
    assert result.outcome == "same"

    # Test 2: Very high neutral, low entailment → neutral
    mock_nli_filter.predict.return_value = ("neutral", [0.05, 0.15, 0.80])
    result = await classifier.classify_pair(new_mem, similar_mem, "primary", None)
    assert result.outcome == "neutral"

    # Test 3: High contradiction → pass through
    mock_nli_filter.predict.return_value = ("contradiction", [0.85, 0.10, 0.05])
    result = await classifier.classify_pair(new_mem, similar_mem, "primary", None)
    assert result is None


def test_nli_classifier_get_metrics(mock_nli_filter):
    """Test that metrics are retrieved from NLI filter."""
    mock_nli_filter.get_metrics.return_value = {
        "nli_prediction_count": 10,
        "nli_cache_hits": 3,
    }

    classifier = NLIClassifier(nli_filter=mock_nli_filter)
    metrics = classifier.get_metrics()

    assert metrics["nli_prediction_count"] == 10
    assert metrics["nli_cache_hits"] == 3
