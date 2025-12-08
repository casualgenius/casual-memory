"""
Unit tests for NLI classifier.

Tests the NLI-based classification logic including:
- High entailment → MERGE
- High neutral + low entailment → ADD
- Uncertain cases → Pass to next classifier
- Error handling
"""

import pytest
from unittest.mock import Mock
from casual_memory.models import MemoryFact
from casual_memory.classifiers.models import MemoryPair, ClassificationRequest
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
def sample_memory_pair():
    """Create a sample memory pair for testing."""
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


@pytest.mark.asyncio
async def test_nli_classifier_high_entailment_merge(mock_nli_filter, sample_memory_pair):
    """Test that high entailment scores result in MERGE classification."""
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

    request = ClassificationRequest(
        pairs=[sample_memory_pair],
        results=[],
        user_id="user123",
    )

    result = await classifier.classify(request)

    # Should classify as MERGE
    assert len(result.results) == 1
    assert len(result.pairs) == 0
    assert result.results[0].classification == "MERGE"
    assert result.results[0].classifier_name == "nli"
    assert result.results[0].confidence == 0.95
    assert "nli_scores" in result.results[0].metadata


@pytest.mark.asyncio
async def test_nli_classifier_high_neutral_add(mock_nli_filter, sample_memory_pair):
    """Test that high neutral + low entailment results in ADD classification."""
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

    request = ClassificationRequest(
        pairs=[sample_memory_pair],
        results=[],
        user_id="user123",
    )

    result = await classifier.classify(request)

    # Should classify as ADD
    assert len(result.results) == 1
    assert len(result.pairs) == 0
    assert result.results[0].classification == "ADD"
    assert result.results[0].classifier_name == "nli"
    assert result.results[0].confidence == 0.7


@pytest.mark.asyncio
async def test_nli_classifier_uncertain_pass(mock_nli_filter, sample_memory_pair):
    """Test that uncertain cases are passed to next classifier."""
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

    request = ClassificationRequest(
        pairs=[sample_memory_pair],
        results=[],
        user_id="user123",
    )

    result = await classifier.classify(request)

    # Should pass to next classifier (not classify)
    assert len(result.results) == 0
    assert len(result.pairs) == 1


@pytest.mark.asyncio
async def test_nli_classifier_neutral_but_high_entailment_pass(mock_nli_filter, sample_memory_pair):
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

    request = ClassificationRequest(
        pairs=[sample_memory_pair],
        results=[],
        user_id="user123",
    )

    result = await classifier.classify(request)

    # Entailment is 0.5, below threshold of 0.85, so should pass
    # Neutral is 0.4, below threshold of 0.5, so doesn't meet ADD criteria
    assert len(result.results) == 0
    assert len(result.pairs) == 1


@pytest.mark.asyncio
async def test_nli_classifier_error_handling(mock_nli_filter, sample_memory_pair):
    """Test that NLI errors are handled gracefully."""
    # Mock NLI to raise an exception
    mock_nli_filter.predict.side_effect = Exception("NLI model failed")

    classifier = NLIClassifier(
        nli_filter=mock_nli_filter,
        entailment_threshold=0.85,
        neutral_threshold=0.5,
    )

    request = ClassificationRequest(
        pairs=[sample_memory_pair],
        results=[],
        user_id="user123",
    )

    result = await classifier.classify(request)

    # Should pass to next classifier on error
    assert len(result.results) == 0
    assert len(result.pairs) == 1


@pytest.mark.asyncio
async def test_nli_classifier_multiple_pairs(mock_nli_filter):
    """Test classification of multiple pairs."""
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

    pair3 = MemoryPair(
        existing_memory=MemoryFact(
            text="I work as a teacher",
            type="fact",
            tags=[],
            importance=0.7,
            user_id="user123",
        ),
        new_memory=MemoryFact(
            text="I work as a doctor",
            type="fact",
            tags=[],
            importance=0.7,
            user_id="user123",
        ),
        similarity_score=0.91,
        existing_memory_id="mem_3",
    )

    # Mock NLI to return different results for each pair
    mock_nli_filter.predict.side_effect = [
        ("entailment", [0.1, 0.95, 0.05]),  # pair1: high entailment → MERGE
        ("neutral", [0.1, 0.2, 0.7]),  # pair2: high neutral → ADD
        ("contradiction", [0.8, 0.1, 0.1]),  # pair3: contradiction → PASS
    ]

    classifier = NLIClassifier(
        nli_filter=mock_nli_filter,
        entailment_threshold=0.85,
        neutral_threshold=0.5,
    )

    request = ClassificationRequest(
        pairs=[pair1, pair2, pair3],
        results=[],
        user_id="user123",
    )

    result = await classifier.classify(request)

    # Should classify 2, pass 1
    assert len(result.results) == 2
    assert len(result.pairs) == 1

    # Check classifications
    assert result.results[0].classification == "MERGE"
    assert result.results[0].pair.existing_memory_id == "mem_1"

    assert result.results[1].classification == "ADD"
    assert result.results[1].pair.existing_memory_id == "mem_2"

    assert result.pairs[0].existing_memory_id == "mem_3"


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
