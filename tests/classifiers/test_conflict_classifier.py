"""
Unit tests for Conflict classifier.

Tests the LLM-based conflict detection logic including:
- Conflict detection → CONFLICT classification
- No conflict → Pass to next classifier
- Auto-resolution metadata handling
- Error handling
"""

import pytest
from unittest.mock import Mock, AsyncMock
from casual_memory.models import MemoryFact, MemoryConflict
from casual_memory.classifiers.models import MemoryPair, ClassificationRequest
from casual_memory.classifiers.conflict_classifier import ConflictClassifier


@pytest.fixture
def mock_llm_verifier():
    """Create a mock LLM conflict verifier."""
    mock = Mock()
    mock.verify_conflict = AsyncMock()
    mock.get_metrics.return_value = {
        "conflict_verifier_llm_call_count": 0,
        "conflict_verifier_llm_success_count": 0,
    }
    return mock


@pytest.fixture
def conflicting_memory_pair():
    """Create a pair of conflicting memories."""
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
        text="I live in Paris",
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
        similarity_score=0.91,
        existing_memory_id="mem_123",
    )


@pytest.fixture
def non_conflicting_memory_pair():
    """Create a pair of non-conflicting memories."""
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
        tags=["location", "work"],
        importance=0.8,
        user_id="user123",
        confidence=0.5,
        mention_count=1,
    )

    return MemoryPair(
        existing_memory=existing,
        new_memory=new,
        similarity_score=0.88,
        existing_memory_id="mem_456",
    )


@pytest.mark.asyncio
async def test_conflict_classifier_detects_conflict(mock_llm_verifier, conflicting_memory_pair):
    """Test that detected conflicts result in CONFLICT classification."""
    # Mock LLM to detect conflict
    mock_llm_verifier.verify_conflict.return_value = (True, "llm")

    classifier = ConflictClassifier(llm_conflict_verifier=mock_llm_verifier)

    request = ClassificationRequest(
        pairs=[conflicting_memory_pair],
        results=[],
        user_id="user123",
    )

    result = await classifier.classify(request)

    # Should classify as CONFLICT
    assert len(result.results) == 1
    assert len(result.pairs) == 0
    assert result.results[0].classification == "CONFLICT"
    assert result.results[0].classifier_name == "conflict"

    # Should have conflict metadata
    assert "detection_method" in result.results[0].metadata
    assert result.results[0].metadata["detection_method"] == "llm"
    assert "conflict" in result.results[0].metadata

    # Verify conflict object structure
    conflict = result.results[0].metadata["conflict"]
    assert isinstance(conflict, MemoryConflict)
    assert conflict.memory_a_id == "mem_123"
    assert conflict.memory_b_id == "pending"
    assert conflict.user_id == "user123"


@pytest.mark.asyncio
async def test_conflict_classifier_no_conflict_pass(mock_llm_verifier, non_conflicting_memory_pair):
    """Test that non-conflicts are passed to next classifier."""
    # Mock LLM to not detect conflict
    mock_llm_verifier.verify_conflict.return_value = (False, "llm")

    classifier = ConflictClassifier(llm_conflict_verifier=mock_llm_verifier)

    request = ClassificationRequest(
        pairs=[non_conflicting_memory_pair],
        results=[],
        user_id="user123",
    )

    result = await classifier.classify(request)

    # Should pass to next classifier
    assert len(result.results) == 0
    assert len(result.pairs) == 1


@pytest.mark.asyncio
async def test_conflict_classifier_heuristic_fallback(mock_llm_verifier, conflicting_memory_pair):
    """Test that heuristic fallback is captured in metadata."""
    # Mock LLM to use heuristic fallback
    mock_llm_verifier.verify_conflict.return_value = (True, "heuristic_fallback")

    classifier = ConflictClassifier(llm_conflict_verifier=mock_llm_verifier)

    request = ClassificationRequest(
        pairs=[conflicting_memory_pair],
        results=[],
        user_id="user123",
    )

    result = await classifier.classify(request)

    # Should classify as CONFLICT with heuristic method
    assert len(result.results) == 1
    assert result.results[0].classification == "CONFLICT"
    assert result.results[0].metadata["detection_method"] == "heuristic_fallback"


@pytest.mark.asyncio
async def test_conflict_classifier_error_handling(mock_llm_verifier, conflicting_memory_pair):
    """Test that errors are handled gracefully."""
    # Mock LLM to raise exception
    mock_llm_verifier.verify_conflict.side_effect = Exception("LLM call failed")

    classifier = ConflictClassifier(llm_conflict_verifier=mock_llm_verifier)

    request = ClassificationRequest(
        pairs=[conflicting_memory_pair],
        results=[],
        user_id="user123",
    )

    result = await classifier.classify(request)

    # Should pass to next classifier on error
    assert len(result.results) == 0
    assert len(result.pairs) == 1


@pytest.mark.asyncio
async def test_conflict_classifier_multiple_pairs(mock_llm_verifier):
    """Test classification of multiple pairs."""
    pair1 = MemoryPair(
        existing_memory=MemoryFact(
            text="I live in London",
            type="fact",
            tags=[],
            importance=0.8,
            user_id="user123",
        ),
        new_memory=MemoryFact(
            text="I live in Paris",
            type="fact",
            tags=[],
            importance=0.8,
            user_id="user123",
        ),
        similarity_score=0.91,
        existing_memory_id="mem_1",
    )

    pair2 = MemoryPair(
        existing_memory=MemoryFact(
            text="I live in Bangkok",
            type="fact",
            tags=[],
            importance=0.8,
            user_id="user123",
        ),
        new_memory=MemoryFact(
            text="I work in Bangkok",
            type="fact",
            tags=[],
            importance=0.8,
            user_id="user123",
        ),
        similarity_score=0.88,
        existing_memory_id="mem_2",
    )

    # Mock LLM to return different results
    mock_llm_verifier.verify_conflict.side_effect = [
        (True, "llm"),  # pair1: conflict
        (False, "llm"),  # pair2: no conflict
    ]

    classifier = ConflictClassifier(llm_conflict_verifier=mock_llm_verifier)

    request = ClassificationRequest(
        pairs=[pair1, pair2],
        results=[],
        user_id="user123",
    )

    result = await classifier.classify(request)

    # Should classify 1, pass 1
    assert len(result.results) == 1
    assert len(result.pairs) == 1

    assert result.results[0].classification == "CONFLICT"
    assert result.results[0].pair.existing_memory_id == "mem_1"
    assert result.pairs[0].existing_memory_id == "mem_2"


@pytest.mark.asyncio
async def test_conflict_classifier_categorization(mock_llm_verifier):
    """Test that conflicts are categorized correctly."""
    # Location conflict
    location_pair = MemoryPair(
        existing_memory=MemoryFact(
            text="I live in London",
            type="fact",
            tags=[],
            importance=0.8,
            user_id="user123",
        ),
        new_memory=MemoryFact(
            text="I live in Paris",
            type="fact",
            tags=[],
            importance=0.8,
            user_id="user123",
        ),
        similarity_score=0.91,
        existing_memory_id="mem_1",
    )

    mock_llm_verifier.verify_conflict.return_value = (True, "llm")

    classifier = ConflictClassifier(llm_conflict_verifier=mock_llm_verifier)

    request = ClassificationRequest(
        pairs=[location_pair],
        results=[],
        user_id="user123",
    )

    result = await classifier.classify(request)

    conflict = result.results[0].metadata["conflict"]
    assert conflict.category == "location"


@pytest.mark.asyncio
async def test_conflict_classifier_preference_category(mock_llm_verifier):
    """Test preference conflict categorization."""
    preference_pair = MemoryPair(
        existing_memory=MemoryFact(
            text="I love coffee",
            type="preference",
            tags=[],
            importance=0.6,
            user_id="user123",
        ),
        new_memory=MemoryFact(
            text="I hate coffee",
            type="preference",
            tags=[],
            importance=0.6,
            user_id="user123",
        ),
        similarity_score=0.93,
        existing_memory_id="mem_2",
    )

    mock_llm_verifier.verify_conflict.return_value = (True, "llm")

    classifier = ConflictClassifier(llm_conflict_verifier=mock_llm_verifier)

    request = ClassificationRequest(
        pairs=[preference_pair],
        results=[],
        user_id="user123",
    )

    result = await classifier.classify(request)

    conflict = result.results[0].metadata["conflict"]
    assert conflict.category == "preference"


def test_conflict_classifier_get_metrics(mock_llm_verifier):
    """Test that metrics are retrieved from verifier."""
    mock_llm_verifier.get_metrics.return_value = {
        "conflict_verifier_llm_call_count": 5,
        "conflict_verifier_llm_success_count": 4,
    }

    classifier = ConflictClassifier(llm_conflict_verifier=mock_llm_verifier)
    metrics = classifier.get_metrics()

    assert metrics["conflict_verifier_llm_call_count"] == 5
    assert metrics["conflict_verifier_llm_success_count"] == 4
