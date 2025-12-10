"""
Unit tests for Conflict classifier with pass-through architecture.

Tests the LLM-based conflict detection logic including:
- Conflict detection → conflict outcome
- No conflict → pass through (return None)
- Categorization and clarification hints
- Pass-through of existing results
- Error handling
"""

import pytest
from unittest.mock import Mock, AsyncMock
from casual_memory.models import MemoryFact
from casual_memory.classifiers.models import SimilarMemory, SimilarityResult
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
def new_memory_conflicting():
    """Create a new memory that conflicts."""
    return MemoryFact(
        text="I live in Paris",
        type="fact",
        tags=["location"],
        importance=0.8,
        user_id="user123",
        confidence=0.5,
        mention_count=1,
    )


@pytest.fixture
def new_memory_non_conflicting():
    """Create a new memory that doesn't conflict."""
    return MemoryFact(
        text="I work in Bangkok",
        type="fact",
        tags=["location", "work"],
        importance=0.8,
        user_id="user123",
        confidence=0.5,
        mention_count=1,
    )


@pytest.fixture
def similar_memory_location():
    """Create a similar memory about location."""
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
        similarity_score=0.91,
    )


@pytest.mark.asyncio
async def test_conflict_classifier_detects_conflict(
    mock_llm_verifier, new_memory_conflicting, similar_memory_location
):
    """Test that detected conflicts result in conflict classification."""
    # Mock LLM to detect conflict
    mock_llm_verifier.verify_conflict.return_value = (True, "llm")

    classifier = ConflictClassifier(llm_conflict_verifier=mock_llm_verifier)

    result = await classifier.classify_pair(
        new_memory_conflicting,
        similar_memory_location,
        check_type="primary",
        existing_result=None,
    )

    # Should classify as conflict
    assert result is not None
    assert result.outcome == "conflict"
    assert result.classifier_name == "conflict"
    assert result.confidence == 0.9

    # Should have conflict metadata
    assert "detection_method" in result.metadata
    assert result.metadata["detection_method"] == "llm"
    assert "category" in result.metadata
    assert result.metadata["category"] == "location"
    assert "clarification_hint" in result.metadata


@pytest.mark.asyncio
async def test_conflict_classifier_no_conflict_pass(
    mock_llm_verifier, new_memory_non_conflicting, similar_memory_location
):
    """Test that non-conflicts are passed to next classifier (return None)."""
    # Mock LLM to not detect conflict
    mock_llm_verifier.verify_conflict.return_value = (False, "llm")

    classifier = ConflictClassifier(llm_conflict_verifier=mock_llm_verifier)

    result = await classifier.classify_pair(
        new_memory_non_conflicting,
        similar_memory_location,
        check_type="primary",
        existing_result=None,
    )

    # Should pass to next classifier
    assert result is None


@pytest.mark.asyncio
async def test_conflict_classifier_pass_through_existing_result(
    mock_llm_verifier, new_memory_conflicting, similar_memory_location
):
    """Test that existing results are passed through unchanged."""
    # Create an existing result from a previous classifier
    existing_result = SimilarityResult(
        similar_memory=similar_memory_location,
        outcome="same",
        confidence=0.95,
        classifier_name="nli",
        metadata={"nli_scores": [0.1, 0.95, 0.05]},
    )

    classifier = ConflictClassifier(llm_conflict_verifier=mock_llm_verifier)

    result = await classifier.classify_pair(
        new_memory_conflicting,
        similar_memory_location,
        check_type="primary",
        existing_result=existing_result,
    )

    # Should pass through the existing result
    assert result == existing_result
    assert result.outcome == "same"
    assert result.classifier_name == "nli"

    # LLM should not be called
    mock_llm_verifier.verify_conflict.assert_not_called()


@pytest.mark.asyncio
async def test_conflict_classifier_secondary_check(
    mock_llm_verifier, new_memory_conflicting, similar_memory_location
):
    """Test that conflict detection works on secondary checks."""
    # Mock LLM to detect conflict
    mock_llm_verifier.verify_conflict.return_value = (True, "llm")

    classifier = ConflictClassifier(llm_conflict_verifier=mock_llm_verifier)

    result = await classifier.classify_pair(
        new_memory_conflicting,
        similar_memory_location,
        check_type="secondary",
        existing_result=None,
    )

    # Should detect conflict on secondary check too
    assert result is not None
    assert result.outcome == "conflict"

    # Verify LLM was called
    mock_llm_verifier.verify_conflict.assert_called_once()


@pytest.mark.asyncio
async def test_conflict_classifier_heuristic_fallback(
    mock_llm_verifier, new_memory_conflicting, similar_memory_location
):
    """Test that heuristic fallback is captured in metadata."""
    # Mock LLM to use heuristic fallback
    mock_llm_verifier.verify_conflict.return_value = (True, "heuristic_fallback")

    classifier = ConflictClassifier(llm_conflict_verifier=mock_llm_verifier)

    result = await classifier.classify_pair(
        new_memory_conflicting,
        similar_memory_location,
        check_type="primary",
        existing_result=None,
    )

    # Should classify as conflict with heuristic method
    assert result is not None
    assert result.outcome == "conflict"
    assert result.metadata["detection_method"] == "heuristic_fallback"


@pytest.mark.asyncio
async def test_conflict_classifier_error_handling(
    mock_llm_verifier, new_memory_conflicting, similar_memory_location
):
    """Test that errors are handled gracefully."""
    # Mock LLM to raise exception
    mock_llm_verifier.verify_conflict.side_effect = Exception("LLM call failed")

    classifier = ConflictClassifier(llm_conflict_verifier=mock_llm_verifier)

    result = await classifier.classify_pair(
        new_memory_conflicting,
        similar_memory_location,
        check_type="primary",
        existing_result=None,
    )

    # Should pass to next classifier on error (return None)
    assert result is None


@pytest.mark.asyncio
async def test_conflict_classifier_categorization():
    """Test that conflicts are categorized correctly."""
    mock_llm_verifier = Mock()
    mock_llm_verifier.verify_conflict = AsyncMock(return_value=(True, "llm"))
    mock_llm_verifier.get_metrics.return_value = {}

    classifier = ConflictClassifier(llm_conflict_verifier=mock_llm_verifier)

    # Location conflict
    new_location = MemoryFact(
        text="I live in Paris",
        type="fact",
        tags=[],
        importance=0.8,
        user_id="user123",
    )
    similar_location = SimilarMemory(
        memory_id="mem_1",
        memory=MemoryFact(
            text="I live in London",
            type="fact",
            tags=[],
            importance=0.8,
            user_id="user123",
        ),
        similarity_score=0.91,
    )

    result = await classifier.classify_pair(
        new_location, similar_location, "primary", None
    )
    assert result.metadata["category"] == "location"

    # Job conflict
    new_job = MemoryFact(
        text="I work as a teacher",
        type="fact",
        tags=[],
        importance=0.7,
        user_id="user123",
    )
    similar_job = SimilarMemory(
        memory_id="mem_2",
        memory=MemoryFact(
            text="I work as a doctor",
            type="fact",
            tags=[],
            importance=0.7,
            user_id="user123",
        ),
        similarity_score=0.88,
    )

    result = await classifier.classify_pair(new_job, similar_job, "primary", None)
    assert result.metadata["category"] == "job"


@pytest.mark.asyncio
async def test_conflict_classifier_preference_category():
    """Test preference conflict categorization."""
    mock_llm_verifier = Mock()
    mock_llm_verifier.verify_conflict = AsyncMock(return_value=(True, "llm"))
    mock_llm_verifier.get_metrics.return_value = {}

    classifier = ConflictClassifier(llm_conflict_verifier=mock_llm_verifier)

    new_pref = MemoryFact(
        text="I hate coffee",
        type="preference",
        tags=[],
        importance=0.6,
        user_id="user123",
    )
    similar_pref = SimilarMemory(
        memory_id="mem_2",
        memory=MemoryFact(
            text="I love coffee",
            type="preference",
            tags=[],
            importance=0.6,
            user_id="user123",
        ),
        similarity_score=0.93,
    )

    result = await classifier.classify_pair(new_pref, similar_pref, "primary", None)
    assert result.metadata["category"] == "preference"


@pytest.mark.asyncio
async def test_conflict_classifier_clarification_hints():
    """Test that appropriate clarification hints are generated."""
    mock_llm_verifier = Mock()
    mock_llm_verifier.verify_conflict = AsyncMock(return_value=(True, "llm"))
    mock_llm_verifier.get_metrics.return_value = {}

    classifier = ConflictClassifier(llm_conflict_verifier=mock_llm_verifier)

    # Location conflict
    new_mem = MemoryFact(
        text="I live in Paris",
        type="fact",
        tags=[],
        importance=0.8,
        user_id="user123",
    )
    similar_mem = SimilarMemory(
        memory_id="mem_1",
        memory=MemoryFact(
            text="I live in London",
            type="fact",
            tags=[],
            importance=0.8,
            user_id="user123",
        ),
        similarity_score=0.91,
    )

    result = await classifier.classify_pair(new_mem, similar_mem, "primary", None)
    assert "clarification_hint" in result.metadata
    assert "Where do you currently live?" == result.metadata["clarification_hint"]


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
