"""Tests for LLM-based duplicate detection."""

from unittest.mock import AsyncMock, Mock

import pytest

from casual_memory.intelligence.duplicate_detector import LLMDuplicateDetector
from casual_memory.models import MemoryFact
from tests.mock_helpers import MockModel


@pytest.mark.asyncio
async def test_duplicate_detector_initialization():
    """Test duplicate detector initialization."""
    mock_model = MockModel("DISTINCT")
    detector = LLMDuplicateDetector(model=mock_model)

    assert detector.model_name == "test-model"
    assert detector.llm_call_count == 0
    assert detector.llm_success_count == 0
    assert detector.llm_failure_count == 0
    assert detector.heuristic_fallback_count == 0


@pytest.mark.asyncio
async def test_duplicate_detection_same():
    """Test detection of duplicate/refinement memories."""
    mock_model = MockModel("SAME")
    detector = LLMDuplicateDetector(mock_model)

    # Exact duplicate
    memory_a = MemoryFact(text="I live in London", type="fact", tags=["location"], importance=0.9)
    memory_b = MemoryFact(text="I live in London", type="fact", tags=["location"], importance=0.9)

    is_duplicate = await detector.is_duplicate_or_refinement(memory_a, memory_b, 0.99)

    assert is_duplicate is True
    assert detector.llm_call_count == 1
    assert detector.llm_success_count == 1


@pytest.mark.asyncio
async def test_duplicate_detection_refinement():
    """Test detection of refinements as duplicates."""
    mock_model = MockModel("SAME")
    detector = LLMDuplicateDetector(mock_model)

    # Location refinement (general -> specific)
    memory_a = MemoryFact(text="I live in London", type="fact", tags=["location"], importance=0.8)
    memory_b = MemoryFact(
        text="I live in Central London", type="fact", tags=["location"], importance=0.9
    )

    is_duplicate = await detector.is_duplicate_or_refinement(memory_a, memory_b, 0.92)

    assert is_duplicate is True
    assert detector.llm_call_count == 1


@pytest.mark.asyncio
async def test_duplicate_detection_job_refinement():
    """Test job refinement detection."""
    mock_model = MockModel("SAME")
    detector = LLMDuplicateDetector(mock_model)

    # Job refinement (general -> specific)
    memory_a = MemoryFact(text="I work as an engineer", type="fact", tags=["job"], importance=0.7)
    memory_b = MemoryFact(
        text="I work as a senior software engineer at Google",
        type="fact",
        tags=["job"],
        importance=0.9,
    )

    is_duplicate = await detector.is_duplicate_or_refinement(memory_a, memory_b, 0.85)

    assert is_duplicate is True


@pytest.mark.asyncio
async def test_duplicate_detection_distinct():
    """Test detection of distinct facts."""
    mock_model = MockModel("DISTINCT")
    detector = LLMDuplicateDetector(mock_model)

    # Different facts (residence vs work location)
    memory_a = MemoryFact(text="I live in Bangkok", type="fact", tags=["location"], importance=0.8)
    memory_b = MemoryFact(
        text="I work in Bangkok", type="fact", tags=["location", "job"], importance=0.8
    )

    is_duplicate = await detector.is_duplicate_or_refinement(memory_a, memory_b, 0.88)

    assert is_duplicate is False
    assert detector.llm_call_count == 1
    assert detector.llm_success_count == 1


@pytest.mark.asyncio
async def test_duplicate_detection_case_insensitive():
    """Test that response parsing is case-insensitive."""
    # Test with lowercase "same"
    mock_model = MockModel("same fact")
    detector = LLMDuplicateDetector(mock_model)

    memory_a = MemoryFact(text="I live in London", type="fact", tags=[], importance=0.8)
    memory_b = MemoryFact(text="I live in London", type="fact", tags=[], importance=0.8)

    is_duplicate = await detector.is_duplicate_or_refinement(memory_a, memory_b, 0.99)
    assert is_duplicate is True

    # Test with lowercase "distinct"
    mock_model2 = MockModel("distinct facts")
    detector2 = LLMDuplicateDetector(mock_model2)

    is_duplicate2 = await detector2.is_duplicate_or_refinement(memory_a, memory_b, 0.99)
    assert is_duplicate2 is False


@pytest.mark.asyncio
async def test_duplicate_fallback_high_similarity():
    """Test heuristic fallback with high similarity."""
    mock_model = Mock()
    mock_model.name = "test-model"
    mock_model.chat = AsyncMock(side_effect=Exception("LLM failed"))
    detector = LLMDuplicateDetector(mock_model)

    memory_a = MemoryFact(text="I live in London", type="fact", tags=[], importance=0.8)
    memory_b = MemoryFact(text="I live in London", type="fact", tags=[], importance=0.8)

    # Similarity >= 0.95 should be treated as duplicate
    is_duplicate = await detector.is_duplicate_or_refinement(memory_a, memory_b, 0.96)

    assert is_duplicate is True
    assert detector.llm_call_count == 1
    assert detector.llm_success_count == 0
    assert detector.llm_failure_count == 1
    assert detector.heuristic_fallback_count == 1


@pytest.mark.asyncio
async def test_duplicate_fallback_low_similarity():
    """Test heuristic fallback with low similarity."""
    mock_model = Mock()
    mock_model.name = "test-model"
    mock_model.chat = AsyncMock(side_effect=Exception("LLM failed"))
    detector = LLMDuplicateDetector(mock_model)

    memory_a = MemoryFact(text="I live in London", type="fact", tags=[], importance=0.8)
    memory_b = MemoryFact(text="I work in London", type="fact", tags=[], importance=0.8)

    # Similarity < 0.95 should be treated as distinct (conservative)
    is_duplicate = await detector.is_duplicate_or_refinement(memory_a, memory_b, 0.85)

    assert is_duplicate is False
    assert detector.heuristic_fallback_count == 1


@pytest.mark.asyncio
async def test_duplicate_metrics():
    """Test metrics tracking."""
    mock_model = MockModel("SAME")
    detector = LLMDuplicateDetector(mock_model)

    # Initial metrics
    metrics = detector.get_metrics()
    assert metrics["duplicate_detector_llm_call_count"] == 0
    assert metrics["duplicate_detector_llm_success_count"] == 0
    assert metrics["duplicate_detector_llm_failure_count"] == 0
    assert metrics["duplicate_detector_heuristic_fallback_count"] == 0

    # Make successful call
    memory_a = MemoryFact(text="I live in London", type="fact", tags=[], importance=0.8)
    memory_b = MemoryFact(text="I live in Central London", type="fact", tags=[], importance=0.9)
    await detector.is_duplicate_or_refinement(memory_a, memory_b, 0.92)

    metrics = detector.get_metrics()
    assert metrics["duplicate_detector_llm_call_count"] == 1
    assert metrics["duplicate_detector_llm_success_count"] == 1
    assert metrics["duplicate_detector_llm_success_rate_percent"] == 100.0


@pytest.mark.asyncio
async def test_duplicate_metrics_with_failures():
    """Test metrics with both successes and failures."""
    # First call succeeds
    mock_model = MockModel("SAME")
    detector = LLMDuplicateDetector(mock_model)

    memory_a = MemoryFact(text="Test A", type="fact", tags=[], importance=0.8)
    memory_b = MemoryFact(text="Test B", type="fact", tags=[], importance=0.8)
    await detector.is_duplicate_or_refinement(memory_a, memory_b, 0.92)

    # Inject failure for second call
    detector.model.chat = AsyncMock(side_effect=Exception("LLM failed"))
    await detector.is_duplicate_or_refinement(memory_a, memory_b, 0.96)

    metrics = detector.get_metrics()
    assert metrics["duplicate_detector_llm_call_count"] == 2
    assert metrics["duplicate_detector_llm_success_count"] == 1
    assert metrics["duplicate_detector_llm_failure_count"] == 1
    assert metrics["duplicate_detector_heuristic_fallback_count"] == 1
    assert metrics["duplicate_detector_llm_success_rate_percent"] == 50.0


@pytest.mark.asyncio
async def test_duplicate_custom_prompt():
    """Test using a custom system prompt."""
    custom_prompt = "Custom prompt for duplicate detection"
    mock_model = MockModel("SAME")
    detector = LLMDuplicateDetector(mock_model, system_prompt=custom_prompt)

    assert detector.system_prompt == custom_prompt

    memory_a = MemoryFact(text="Test A", type="fact", tags=[], importance=0.8)
    memory_b = MemoryFact(text="Test B", type="fact", tags=[], importance=0.8)
    await detector.is_duplicate_or_refinement(memory_a, memory_b, 0.92)

    # Verify custom prompt was used in system message
    call_args = mock_model.chat.call_args
    messages = call_args[0][0]
    assert messages[0].content == custom_prompt


@pytest.mark.asyncio
async def test_paraphrase_detection():
    """Test detection of paraphrases as duplicates."""
    mock_model = MockModel("SAME")
    detector = LLMDuplicateDetector(mock_model)

    memory_a = MemoryFact(
        text="I work as a software engineer", type="fact", tags=["job"], importance=0.8
    )
    memory_b = MemoryFact(
        text="I'm employed as a software developer", type="fact", tags=["job"], importance=0.8
    )

    is_duplicate = await detector.is_duplicate_or_refinement(memory_a, memory_b, 0.87)
    assert is_duplicate is True


@pytest.mark.asyncio
async def test_intensity_variations():
    """Test that intensity variations are treated as duplicates."""
    mock_model = MockModel("SAME")
    detector = LLMDuplicateDetector(mock_model)

    memory_a = MemoryFact(text="I like coffee", type="preference", tags=["drink"], importance=0.7)
    memory_b = MemoryFact(text="I love coffee", type="preference", tags=["drink"], importance=0.9)

    is_duplicate = await detector.is_duplicate_or_refinement(memory_a, memory_b, 0.90)
    assert is_duplicate is True


@pytest.mark.asyncio
async def test_contradictions_are_distinct():
    """Test that contradictions are treated as distinct facts."""
    mock_model = MockModel("DISTINCT")
    detector = LLMDuplicateDetector(mock_model)

    memory_a = MemoryFact(text="I live in Paris", type="fact", tags=["location"], importance=0.9)
    memory_b = MemoryFact(text="I live in London", type="fact", tags=["location"], importance=0.9)

    is_duplicate = await detector.is_duplicate_or_refinement(memory_a, memory_b, 0.85)
    assert is_duplicate is False


@pytest.mark.asyncio
async def test_temporal_changes_are_distinct():
    """Test that temporal changes are distinct facts."""
    mock_model = MockModel("DISTINCT")
    detector = LLMDuplicateDetector(mock_model)

    memory_a = MemoryFact(text="I lived in Paris", type="fact", tags=["location"], importance=0.8)
    memory_b = MemoryFact(
        text="I now live in London", type="fact", tags=["location"], importance=0.9
    )

    is_duplicate = await detector.is_duplicate_or_refinement(memory_a, memory_b, 0.75)
    assert is_duplicate is False


@pytest.mark.asyncio
async def test_multiple_duplicate_checks():
    """Test multiple duplicate detections."""
    mock_model = MockModel("SAME")
    detector = LLMDuplicateDetector(mock_model)

    memory_pairs = [
        (
            MemoryFact(text="I live in London", type="fact", tags=[], importance=0.8),
            MemoryFact(text="I live in Central London", type="fact", tags=[], importance=0.9),
        ),
        (
            MemoryFact(text="I like coffee", type="preference", tags=[], importance=0.7),
            MemoryFact(text="I love coffee", type="preference", tags=[], importance=0.9),
        ),
        (
            MemoryFact(text="My name is Alex", type="fact", tags=[], importance=1.0),
            MemoryFact(text="My name is Alex", type="fact", tags=[], importance=1.0),
        ),
    ]

    for memory_a, memory_b in memory_pairs:
        is_duplicate = await detector.is_duplicate_or_refinement(memory_a, memory_b, 0.92)
        assert is_duplicate is True

    assert detector.llm_call_count == 3
    assert detector.llm_success_count == 3
