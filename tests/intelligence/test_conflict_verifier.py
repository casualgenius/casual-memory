"""Tests for LLM-based conflict verification."""

from unittest.mock import AsyncMock, Mock

import pytest

from casual_memory.intelligence.conflict_verifier import LLMConflictVerifier
from casual_memory.models import MemoryFact


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, response_content: str):
        self.response_content = response_content
        self.chat = AsyncMock(return_value=Mock(content=response_content))


@pytest.mark.asyncio
async def test_conflict_verifier_initialization():
    """Test conflict verifier initialization."""
    provider = MockLLMProvider("NO")
    verifier = LLMConflictVerifier(
        llm_provider=provider, model_name="test-model", enable_fallback=True
    )

    assert verifier.model_name == "test-model"
    assert verifier.enable_fallback is True
    assert verifier.llm_call_count == 0
    assert verifier.llm_success_count == 0
    assert verifier.llm_failure_count == 0
    assert verifier.fallback_count == 0


@pytest.mark.asyncio
async def test_conflict_detection_yes():
    """Test detection of conflicting memories."""
    provider = MockLLMProvider("YES")
    verifier = LLMConflictVerifier(provider, "test-model")

    memory_a = MemoryFact(text="I live in London", type="fact", tags=["location"], importance=0.9)
    memory_b = MemoryFact(text="I live in Paris", type="fact", tags=["location"], importance=0.9)

    is_conflict, method = await verifier.verify_conflict(memory_a, memory_b, 0.85)

    assert is_conflict is True
    assert method == "llm"
    assert verifier.llm_call_count == 1
    assert verifier.llm_success_count == 1


@pytest.mark.asyncio
async def test_conflict_detection_no():
    """Test detection of non-conflicting memories."""
    provider = MockLLMProvider("NO")
    verifier = LLMConflictVerifier(provider, "test-model")

    memory_a = MemoryFact(text="I work as an engineer", type="fact", tags=["job"], importance=0.8)
    memory_b = MemoryFact(
        text="I work as a senior software engineer at Google",
        type="fact",
        tags=["job"],
        importance=0.9,
    )

    is_conflict, method = await verifier.verify_conflict(memory_a, memory_b, 0.80)

    assert is_conflict is False
    assert method == "llm"
    assert verifier.llm_call_count == 1
    assert verifier.llm_success_count == 1


@pytest.mark.asyncio
async def test_conflict_detection_case_insensitive():
    """Test that response parsing is case-insensitive."""
    # Test with lowercase "yes"
    provider = MockLLMProvider("yes, these conflict")
    verifier = LLMConflictVerifier(provider, "test-model")

    memory_a = MemoryFact(text="I live in London", type="fact", tags=[], importance=0.8)
    memory_b = MemoryFact(text="I live in Paris", type="fact", tags=[], importance=0.8)

    is_conflict, method = await verifier.verify_conflict(memory_a, memory_b, 0.85)
    assert is_conflict is True

    # Test with lowercase "no"
    provider2 = MockLLMProvider("no conflict here")
    verifier2 = LLMConflictVerifier(provider2, "test-model")

    is_conflict2, method2 = await verifier2.verify_conflict(memory_a, memory_b, 0.85)
    assert is_conflict2 is False


@pytest.mark.asyncio
async def test_conflict_fallback_when_llm_fails():
    """Test that fallback is used when LLM fails."""
    provider = Mock()
    provider.chat = AsyncMock(side_effect=Exception("LLM failed"))
    verifier = LLMConflictVerifier(provider, "test-model", enable_fallback=True)

    # High similarity + location keywords = conflict via heuristic
    memory_a = MemoryFact(text="I live in London", type="fact", tags=["location"], importance=0.9)
    memory_b = MemoryFact(text="I live in Paris", type="fact", tags=["location"], importance=0.9)

    is_conflict, method = await verifier.verify_conflict(memory_a, memory_b, 0.93)

    assert is_conflict is True
    assert method == "heuristic_fallback"
    assert verifier.llm_call_count == 1
    assert verifier.llm_success_count == 0
    assert verifier.llm_failure_count == 1
    assert verifier.fallback_count == 1


@pytest.mark.asyncio
async def test_conflict_fallback_raises_when_disabled():
    """Test that exception is raised when fallback is disabled."""
    provider = Mock()
    provider.chat = AsyncMock(side_effect=Exception("LLM failed"))
    verifier = LLMConflictVerifier(provider, "test-model", enable_fallback=False)

    memory_a = MemoryFact(text="I live in London", type="fact", tags=[], importance=0.8)
    memory_b = MemoryFact(text="I live in Paris", type="fact", tags=[], importance=0.8)

    with pytest.raises(Exception, match="LLM failed"):
        await verifier.verify_conflict(memory_a, memory_b, 0.85)


@pytest.mark.asyncio
async def test_heuristic_negation_detection():
    """Test heuristic detection of negation patterns."""
    provider = Mock()
    provider.chat = AsyncMock(side_effect=Exception("LLM failed"))
    verifier = LLMConflictVerifier(provider, "test-model", enable_fallback=True)

    # Test "like" vs "don't like"
    memory_a = MemoryFact(text="I like coffee", type="preference", tags=[], importance=0.7)
    memory_b = MemoryFact(text="I don't like coffee", type="preference", tags=[], importance=0.7)

    is_conflict, method = await verifier.verify_conflict(memory_a, memory_b, 0.91)
    assert is_conflict is True
    assert method == "heuristic_fallback"

    # Test "can" vs "can't"
    memory_c = MemoryFact(text="I can swim", type="fact", tags=[], importance=0.6)
    memory_d = MemoryFact(text="I can't swim", type="fact", tags=[], importance=0.6)

    is_conflict2, method2 = await verifier.verify_conflict(memory_c, memory_d, 0.92)
    assert is_conflict2 is True


@pytest.mark.asyncio
async def test_heuristic_location_conflict():
    """Test heuristic detection of location conflicts."""
    provider = Mock()
    provider.chat = AsyncMock(side_effect=Exception("LLM failed"))
    verifier = LLMConflictVerifier(provider, "test-model", enable_fallback=True)

    # High similarity + location indicators = conflict
    memory_a = MemoryFact(text="I live in Bangkok", type="fact", tags=[], importance=0.8)
    memory_b = MemoryFact(text="I reside in Tokyo", type="fact", tags=[], importance=0.8)

    # Similarity >= 0.92 required for location conflict
    is_conflict, method = await verifier.verify_conflict(memory_a, memory_b, 0.93)
    assert is_conflict is True

    # Lower similarity should not trigger
    is_conflict2, method2 = await verifier.verify_conflict(memory_a, memory_b, 0.91)
    assert is_conflict2 is False


@pytest.mark.asyncio
async def test_heuristic_job_conflict():
    """Test heuristic detection of job conflicts."""
    provider = Mock()
    provider.chat = AsyncMock(side_effect=Exception("LLM failed"))
    verifier = LLMConflictVerifier(provider, "test-model", enable_fallback=True)

    # High similarity + job indicators = conflict
    memory_a = MemoryFact(text="I work as a teacher", type="fact", tags=[], importance=0.8)
    memory_b = MemoryFact(text="I work as a doctor", type="fact", tags=[], importance=0.8)

    # Similarity >= 0.92 required for job conflict
    is_conflict, method = await verifier.verify_conflict(memory_a, memory_b, 0.93)
    assert is_conflict is True

    # Lower similarity should not trigger
    is_conflict2, method2 = await verifier.verify_conflict(memory_a, memory_b, 0.90)
    assert is_conflict2 is False


@pytest.mark.asyncio
async def test_heuristic_no_conflict_low_similarity():
    """Test that heuristic requires high similarity."""
    provider = Mock()
    provider.chat = AsyncMock(side_effect=Exception("LLM failed"))
    verifier = LLMConflictVerifier(provider, "test-model", enable_fallback=True)

    memory_a = MemoryFact(text="I live in London", type="fact", tags=[], importance=0.8)
    memory_b = MemoryFact(text="I live in Paris", type="fact", tags=[], importance=0.8)

    # Similarity < 0.90 should return False
    is_conflict, method = await verifier.verify_conflict(memory_a, memory_b, 0.85)
    assert is_conflict is False
    assert method == "heuristic_fallback"


@pytest.mark.asyncio
async def test_conflict_metrics():
    """Test metrics tracking."""
    provider = MockLLMProvider("YES")
    verifier = LLMConflictVerifier(provider, "test-model")

    # Initial metrics
    metrics = verifier.get_metrics()
    assert metrics["conflict_verifier_llm_call_count"] == 0
    assert metrics["conflict_verifier_llm_success_count"] == 0
    assert metrics["conflict_verifier_llm_failure_count"] == 0
    assert metrics["conflict_verifier_fallback_count"] == 0

    # Make successful call
    memory_a = MemoryFact(text="I live in London", type="fact", tags=[], importance=0.8)
    memory_b = MemoryFact(text="I live in Paris", type="fact", tags=[], importance=0.8)
    await verifier.verify_conflict(memory_a, memory_b, 0.85)

    metrics = verifier.get_metrics()
    assert metrics["conflict_verifier_llm_call_count"] == 1
    assert metrics["conflict_verifier_llm_success_count"] == 1
    assert metrics["conflict_verifier_llm_success_rate_percent"] == 100.0


@pytest.mark.asyncio
async def test_conflict_custom_prompt():
    """Test using a custom system prompt."""
    custom_prompt = "Custom prompt for conflict detection"
    provider = MockLLMProvider("YES")
    verifier = LLMConflictVerifier(provider, "test-model", system_prompt=custom_prompt)

    assert verifier.system_prompt == custom_prompt

    memory_a = MemoryFact(text="Test A", type="fact", tags=[], importance=0.8)
    memory_b = MemoryFact(text="Test B", type="fact", tags=[], importance=0.8)
    await verifier.verify_conflict(memory_a, memory_b, 0.85)

    # Verify custom prompt was used in system message
    call_args = provider.chat.call_args
    messages = call_args[0][0]
    assert messages[0].content == custom_prompt


@pytest.mark.asyncio
async def test_multiple_conflict_checks():
    """Test multiple conflict verifications."""
    provider = MockLLMProvider("YES")
    verifier = LLMConflictVerifier(provider, "test-model")

    memory_pairs = [
        (
            MemoryFact(text="I live in London", type="fact", tags=[], importance=0.8),
            MemoryFact(text="I live in Paris", type="fact", tags=[], importance=0.8),
        ),
        (
            MemoryFact(text="I work as a teacher", type="fact", tags=[], importance=0.7),
            MemoryFact(text="I work as a doctor", type="fact", tags=[], importance=0.7),
        ),
        (
            MemoryFact(text="I like coffee", type="preference", tags=[], importance=0.6),
            MemoryFact(text="I hate coffee", type="preference", tags=[], importance=0.6),
        ),
    ]

    for memory_a, memory_b in memory_pairs:
        is_conflict, method = await verifier.verify_conflict(memory_a, memory_b, 0.85)
        assert is_conflict is True
        assert method == "llm"

    assert verifier.llm_call_count == 3
    assert verifier.llm_success_count == 3
