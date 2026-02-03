"""Tests for LLM memory extractor."""

import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest
from casual_llm import AssistantMessage, SystemMessage, UserMessage

from casual_memory.extractors.llm_extractor import LLMMemoryExtracter
from casual_memory.extractors.prompts import USER_MEMORY_PROMPT


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, response_content: str):
        self.response_content = response_content
        self.chat = AsyncMock(return_value=Mock(content=response_content))


@pytest.fixture
def mock_prompt():
    """Simplified prompt for testing."""
    return "Extract memories from: {today_natural} (ISO: {isonow})\n\n{conversation}"


@pytest.mark.asyncio
async def test_extract_basic_memory():
    """Test basic memory extraction with valid response."""
    response_json = json.dumps(
        {
            "memories": [
                {
                    "text": "My name is Alex",
                    "type": "fact",
                    "tags": ["name", "identity"],
                    "importance": 0.9,
                    "valid_until": None,
                }
            ]
        }
    )

    provider = MockLLMProvider(response_json)
    extractor = LLMMemoryExtracter(provider, USER_MEMORY_PROMPT)

    messages = [
        UserMessage(content="My name is Alex"),
        AssistantMessage(content="Nice to meet you, Alex!"),
    ]

    memories = await extractor.extract(messages)

    assert len(memories) == 1
    assert memories[0].text == "My name is Alex"
    assert memories[0].type == "fact"
    assert memories[0].importance == 0.9
    # source is system-managed, defaults to None (set by calling code)
    assert memories[0].source is None


@pytest.mark.asyncio
async def test_extract_multiple_memories():
    """Test extraction of multiple memories from conversation."""
    response_json = json.dumps(
        {
            "memories": [
                {
                    "text": "I live in London",
                    "type": "fact",
                    "tags": ["location", "residence"],
                    "importance": 0.8,
                    "valid_until": None,
                },
                {
                    "text": "I work as a software engineer",
                    "type": "fact",
                    "tags": ["job", "career"],
                    "importance": 0.7,
                    "valid_until": None,
                },
                {
                    "text": "I enjoy hiking",
                    "type": "preference",
                    "tags": ["hobby", "outdoor"],
                    "importance": 0.6,
                    "valid_until": None,
                },
            ]
        }
    )

    provider = MockLLMProvider(response_json)
    extractor = LLMMemoryExtracter(provider, USER_MEMORY_PROMPT)

    messages = [
        UserMessage(content="I live in London, work as a software engineer, and enjoy hiking")
    ]

    memories = await extractor.extract(messages)

    assert len(memories) == 3
    assert memories[0].type == "fact"
    assert memories[1].type == "fact"
    assert memories[2].type == "preference"


@pytest.mark.asyncio
async def test_extract_filters_low_importance():
    """Test that memories below importance threshold are filtered."""
    response_json = json.dumps(
        {
            "memories": [
                {
                    "text": "High importance memory",
                    "type": "fact",
                    "tags": ["test"],
                    "importance": 0.9,
                    "valid_until": None,
                },
                {
                    "text": "Low importance memory",
                    "type": "fact",
                    "tags": ["test"],
                    "importance": 0.3,  # Below 0.5 threshold
                    "valid_until": None,
                },
                {
                    "text": "Medium importance memory",
                    "type": "fact",
                    "tags": ["test"],
                    "importance": 0.5,  # Exactly at threshold
                    "valid_until": None,
                },
            ]
        }
    )

    provider = MockLLMProvider(response_json)
    extractor = LLMMemoryExtracter(provider, USER_MEMORY_PROMPT)

    messages = [UserMessage(content="Test message")]
    memories = await extractor.extract(messages)

    # Should only include memories with importance >= 0.5
    assert len(memories) == 2
    assert memories[0].importance == 0.9
    assert memories[1].importance == 0.5


@pytest.mark.asyncio
async def test_extract_with_temporal_memory():
    """Test extraction with valid_until timestamps."""
    now = datetime.now()
    expires = (now + timedelta(days=1)).isoformat()

    response_json = json.dumps(
        {
            "memories": [
                {
                    "text": "I have a meeting tomorrow",
                    "type": "event",
                    "tags": ["meeting", "reminder"],
                    "importance": 0.8,
                    "source": "user",
                    "valid_until": expires,
                }
            ]
        }
    )

    provider = MockLLMProvider(response_json)
    extractor = LLMMemoryExtracter(provider, USER_MEMORY_PROMPT)

    messages = [UserMessage(content="I have a meeting tomorrow")]
    memories = await extractor.extract(messages)

    assert len(memories) == 1
    assert memories[0].valid_until is not None
    # Check that valid_until is roughly tomorrow
    assert memories[0].valid_until is not None


@pytest.mark.asyncio
async def test_extract_handles_invalid_json():
    """Test that invalid JSON responses raise ValueError."""
    provider = MockLLMProvider("This is not valid JSON")
    extractor = LLMMemoryExtracter(provider, USER_MEMORY_PROMPT)

    messages = [UserMessage(content="Test message")]

    # Should raise ValueError on JSON parse error
    with pytest.raises(ValueError, match="LLM response did not match expected schema"):
        await extractor.extract(messages)


@pytest.mark.asyncio
async def test_extract_handles_llm_exception():
    """Test that LLM exceptions are propagated."""
    provider = Mock()
    provider.chat = AsyncMock(side_effect=Exception("LLM failed"))
    extractor = LLMMemoryExtracter(provider, USER_MEMORY_PROMPT)

    messages = [UserMessage(content="Test message")]

    # Should propagate the exception
    with pytest.raises(Exception, match="LLM failed"):
        await extractor.extract(messages)


@pytest.mark.asyncio
async def test_extract_with_empty_conversation():
    """Test extraction with no messages."""
    response_json = json.dumps({"memories": []})

    provider = MockLLMProvider(response_json)
    extractor = LLMMemoryExtracter(provider, USER_MEMORY_PROMPT)

    memories = await extractor.extract([])

    assert len(memories) == 0


@pytest.mark.asyncio
async def test_extract_different_sources():
    """Test that source field is system-managed, not extracted by LLM."""
    response_json = json.dumps(
        {
            "memories": [
                {
                    "text": "User stated fact",
                    "type": "fact",
                    "tags": ["test"],
                    "importance": 0.7,
                    "valid_until": None,
                },
                {
                    "text": "Assistant observed fact",
                    "type": "fact",
                    "tags": ["test"],
                    "importance": 0.6,
                    "valid_until": None,
                },
            ]
        }
    )

    provider = MockLLMProvider(response_json)
    extractor = LLMMemoryExtracter(provider, USER_MEMORY_PROMPT)

    messages = [
        UserMessage(content="I like pizza"),
        AssistantMessage(content="I notice you seem happy today"),
    ]

    memories = await extractor.extract(messages)

    assert len(memories) == 2
    # Source is system-managed, not extracted by LLM (defaults to None)
    # Calling code should set this based on message role
    assert memories[0].source is None
    assert memories[1].source is None


@pytest.mark.asyncio
async def test_extract_preserves_tags():
    """Test that tags are properly extracted and preserved."""
    response_json = json.dumps(
        {
            "memories": [
                {
                    "text": "I am allergic to peanuts",
                    "type": "fact",
                    "tags": ["allergy", "medical", "safety"],
                    "importance": 1.0,
                    "valid_until": None,
                }
            ]
        }
    )

    provider = MockLLMProvider(response_json)
    extractor = LLMMemoryExtracter(provider, USER_MEMORY_PROMPT)

    messages = [UserMessage(content="I'm allergic to peanuts")]
    memories = await extractor.extract(messages)

    assert len(memories) == 1
    assert set(memories[0].tags) == {"allergy", "medical", "safety"}


@pytest.mark.asyncio
async def test_extract_all_memory_types():
    """Test extraction of all memory types."""
    response_json = json.dumps(
        {
            "memories": [
                {
                    "text": "My name is Alex",
                    "type": "fact",
                    "tags": ["name"],
                    "importance": 0.9,
                    "valid_until": None,
                },
                {
                    "text": "I enjoy hiking",
                    "type": "preference",
                    "tags": ["hobby"],
                    "importance": 0.7,
                    "valid_until": None,
                },
                {
                    "text": "I want to learn Spanish",
                    "type": "goal",
                    "tags": ["learning", "language"],
                    "importance": 0.8,
                    "valid_until": None,
                },
                {
                    "text": "I have a dentist appointment tomorrow",
                    "type": "event",
                    "tags": ["appointment", "dental"],
                    "importance": 0.9,
                    "valid_until": None,
                },
            ]
        }
    )

    provider = MockLLMProvider(response_json)
    extractor = LLMMemoryExtracter(provider, USER_MEMORY_PROMPT)

    messages = [UserMessage(content="Complex multi-type message")]
    memories = await extractor.extract(messages)

    assert len(memories) == 4
    types = {m.type for m in memories}
    assert types == {"fact", "preference", "goal", "event"}


@pytest.mark.asyncio
async def test_prompt_formatting():
    """Test that prompt is formatted with correct date information."""
    response_json = json.dumps({"memories": []})
    provider = MockLLMProvider(response_json)
    extractor = LLMMemoryExtracter(provider, USER_MEMORY_PROMPT)

    messages = [UserMessage(content="Test")]
    await extractor.extract(messages)

    # Verify chat was called
    provider.chat.assert_called_once()

    # Get the arguments passed to chat
    call_args = provider.chat.call_args
    llm_messages = call_args[1]["messages"]

    # Verify system message was created with formatted prompt
    assert len(llm_messages) == 2
    assert isinstance(llm_messages[0], SystemMessage)
    assert "memories" in llm_messages[0].content.lower()


@pytest.mark.asyncio
async def test_extract_with_defaults():
    """Test that optional fields use appropriate defaults and required fields are enforced."""
    # Test with valid extraction including all required fields
    response_json = json.dumps(
        {
            "memories": [
                {
                    "text": "Minimal memory",
                    "type": "fact",  # Required
                    "tags": [],  # Required
                    "importance": 0.7,  # Required (no longer optional!)
                    # Optional: valid_until defaults to None
                }
            ]
        }
    )

    provider = MockLLMProvider(response_json)
    extractor = LLMMemoryExtracter(provider, USER_MEMORY_PROMPT)

    messages = [UserMessage(content="Test")]
    memories = await extractor.extract(messages)

    assert len(memories) == 1
    assert memories[0].text == "Minimal memory"
    assert memories[0].type == "fact"
    assert memories[0].tags == []
    assert memories[0].importance == 0.7  # LLM must provide this
    assert memories[0].valid_until is None  # Optional, defaults to None
    assert memories[0].source is None  # System-managed, not extracted
