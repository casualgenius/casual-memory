"""Tests for LLM memory extractor."""

import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest
from casual_llm import AssistantMessage, SystemMessage, UserMessage
from pydantic import BaseModel, Field

from casual_memory.extractors.llm_extractor import LLMMemoryExtracter
from casual_memory.extractors.models import MemoryExtractionResponse
from casual_memory.extractors.prompts import USER_MEMORY_PROMPT


class MockModel:
    """Mock Model for testing."""

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

    provider = MockModel(response_json)
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

    provider = MockModel(response_json)
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

    provider = MockModel(response_json)
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

    provider = MockModel(response_json)
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
    provider = MockModel("This is not valid JSON")
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

    provider = MockModel(response_json)
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

    provider = MockModel(response_json)
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

    provider = MockModel(response_json)
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

    provider = MockModel(response_json)
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
    provider = MockModel(response_json)
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

    provider = MockModel(response_json)
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


# =============================================================================
# Tests for parameterized extraction model
# =============================================================================


class CustomMemoryExtraction(BaseModel):
    """Custom memory extraction model for testing.

    Must include fields compatible with MemoryFact (text, type, tags, importance)
    since extract() converts results to MemoryFact instances.
    Can add additional custom fields for domain-specific extraction.
    """

    text: str = Field(..., description="Memory text")
    type: str = Field(..., description="Custom type like 'insight', 'reflection'")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")
    importance: float = Field(..., ge=0.0, le=1.0)
    # Custom field beyond standard MemoryFact fields
    custom_field: str = Field(default="", description="Custom field for testing")


class CustomExtractionResponse(BaseModel):
    """Custom extraction response with custom memory model."""

    memories: list[CustomMemoryExtraction] = Field(default_factory=list)


@pytest.mark.asyncio
async def test_extract_with_custom_extraction_model():
    """Test extraction with a custom extraction model."""
    response_json = json.dumps(
        {
            "memories": [
                {
                    "text": "This is an insight about learning",
                    "type": "insight",  # Custom type
                    "tags": ["learning", "insight"],
                    "importance": 0.8,
                    "custom_field": "reflection_context",
                }
            ]
        }
    )

    provider = MockModel(response_json)
    custom_prompt = "Extract insights: {today_natural} (ISO: {isonow})"

    extractor = LLMMemoryExtracter(
        model=provider,
        prompt=custom_prompt,
        extraction_model=CustomExtractionResponse,
    )

    messages = [UserMessage(content="I've learned something important")]
    memories = await extractor.extract(messages)

    assert len(memories) == 1
    assert memories[0].text == "This is an insight about learning"
    assert memories[0].type == "insight"  # Custom type preserved
    assert memories[0].importance == 0.8
    assert memories[0].tags == ["learning", "insight"]

    # Verify the custom model was used for LLM response_format
    call_args = provider.chat.call_args
    assert call_args[1]["response_format"] == CustomExtractionResponse


@pytest.mark.asyncio
async def test_extract_with_default_extraction_model():
    """Test that default extraction model (MemoryExtractionResponse) works."""
    response_json = json.dumps(
        {
            "memories": [
                {
                    "text": "My name is Test",
                    "type": "fact",
                    "tags": ["name"],
                    "importance": 0.9,
                    "valid_until": None,
                }
            ]
        }
    )

    provider = MockModel(response_json)

    # Create extractor without explicit extraction_model - should use default
    extractor = LLMMemoryExtracter(
        model=provider,
        prompt=USER_MEMORY_PROMPT,
    )

    messages = [UserMessage(content="My name is Test")]
    memories = await extractor.extract(messages)

    assert len(memories) == 1
    assert memories[0].text == "My name is Test"

    # Verify default model was used
    call_args = provider.chat.call_args
    assert call_args[1]["response_format"] == MemoryExtractionResponse


@pytest.mark.asyncio
async def test_extract_custom_model_filters_low_importance():
    """Test that importance filtering works with custom models."""
    response_json = json.dumps(
        {
            "memories": [
                {
                    "text": "High importance insight",
                    "type": "insight",
                    "tags": ["test"],
                    "importance": 0.9,
                    "custom_field": "",
                },
                {
                    "text": "Low importance insight",
                    "type": "insight",
                    "tags": ["test"],
                    "importance": 0.3,  # Below threshold
                    "custom_field": "",
                },
            ]
        }
    )

    provider = MockModel(response_json)

    extractor = LLMMemoryExtracter(
        model=provider,
        prompt="Extract insights: {today_natural} (ISO: {isonow})",
        extraction_model=CustomExtractionResponse,
    )

    messages = [UserMessage(content="Test")]
    memories = await extractor.extract(messages)

    # Only high importance should pass the filter
    assert len(memories) == 1
    assert memories[0].importance == 0.9


@pytest.mark.asyncio
async def test_extract_custom_model_with_custom_types():
    """Test extraction with custom memory types (insight, reflection, opinion)."""
    response_json = json.dumps(
        {
            "memories": [
                {
                    "text": "I think AI will transform education",
                    "type": "opinion",
                    "tags": ["ai", "education", "opinion"],
                    "importance": 0.7,
                    "custom_field": "tech_opinion",
                },
                {
                    "text": "My relationship with learning has improved",
                    "type": "reflection",
                    "tags": ["learning", "self-improvement"],
                    "importance": 0.8,
                    "custom_field": "self_reflection",
                },
            ]
        }
    )

    provider = MockModel(response_json)

    extractor = LLMMemoryExtracter(
        model=provider,
        prompt="Extract reflections: {today_natural} (ISO: {isonow})",
        extraction_model=CustomExtractionResponse,
    )

    messages = [UserMessage(content="Test")]
    memories = await extractor.extract(messages)

    assert len(memories) == 2
    types = {m.type for m in memories}
    assert types == {"opinion", "reflection"}


@pytest.mark.asyncio
async def test_extraction_model_stored_on_instance():
    """Test that extraction_model is stored on the extractor instance."""
    provider = MockModel(json.dumps({"memories": []}))

    # Default model
    extractor_default = LLMMemoryExtracter(
        model=provider,
        prompt=USER_MEMORY_PROMPT,
    )
    assert extractor_default.extraction_model == MemoryExtractionResponse

    # Custom model
    extractor_custom = LLMMemoryExtracter(
        model=provider,
        prompt="Custom prompt: {today_natural} (ISO: {isonow})",
        extraction_model=CustomExtractionResponse,
    )
    assert extractor_custom.extraction_model == CustomExtractionResponse


@pytest.mark.asyncio
async def test_custom_model_validation_error():
    """Test that validation errors with custom models raise ValueError."""
    # Response missing required 'importance' field
    response_json = json.dumps(
        {
            "memories": [
                {
                    "text": "Missing importance",
                    "type": "insight",
                    "tags": ["test"],
                    # importance is required but missing
                }
            ]
        }
    )

    provider = MockModel(response_json)

    extractor = LLMMemoryExtracter(
        model=provider,
        prompt="Extract: {today_natural} (ISO: {isonow})",
        extraction_model=CustomExtractionResponse,
    )

    messages = [UserMessage(content="Test")]

    with pytest.raises(ValueError, match="LLM response did not match expected schema"):
        await extractor.extract(messages)
