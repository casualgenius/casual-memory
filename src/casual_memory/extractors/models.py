"""
Data models for memory extraction.

This module defines the response models used by the LLM memory extractor.
These models are used to generate JSON schemas for structured LLM outputs.
"""

from pydantic import BaseModel, Field

from casual_memory.models import MemoryFact


class MemoryExtractionResponse(BaseModel):
    """Response model for LLM memory extraction.

    This model defines the JSON schema that the LLM must follow
    when extracting memories from conversations.
    """

    memories: list[MemoryFact] = Field(
        default_factory=list,
        description=(
            "List of extracted memories from the conversation. "
            "Return empty list if no significant memories are found. "
            "Only extract facts, preferences, goals, and events - not conversational filler."
        ),
    )
