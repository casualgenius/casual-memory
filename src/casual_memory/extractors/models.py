"""
Data models for memory extraction.

This module defines the response models used by the LLM memory extractor.
These models are used to generate JSON schemas for structured LLM outputs.
"""

from pydantic import BaseModel, Field

from casual_memory.models import MemoryFactExtraction


class MemoryExtractionResponse(BaseModel):
    """Response model for LLM memory extraction.

    This model defines the JSON schema that the LLM must follow
    when extracting memories from conversations.

    Uses MemoryFactExtraction (not MemoryFact) to ensure the LLM only
    sees and populates extraction fields, not system-managed fields like
    confidence, mention_count, etc.
    """

    memories: list[MemoryFactExtraction] = Field(
        default_factory=list,
        description=(
            "List of extracted memories from the conversation. "
            "Return empty list if no significant memories are found. "
            "Only extract facts, preferences, goals, and events - not conversational filler."
        ),
    )
