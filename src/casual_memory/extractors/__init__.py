"""
Memory extraction from conversations.

Provides extractors for user and assistant messages with structured prompts.
"""

from casual_memory.extractors.base import MemoryExtracter
from casual_memory.extractors.llm_extractor import LLMMemoryExtracter
from casual_memory.extractors.models import MemoryExtractionResponse
from casual_memory.extractors.prompts import ASSISTANT_MEMORY_PROMPT, USER_MEMORY_PROMPT

__all__ = [
    "MemoryExtracter",
    "LLMMemoryExtracter",
    "MemoryExtractionResponse",
    # Prompts
    "USER_MEMORY_PROMPT",
    "ASSISTANT_MEMORY_PROMPT",
]
