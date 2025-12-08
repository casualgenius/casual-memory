"""
Memory extraction from conversations.

Provides extractors for user and assistant messages with structured prompts.
"""

from casual_memory.extractors.base import MemoryExtracter
from casual_memory.extractors.user_extractor import UserMemoryExtracter
from casual_memory.extractors.assistant_extractor import AssistantMemoryExtracter
from casual_memory.extractors.llm_extractor import LLMMemoryExtracter


__all__ = [
    "MemoryExtracter",
    "LLMMemoryExtracter",
    "UserMemoryExtracter",
    "AssistantMemoryExtracter",
]
