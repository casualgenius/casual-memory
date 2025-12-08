"""
Memory extraction from conversations.

Provides extractors for user and assistant messages with structured prompts.
"""

from casual_memory.extractors.base import MemoryExtractor
from casual_memory.extractors.user_extractor import UserMemoryExtractor
from casual_memory.extractors.assistant_extractor import AssistantMemoryExtractor

__all__ = [
    "MemoryExtractor",
    "UserMemoryExtractor",
    "AssistantMemoryExtractor",
]
