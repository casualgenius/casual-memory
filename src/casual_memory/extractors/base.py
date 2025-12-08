"""
Base protocol for Memory Extractors.
"""

from __future__ import annotations

from typing import Protocol, Literal, List

from casual_llm.messages import ChatMessage
from casual_memory.models import MemoryFact


class MemoryExtracter(Protocol):
    """
    Protocol for Memory Extracters.

    This is a Protocol (PEP 544), meaning any class that implements
    the extract() method with this signature is compatible - no
    inheritance required.
    """
    async def extract(self, messages: List[ChatMessage]) -> List[MemoryFact]:
        """
        Extract memories from a list of ChatMessages.

        Args:
            messages: List of ChatMessage (UserMessage, AssistantMessage, SystemMessage, etc.)
        
        Returns:
            List of MemoryFact
        """
        ...
