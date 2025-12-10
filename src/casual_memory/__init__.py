"""
casual-memory: Intelligent semantic memory with conflict detection and classification pipeline.

Core components:
- classifiers: Memory classification pipeline (NLI, conflict, duplicate, auto-resolution)
- intelligence: Supporting components (NLI filter, verifiers, confidence scoring)
- extractors: Memory extraction from conversations
- storage: Protocol abstractions for vector stores, conflict stores, etc.
- models: Core data models (MemoryFact, MemoryConflict, etc.)
"""

__version__ = "0.1.0"

from casual_memory.models import (
    MemoryFact,
    MemoryBlock,
    MemoryConflict,
    ConflictResolution,
    ShortTermMemory,
    MemoryQueryFilter
)
from casual_memory.memory_service import MemoryService

__all__ = [
    "__version__",
    # Models
    "MemoryFact",
    "MemoryBlock",
    "MemoryConflict",
    "ConflictResolution",
    "ShortTermMemory",
    "MemoryQueryFilter",
    "MemoryService"
]
