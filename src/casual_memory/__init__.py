"""
casual-memory: Intelligent semantic memory with conflict detection and classification pipeline.

Core components:
- classifiers: Memory classification pipeline (NLI, conflict, duplicate, auto-resolution)
- intelligence: Supporting components (NLI filter, verifiers, confidence scoring)
- extractors: Memory extraction from conversations
- storage: Protocol abstractions for vector stores, conflict stores, etc.
- models: Core data models (MemoryFact, MemoryConflict, etc.)
- services: High-level service APIs (MemoryService, ContextService)
"""

__version__ = "0.2.1"

from casual_memory.context_service import ContextService
from casual_memory.memory_service import MemoryService
from casual_memory.models import (
    ConflictResolution,
    MemoryBlock,
    MemoryConflict,
    MemoryFact,
    MemoryFactExtraction,
    MemoryQueryFilter,
    ShortTermMemory,
)

__all__ = [
    "__version__",
    # Models
    "MemoryFact",
    "MemoryFactExtraction",
    "MemoryBlock",
    "MemoryConflict",
    "ConflictResolution",
    "ShortTermMemory",
    "MemoryQueryFilter",
    # Services
    "MemoryService",
    "ContextService",
]
