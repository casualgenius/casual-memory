"""
Storage abstraction protocols.

Provides protocol definitions for vector stores, conflict stores, short-term stores,
and embedding providers.
"""

from casual_memory.storage.protocols import (
    VectorStore,
    ConflictStore,
    ShortTermStore,
    EmbeddingProvider,
)

__all__ = [
    "VectorStore",
    "ConflictStore",
    "ShortTermStore",
    "EmbeddingProvider",
]
