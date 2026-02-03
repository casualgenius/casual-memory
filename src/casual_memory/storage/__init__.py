"""
Storage protocols for memory and conflict management.

Provides protocol definitions for storage backends. Implementations can use
various databases (Qdrant, PostgreSQL, in-memory, etc.) as long as they
satisfy the protocol interface.
"""

from casual_memory.storage.protocols import ConflictStore, ShortTermStore, VectorMemoryStore

__all__ = [
    "VectorMemoryStore",
    "ConflictStore",
    "ShortTermStore",
]

# Vector storage implementations
try:
    from casual_memory.storage.vector.memory import InMemoryVectorStore  # noqa: F401

    __all__.append("InMemoryVectorStore")
except ImportError:
    pass

try:
    from casual_memory.storage.vector.qdrant import QdrantMemoryStore  # noqa: F401

    __all__.append("QdrantMemoryStore")
except ImportError:
    pass

# Conflict storage implementations
try:
    from casual_memory.storage.conflicts.memory import InMemoryConflictStore  # noqa: F401

    __all__.append("InMemoryConflictStore")
except ImportError:
    pass

try:
    from casual_memory.storage.conflicts.sqlalchemy import SQLAlchemyConflictStore  # noqa: F401

    __all__.append("SQLAlchemyConflictStore")
except ImportError:
    pass

# Short-term storage implementations
try:
    from casual_memory.storage.short_term.memory import InMemoryShortTermStore  # noqa: F401

    __all__.append("InMemoryShortTermStore")
except ImportError:
    pass

try:
    from casual_memory.storage.short_term.redis import RedisShortTermStore  # noqa: F401

    __all__.append("RedisShortTermStore")
except ImportError:
    pass
