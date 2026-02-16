"""
Storage protocol definitions for memory and conflict management.

These protocols define the interface that storage implementations must provide.
They are implementation-agnostic and can be backed by various databases
(Qdrant, PostgreSQL, in-memory, etc.).
"""

from typing import Any, Optional, Protocol

from casual_memory.models import ConflictResolution, MemoryConflict, ShortTermMemory


class VectorMemoryStore(Protocol):
    """
    Protocol for vector-based memory storage.

    Implementations should provide vector similarity search and memory management
    capabilities. This is typically backed by a vector database like Qdrant,
    but could also be implemented with other vector stores or in-memory for testing.
    """

    def add(self, vector: list[float], payload: dict[str, Any]) -> str:
        """
        Add a memory to the store.

        Args:
            vector: The embedding vector for the memory
            payload: Dictionary of memory fields (should match MemoryFact structure).
                     Should include 'namespace' and 'entity_id' keys for isolation.

        Returns:
            The generated memory ID
        """
        ...

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        min_score: float = 0.7,
        filters: Optional[Any] = None,
    ) -> list[Any]:
        """
        Search for memories by vector similarity.

        Args:
            query_embedding: The query embedding vector
            top_k: Maximum number of results to return
            min_score: Minimum similarity score (0.0-1.0)
            filters: Optional filters dict. Supported keys:
                     - 'entity_id': Filter by entity ID (replaces deprecated 'user_id')
                     - 'namespace': Filter by namespace
                     - 'type': Filter by memory type (str or list[str])
                     - 'min_importance': Filter by minimum importance
                     - 'user_id': Deprecated, use 'entity_id' instead

        Returns:
            List of memory points matching the query
        """
        ...

    def find_similar_memories(
        self,
        embedding: list[float],
        entity_id: str | None = None,
        namespace: str = "default",
        threshold: float | None = None,
        limit: int = 5,
        exclude_archived: bool = True,
    ) -> list[tuple[Any, float]]:
        """
        Find similar memories based on vector similarity.

        Args:
            embedding: The embedding vector to search for
            entity_id: Filter by entity ID (for multi-entity isolation)
            namespace: Namespace for memory isolation (default: "default")
            threshold: Similarity threshold (0.0-1.0)
            limit: Maximum number of results to return
            exclude_archived: Whether to exclude archived memories

        Returns:
            List of tuples containing (memory_point, similarity_score)
        """
        ...

    def update_memory(self, memory_id: str, payload_updates: dict[str, Any]) -> bool:
        """
        Update specific fields in a memory's payload.

        Args:
            memory_id: The ID of the memory to update
            payload_updates: Dictionary of fields to update

        Returns:
            True if successful, False otherwise
        """
        ...

    def get_by_id(self, memory_id: str) -> Optional[Any]:
        """
        Retrieve a specific memory by its ID.

        Args:
            memory_id: The ID of the memory to retrieve

        Returns:
            Memory point if found, None otherwise
        """
        ...

    def archive_memory(self, memory_id: str, superseded_by: Optional[str] = None) -> bool:
        """
        Archive a memory by marking it as archived.

        Args:
            memory_id: The ID of the memory to archive
            superseded_by: Optional ID of the memory that supersedes this one

        Returns:
            True if successful, False otherwise
        """
        ...

    def clear_memories(self, entity_id: str, namespace: str = "default") -> int:
        """
        Clear all memories for a specific entity within a namespace.

        Args:
            entity_id: The ID of the entity whose memories to clear
            namespace: Namespace to clear within (default: "default")

        Returns:
            Number of memories deleted
        """
        ...

    def clear_user_memories(self, user_id: str) -> int:
        """
        Deprecated: Use clear_memories() instead.

        Clear all memories for a specific user across all namespaces.

        Args:
            user_id: The ID of the user whose memories to clear

        Returns:
            Number of memories deleted
        """
        ...


class ConflictStore(Protocol):
    """
    Protocol for memory conflict storage and management.

    Implementations should provide conflict tracking, resolution, and querying
    capabilities. This could be backed by PostgreSQL, in-memory storage, or
    other databases.
    """

    def add_conflict(self, conflict: MemoryConflict) -> str:
        """
        Store a detected conflict.

        Args:
            conflict: The conflict to store

        Returns:
            The conflict ID
        """
        ...

    def get_conflict(self, conflict_id: str) -> Optional[MemoryConflict]:
        """
        Retrieve a conflict by ID.

        Args:
            conflict_id: The conflict ID

        Returns:
            The conflict if found, None otherwise
        """
        ...

    def get_pending_conflicts(
        self, user_id: str, limit: Optional[int] = None
    ) -> list[MemoryConflict]:
        """
        Get all pending conflicts for a user.

        Args:
            user_id: The user ID
            limit: Maximum number of conflicts to return

        Returns:
            List of pending conflicts, sorted by importance
        """
        ...

    def resolve_conflict(self, conflict_id: str, resolution: ConflictResolution) -> bool:
        """
        Mark a conflict as resolved.

        Args:
            conflict_id: The conflict ID
            resolution: Resolution details

        Returns:
            True if successful, False if conflict not found
        """
        ...

    def get_conflict_count(self, user_id: str, status: Optional[str] = None) -> int:
        """
        Count conflicts for a user.

        Args:
            user_id: The user ID
            status: Optional status filter ("pending", "resolved", "escalated")

        Returns:
            Number of conflicts matching the criteria
        """
        ...

    def escalate_conflict(self, conflict_id: str) -> bool:
        """
        Escalate a conflict that couldn't be auto-resolved.

        Args:
            conflict_id: The conflict ID

        Returns:
            True if successful, False if conflict not found
        """
        ...

    def clear_user_conflicts(self, user_id: str, status: Optional[str] = None) -> int:
        """
        Clear conflicts for a user.

        Args:
            user_id: The user ID
            status: Optional status filter (only clear conflicts with this status)

        Returns:
            Number of conflicts cleared
        """
        ...


class ShortTermStore(Protocol):
    """
    Protocol for short-term memory storage (conversation history).

    Implementations should provide fast access to recent conversation messages.
    This is typically backed by Redis for production or in-memory for testing.
    Stores the last N messages per user for immediate conversational context.
    """

    def add_messages(self, user_id: str, messages: list[ShortTermMemory]) -> int:
        """
        Add messages to short-term storage.

        Args:
            user_id: The user ID
            messages: List of messages to add

        Returns:
            Number of messages added
        """
        ...

    def get_recent_messages(self, user_id: str, limit: int = 20) -> list[ShortTermMemory]:
        """
        Get recent messages for a user.

        Args:
            user_id: The user ID
            limit: Maximum number of messages to return (default: 20)

        Returns:
            List of recent messages, ordered by timestamp (oldest first)
        """
        ...

    def clear_user_messages(self, user_id: str) -> int:
        """
        Clear all messages for a user.

        Args:
            user_id: The user ID

        Returns:
            Number of messages deleted
        """
        ...

    def get_message_count(self, user_id: str) -> int:
        """
        Get the number of messages stored for a user.

        Args:
            user_id: The user ID

        Returns:
            Number of messages
        """
        ...
