"""
Models for memory action execution results.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class MemoryActionResult:
    """
    Result of executing memory actions.

    Provides structured information about what action was taken
    and the relevant IDs for tracking.

    Attributes:
        action: The action that was executed ("added", "updated", or "conflict")
        memory_id: ID of memory (added or updated), None for conflicts
        conflict_ids: List of conflict IDs created, empty for add/skip
        superseded_ids: List of memory IDs that were superseded (archived)
        metadata: Additional context about the action

    Examples:
        >>> # Memory added
        >>> result = MemoryActionResult(
        ...     action="added",
        ...     memory_id="mem_123",
        ...     superseded_ids=["mem_100", "mem_101"]
        ... )

        >>> # Memory updated (duplicate)
        >>> result = MemoryActionResult(
        ...     action="updated",
        ...     memory_id="mem_456"
        ... )

        >>> # Conflict detected
        >>> result = MemoryActionResult(
        ...     action="conflict",
        ...     conflict_ids=["conflict_001"]
        ... )
    """

    action: Literal["added", "updated", "conflict"]
    memory_id: Optional[str] = None
    conflict_ids: list[str] = field(default_factory=list)
    superseded_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
