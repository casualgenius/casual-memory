"""
In-memory conflict storage implementation.

Provides a simple in-memory store for conflicts, suitable for testing
and single-instance deployments. For production with multiple replicas,
use PostgreSQL or SQLite implementations instead.
"""

import logging
import warnings
from datetime import datetime

from casual_memory.models import ConflictResolution, MemoryConflict

logger = logging.getLogger(__name__)


class InMemoryConflictStore:
    """
    In-memory implementation of the ConflictStore protocol.

    Stores conflicts in dictionaries for fast lookup. Data is lost on restart.
    Indexes conflicts by (namespace, entity_id) tuple for namespace-aware isolation.
    """

    def __init__(self) -> None:
        # Store conflicts by conflict ID
        self._conflicts: dict[str, MemoryConflict] = {}

        # Index by (namespace, entity_id) for fast lookup
        self._entity_conflicts: dict[tuple[str, str], list[str]] = {}

        logger.info("InMemoryConflictStore initialized")

    def add_conflict(self, conflict: MemoryConflict) -> str:
        """Store a detected conflict."""
        conflict_id = conflict.id

        # Store the conflict
        self._conflicts[conflict_id] = conflict

        # Add to (namespace, entity_id) index
        key = (conflict.namespace, conflict.entity_id)
        if key not in self._entity_conflicts:
            self._entity_conflicts[key] = []
        self._entity_conflicts[key].append(conflict_id)

        logger.info(
            f"Stored conflict {conflict_id} for entity {conflict.entity_id} "
            f"in namespace {conflict.namespace}: "
            f"{conflict.category} ({conflict.status})"
        )

        return conflict_id

    def get_conflict(self, conflict_id: str) -> MemoryConflict | None:
        """Retrieve a conflict by ID."""
        return self._conflicts.get(conflict_id)

    def get_pending_conflicts(
        self, entity_id: str, namespace: str = "default", limit: int | None = None
    ) -> list[MemoryConflict]:
        """Get all pending conflicts for an entity within a namespace."""
        key = (namespace, entity_id)
        if key not in self._entity_conflicts:
            return []

        conflict_ids = self._entity_conflicts[key]
        conflicts = [
            self._conflicts[cid]
            for cid in conflict_ids
            if cid in self._conflicts and self._conflicts[cid].status == "pending"
        ]

        # Sort by average importance (high to low)
        conflicts.sort(key=lambda c: c.avg_importance, reverse=True)

        if limit:
            conflicts = conflicts[:limit]

        return conflicts

    def resolve_conflict(self, conflict_id: str, resolution: ConflictResolution) -> bool:
        """Mark a conflict as resolved."""
        conflict = self._conflicts.get(conflict_id)
        if not conflict:
            logger.warning(f"Cannot resolve conflict {conflict_id}: not found")
            return False

        # Update conflict status
        conflict.status = "resolved"
        conflict.resolved_at = datetime.now()
        conflict.resolution_type = resolution.resolution_type

        # Set winning memory based on resolution decision
        if resolution.decision == "keep_a":
            conflict.winning_memory_id = conflict.memory_a_id
        elif resolution.decision == "keep_b":
            conflict.winning_memory_id = conflict.memory_b_id
        elif resolution.decision == "merge":
            # Could create a merged memory and set that as the winner
            # For now, just mark as resolved
            pass

        # Update metadata
        conflict.metadata.update(
            {
                "resolution_decision": resolution.decision,
                "resolution_notes": resolution.notes,
                "resolved_by": resolution.resolved_by,
                "resolved_at": datetime.now().isoformat(),
            }
        )

        logger.info(
            f"Resolved conflict {conflict_id}: {resolution.decision} "
            f"(by {resolution.resolved_by})"
        )

        return True

    def get_conflict_count(
        self, entity_id: str, namespace: str = "default", status: str | None = None
    ) -> int:
        """Count conflicts for an entity within a namespace."""
        key = (namespace, entity_id)
        if key not in self._entity_conflicts:
            return 0

        conflict_ids = self._entity_conflicts[key]
        conflicts = [self._conflicts[cid] for cid in conflict_ids if cid in self._conflicts]

        if status:
            conflicts = [c for c in conflicts if c.status == status]

        return len(conflicts)

    def escalate_conflict(self, conflict_id: str) -> bool:
        """Escalate a conflict that couldn't be auto-resolved."""
        conflict = self._conflicts.get(conflict_id)
        if not conflict:
            logger.warning(f"Cannot escalate conflict {conflict_id}: not found")
            return False

        conflict.status = "escalated"
        conflict.resolution_attempts += 1

        logger.info(f"Escalated conflict {conflict_id} (attempts={conflict.resolution_attempts})")

        return True

    def clear_conflicts(
        self, entity_id: str, namespace: str = "default", status: str | None = None
    ) -> int:
        """Clear conflicts for an entity within a namespace."""
        key = (namespace, entity_id)
        if key not in self._entity_conflicts:
            return 0

        conflict_ids = self._entity_conflicts[key].copy()
        cleared_count = 0

        for conflict_id in conflict_ids:
            conflict = self._conflicts.get(conflict_id)
            if not conflict:
                continue

            if status is None or conflict.status == status:
                # Remove from conflicts
                del self._conflicts[conflict_id]

                # Remove from entity index
                self._entity_conflicts[key].remove(conflict_id)

                cleared_count += 1

        logger.info(
            f"Cleared {cleared_count} conflicts for entity {entity_id} "
            f"in namespace {namespace} (status={status or 'all'})"
        )

        return cleared_count

    def clear_user_conflicts(self, user_id: str, status: str | None = None) -> int:
        """Deprecated: Use clear_conflicts() instead.

        Clear conflicts for a user across all namespaces.
        """
        warnings.warn(
            "clear_user_conflicts() is deprecated, use clear_conflicts() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        total_cleared = 0
        # Find all keys matching this entity_id across all namespaces
        matching_keys = [key for key in self._entity_conflicts if key[1] == user_id]
        for key in matching_keys:
            namespace = key[0]
            total_cleared += self.clear_conflicts(user_id, namespace=namespace, status=status)
        return total_cleared
