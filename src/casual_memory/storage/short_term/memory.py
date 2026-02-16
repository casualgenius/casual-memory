"""
In-memory short-term storage implementation.

Provides a simple in-memory store for recent conversation messages,
suitable for testing and single-instance deployments. For production
with multiple replicas, use Redis implementation instead.
"""

import logging
import warnings
from collections import deque

from casual_memory.models import ShortTermMemory

logger = logging.getLogger(__name__)


class InMemoryShortTermStore:
    """
    In-memory implementation of the ShortTermStore protocol.

    Stores recent messages in a deque (double-ended queue) for efficient
    FIFO operations. Data is lost on restart.

    Messages are indexed by (namespace, entity_id) tuple for namespace isolation.
    """

    def __init__(self, max_messages: int = 20):
        """
        Initialize the store.

        Args:
            max_messages: Maximum number of messages to store per entity (default: 20)
        """
        self._messages: dict[tuple[str, str], deque[ShortTermMemory]] = {}
        self._max_messages = max_messages

        logger.info(f"InMemoryShortTermStore initialized (max_messages={max_messages})")

    def add_messages(
        self, entity_id: str, messages: list[ShortTermMemory], namespace: str = "default"
    ) -> int:
        """Add messages to short-term storage."""
        key = (namespace, entity_id)

        if key not in self._messages:
            self._messages[key] = deque(maxlen=self._max_messages)

        queue = self._messages[key]
        count = 0

        for message in messages:
            queue.append(message)
            count += 1

        logger.debug(
            f"Added {count} messages for entity {entity_id} in namespace {namespace} "
            f"(total: {len(queue)})"
        )

        return count

    def get_recent_messages(
        self, entity_id: str, limit: int = 20, namespace: str = "default"
    ) -> list[ShortTermMemory]:
        """Get recent messages for an entity."""
        key = (namespace, entity_id)

        if key not in self._messages:
            return []

        queue = self._messages[key]

        # Return last N messages (most recent)
        messages = list(queue)[-limit:]

        logger.debug(
            f"Retrieved {len(messages)} messages for entity {entity_id} "
            f"in namespace {namespace}"
        )

        return messages

    def clear_messages(self, entity_id: str, namespace: str = "default") -> int:
        """Clear all messages for an entity within a namespace."""
        key = (namespace, entity_id)

        if key not in self._messages:
            return 0

        count = len(self._messages[key])
        del self._messages[key]

        logger.info(f"Cleared {count} messages for entity {entity_id} in namespace {namespace}")

        return count

    def clear_user_messages(self, user_id: str) -> int:
        """Deprecated: Use clear_messages() instead.

        Clear all messages for a user across all namespaces.
        """
        warnings.warn(
            "clear_user_messages() is deprecated, use clear_messages() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        total = 0
        keys_to_delete = [key for key in self._messages if key[1] == user_id]
        for key in keys_to_delete:
            total += len(self._messages[key])
            del self._messages[key]

        logger.info(f"Cleared {total} messages for user {user_id} across all namespaces")

        return total

    def get_message_count(self, entity_id: str, namespace: str = "default") -> int:
        """Get the number of messages stored for an entity."""
        key = (namespace, entity_id)

        if key not in self._messages:
            return 0

        return len(self._messages[key])
