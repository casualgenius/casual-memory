"""
In-memory short-term storage implementation.

Provides a simple in-memory store for recent conversation messages,
suitable for testing and single-instance deployments. For production
with multiple replicas, use Redis implementation instead.
"""

import logging
from collections import deque
from typing import Dict, List

from casual_memory.models import ShortTermMemory

logger = logging.getLogger(__name__)


class InMemoryShortTermStore:
    """
    In-memory implementation of the ShortTermStore protocol.

    Stores recent messages in a deque (double-ended queue) for efficient
    FIFO operations. Data is lost on restart.
    """

    def __init__(self, max_messages: int = 20):
        """
        Initialize the store.

        Args:
            max_messages: Maximum number of messages to store per user (default: 20)
        """
        # Store messages by user_id
        self._messages: Dict[str, deque] = {}
        self._max_messages = max_messages

        logger.info(f"InMemoryShortTermStore initialized (max_messages={max_messages})")

    def add_messages(self, user_id: str, messages: List[ShortTermMemory]) -> int:
        """Add messages to short-term storage."""
        if user_id not in self._messages:
            self._messages[user_id] = deque(maxlen=self._max_messages)

        queue = self._messages[user_id]
        count = 0

        for message in messages:
            queue.append(message)
            count += 1

        logger.debug(f"Added {count} messages for user {user_id} (total: {len(queue)})")

        return count

    def get_recent_messages(self, user_id: str, limit: int = 20) -> List[ShortTermMemory]:
        """Get recent messages for a user."""
        if user_id not in self._messages:
            return []

        queue = self._messages[user_id]

        # Return last N messages (most recent)
        messages = list(queue)[-limit:]

        logger.debug(f"Retrieved {len(messages)} messages for user {user_id}")

        return messages

    def clear_user_messages(self, user_id: str) -> int:
        """Clear all messages for a user."""
        if user_id not in self._messages:
            return 0

        count = len(self._messages[user_id])
        del self._messages[user_id]

        logger.info(f"Cleared {count} messages for user {user_id}")

        return count

    def get_message_count(self, user_id: str) -> int:
        """Get the number of messages stored for a user."""
        if user_id not in self._messages:
            return 0

        return len(self._messages[user_id])
