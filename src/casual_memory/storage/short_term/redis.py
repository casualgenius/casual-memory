"""
Redis short-term storage implementation.

Provides a Redis-backed store for recent conversation messages,
suitable for production deployments with multiple replicas.
"""

import logging
import json
from typing import List
from casual_memory.models import ShortTermMemory

try:
    import redis
except ImportError:
    redis = None  # type: ignore

logger = logging.getLogger(__name__)


class RedisShortTermStore:
    """
    Redis implementation of the ShortTermStore protocol.

    Stores recent messages in Redis lists for fast FIFO operations.
    Survives restarts and works across multiple replicas.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        max_messages: int = 20,
        key_prefix: str = "memory:",
    ):
        """
        Initialize the Redis store.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            max_messages: Maximum number of messages to store per user
            key_prefix: Prefix for Redis keys (default: "memory:")
        """
        if redis is None:
            raise ImportError(
                "redis package is required for RedisShortTermStore. "
                "Install with: pip install redis"
            )

        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self._max_messages = max_messages
        self._key_prefix = key_prefix

        # Test connection
        try:
            self.client.ping()
            logger.info(
                f"RedisShortTermStore initialized (host={host}:{port}, "
                f"max_messages={max_messages})"
            )
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def _get_key(self, user_id: str) -> str:
        """Get the Redis key for a user's messages."""
        return f"{self._key_prefix}{user_id}"

    def add_messages(self, user_id: str, messages: List[ShortTermMemory]) -> int:
        """Add messages to short-term storage."""
        key = self._get_key(user_id)
        count = 0

        pipeline = self.client.pipeline()

        for message in messages:
            # Serialize message to JSON
            message_json = message.model_dump_json()
            pipeline.rpush(key, message_json)
            count += 1

        # Trim to max_messages
        pipeline.ltrim(key, -self._max_messages, -1)

        # Execute pipeline
        pipeline.execute()

        logger.debug(f"Added {count} messages for user {user_id}")

        return count

    def get_recent_messages(self, user_id: str, limit: int = 20) -> List[ShortTermMemory]:
        """Get recent messages for a user."""
        key = self._get_key(user_id)

        # Get last N messages
        messages_json = self.client.lrange(key, -limit, -1)

        messages = []
        for msg_json in messages_json:
            try:
                message = ShortTermMemory.model_validate_json(msg_json)
                messages.append(message)
            except Exception as e:
                logger.warning(f"Failed to deserialize message: {e}")
                continue

        logger.debug(f"Retrieved {len(messages)} messages for user {user_id}")

        return messages

    def clear_user_messages(self, user_id: str) -> int:
        """Clear all messages for a user."""
        key = self._get_key(user_id)

        # Get count before deletion
        count = self.client.llen(key)

        # Delete the key
        self.client.delete(key)

        logger.info(f"Cleared {count} messages for user {user_id}")

        return count

    def get_message_count(self, user_id: str) -> int:
        """Get the number of messages stored for a user."""
        key = self._get_key(user_id)
        return self.client.llen(key)
