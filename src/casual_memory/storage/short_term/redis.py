"""
Redis short-term storage implementation.

Provides a Redis-backed store for recent conversation messages,
suitable for production deployments with multiple replicas.
"""

import logging
import warnings

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

    Keys follow the pattern ``{prefix}{namespace}:{entity_id}`` for
    namespace isolation.
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
            max_messages: Maximum number of messages to store per entity
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

    def _get_key(self, entity_id: str, namespace: str = "default") -> str:
        """Get the Redis key for an entity's messages within a namespace."""
        return f"{self._key_prefix}{namespace}:{entity_id}"

    def add_messages(
        self, entity_id: str, messages: list[ShortTermMemory], namespace: str = "default"
    ) -> int:
        """Add messages to short-term storage."""
        key = self._get_key(entity_id, namespace)
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

        logger.debug(f"Added {count} messages for entity {entity_id} in namespace {namespace}")

        return count

    def get_recent_messages(
        self, entity_id: str, limit: int = 20, namespace: str = "default"
    ) -> list[ShortTermMemory]:
        """Get recent messages for an entity."""
        key = self._get_key(entity_id, namespace)

        # Get last N messages
        messages_json_list: list[str] = self.client.lrange(key, -limit, -1)

        messages = []
        for msg_json in messages_json_list:
            try:
                message = ShortTermMemory.model_validate_json(msg_json)
                messages.append(message)
            except Exception as e:
                logger.warning(f"Failed to deserialize message: {e}")
                continue

        logger.debug(
            f"Retrieved {len(messages)} messages for entity {entity_id} "
            f"in namespace {namespace}"
        )

        return messages

    def clear_messages(self, entity_id: str, namespace: str = "default") -> int:
        """Clear all messages for an entity within a namespace."""
        key = self._get_key(entity_id, namespace)

        # Get count before deletion
        count: int = self.client.llen(key)

        # Delete the key
        self.client.delete(key)

        logger.info(f"Cleared {count} messages for entity {entity_id} in namespace {namespace}")

        return count

    def clear_user_messages(self, user_id: str) -> int:
        """Deprecated: Use clear_messages() instead.

        Clear all messages for a user across all namespaces.

        Note: This scans for all keys matching the user_id pattern across
        namespaces. For large key spaces, prefer clear_messages() with an
        explicit namespace.
        """
        warnings.warn(
            "clear_user_messages() is deprecated, use clear_messages() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Scan for keys matching any namespace for this user_id
        pattern = f"{self._key_prefix}*:{user_id}"
        total = 0

        for key in self.client.scan_iter(match=pattern):
            count: int = self.client.llen(key)
            total += count
            self.client.delete(key)

        logger.info(f"Cleared {total} messages for user {user_id} across all namespaces")

        return total

    def get_message_count(self, entity_id: str, namespace: str = "default") -> int:
        """Get the number of messages stored for an entity."""
        key = self._get_key(entity_id, namespace)
        count: int = self.client.llen(key)
        return count
