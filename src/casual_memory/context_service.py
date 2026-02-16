import logging
import warnings
from datetime import datetime

from casual_llm import ChatMessage

from casual_memory.models import ShortTermMemory
from casual_memory.storage.protocols import ShortTermStore
from casual_memory.storage.short_term.utils import OVERFETCH_BUFFER, trim_to_safe_boundary

logger = logging.getLogger(__name__)


class ContextService:
    """Service for managing short-term conversation context.

    Handles storing and retrieving recent conversation messages with
    safe boundary trimming to ensure tool-call sequences are never split.
    """

    def __init__(
        self,
        short_term_store: ShortTermStore,
        short_term_limit: int = 50,
    ):
        self.short_term_store = short_term_store
        self.short_term_limit = short_term_limit

    def _compose_key(self, entity_id: str, session_id: str) -> str:
        """Compose a storage key from entity_id and session_id."""
        return f"{entity_id}:{session_id}"

    def add(
        self,
        entity_id: str,
        session_id: str,
        messages: list[ChatMessage],
        namespace: str = "default",
        *,
        user_id: str | None = None,
    ) -> list[ShortTermMemory]:
        """Add ChatMessages to short-term storage.

        Filters out system messages. Each message is wrapped in a
        ShortTermMemory with a timestamp before storage.

        Args:
            entity_id: The entity ID (or user ID for backward compatibility)
            session_id: The session ID
            messages: List of ChatMessages to store
            namespace: Namespace for message isolation (default: "default")
            user_id: Deprecated. Use entity_id instead.

        Returns:
            List of ShortTermMemory objects that were stored.
        """
        entity_id = self._resolve_entity_id(entity_id, user_id)

        memories: list[ShortTermMemory] = []
        for message in messages:
            if message.role not in ("user", "assistant", "tool"):
                logger.debug(f"Skipping message with role '{message.role}'")
                continue
            memory = ShortTermMemory(message=message, timestamp=str(datetime.now()))
            memories.append(memory)

        if memories:
            key = self._compose_key(entity_id, session_id)
            self.short_term_store.add_messages(key, memories, namespace=namespace)
            logger.info(f"Saved {len(memories)} messages to short-term store")

        return memories

    def get(
        self,
        entity_id: str,
        session_id: str,
        limit: int | None = None,
        namespace: str = "default",
        *,
        user_id: str | None = None,
    ) -> list[ShortTermMemory]:
        """Get recent messages with safe boundary trimming.

        Over-fetches from the store and trims the result so the first
        message has role "user", ensuring tool-call sequences are never
        split.

        Args:
            entity_id: The entity ID (or user ID for backward compatibility)
            session_id: The session ID
            limit: Max messages to return (defaults to short_term_limit)
            namespace: Namespace for message isolation (default: "default")
            user_id: Deprecated. Use entity_id instead.

        Returns:
            List of messages starting at a user message boundary.
            May return more or fewer than limit messages, or an empty
            list if no user message is found.
        """
        entity_id = self._resolve_entity_id(entity_id, user_id)

        effective_limit = self.short_term_limit if limit is None else limit
        key = self._compose_key(entity_id, session_id)

        fetch_count = effective_limit + OVERFETCH_BUFFER
        messages = self.short_term_store.get_recent_messages(key, fetch_count, namespace=namespace)

        return trim_to_safe_boundary(messages, target_limit=effective_limit)

    def clear(
        self,
        entity_id: str,
        session_id: str,
        namespace: str = "default",
        *,
        user_id: str | None = None,
    ) -> int:
        """Clear all messages for a session.

        Args:
            entity_id: The entity ID (or user ID for backward compatibility)
            session_id: The session ID
            namespace: Namespace for message isolation (default: "default")
            user_id: Deprecated. Use entity_id instead.

        Returns:
            Number of messages deleted.
        """
        entity_id = self._resolve_entity_id(entity_id, user_id)

        key = self._compose_key(entity_id, session_id)
        return self.short_term_store.clear_messages(key, namespace=namespace)

    @staticmethod
    def _resolve_entity_id(entity_id: str, user_id: str | None) -> str:
        """Resolve entity_id from positional arg and deprecated user_id kwarg.

        If user_id is provided as a keyword argument, it is used as entity_id
        with a deprecation warning. If both are provided and differ, entity_id
        takes precedence and user_id is ignored with a warning.
        """
        if user_id is not None:
            warnings.warn(
                "ContextService user_id parameter is deprecated, use entity_id instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            # If entity_id was also provided via the positional arg
            # (always required), use it. The user_id kwarg is just ignored
            # with the deprecation warning above.
        return entity_id
