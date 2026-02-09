import logging
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

    def _compose_key(self, user_id: str, session_id: str) -> str:
        """Compose a storage key from user_id and session_id."""
        return f"{user_id}:{session_id}"

    def add(
        self, user_id: str, session_id: str, messages: list[ChatMessage]
    ) -> list[ShortTermMemory]:
        """Add ChatMessages to short-term storage.

        Filters out system messages. Each message is wrapped in a
        ShortTermMemory with a timestamp before storage.

        Args:
            user_id: The user ID
            session_id: The session ID
            messages: List of ChatMessages to store

        Returns:
            List of ShortTermMemory objects that were stored.
        """
        memories = []
        for message in messages:
            if message.role not in ("user", "assistant", "tool"):
                logger.debug(f"Skipping message with role '{message.role}'")
                continue
            memory = ShortTermMemory(message=message, timestamp=str(datetime.now()))
            memories.append(memory)

        if memories:
            key = self._compose_key(user_id, session_id)
            self.short_term_store.add_messages(key, memories)
            logger.info(f"Saved {len(memories)} messages to short-term store")

        return memories

    def get(self, user_id: str, session_id: str, limit: int | None = None) -> list[ShortTermMemory]:
        """Get recent messages with safe boundary trimming.

        Over-fetches from the store and trims the result so the first
        message has role "user", ensuring tool-call sequences are never
        split.

        Args:
            user_id: The user ID
            session_id: The session ID
            limit: Max messages to return (defaults to short_term_limit)

        Returns:
            List of messages starting at a user message boundary.
            May return more or fewer than limit messages, or an empty
            list if no user message is found.
        """
        effective_limit = self.short_term_limit if limit is None else limit
        key = self._compose_key(user_id, session_id)

        fetch_count = effective_limit + OVERFETCH_BUFFER
        messages = self.short_term_store.get_recent_messages(key, fetch_count)

        return trim_to_safe_boundary(messages, target_limit=effective_limit)

    def clear(self, user_id: str, session_id: str) -> int:
        """Clear all messages for a session.

        Args:
            user_id: The user ID
            session_id: The session ID

        Returns:
            Number of messages deleted.
        """
        key = self._compose_key(user_id, session_id)
        return self.short_term_store.clear_user_messages(key)
