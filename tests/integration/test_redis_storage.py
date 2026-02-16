"""Integration tests for Redis short-term memory storage backend."""

import pytest
from casual_llm.messages import AssistantMessage, UserMessage

from casual_memory.models import ShortTermMemory
from casual_memory.storage.short_term.redis import RedisShortTermStore


@pytest.mark.integration
def test_redis_add_and_get_messages(skip_if_no_redis):
    """Test adding and retrieving messages with Redis."""
    pytest.importorskip("redis")

    host = skip_if_no_redis  # Fixture returns the host

    # Create storage instance (use separate DB for testing)
    storage = RedisShortTermStore(host=host, port=6379, db=15)

    try:
        # Create test messages
        messages = [
            ShortTermMemory(
                message=UserMessage(content="Hello, how are you?"),
                timestamp="2024-01-01T10:00:00",
            ),
            ShortTermMemory(
                message=AssistantMessage(content="I'm doing well, thank you!"),
                timestamp="2024-01-01T10:00:05",
            ),
            ShortTermMemory(
                message=UserMessage(content="What's the weather like?"),
                timestamp="2024-01-01T10:00:10",
            ),
        ]

        # Add messages
        storage.add_messages(entity_id="test_user", messages=messages)

        # Get messages
        retrieved = storage.get_recent_messages(entity_id="test_user", limit=10)

        assert len(retrieved) == 3
        assert retrieved[0].message.content == "Hello, how are you?"
        assert retrieved[1].message.role == "assistant"
        assert retrieved[2].message.content == "What's the weather like?"

    finally:
        # Cleanup
        try:
            storage.clear_messages(entity_id="test_user")
        except Exception:
            pass


@pytest.mark.integration
def test_redis_message_limit(skip_if_no_redis):
    """Test that Redis respects the message limit."""
    pytest.importorskip("redis")

    host = skip_if_no_redis

    storage = RedisShortTermStore(
        host=host, port=6379, db=15, max_messages=5  # Limit to 5 messages
    )

    try:
        # Add 10 messages
        messages = [
            ShortTermMemory(
                message=(
                    UserMessage(content=f"Message {i}")
                    if i % 2 == 0
                    else AssistantMessage(content=f"Message {i}")
                ),
                timestamp=f"2024-01-01T10:{i:02d}:00",
            )
            for i in range(10)
        ]

        storage.add_messages(entity_id="test_user", messages=messages)

        # Should only have the last 5 messages
        retrieved = storage.get_recent_messages(entity_id="test_user", limit=10)

        assert len(retrieved) <= 5
        # Should be the most recent messages (5-9)
        assert any("Message 9" in m.message.content for m in retrieved)

    finally:
        try:
            storage.clear_messages(entity_id="test_user")
        except Exception:
            pass


@pytest.mark.integration
def test_redis_clear_messages(skip_if_no_redis):
    """Test clearing messages from Redis."""
    pytest.importorskip("redis")

    host = skip_if_no_redis

    storage = RedisShortTermStore(host=host, port=6379, db=15)

    try:
        # Add messages
        messages = [
            ShortTermMemory(
                message=UserMessage(content="Test message"), timestamp="2024-01-01T10:00:00"
            )
        ]

        storage.add_messages(entity_id="test_user", messages=messages)

        # Verify messages exist
        retrieved = storage.get_recent_messages(entity_id="test_user")
        assert len(retrieved) > 0

        # Clear messages
        storage.clear_messages(entity_id="test_user")

        # Verify messages are cleared
        retrieved = storage.get_recent_messages(entity_id="test_user")
        assert len(retrieved) == 0

    finally:
        try:
            storage.clear_messages(entity_id="test_user")
        except Exception:
            pass


@pytest.mark.integration
def test_redis_user_isolation(skip_if_no_redis):
    """Test that messages are isolated by entity_id."""
    pytest.importorskip("redis")

    host = skip_if_no_redis

    storage = RedisShortTermStore(host=host, port=6379, db=15)

    try:
        # Add messages for user1
        messages_user1 = [
            ShortTermMemory(
                message=UserMessage(content="User 1 message"), timestamp="2024-01-01T10:00:00"
            )
        ]
        storage.add_messages(entity_id="user_1", messages=messages_user1)

        # Add messages for user2
        messages_user2 = [
            ShortTermMemory(
                message=UserMessage(content="User 2 message"), timestamp="2024-01-01T10:00:00"
            )
        ]
        storage.add_messages(entity_id="user_2", messages=messages_user2)

        # Get messages for each user
        user1_messages = storage.get_recent_messages(entity_id="user_1")
        user2_messages = storage.get_recent_messages(entity_id="user_2")

        # Each user should only see their own messages
        assert len(user1_messages) == 1
        assert user1_messages[0].message.content == "User 1 message"

        assert len(user2_messages) == 1
        assert user2_messages[0].message.content == "User 2 message"

    finally:
        try:
            storage.clear_messages(entity_id="user_1")
            storage.clear_messages(entity_id="user_2")
        except Exception:
            pass


@pytest.mark.integration
def test_redis_get_with_limit(skip_if_no_redis):
    """Test retrieving messages with a limit."""
    pytest.importorskip("redis")

    host = skip_if_no_redis

    storage = RedisShortTermStore(host=host, port=6379, db=15)

    try:
        # Add 10 messages
        messages = [
            ShortTermMemory(
                message=(
                    UserMessage(content=f"Message {i}")
                    if i % 2 == 0
                    else AssistantMessage(content=f"Message {i}")
                ),
                timestamp=f"2024-01-01T10:{i:02d}:00",
            )
            for i in range(10)
        ]

        storage.add_messages(entity_id="test_user", messages=messages)

        # Get only 3 messages
        retrieved = storage.get_recent_messages(entity_id="test_user", limit=3)

        assert len(retrieved) == 3

    finally:
        try:
            storage.clear_messages(entity_id="test_user")
        except Exception:
            pass


@pytest.mark.integration
def test_redis_message_persistence(skip_if_no_redis):
    """Test that messages persist across storage instances."""
    pytest.importorskip("redis")

    host = skip_if_no_redis

    # First instance adds messages
    storage1 = RedisShortTermStore(host=host, port=6379, db=15)

    try:
        messages = [
            ShortTermMemory(
                message=UserMessage(content="Persistent message"),
                timestamp="2024-01-01T10:00:00",
            )
        ]

        storage1.add_messages(entity_id="test_user", messages=messages)

        # Second instance retrieves messages
        storage2 = RedisShortTermStore(host=host, port=6379, db=15)

        retrieved = storage2.get_recent_messages(entity_id="test_user")

        assert len(retrieved) == 1
        assert retrieved[0].message.content == "Persistent message"

    finally:
        try:
            storage1.clear_messages(entity_id="test_user")
        except Exception:
            pass
