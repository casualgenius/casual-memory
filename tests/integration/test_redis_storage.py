"""Integration tests for Redis short-term memory storage backend."""

import pytest

from casual_memory.models import ShortTermMemory
from casual_memory.storage.short_term.redis import RedisShortTermStore


@pytest.mark.integration
@pytest.mark.asyncio
async def test_redis_add_and_get_messages(skip_if_no_redis):
    """Test adding and retrieving messages with Redis."""
    pytest.importorskip("redis")

    # Create storage instance
    storage = RedisShortTermStore(host="localhost", port=6379, db=15)  # Use separate DB for testing

    await storage.initialize()

    try:
        # Create test messages
        messages = [
            ShortTermMemory(
                role="user", content="Hello, how are you?", timestamp="2024-01-01T10:00:00"
            ),
            ShortTermMemory(
                role="assistant",
                content="I'm doing well, thank you!",
                timestamp="2024-01-01T10:00:05",
            ),
            ShortTermMemory(
                role="user", content="What's the weather like?", timestamp="2024-01-01T10:00:10"
            ),
        ]

        # Add messages
        await storage.add(messages, user_id="test_user")

        # Get messages
        retrieved = await storage.get(user_id="test_user", limit=10)

        assert len(retrieved) == 3
        assert retrieved[0].content == "Hello, how are you?"
        assert retrieved[1].role == "assistant"
        assert retrieved[2].content == "What's the weather like?"

    finally:
        # Cleanup
        try:
            await storage.clear(user_id="test_user")
        except Exception:
            pass


@pytest.mark.integration
@pytest.mark.asyncio
async def test_redis_message_limit(skip_if_no_redis):
    """Test that Redis respects the message limit."""
    pytest.importorskip("redis")

    storage = RedisShortTermStorage(
        host="localhost", port=6379, db=15, max_messages=5  # Limit to 5 messages
    )

    await storage.initialize()

    try:
        # Add 10 messages
        messages = [
            ShortTermMemory(
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}",
                timestamp=f"2024-01-01T10:{i:02d}:00",
            )
            for i in range(10)
        ]

        await storage.add(messages, user_id="test_user")

        # Should only have the last 5 messages
        retrieved = await storage.get(user_id="test_user", limit=10)

        assert len(retrieved) <= 5
        # Should be the most recent messages (5-9)
        assert any("Message 9" in m.content for m in retrieved)

    finally:
        try:
            await storage.clear(user_id="test_user")
        except Exception:
            pass


@pytest.mark.integration
@pytest.mark.asyncio
async def test_redis_clear_messages(skip_if_no_redis):
    """Test clearing messages from Redis."""
    pytest.importorskip("redis")

    storage = RedisShortTermStorage(host="localhost", port=6379, db=15)

    await storage.initialize()

    try:
        # Add messages
        messages = [
            ShortTermMemory(role="user", content="Test message", timestamp="2024-01-01T10:00:00")
        ]

        await storage.add(messages, user_id="test_user")

        # Verify messages exist
        retrieved = await storage.get(user_id="test_user")
        assert len(retrieved) > 0

        # Clear messages
        await storage.clear(user_id="test_user")

        # Verify messages are cleared
        retrieved = await storage.get(user_id="test_user")
        assert len(retrieved) == 0

    finally:
        try:
            await storage.clear(user_id="test_user")
        except Exception:
            pass


@pytest.mark.integration
@pytest.mark.asyncio
async def test_redis_user_isolation(skip_if_no_redis):
    """Test that messages are isolated by user_id."""
    pytest.importorskip("redis")

    storage = RedisShortTermStorage(host="localhost", port=6379, db=15)

    await storage.initialize()

    try:
        # Add messages for user1
        messages_user1 = [
            ShortTermMemory(role="user", content="User 1 message", timestamp="2024-01-01T10:00:00")
        ]
        await storage.add(messages_user1, user_id="user_1")

        # Add messages for user2
        messages_user2 = [
            ShortTermMemory(role="user", content="User 2 message", timestamp="2024-01-01T10:00:00")
        ]
        await storage.add(messages_user2, user_id="user_2")

        # Get messages for each user
        user1_messages = await storage.get(user_id="user_1")
        user2_messages = await storage.get(user_id="user_2")

        # Each user should only see their own messages
        assert len(user1_messages) == 1
        assert user1_messages[0].content == "User 1 message"

        assert len(user2_messages) == 1
        assert user2_messages[0].content == "User 2 message"

    finally:
        try:
            await storage.clear(user_id="user_1")
            await storage.clear(user_id="user_2")
        except Exception:
            pass


@pytest.mark.integration
@pytest.mark.asyncio
async def test_redis_get_with_limit(skip_if_no_redis):
    """Test retrieving messages with a limit."""
    pytest.importorskip("redis")

    storage = RedisShortTermStorage(host="localhost", port=6379, db=15)

    await storage.initialize()

    try:
        # Add 10 messages
        messages = [
            ShortTermMemory(
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}",
                timestamp=f"2024-01-01T10:{i:02d}:00",
            )
            for i in range(10)
        ]

        await storage.add(messages, user_id="test_user")

        # Get only 3 messages
        retrieved = await storage.get(user_id="test_user", limit=3)

        assert len(retrieved) == 3

    finally:
        try:
            await storage.clear(user_id="test_user")
        except Exception:
            pass


@pytest.mark.integration
@pytest.mark.asyncio
async def test_redis_message_persistence(skip_if_no_redis):
    """Test that messages persist across storage instances."""
    pytest.importorskip("redis")

    # First instance adds messages
    storage1 = RedisShortTermStore(host="localhost", port=6379, db=15)

    await storage1.initialize()

    try:
        messages = [
            ShortTermMemory(
                role="user", content="Persistent message", timestamp="2024-01-01T10:00:00"
            )
        ]

        await storage1.add(messages, user_id="test_user")

        # Second instance retrieves messages
        storage2 = RedisShortTermStore(host="localhost", port=6379, db=15)

        await storage2.initialize()

        retrieved = await storage2.get(user_id="test_user")

        assert len(retrieved) == 1
        assert retrieved[0].content == "Persistent message"

    finally:
        try:
            await storage1.clear(user_id="test_user")
        except Exception:
            pass
