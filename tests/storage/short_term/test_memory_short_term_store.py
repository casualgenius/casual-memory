"""
Unit tests for in-memory short-term storage.

Tests message storage, retrieval, and FIFO behavior.
"""

from datetime import datetime

import pytest
from casual_llm.messages import AssistantMessage, UserMessage

from casual_memory.models import ShortTermMemory
from casual_memory.storage.short_term.memory import InMemoryShortTermStore


@pytest.fixture
def short_term_store():
    """Create a fresh in-memory short-term store."""
    return InMemoryShortTermStore(max_messages=20)


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        ShortTermMemory(
            message=UserMessage(content="Hello!"),
            timestamp=datetime.now().isoformat(),
        ),
        ShortTermMemory(
            message=AssistantMessage(content="Hi! How can I help?"),
            timestamp=datetime.now().isoformat(),
        ),
        ShortTermMemory(
            message=UserMessage(content="What's the weather?"),
            timestamp=datetime.now().isoformat(),
        ),
    ]


def test_add_messages(short_term_store, sample_messages):
    """Test adding messages to storage."""
    count = short_term_store.add_messages("user1", sample_messages)

    assert count == 3


def test_get_recent_messages(short_term_store, sample_messages):
    """Test retrieving recent messages."""
    short_term_store.add_messages("user1", sample_messages)

    messages = short_term_store.get_recent_messages("user1")

    assert len(messages) == 3
    assert messages[0].message.content == "Hello!"
    assert messages[1].message.role == "assistant"


def test_get_recent_messages_with_limit(short_term_store):
    """Test retrieving messages with limit."""
    # Add 10 messages
    messages = [
        ShortTermMemory(
            message=UserMessage(content=f"Message {i}"),
            timestamp=datetime.now().isoformat(),
        )
        for i in range(10)
    ]
    short_term_store.add_messages("user1", messages)

    # Get only last 5
    recent = short_term_store.get_recent_messages("user1", limit=5)

    assert len(recent) == 5
    # Should get the most recent messages
    assert recent[0].message.content == "Message 5"
    assert recent[4].message.content == "Message 9"


def test_get_recent_messages_empty(short_term_store):
    """Test retrieving messages when none exist."""
    messages = short_term_store.get_recent_messages("user1")

    assert len(messages) == 0


def test_max_messages_limit(short_term_store):
    """Test that max_messages limit is enforced."""
    # Add more than max_messages (20)
    messages = [
        ShortTermMemory(
            message=UserMessage(content=f"Message {i}"),
            timestamp=datetime.now().isoformat(),
        )
        for i in range(25)
    ]
    short_term_store.add_messages("user1", messages)

    # Should only keep last 20
    all_messages = short_term_store.get_recent_messages("user1", limit=30)

    assert len(all_messages) == 20
    # Should have dropped first 5 messages
    assert all_messages[0].message.content == "Message 5"
    assert all_messages[-1].message.content == "Message 24"


def test_fifo_behavior(short_term_store):
    """Test FIFO (First In, First Out) behavior."""
    # Add messages in batches
    batch1 = [
        ShortTermMemory(
            message=UserMessage(content=f"Batch1 Message {i}"),
            timestamp=datetime.now().isoformat(),
        )
        for i in range(10)
    ]

    batch2 = [
        ShortTermMemory(
            message=UserMessage(content=f"Batch2 Message {i}"),
            timestamp=datetime.now().isoformat(),
        )
        for i in range(15)
    ]

    short_term_store.add_messages("user1", batch1)
    short_term_store.add_messages("user1", batch2)

    # Should have only last 20 messages (5 from batch2)
    all_messages = short_term_store.get_recent_messages("user1", limit=30)

    assert len(all_messages) == 20
    # First 5 should be from end of batch1
    assert all_messages[0].message.content.startswith("Batch1")
    # Rest should be from batch2
    assert all_messages[-1].message.content.startswith("Batch2")


def test_clear_messages(short_term_store, sample_messages):
    """Test clearing all messages for an entity."""
    short_term_store.add_messages("user1", sample_messages)

    count = short_term_store.clear_messages("user1")

    assert count == 3

    # Verify messages are cleared
    messages = short_term_store.get_recent_messages("user1")
    assert len(messages) == 0


def test_clear_nonexistent_entity(short_term_store):
    """Test clearing messages for an entity with no messages."""
    count = short_term_store.clear_messages("nonexistent")

    assert count == 0


def test_get_message_count(short_term_store, sample_messages):
    """Test getting message count."""
    short_term_store.add_messages("user1", sample_messages)

    count = short_term_store.get_message_count("user1")

    assert count == 3


def test_get_message_count_empty(short_term_store):
    """Test getting message count when no messages exist."""
    count = short_term_store.get_message_count("user1")

    assert count == 0


def test_entity_isolation(short_term_store):
    """Test that messages are isolated per entity."""
    messages_user1 = [
        ShortTermMemory(
            message=UserMessage(content="User1 message"),
            timestamp=datetime.now().isoformat(),
        )
    ]

    messages_user2 = [
        ShortTermMemory(
            message=UserMessage(content="User2 message"),
            timestamp=datetime.now().isoformat(),
        )
    ]

    short_term_store.add_messages("user1", messages_user1)
    short_term_store.add_messages("user2", messages_user2)

    # Each entity should only see their own messages
    user1_messages = short_term_store.get_recent_messages("user1")
    user2_messages = short_term_store.get_recent_messages("user2")

    assert len(user1_messages) == 1
    assert len(user2_messages) == 1
    assert user1_messages[0].message.content == "User1 message"
    assert user2_messages[0].message.content == "User2 message"


def test_multiple_entities_independent_limits(short_term_store):
    """Test that each entity has independent max_messages limit."""
    # Add 25 messages for each entity
    for entity_id in ["user1", "user2"]:
        messages = [
            ShortTermMemory(
                message=UserMessage(content=f"{entity_id} message {i}"),
                timestamp=datetime.now().isoformat(),
            )
            for i in range(25)
        ]
        short_term_store.add_messages(entity_id, messages)

    # Each entity should have 20 messages (max limit)
    user1_count = short_term_store.get_message_count("user1")
    user2_count = short_term_store.get_message_count("user2")

    assert user1_count == 20
    assert user2_count == 20


def test_incremental_adds(short_term_store):
    """Test adding messages incrementally."""
    # Add messages one at a time
    for i in range(5):
        msg = (
            UserMessage(content=f"Message {i}")
            if i % 2 == 0
            else AssistantMessage(content=f"Message {i}")
        )
        message = ShortTermMemory(
            message=msg,
            timestamp=datetime.now().isoformat(),
        )
        short_term_store.add_messages("user1", [message])

    # Should have all 5 messages
    messages = short_term_store.get_recent_messages("user1")

    assert len(messages) == 5
    # Verify order
    assert messages[0].message.content == "Message 0"
    assert messages[4].message.content == "Message 4"


def test_message_roles_preserved(short_term_store):
    """Test that message roles are preserved correctly."""
    messages = [
        ShortTermMemory(
            message=UserMessage(content="User message"),
            timestamp=datetime.now().isoformat(),
        ),
        ShortTermMemory(
            message=AssistantMessage(content="Assistant message"),
            timestamp=datetime.now().isoformat(),
        ),
    ]

    short_term_store.add_messages("user1", messages)

    retrieved = short_term_store.get_recent_messages("user1")

    assert retrieved[0].message.role == "user"
    assert retrieved[1].message.role == "assistant"


def test_custom_max_messages():
    """Test creating store with custom max_messages."""
    store = InMemoryShortTermStore(max_messages=5)

    # Add 10 messages
    messages = [
        ShortTermMemory(
            message=UserMessage(content=f"Message {i}"),
            timestamp=datetime.now().isoformat(),
        )
        for i in range(10)
    ]
    store.add_messages("user1", messages)

    # Should only keep last 5
    all_messages = store.get_recent_messages("user1", limit=20)

    assert len(all_messages) == 5
    assert all_messages[0].message.content == "Message 5"
    assert all_messages[-1].message.content == "Message 9"
