"""
Unit tests for ContextService.

Tests message storage, retrieval with safe boundary trimming,
and session management.
"""

from datetime import datetime

import pytest
from casual_llm.messages import (
    AssistantMessage,
    AssistantToolCall,
    AssistantToolCallFunction,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)

from casual_memory.context_service import ContextService
from casual_memory.models import ShortTermMemory
from casual_memory.storage.short_term.memory import InMemoryShortTermStore


@pytest.fixture
def store():
    return InMemoryShortTermStore(max_messages=100)


@pytest.fixture
def service(store):
    return ContextService(short_term_store=store, short_term_limit=50)


def test_add_stores_messages(service):
    """add() wraps ChatMessages in ShortTermMemory and stores them."""
    messages = [UserMessage(content="hello"), AssistantMessage(content="hi")]
    result = service.add("user1", "sess1", messages)

    assert len(result) == 2
    assert result[0].message.role == "user"
    assert result[1].message.role == "assistant"
    assert result[0].timestamp  # has a timestamp


def test_add_filters_system_messages(service):
    """add() skips system messages."""
    messages = [
        SystemMessage(content="system prompt"),
        UserMessage(content="hello"),
    ]
    result = service.add("user1", "sess1", messages)

    assert len(result) == 1
    assert result[0].message.role == "user"


def test_add_empty_list(service):
    """add() with empty list returns empty and doesn't call store."""
    result = service.add("user1", "sess1", [])
    assert result == []


def test_add_all_system_messages(service):
    """add() with only system messages returns empty."""
    result = service.add("user1", "sess1", [SystemMessage(content="sys")])
    assert result == []


def test_get_returns_messages(service):
    """get() returns stored messages."""
    service.add(
        "user1",
        "sess1",
        [
            UserMessage(content="hello"),
            AssistantMessage(content="hi"),
        ],
    )
    result = service.get("user1", "sess1")

    assert len(result) == 2
    assert result[0].message.content == "hello"


def test_get_uses_default_limit(store):
    """get() uses short_term_limit from constructor."""
    svc = ContextService(short_term_store=store, short_term_limit=3)

    # Add 5 user messages directly to store
    memories = [
        ShortTermMemory(
            message=UserMessage(content=f"msg{i}"),
            timestamp=datetime.now().isoformat(),
        )
        for i in range(5)
    ]
    store.add_messages("user1:sess1", memories)

    result = svc.get("user1", "sess1")
    assert len(result) == 3
    assert result[0].message.content == "msg2"


def test_get_custom_limit(service):
    """get() accepts a custom limit override."""
    service.add("user1", "sess1", [UserMessage(content=f"msg{i}") for i in range(10)])
    result = service.get("user1", "sess1", limit=3)

    assert len(result) == 3
    assert result[0].message.content == "msg7"


def test_get_trims_tool_boundary(service):
    """get() searches buffer backward when window starts at tool_result."""
    service.add(
        "user1",
        "sess1",
        [
            UserMessage(content="First question"),
            AssistantMessage(
                content=None,
                tool_calls=[
                    AssistantToolCall(
                        id="call_1",
                        function=AssistantToolCallFunction(name="search", arguments="{}"),
                    )
                ],
            ),
            ToolResultMessage(name="search", tool_call_id="call_1", content="result"),
            UserMessage(content="Second question"),
            AssistantMessage(content="Answer to second"),
        ],
    )

    # limit=3 naive slice would be [tool_result, user, assistant]
    # buffer-first: finds user("First question") backward → returns all 5
    result = service.get("user1", "sess1", limit=3)

    assert result[0].message.role == "user"
    assert result[0].message.content == "First question"
    assert len(result) == 5


def test_get_no_trimming_when_already_safe(service):
    """get() returns full limit when window already starts at user."""
    service.add("user1", "sess1", [UserMessage(content=f"msg{i}") for i in range(10)])
    result = service.get("user1", "sess1", limit=5)

    assert len(result) == 5
    assert result[0].message.role == "user"


def test_get_empty_session(service):
    """get() returns empty list for non-existent session."""
    result = service.get("user1", "sess1")
    assert result == []


def test_clear_removes_messages(service):
    """clear() removes all messages for the session."""
    service.add("user1", "sess1", [UserMessage(content="hello")])
    count = service.clear("user1", "sess1")

    assert count == 1
    assert service.get("user1", "sess1") == []


def test_clear_empty_session(service):
    """clear() returns 0 for non-existent session."""
    count = service.clear("user1", "sess1")
    assert count == 0


def test_session_isolation(service):
    """Different sessions are isolated."""
    service.add("user1", "sess1", [UserMessage(content="session 1")])
    service.add("user1", "sess2", [UserMessage(content="session 2")])

    result1 = service.get("user1", "sess1")
    result2 = service.get("user1", "sess2")

    assert len(result1) == 1
    assert len(result2) == 1
    assert result1[0].message.content == "session 1"
    assert result2[0].message.content == "session 2"


def test_key_composition(service):
    """_compose_key creates user_id:session_id format."""
    assert service._compose_key("alice", "chat1") == "alice:chat1"
