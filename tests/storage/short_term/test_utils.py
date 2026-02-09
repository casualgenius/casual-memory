"""
Unit tests for short-term storage utility functions.

Tests the trim_to_safe_boundary function that ensures message windows
never start with orphaned tool results or assistant messages.
"""

from datetime import datetime

from casual_llm.messages import (
    AssistantMessage,
    AssistantToolCall,
    AssistantToolCallFunction,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)

from casual_memory.models import ShortTermMemory
from casual_memory.storage.short_term.utils import trim_to_safe_boundary


def _ts() -> str:
    return datetime.now().isoformat()


def _user(text="hi") -> ShortTermMemory:
    return ShortTermMemory(message=UserMessage(content=text), timestamp=_ts())


def _assistant(text="hello") -> ShortTermMemory:
    return ShortTermMemory(message=AssistantMessage(content=text), timestamp=_ts())


def _assistant_with_tool_calls() -> ShortTermMemory:
    return ShortTermMemory(
        message=AssistantMessage(
            content=None,
            tool_calls=[
                AssistantToolCall(
                    id="call_1",
                    function=AssistantToolCallFunction(name="get_weather", arguments="{}"),
                )
            ],
        ),
        timestamp=_ts(),
    )


def _tool_result(call_id="call_1") -> ShortTermMemory:
    return ShortTermMemory(
        message=ToolResultMessage(name="get_weather", tool_call_id=call_id, content="Sunny"),
        timestamp=_ts(),
    )


def _system(text="You are helpful") -> ShortTermMemory:
    return ShortTermMemory(message=SystemMessage(content=text), timestamp=_ts())


def test_empty_list():
    assert trim_to_safe_boundary([], target_limit=10) == []


def test_no_trimming_needed():
    messages = [_user("1"), _assistant("2"), _user("3")]
    result = trim_to_safe_boundary(messages, target_limit=10)
    assert len(result) == 3
    assert result[0].message.role == "user"


def test_trims_leading_tool_result():
    messages = [
        _tool_result(),
        _tool_result("call_2"),
        _user("question"),
        _assistant("answer"),
    ]
    result = trim_to_safe_boundary(messages, target_limit=4)
    assert len(result) == 2
    assert result[0].message.role == "user"
    assert result[0].message.content == "question"


def test_trims_leading_assistant_and_tool_results():
    messages = [
        _assistant_with_tool_calls(),
        _tool_result(),
        _user("follow up"),
        _assistant("response"),
    ]
    result = trim_to_safe_boundary(messages, target_limit=4)
    assert len(result) == 2
    assert result[0].message.role == "user"


def test_all_unsafe_returns_empty():
    messages = [
        _assistant_with_tool_calls(),
        _tool_result(),
        _assistant("no user message"),
    ]
    result = trim_to_safe_boundary(messages, target_limit=10)
    assert result == []


def test_system_message_is_safe_start():
    messages = [_system(), _user("hi"), _assistant("hello")]
    result = trim_to_safe_boundary(messages, target_limit=10)
    assert len(result) == 3
    assert result[0].message.role == "system"


def test_caps_to_target_limit():
    messages = [_user(f"msg{i}") for i in range(15)]
    result = trim_to_safe_boundary(messages, target_limit=5)
    assert len(result) == 5
    assert result[0].message.content == "msg10"


def test_single_user_message():
    result = trim_to_safe_boundary([_user("only")], target_limit=10)
    assert len(result) == 1
    assert result[0].message.content == "only"


def test_overfetch_buffer_used():
    """When over-fetched, the buffer helps find a safe boundary."""
    # Simulate 60 messages fetched (50 target + 10 buffer)
    # Buffer zone: messages[0..9], target zone: messages[10..59]
    messages = (
        [_user("buffer_start")]  # index 0 - in buffer
        + [_assistant(f"buf{i}") for i in range(9)]  # fill buffer
        + [_tool_result()]  # index 10 - would be naive start for limit=50
        + [_user("real_start")]  # index 11 - safe boundary
        + [_assistant(f"msg{i}") for i in range(48)]
    )
    result = trim_to_safe_boundary(messages, target_limit=50)
    assert result[0].message.role == "user"
    assert result[0].message.content == "real_start"
    assert len(result) == 49  # trimmed 1 tool_result from target zone


def test_real_scenario_tool_break():
    """Simulate the exact bug: limit window starts at orphaned tool result."""
    messages = [
        _user("What's the weather?"),
        _assistant_with_tool_calls(),
        _tool_result(),
        _user("Thanks! And tomorrow?"),
        _assistant_with_tool_calls(),
        _tool_result(),
        _user("Great, thanks!"),
        _assistant("You're welcome!"),
    ]
    # limit=6: naive slice gives messages[2:] starting at tool_result
    result = trim_to_safe_boundary(messages, target_limit=6)
    assert result[0].message.role == "user"
    assert result[0].message.content == "Thanks! And tomorrow?"
    assert len(result) == 5


def test_target_limit_larger_than_messages():
    messages = [_user("a"), _assistant("b")]
    result = trim_to_safe_boundary(messages, target_limit=100)
    assert len(result) == 2
    assert result[0].message.role == "user"


def test_trims_single_tool_result_at_start():
    messages = [
        _tool_result(),
        _user("hi"),
    ]
    result = trim_to_safe_boundary(messages, target_limit=2)
    assert len(result) == 1
    assert result[0].message.role == "user"
