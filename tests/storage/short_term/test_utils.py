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
    """No buffer available (ideal_start=0), so trims forward."""
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
    """No buffer available (ideal_start=0), so trims forward."""
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


def test_system_message_is_not_safe_start():
    """System messages are filtered by add(), but if one were present it is not a safe start."""
    messages = [_system(), _user("hi"), _assistant("hello")]
    result = trim_to_safe_boundary(messages, target_limit=10)
    assert len(result) == 2
    assert result[0].message.role == "user"


def test_caps_to_target_limit():
    messages = [_user(f"msg{i}") for i in range(15)]
    result = trim_to_safe_boundary(messages, target_limit=5)
    assert len(result) == 5
    assert result[0].message.content == "msg10"


def test_single_user_message():
    result = trim_to_safe_boundary([_user("only")], target_limit=10)
    assert len(result) == 1
    assert result[0].message.content == "only"


def test_buffer_search_prefers_backward():
    """When ideal start is unsafe, search backward into buffer first."""
    # Buffer: [user, assistant, assistant]  Target: [tool_result, user, assistant, ...]
    messages = (
        [_user("buffer_user")]
        + [_assistant(f"buf{i}") for i in range(2)]
        + [_tool_result()]  # index 3 = ideal start for limit=4
        + [_user("target_user")]
        + [_assistant("a1")]
        + [_assistant("a2")]
    )
    # ideal_start = 7-4 = 3 (tool_result)
    # backward: index 2 (assistant), 1 (assistant), 0 (user) → use index 0
    result = trim_to_safe_boundary(messages, target_limit=4)
    assert result[0].message.role == "user"
    assert result[0].message.content == "buffer_user"
    assert len(result) == 7  # all messages from buffer_user onward


def test_buffer_finds_nearest_user():
    """Backward scan finds the user closest to ideal_start in the buffer."""
    messages = [
        _user("far_user"),       # index 0
        _assistant("a1"),        # index 1
        _user("near_user"),      # index 2
        _assistant("a2"),        # index 3
        _tool_result(),          # index 4 = ideal start for limit=3
        _user("target_user"),    # index 5
        _assistant("a3"),        # index 6
    ]
    # ideal_start = 7-3 = 4 (tool_result)
    # backward: index 3 (assistant), 2 (user "near_user") → use index 2
    result = trim_to_safe_boundary(messages, target_limit=3)
    assert result[0].message.content == "near_user"
    assert len(result) == 5  # messages[2:]


def test_forward_trim_when_no_buffer_user():
    """Falls back to forward trim when buffer has no user messages."""
    messages = [
        _assistant("buf1"),      # index 0 (buffer)
        _tool_result(),          # index 1 (buffer)
        _assistant("a1"),        # index 2 = ideal start for limit=3
        _user("forward_user"),   # index 3
        _assistant("a2"),        # index 4
    ]
    # ideal_start = 5-3 = 2 (assistant)
    # backward: index 1 (tool_result), 0 (assistant) → no user in buffer
    # forward: index 3 (user "forward_user") → use index 3
    result = trim_to_safe_boundary(messages, target_limit=3)
    assert result[0].message.content == "forward_user"
    assert len(result) == 2  # fewer than target


def test_overfetch_buffer_used():
    """When over-fetched, the buffer helps find a safe boundary."""
    # Simulate 60 messages fetched (50 target + 10 buffer)
    # Buffer zone: messages[0..9], target zone: messages[10..59]
    messages = (
        [_user("buffer_start")]  # index 0 - in buffer
        + [_assistant(f"buf{i}") for i in range(9)]  # fill buffer
        + [_tool_result()]  # index 10 - would be naive start for limit=50
        + [_user("real_start")]  # index 11 - forward boundary
        + [_assistant(f"msg{i}") for i in range(48)]
    )
    # ideal_start = 60-50 = 10 (tool_result)
    # backward: index 9..1 (assistants), 0 (user "buffer_start") → use index 0
    result = trim_to_safe_boundary(messages, target_limit=50)
    assert result[0].message.role == "user"
    assert result[0].message.content == "buffer_start"
    assert len(result) == 60  # all messages returned (slightly more than target)


def test_real_scenario_tool_break():
    """Simulate the exact bug: limit window starts at orphaned tool result."""
    messages = [
        _user("What's the weather?"),       # index 0
        _assistant_with_tool_calls(),        # index 1
        _tool_result(),                      # index 2 = ideal start for limit=6
        _user("Thanks! And tomorrow?"),      # index 3
        _assistant_with_tool_calls(),        # index 4
        _tool_result(),                      # index 5
        _user("Great, thanks!"),             # index 6
        _assistant("You're welcome!"),       # index 7
    ]
    # ideal_start = 8-6 = 2 (tool_result)
    # backward: index 1 (assistant), 0 (user) → use index 0
    result = trim_to_safe_boundary(messages, target_limit=6)
    assert result[0].message.role == "user"
    assert result[0].message.content == "What's the weather?"
    assert len(result) == 8  # all messages returned (buffer used)


def test_target_limit_larger_than_messages():
    messages = [_user("a"), _assistant("b")]
    result = trim_to_safe_boundary(messages, target_limit=100)
    assert len(result) == 2
    assert result[0].message.role == "user"


def test_trims_single_tool_result_at_start():
    """No buffer (ideal_start=0), trims forward."""
    messages = [
        _tool_result(),
        _user("hi"),
    ]
    result = trim_to_safe_boundary(messages, target_limit=2)
    assert len(result) == 1
    assert result[0].message.role == "user"
