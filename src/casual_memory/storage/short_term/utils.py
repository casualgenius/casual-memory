"""
Utility functions for short-term memory stores.

Provides message boundary trimming to ensure tool-call sequences
are never split when fetching recent messages.
"""

from casual_memory.models import ShortTermMemory

_SAFE_START_ROLES = {"user", "system"}

_OVERFETCH_BUFFER = 10


def trim_to_safe_boundary(
    messages: list[ShortTermMemory],
    target_limit: int,
) -> list[ShortTermMemory]:
    """
    Trim a list of messages so the first message has role "user" or "system".

    When fetching the last N messages from a conversation, the cut boundary
    may land in the middle of a tool-call sequence (e.g. starting at a
    tool_result without the preceding assistant tool_call). This function
    ensures the returned window starts at a safe boundary.

    Args:
        messages: The fetched messages (ordered oldest-first), typically
            over-fetched by a small buffer beyond target_limit.
        target_limit: The desired number of messages to return.

    Returns:
        A list where the first message has role "user" or "system".
        May return fewer than target_limit messages. Returns an empty
        list if no safe boundary is found.
    """
    if not messages:
        return []

    ideal_start = max(0, len(messages) - target_limit)

    if messages[ideal_start].message.role in _SAFE_START_ROLES:
        return messages[ideal_start:]

    # Search forward for the first safe start
    for i in range(ideal_start + 1, len(messages)):
        if messages[i].message.role in _SAFE_START_ROLES:
            return messages[i:]

    return []
