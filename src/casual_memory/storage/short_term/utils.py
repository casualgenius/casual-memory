"""
Utility functions for short-term memory stores.

Provides message boundary trimming to ensure tool-call sequences
are never split when fetching recent messages.
"""

from casual_memory.models import ShortTermMemory

_SAFE_START_ROLES = {"user"}

OVERFETCH_BUFFER = 10


def trim_to_safe_boundary(
    messages: list[ShortTermMemory],
    target_limit: int,
) -> list[ShortTermMemory]:
    """
    Trim a list of messages so the first message has role "user".

    ContextService.add() filters out system messages before storage, so only
    user, assistant, and tool messages are persisted.  When fetching the last
    N messages, the cut boundary may land in the middle of a tool-call
    sequence (e.g. starting at a tool_result without the preceding assistant
    tool_call).  This function ensures the returned window starts at a user
    message — the only safe conversation boundary.

    The algorithm prefers returning slightly more messages over fewer:

    1. If the ideal start position is already a user message, return from there.
    2. Otherwise search backward into the over-fetch buffer for the nearest
       user message (returns between target_limit and target_limit + buffer).
    3. If none found in the buffer, search forward from the ideal start,
       trimming messages until a user message is found (returns < target_limit).
    4. If no user message exists at all, return an empty list.

    Args:
        messages: The fetched messages (ordered oldest-first), typically
            over-fetched by a small buffer beyond target_limit.
        target_limit: The desired number of messages to return.

    Returns:
        A list where the first message has role "user".
        May return more or fewer than target_limit messages depending on
        where the nearest user boundary falls.  Returns an empty list if
        no user message is found.
    """
    if not messages:
        return []

    ideal_start = max(0, len(messages) - target_limit)

    if messages[ideal_start].message.role in _SAFE_START_ROLES:
        return messages[ideal_start:]

    # Step 1: Search backward in buffer for nearest user message.
    # Gives slightly more than target_limit messages.
    for i in range(ideal_start - 1, -1, -1):
        if messages[i].message.role in _SAFE_START_ROLES:
            return messages[i:]

    # Step 2: Search forward, trimming messages from the front.
    # Gives fewer than target_limit messages.
    for i in range(ideal_start + 1, len(messages)):
        if messages[i].message.role in _SAFE_START_ROLES:
            return messages[i:]

    # Step 3: No user message found anywhere.
    return []
