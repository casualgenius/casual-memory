"""
Date normalization utilities for memory extraction.

This module handles converting relative date references (tomorrow, Friday, in 2 days, etc.)
to absolute dates in memory text and calculates appropriate valid_until timestamps.
"""

import re
from datetime import datetime, timedelta
from typing import Optional
import dateparser
import logging

logger = logging.getLogger(__name__)


# Map weekday names to numbers (Monday=0, Sunday=6)
WEEKDAY_MAP = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def get_next_weekday(current_date: datetime, target_weekday: int, min_days_ahead: int = 1) -> datetime:
    """
    Get the next occurrence of a target weekday.

    Args:
        current_date: The reference date
        target_weekday: Target day (0=Monday, 6=Sunday)
        min_days_ahead: Minimum days in the future (1 = at least tomorrow)

    Returns:
        datetime object for the next occurrence of that weekday
    """
    days_ahead = (target_weekday - current_date.weekday()) % 7
    if days_ahead < min_days_ahead:
        days_ahead += 7
    return current_date + timedelta(days=days_ahead)


def extract_and_normalize_date(text: str, reference_date: datetime) -> tuple[str, Optional[datetime]]:
    """
    Extract relative date references from text and normalize to absolute dates.

    Args:
        text: Memory text potentially containing relative dates
        reference_date: The reference date (usually "now")

    Returns:
        Tuple of (normalized_text, absolute_date)
        - normalized_text: Text with relative dates replaced by absolute dates
        - absolute_date: The parsed absolute date, or None if no date found
    """
    original_text = text
    absolute_date = None

    # Patterns to detect and normalize (in priority order)
    patterns = [
        # "tomorrow" or "tomorrow morning/afternoon/evening"
        (
            r'\btomorrow(?:\s+(?:morning|afternoon|evening|night))?\b',
            lambda m: reference_date + timedelta(days=1),
            lambda m, d: f"on {d.strftime('%B %d')}"
        ),
        # "in X days" or "in X day"
        (
            r'\bin\s+(\d+)\s+days?\b',
            lambda m: reference_date + timedelta(days=int(m.group(1))),
            lambda m, d: f"on {d.strftime('%B %d')}"
        ),
        # "next Monday", "next Tuesday", etc.
        (
            r'\bnext\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            lambda m: get_next_weekday(reference_date, WEEKDAY_MAP[m.group(1).lower()]),
            lambda m, d: f"on {d.strftime('%B %d')}"
        ),
        # Standalone weekday: "on Friday", "Friday morning", etc.
        (
            r'\b(?:on\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)(?:\s+(?:morning|afternoon|evening|at))?\b',
            lambda m: get_next_weekday(reference_date, WEEKDAY_MAP[m.group(1).lower()]),
            lambda m, d: f"on {d.strftime('%B %d')}"
        ),
    ]

    for pattern, date_calc, replacement_func in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                # Calculate the absolute date
                calculated_date = date_calc(match)

                # Only use the first match we find
                if absolute_date is None:
                    absolute_date = calculated_date

                    # Replace the relative reference with absolute date
                    replacement = replacement_func(match, calculated_date)
                    text = text[:match.start()] + replacement + text[match.end():]

                    logger.debug(
                        f"Normalized date: '{match.group(0)}' -> '{replacement}' "
                        f"({calculated_date.strftime('%Y-%m-%d')})"
                    )
                    break

            except Exception as e:
                logger.warning(f"Failed to normalize date pattern '{match.group(0)}': {e}")
                continue

    if text != original_text:
        logger.info(f"Date normalized: '{original_text}' -> '{text}'")

    return text, absolute_date


def calculate_valid_until(
    memory_type: str,
    absolute_date: Optional[datetime],
    reference_date: datetime
) -> Optional[str]:
    """
    Calculate the valid_until timestamp for a memory.

    Args:
        memory_type: Type of memory (fact, preference, goal, event)
        absolute_date: Absolute date extracted from the memory text
        reference_date: Current date/time

    Returns:
        ISO8601 timestamp string for valid_until, or None if memory doesn't expire
    """
    # Only temporal memories (events, goals) with future dates should have expiry
    if memory_type not in ["event", "goal"]:
        return None

    if absolute_date is None:
        # No specific date mentioned (e.g., "someday")
        return None

    if absolute_date.date() <= reference_date.date():
        # Past or today's event - permanent memory
        return None

    # Future event - expires at end of that day
    expiry = absolute_date.replace(hour=23, minute=59, second=59, microsecond=0)
    return expiry.isoformat()


def normalize_memory_dates(memory_data: dict, reference_date: datetime) -> dict:
    """
    Normalize dates in a single memory dictionary.

    Args:
        memory_data: Memory dictionary from LLM response
        reference_date: Current date/time

    Returns:
        Updated memory dictionary with normalized text and valid_until
    """
    text = memory_data.get("text", "")
    memory_type = memory_data.get("type", "fact")

    # Normalize the text and extract absolute date
    normalized_text, absolute_date = extract_and_normalize_date(text, reference_date)

    # Update the text
    memory_data["text"] = normalized_text

    # Calculate valid_until if not already set by LLM
    if not memory_data.get("valid_until"):
        valid_until = calculate_valid_until(memory_type, absolute_date, reference_date)
        if valid_until:
            memory_data["valid_until"] = valid_until

    return memory_data
