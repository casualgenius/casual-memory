"""
Confidence calculation for memory intelligence.

Implements the confidence scoring system based on mention frequency,
recency, and temporal spread.
"""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Configuration constants (previously from app.config)
MEMORY_MIN_CONFIDENCE = 0.5
MEMORY_MAX_CONFIDENCE = 0.95
MEMORY_HIGH_CONFIDENCE_MENTIONS = 5


def calculate_confidence(mention_count: int, days_span: int = 0, days_since_last: int = 0) -> float:
    """
    Calculate confidence score based on how often a memory is confirmed.

    Formula based on mention frequency with diminishing returns, plus
    optional recency and spread factors.

    Args:
        mention_count: Number of times this memory has been mentioned
        days_span: Number of days between first and last mention (for spread calculation)
        days_since_last: Number of days since last mention (for recency calculation)

    Returns:
        Confidence score between 0.0 and 1.0
    """
    # Base confidence from mention frequency
    if mention_count == 1:
        base = MEMORY_MIN_CONFIDENCE  # 0.5 - Low confidence, single mention
    elif mention_count == 2:
        base = 0.7  # Medium-low
    elif mention_count == 3:
        base = 0.8  # Medium
    elif mention_count >= MEMORY_HIGH_CONFIDENCE_MENTIONS:  # Default: 5
        base = MEMORY_MAX_CONFIDENCE  # 0.95 - High confidence, well-established
    else:
        # Linear interpolation for 4 mentions
        base = min(MEMORY_MAX_CONFIDENCE, 0.5 + (mention_count * 0.1))

    # Optional: Apply recency factor (reduce confidence if not mentioned recently)
    recency_factor = 1.0
    if days_since_last > 30:
        recency_factor = 0.95
        logger.debug(f"Applying recency decay: {days_since_last} days since last mention")

    # Optional: Apply spread factor (boost if mentions spread over time)
    spread_factor = 1.0
    if mention_count > 1 and days_span > 0:
        # Max 10% boost over 3 months (90 days)
        spread_factor = min(1.1, 1.0 + (days_span / 900))
        logger.debug(f"Applying spread boost: {days_span} days between mentions")

    final_confidence = min(MEMORY_MAX_CONFIDENCE, base * recency_factor * spread_factor)

    logger.debug(
        f"Confidence calculation: mentions={mention_count}, "
        f"base={base:.2f}, recency={recency_factor:.2f}, "
        f"spread={spread_factor:.2f}, final={final_confidence:.2f}"
    )

    return final_confidence


def calculate_days_span(first_seen: str, last_seen: str) -> int:
    """
    Calculate the number of days between first and last mention.

    Args:
        first_seen: ISO format timestamp of first mention
        last_seen: ISO format timestamp of last mention

    Returns:
        Number of days (0 if same day or invalid)
    """
    try:
        first = datetime.fromisoformat(first_seen)
        last = datetime.fromisoformat(last_seen)
        delta = last - first
        return max(0, delta.days)
    except (ValueError, TypeError):
        return 0


def calculate_days_since(last_seen: str) -> int:
    """
    Calculate the number of days since last mention.

    Args:
        last_seen: ISO format timestamp of last mention

    Returns:
        Number of days (0 if today or invalid)
    """
    try:
        last = datetime.fromisoformat(last_seen)
        now = datetime.now()
        delta = now - last
        return max(0, delta.days)
    except (ValueError, TypeError):
        return 0
