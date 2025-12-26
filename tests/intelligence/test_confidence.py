"""Tests for confidence scoring."""

from datetime import datetime, timedelta

from casual_memory.intelligence.confidence import (
    MEMORY_HIGH_CONFIDENCE_MENTIONS,
    MEMORY_MAX_CONFIDENCE,
    MEMORY_MIN_CONFIDENCE,
    calculate_confidence,
    calculate_days_since,
    calculate_days_span,
)


def test_confidence_single_mention():
    """Test confidence for single mention."""
    confidence = calculate_confidence(mention_count=1)
    assert confidence == MEMORY_MIN_CONFIDENCE  # 0.5


def test_confidence_two_mentions():
    """Test confidence for two mentions."""
    confidence = calculate_confidence(mention_count=2)
    assert confidence == 0.7


def test_confidence_three_mentions():
    """Test confidence for three mentions."""
    confidence = calculate_confidence(mention_count=3)
    assert confidence == 0.8


def test_confidence_four_mentions():
    """Test confidence for four mentions."""
    confidence = calculate_confidence(mention_count=4)
    # Linear interpolation: 0.5 + (4 * 0.1) = 0.9
    assert confidence == 0.9


def test_confidence_high_mentions():
    """Test confidence for high mention count."""
    confidence = calculate_confidence(mention_count=MEMORY_HIGH_CONFIDENCE_MENTIONS)
    assert confidence == MEMORY_MAX_CONFIDENCE  # 0.95

    # Should cap at max even with more mentions
    confidence = calculate_confidence(mention_count=10)
    assert confidence == MEMORY_MAX_CONFIDENCE


def test_confidence_zero_mentions():
    """Test edge case: zero mentions."""
    confidence = calculate_confidence(mention_count=0)
    # Should return minimum confidence for safety
    assert confidence >= 0.0


def test_confidence_recency_factor():
    """Test recency factor reduces confidence for old memories."""
    # Fresh memory (no recency penalty)
    fresh = calculate_confidence(mention_count=3, days_since_last=0)

    # Old memory (recency penalty after 30 days)
    old = calculate_confidence(mention_count=3, days_since_last=31)

    # Old should be slightly lower due to recency decay
    assert old < fresh
    assert old == fresh * 0.95  # 5% decay after 30 days


def test_confidence_no_recency_penalty_within_30_days():
    """Test that recency penalty only applies after 30 days."""
    base = calculate_confidence(mention_count=3, days_since_last=0)
    recent = calculate_confidence(mention_count=3, days_since_last=15)

    # No penalty within 30 days
    assert recent == base


def test_confidence_spread_factor():
    """Test spread factor boosts confidence for mentions over time."""
    # Concentrated mentions (no spread boost)
    concentrated = calculate_confidence(mention_count=3, days_span=0)

    # Spread out mentions (spread boost)
    spread = calculate_confidence(mention_count=3, days_span=90)

    # Spread should be slightly higher
    assert spread > concentrated


def test_confidence_spread_factor_max_boost():
    """Test that spread factor caps at 10% boost."""
    # Maximum spread (90 days): 1.0 + (90/900) = 1.1 (10% boost)
    max_spread = calculate_confidence(mention_count=3, days_span=90)

    # Excessive spread should not boost beyond 10%
    excessive_spread = calculate_confidence(mention_count=3, days_span=900)

    # Both should hit the cap
    assert max_spread > 0.8  # Base 0.8 * 1.1 = 0.88
    assert excessive_spread <= MEMORY_MAX_CONFIDENCE  # Capped at 0.95


def test_confidence_combined_factors():
    """Test confidence calculation with all factors combined."""
    # High mentions, spread over time, but old
    confidence = calculate_confidence(mention_count=5, days_span=90, days_since_last=31)

    # Should have:
    # - Base: 0.95 (high mentions)
    # - Recency: 0.95 (old memory)
    # - Spread: 1.1 (max boost)
    # - Final: min(0.95, 0.95 * 0.95 * 1.1) = min(0.95, 0.99275) = 0.95
    assert confidence == MEMORY_MAX_CONFIDENCE


def test_confidence_never_exceeds_max():
    """Test that confidence never exceeds maximum."""
    # Try various combinations that might push above max
    confidence1 = calculate_confidence(mention_count=100, days_span=1000)
    confidence2 = calculate_confidence(mention_count=10, days_span=90)

    assert confidence1 <= MEMORY_MAX_CONFIDENCE
    assert confidence2 <= MEMORY_MAX_CONFIDENCE


def test_days_span_calculation():
    """Test calculation of days between timestamps."""
    now = datetime.now()
    first = (now - timedelta(days=10)).isoformat()
    last = now.isoformat()

    days = calculate_days_span(first, last)
    assert days == 10


def test_days_span_same_day():
    """Test days span when both timestamps are the same day."""
    now = datetime.now()
    timestamp = now.isoformat()

    days = calculate_days_span(timestamp, timestamp)
    assert days == 0


def test_days_span_reverse_order():
    """Test days span with reverse order (negative delta)."""
    now = datetime.now()
    future = (now + timedelta(days=5)).isoformat()
    past = now.isoformat()

    # Should return 0 for negative delta (max(0, delta.days))
    days = calculate_days_span(future, past)
    assert days == 0


def test_days_span_invalid_format():
    """Test days span with invalid timestamp format."""
    days = calculate_days_span("invalid", "also invalid")
    assert days == 0


def test_days_span_none_values():
    """Test days span with None values."""
    now = datetime.now().isoformat()

    days = calculate_days_span(None, now)
    assert days == 0

    days = calculate_days_span(now, None)
    assert days == 0


def test_days_since_calculation():
    """Test calculation of days since last mention."""
    past = (datetime.now() - timedelta(days=15)).isoformat()

    days = calculate_days_since(past)
    assert days == 15


def test_days_since_today():
    """Test days since for today's timestamp."""
    now = datetime.now().isoformat()

    days = calculate_days_since(now)
    assert days == 0


def test_days_since_future():
    """Test days since for future timestamp."""
    future = (datetime.now() + timedelta(days=5)).isoformat()

    # Should return 0 for future timestamps
    days = calculate_days_since(future)
    assert days == 0


def test_days_since_invalid_format():
    """Test days since with invalid timestamp format."""
    days = calculate_days_since("invalid")
    assert days == 0


def test_days_since_none():
    """Test days since with None value."""
    days = calculate_days_since(None)
    assert days == 0


def test_confidence_realistic_scenarios():
    """Test confidence calculation for realistic scenarios."""
    now = datetime.now()

    # Scenario 1: User just mentioned something once
    confidence = calculate_confidence(mention_count=1)
    assert confidence == 0.5  # Low confidence

    # Scenario 2: User mentioned it twice over a week
    first = (now - timedelta(days=7)).isoformat()
    last = now.isoformat()
    days_span = calculate_days_span(first, last)
    confidence = calculate_confidence(mention_count=2, days_span=days_span)
    assert confidence > 0.7  # Slight boost from spread

    # Scenario 3: Well-established fact (5 mentions over 3 months)
    first = (now - timedelta(days=90)).isoformat()
    last = (now - timedelta(days=5)).isoformat()
    days_span = calculate_days_span(first, last)
    days_since_last = calculate_days_since(last)
    confidence = calculate_confidence(
        mention_count=5, days_span=days_span, days_since_last=days_since_last
    )
    assert confidence >= 0.90  # High confidence, recent, spread out

    # Scenario 4: Old memory that hasn't been mentioned in a while
    first = (now - timedelta(days=180)).isoformat()
    last = (now - timedelta(days=60)).isoformat()
    days_since_last = calculate_days_since(last)
    confidence = calculate_confidence(mention_count=3, days_since_last=days_since_last)
    assert confidence < 0.8  # Reduced by recency decay
