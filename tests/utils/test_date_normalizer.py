"""
Tests for date normalization utilities.

Tests extract_and_normalize_date, calculate_valid_until, and normalize_memory_dates.
"""

from datetime import datetime

from casual_memory.utils.date_normalizer import (
    WEEKDAY_MAP,
    calculate_valid_until,
    extract_and_normalize_date,
    get_next_weekday,
    normalize_memory_dates,
)


class TestGetNextWeekday:
    """Tests for get_next_weekday helper function."""

    def test_next_weekday_basic(self):
        """Test getting next occurrence of a weekday."""
        # Monday 2024-01-01
        current = datetime(2024, 1, 1, 10, 0)

        # Next Friday (5 days ahead)
        result = get_next_weekday(current, WEEKDAY_MAP["friday"])
        assert result.date() == datetime(2024, 1, 5).date()

    def test_next_weekday_same_day(self):
        """Test getting next occurrence when it's the same weekday."""
        # Monday 2024-01-01
        current = datetime(2024, 1, 1, 10, 0)

        # Next Monday (7 days ahead, not today)
        result = get_next_weekday(current, WEEKDAY_MAP["monday"])
        assert result.date() == datetime(2024, 1, 8).date()

    def test_next_weekday_with_min_days(self):
        """Test min_days_ahead parameter."""
        # Monday 2024-01-01
        current = datetime(2024, 1, 1, 10, 0)

        # Next Tuesday with min 2 days
        result = get_next_weekday(current, WEEKDAY_MAP["tuesday"], min_days_ahead=2)
        assert result.date() == datetime(2024, 1, 9).date()  # Week later


class TestExtractAndNormalizeDate:
    """Tests for extract_and_normalize_date function."""

    def test_tomorrow_basic(self):
        """Test 'tomorrow' normalization."""
        reference = datetime(2024, 1, 1, 10, 0)
        text = "Remind me to call John tomorrow"

        normalized, abs_date = extract_and_normalize_date(text, reference)

        assert "on January 02" in normalized
        assert abs_date.date() == datetime(2024, 1, 2).date()

    def test_tomorrow_with_time_of_day(self):
        """Test 'tomorrow morning/afternoon/evening'."""
        reference = datetime(2024, 1, 1, 10, 0)

        for time_of_day in ["morning", "afternoon", "evening", "night"]:
            text = f"Meeting tomorrow {time_of_day}"
            normalized, abs_date = extract_and_normalize_date(text, reference)

            assert "on January 02" in normalized
            assert abs_date.date() == datetime(2024, 1, 2).date()

    def test_in_x_days(self):
        """Test 'in X days' normalization."""
        reference = datetime(2024, 1, 1, 10, 0)

        text = "Appointment in 5 days"
        normalized, abs_date = extract_and_normalize_date(text, reference)

        assert "on January 06" in normalized
        assert abs_date.date() == datetime(2024, 1, 6).date()

    def test_in_x_day_singular(self):
        """Test 'in X day' (singular) normalization."""
        reference = datetime(2024, 1, 1, 10, 0)

        text = "Call back in 1 day"
        normalized, abs_date = extract_and_normalize_date(text, reference)

        assert "on January 02" in normalized
        assert abs_date.date() == datetime(2024, 1, 2).date()

    def test_next_weekday(self):
        """Test 'next Monday/Tuesday/etc' normalization."""
        # Monday 2024-01-01
        reference = datetime(2024, 1, 1, 10, 0)

        text = "Meeting next Friday"
        normalized, abs_date = extract_and_normalize_date(text, reference)

        assert "on January 05" in normalized
        assert abs_date.date() == datetime(2024, 1, 5).date()

    def test_standalone_weekday(self):
        """Test standalone weekday (e.g., 'on Friday', 'Friday morning')."""
        # Monday 2024-01-01
        reference = datetime(2024, 1, 1, 10, 0)

        text = "Dentist appointment on Friday"
        normalized, abs_date = extract_and_normalize_date(text, reference)

        assert "on January 05" in normalized
        assert abs_date.date() == datetime(2024, 1, 5).date()

    def test_standalone_weekday_with_time(self):
        """Test weekday with time of day."""
        # Monday 2024-01-01
        reference = datetime(2024, 1, 1, 10, 0)

        text = "Meeting Friday morning"
        normalized, abs_date = extract_and_normalize_date(text, reference)

        assert "on January 05" in normalized
        assert abs_date.date() == datetime(2024, 1, 5).date()

    def test_no_date_reference(self):
        """Test text with no date reference."""
        reference = datetime(2024, 1, 1, 10, 0)

        text = "I like pizza"
        normalized, abs_date = extract_and_normalize_date(text, reference)

        assert normalized == text  # Unchanged
        assert abs_date is None

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        reference = datetime(2024, 1, 1, 10, 0)

        text = "Call TOMORROW"
        normalized, abs_date = extract_and_normalize_date(text, reference)

        assert "on January 02" in normalized
        assert abs_date is not None

    def test_only_first_match_processed(self):
        """Test that only the first date reference is normalized."""
        reference = datetime(2024, 1, 1, 10, 0)

        text = "Call tomorrow and next Friday"
        normalized, abs_date = extract_and_normalize_date(text, reference)

        # Only "tomorrow" should be replaced
        assert "on January 02" in normalized
        assert "next Friday" in normalized  # Not replaced
        assert abs_date.date() == datetime(2024, 1, 2).date()


class TestCalculateValidUntil:
    """Tests for calculate_valid_until function."""

    def test_fact_no_expiry(self):
        """Test that facts don't expire."""
        reference = datetime(2024, 1, 1, 10, 0)
        future_date = datetime(2024, 1, 5, 10, 0)

        result = calculate_valid_until("fact", future_date, reference)

        assert result is None

    def test_preference_no_expiry(self):
        """Test that preferences don't expire."""
        reference = datetime(2024, 1, 1, 10, 0)
        future_date = datetime(2024, 1, 5, 10, 0)

        result = calculate_valid_until("preference", future_date, reference)

        assert result is None

    def test_event_no_date_no_expiry(self):
        """Test event with no date doesn't expire."""
        reference = datetime(2024, 1, 1, 10, 0)

        result = calculate_valid_until("event", None, reference)

        assert result is None

    def test_event_past_date_no_expiry(self):
        """Test event in the past doesn't expire (permanent memory)."""
        reference = datetime(2024, 1, 5, 10, 0)
        past_date = datetime(2024, 1, 1, 10, 0)

        result = calculate_valid_until("event", past_date, reference)

        assert result is None

    def test_event_today_no_expiry(self):
        """Test event today doesn't expire (becomes permanent)."""
        reference = datetime(2024, 1, 1, 10, 0)
        same_day = datetime(2024, 1, 1, 15, 0)

        result = calculate_valid_until("event", same_day, reference)

        assert result is None

    def test_event_future_date_expires(self):
        """Test future event expires at end of that day."""
        reference = datetime(2024, 1, 1, 10, 0)
        future_date = datetime(2024, 1, 5, 10, 0)

        result = calculate_valid_until("event", future_date, reference)

        assert result is not None
        expiry = datetime.fromisoformat(result)
        assert expiry.date() == future_date.date()
        assert expiry.hour == 23
        assert expiry.minute == 59
        assert expiry.second == 59

    def test_goal_future_date_expires(self):
        """Test future goal expires at end of that day."""
        reference = datetime(2024, 1, 1, 10, 0)
        future_date = datetime(2024, 1, 5, 10, 0)

        result = calculate_valid_until("goal", future_date, reference)

        assert result is not None
        expiry = datetime.fromisoformat(result)
        assert expiry.date() == future_date.date()
        assert expiry.hour == 23
        assert expiry.minute == 59

    def test_goal_no_date_no_expiry(self):
        """Test goal with no specific date doesn't expire."""
        reference = datetime(2024, 1, 1, 10, 0)

        result = calculate_valid_until("goal", None, reference)

        assert result is None


class TestNormalizeMemoryDates:
    """Tests for normalize_memory_dates function."""

    def test_normalize_event_with_tomorrow(self):
        """Test normalizing event memory with 'tomorrow'."""
        reference = datetime(2024, 1, 1, 10, 0)
        memory_data = {
            "text": "Doctor appointment tomorrow",
            "type": "event",
            "importance": 0.8,
        }

        result = normalize_memory_dates(memory_data, reference)

        assert "on January 02" in result["text"]
        assert result["valid_until"] is not None
        expiry = datetime.fromisoformat(result["valid_until"])
        assert expiry.date() == datetime(2024, 1, 2).date()

    def test_normalize_goal_with_next_friday(self):
        """Test normalizing goal with 'next Friday'."""
        # Monday 2024-01-01
        reference = datetime(2024, 1, 1, 10, 0)
        memory_data = {
            "text": "Finish project next Friday",
            "type": "goal",
            "importance": 0.9,
        }

        result = normalize_memory_dates(memory_data, reference)

        assert "on January 05" in result["text"]
        assert result["valid_until"] is not None

    def test_normalize_fact_no_expiry(self):
        """Test normalizing fact doesn't add expiry."""
        reference = datetime(2024, 1, 1, 10, 0)
        memory_data = {
            "text": "Meeting tomorrow",
            "type": "fact",
            "importance": 0.5,
        }

        result = normalize_memory_dates(memory_data, reference)

        # Text normalized but no expiry
        assert "on January 02" in result["text"]
        assert "valid_until" not in result or result["valid_until"] is None

    def test_preserve_existing_valid_until(self):
        """Test that existing valid_until is preserved."""
        reference = datetime(2024, 1, 1, 10, 0)
        existing_expiry = "2024-12-31T23:59:59"
        memory_data = {
            "text": "Appointment tomorrow",
            "type": "event",
            "importance": 0.8,
            "valid_until": existing_expiry,
        }

        result = normalize_memory_dates(memory_data, reference)

        # Should preserve existing valid_until
        assert result["valid_until"] == existing_expiry

    def test_no_date_in_text(self):
        """Test memory with no date reference."""
        reference = datetime(2024, 1, 1, 10, 0)
        memory_data = {
            "text": "My name is Alice",
            "type": "fact",
            "importance": 0.7,
        }

        result = normalize_memory_dates(memory_data, reference)

        assert result["text"] == "My name is Alice"
        assert "valid_until" not in result or result["valid_until"] is None

    def test_preserve_other_fields(self):
        """Test that other memory fields are preserved."""
        reference = datetime(2024, 1, 1, 10, 0)
        memory_data = {
            "text": "Call Bob tomorrow",
            "type": "goal",
            "importance": 0.9,
            "tags": ["reminder", "phone"],
            "source": "user",
            "custom_field": "custom_value",
        }

        result = normalize_memory_dates(memory_data, reference)

        assert result["type"] == "goal"
        assert result["importance"] == 0.9
        assert result["tags"] == ["reminder", "phone"]
        assert result["source"] == "user"
        assert result["custom_field"] == "custom_value"


class TestDateNormalizationEdgeCases:
    """Tests for edge cases in date normalization."""

    def test_year_boundary(self):
        """Test date normalization crossing year boundary."""
        # December 31st
        reference = datetime(2024, 12, 31, 10, 0)

        text = "Party tomorrow"
        normalized, abs_date = extract_and_normalize_date(text, reference)

        assert "on January 01" in normalized
        assert abs_date.date() == datetime(2025, 1, 1).date()

    def test_month_boundary(self):
        """Test date normalization crossing month boundary."""
        # January 30th
        reference = datetime(2024, 1, 30, 10, 0)

        text = "Meeting in 5 days"
        normalized, abs_date = extract_and_normalize_date(text, reference)

        assert "on February 04" in normalized
        assert abs_date.date() == datetime(2024, 2, 4).date()

    def test_leap_year_february(self):
        """Test leap year handling."""
        # February 28, 2024 (leap year)
        reference = datetime(2024, 2, 28, 10, 0)

        text = "Event in 2 days"
        normalized, abs_date = extract_and_normalize_date(text, reference)

        assert "on March 01" in normalized
        assert abs_date.date() == datetime(2024, 3, 1).date()

    def test_empty_text(self):
        """Test with empty text."""
        reference = datetime(2024, 1, 1, 10, 0)

        text = ""
        normalized, abs_date = extract_and_normalize_date(text, reference)

        assert normalized == ""
        assert abs_date is None

    def test_text_with_only_whitespace(self):
        """Test with whitespace-only text."""
        reference = datetime(2024, 1, 1, 10, 0)

        text = "   "
        normalized, abs_date = extract_and_normalize_date(text, reference)

        assert normalized == "   "
        assert abs_date is None


class TestWeekdayMap:
    """Tests for WEEKDAY_MAP constant."""

    def test_weekday_map_completeness(self):
        """Test that all weekdays are in the map."""
        expected_days = [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ]
        assert list(WEEKDAY_MAP.keys()) == expected_days

    def test_weekday_map_values(self):
        """Test that weekday numbers are correct (0=Monday, 6=Sunday)."""
        assert WEEKDAY_MAP["monday"] == 0
        assert WEEKDAY_MAP["tuesday"] == 1
        assert WEEKDAY_MAP["wednesday"] == 2
        assert WEEKDAY_MAP["thursday"] == 3
        assert WEEKDAY_MAP["friday"] == 4
        assert WEEKDAY_MAP["saturday"] == 5
        assert WEEKDAY_MAP["sunday"] == 6
