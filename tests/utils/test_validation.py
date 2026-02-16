"""
Tests for identifier validation utilities.

Tests validate_identifier function and its integration with model validators.
"""

import pytest

from casual_memory.utils.validation import validate_identifier


class TestValidateIdentifier:
    """Tests for the validate_identifier utility function."""

    # --- Valid identifiers ---

    def test_simple_lowercase(self):
        assert validate_identifier("default", "namespace") == "default"

    def test_with_digits(self):
        assert validate_identifier("user123", "entity_id") == "user123"

    def test_starts_with_digit(self):
        assert validate_identifier("123abc", "entity_id") == "123abc"

    def test_with_hyphens(self):
        assert validate_identifier("my-namespace", "namespace") == "my-namespace"

    def test_with_underscores(self):
        assert validate_identifier("my_namespace", "namespace") == "my_namespace"

    def test_mixed_valid_chars(self):
        assert validate_identifier("a1-b2_c3", "entity_id") == "a1-b2_c3"

    def test_single_char(self):
        assert validate_identifier("a", "namespace") == "a"

    def test_single_digit(self):
        assert validate_identifier("0", "namespace") == "0"

    def test_100_chars(self):
        """Exactly 100 characters should be valid."""
        value = "a" * 100
        assert validate_identifier(value, "namespace") == value

    # --- Invalid identifiers ---

    def test_empty_string(self):
        with pytest.raises(ValueError, match="must not be empty"):
            validate_identifier("", "namespace")

    def test_too_long(self):
        """101 characters should be rejected."""
        value = "a" * 101
        with pytest.raises(ValueError, match="too long"):
            validate_identifier(value, "namespace")

    def test_uppercase(self):
        with pytest.raises(ValueError, match="does not match required format"):
            validate_identifier("MyNamespace", "namespace")

    def test_spaces(self):
        with pytest.raises(ValueError, match="does not match required format"):
            validate_identifier("my namespace", "namespace")

    def test_colons(self):
        with pytest.raises(ValueError, match="does not match required format"):
            validate_identifier("my:namespace", "namespace")

    def test_double_underscores(self):
        with pytest.raises(ValueError, match="double underscores"):
            validate_identifier("my__namespace", "namespace")

    def test_old_agent_format(self):
        """The old __agent:X__ format should be rejected."""
        with pytest.raises(ValueError, match="double underscores"):
            validate_identifier("__agent:test__", "entity_id")

    def test_starts_with_hyphen(self):
        with pytest.raises(ValueError, match="does not match required format"):
            validate_identifier("-start", "namespace")

    def test_starts_with_underscore(self):
        with pytest.raises(ValueError, match="does not match required format"):
            validate_identifier("_start", "namespace")

    def test_special_characters(self):
        with pytest.raises(ValueError, match="does not match required format"):
            validate_identifier("test@value", "entity_id")

    def test_dots(self):
        with pytest.raises(ValueError, match="does not match required format"):
            validate_identifier("test.value", "entity_id")

    def test_field_name_in_error_message(self):
        """Error messages should include the field name."""
        with pytest.raises(ValueError, match="entity_id"):
            validate_identifier("BAD", "entity_id")

        with pytest.raises(ValueError, match="namespace"):
            validate_identifier("BAD", "namespace")
