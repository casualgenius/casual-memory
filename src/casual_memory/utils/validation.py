"""Validation utilities for memory identifiers (namespace, entity_id)."""

import re

# Pattern: lowercase alphanumeric start, then lowercase alphanumeric, hyphens, underscores.
# Length 1-100 characters. No double underscores (to prevent old __agent:X__ format).
_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,99}$")
_DOUBLE_UNDERSCORE_PATTERN = re.compile(r"__")


def validate_identifier(value: str, field_name: str) -> str:
    """Validate a namespace or entity_id identifier.

    Rules:
    - Must be 1-100 characters long
    - Must start with a lowercase letter or digit
    - Only lowercase alphanumeric characters, hyphens, and underscores allowed
    - No double underscores (prevents old __agent:X__ format)

    Args:
        value: The identifier string to validate.
        field_name: Name of the field being validated (for error messages).

    Returns:
        The validated identifier string (unchanged).

    Raises:
        ValueError: If the identifier does not match the required format.
    """
    if not value:
        raise ValueError(
            f"Invalid {field_name}: must not be empty. "
            f"Expected lowercase alphanumeric with hyphens/underscores, 1-100 chars."
        )

    if _DOUBLE_UNDERSCORE_PATTERN.search(value):
        raise ValueError(
            f"Invalid {field_name}: '{value}' contains double underscores. "
            f"Double underscores are not allowed (reserved format)."
        )

    if not _IDENTIFIER_PATTERN.match(value):
        if len(value) > 100:
            raise ValueError(
                f"Invalid {field_name}: '{value[:20]}...' is too long "
                f"({len(value)} chars, max 100)."
            )
        raise ValueError(
            f"Invalid {field_name}: '{value}' does not match required format. "
            f"Must be 1-100 chars, start with lowercase letter or digit, "
            f"and contain only lowercase alphanumeric, hyphens, or underscores."
        )

    return value
