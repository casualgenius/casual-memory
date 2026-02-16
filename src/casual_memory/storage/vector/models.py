"""
Models for vector storage.

Defines the data structures used by vector storage implementations
for storing memory points with embeddings and payloads.
"""

import warnings
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class MemoryPointPayload(BaseModel):
    """
    Payload for a memory point in vector storage.

    Contains all the metadata and fields for a stored memory.
    This is the single source of truth for memory point payloads.
    """

    # Core fields
    text: str
    tags: list[str] = []
    importance: float = 0.5
    type: str = "fact"
    source: Optional[str] = None
    valid_until: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: str

    # Namespace and entity identification
    namespace: str = Field(
        default="default",
        description="Namespace for memory isolation (e.g., 'default', 'work', 'personal')",
    )
    entity_id: Optional[str] = Field(
        default=None,
        description="Entity this memory belongs to (for multi-entity isolation)",
    )

    # Intelligence fields
    confidence: float = 0.5
    mention_count: int = 1
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    archived: bool = False
    archived_at: Optional[str] = None
    superseded_by: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _handle_user_id_deprecation(cls, data: Any) -> Any:
        """Map deprecated 'user_id' to 'entity_id' with a deprecation warning."""
        if isinstance(data, dict):
            has_user_id = "user_id" in data
            has_entity_id = "entity_id" in data

            if has_user_id and has_entity_id:
                user_id_val = data["user_id"]
                entity_id_val = data["entity_id"]

                if (
                    user_id_val is not None
                    and entity_id_val is not None
                    and user_id_val != entity_id_val
                ):
                    raise ValueError(
                        "Cannot specify both 'user_id' and 'entity_id'. "
                        "Use 'entity_id' (user_id is deprecated)."
                    )

                # Same value or user_id is None: just drop user_id (roundtrip safe)
                data.pop("user_id")

            elif has_user_id:
                warnings.warn(
                    "MemoryPointPayload(user_id=...) is deprecated, use entity_id instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                data["entity_id"] = data.pop("user_id")

        return data

    @field_validator("namespace")
    @classmethod
    def _validate_namespace(cls, v: str) -> str:
        from casual_memory.utils.validation import validate_identifier

        return validate_identifier(v, "namespace")

    @field_validator("entity_id")
    @classmethod
    def _validate_entity_id(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        from casual_memory.utils.validation import validate_identifier

        return validate_identifier(v, "entity_id")

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Override model_dump to include 'user_id' key for backward compatibility.

        The storage layer and other components may still reference 'user_id'.
        Until those layers are fully updated, we include both keys.
        """
        data = super().model_dump(**kwargs)
        # Include user_id as alias for entity_id in serialized output,
        # but respect exclude_none semantics
        user_id_val = data.get("entity_id")
        if not (kwargs.get("exclude_none") and user_id_val is None):
            data["user_id"] = user_id_val
        return data

    @property
    def user_id(self) -> Optional[str]:
        """Deprecated: Use entity_id instead."""
        warnings.warn(
            "MemoryPointPayload.user_id is deprecated, use entity_id instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.entity_id


class MemoryPoint(BaseModel):
    """
    A memory point in vector storage.

    Combines a vector embedding with its associated payload.
    """

    id: str
    vector: list[float]
    payload: MemoryPointPayload
