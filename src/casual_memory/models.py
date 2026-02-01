import uuid
from datetime import datetime
from typing import Any, Literal, Optional

from casual_llm.messages import ChatMessage
from pydantic import BaseModel, Field


class MemoryFactExtraction(BaseModel):
    """
    Model for LLM memory extraction - contains ONLY fields the LLM should populate.

    This is used as the response format for LLM extraction to ensure:
    1. Clean separation: LLM only sees/populates extraction fields
    2. Smaller JSON schema: No system-managed fields in the schema
    3. Explicit contract: Clear about what LLM provides vs system manages
    """

    text: str = Field(
        ...,
        description=(
            "Concise, self-contained memory statement. "
            "For user memories: Use first person (e.g., 'My name is Alex'). "
            "Must be understandable without conversation context."
        ),
    )
    type: Literal["fact", "preference", "event", "goal", "weather"] = Field(
        ...,
        description=(
            "Memory category: "
            "'fact' (verifiable information like name, location, job, allergies), "
            "'preference' (subjective likes/dislikes or opinions), "
            "'event' (specific occurrences with dates like appointments, trips), "
            "'goal' (intentions or tasks to accomplish like reminders, learning objectives), "
            "'weather' (weather forecasts or conditions)"
        ),
    )
    tags: list[str] = Field(
        ...,
        description="Relevant lowercase keywords for categorization (e.g., ['allergy', 'health'])",
    )
    importance: float = Field(
        ...,  # REQUIRED: Force LLM to provide importance score (no default)
        ge=0.0,
        le=1.0,
        description=(
            "Importance score between 0.0 and 1.0. REQUIRED FIELD. "
            "Medical/safety info (allergies) = 1.0, "
            "Personal facts (name, job, birthday) = 0.9, "
            "Location/contact info = 0.8, "
            "Preferences/opinions = 0.6-0.8, "
            "Goals/reminders = 0.7-0.8, "
            "Events = 0.7-0.8"
        ),
    )
    valid_until: Optional[str | None] = Field(
        default=None,
        description=(
            "ISO8601 timestamp when memory expires. "
            "Use for temporary information (weather forecasts, reminders, appointments). "
            "null = permanent memory. "
            "Leave as null - the system will calculate expiry times for temporal memories."
        ),
    )


class MemoryFact(BaseModel):
    # Core extraction fields (used by LLM)
    text: str = Field(
        ...,
        description=(
            "Concise, self-contained memory statement. "
            "For user memories: Use first person (e.g., 'My name is Alex'). "
            "Must be understandable without conversation context."
        ),
    )
    type: Literal["fact", "preference", "event", "goal", "weather"] = Field(
        ...,
        description=(
            "Memory category: "
            "'fact' (verifiable information like name, location, job, allergies), "
            "'preference' (subjective likes/dislikes or opinions), "
            "'event' (specific occurrences with dates like appointments, trips), "
            "'goal' (intentions or tasks to accomplish like reminders, learning objectives), "
            "'weather' (weather forecasts or conditions)"
        ),
    )
    tags: list[str] = Field(
        ...,
        description="Relevant lowercase keywords for categorization (e.g., ['allergy', 'health'])",
    )
    importance: float = Field(
        ...,  # REQUIRED: Force LLM to provide importance score (no default)
        ge=0.0,
        le=1.0,
        description=(
            "Importance score between 0.0 and 1.0. REQUIRED FIELD. "
            "Medical/safety info (allergies) = 1.0, "
            "Personal facts (name, job, birthday) = 0.9, "
            "Location/contact info = 0.8, "
            "Preferences/opinions = 0.6-0.8, "
            "Goals/reminders = 0.7-0.8, "
            "Events = 0.7-0.8"
        ),
    )
    source: Optional[Literal["assistant", "tool", "user"]] = Field(
        default=None,
        description=(
            "Origin of the memory: "
            "'user' (from user messages), "
            "'assistant' (from assistant responses), "
            "'tool' (from tool outputs)"
        ),
    )
    valid_until: Optional[str | None] = Field(
        default=None,
        description=(
            "ISO8601 timestamp when memory expires. "
            "Use for temporary information (weather forecasts, reminders, appointments). "
            "null = permanent memory. "
            "Leave as null - the system will calculate expiry times for temporal memories."
        ),
    )

    # NEW fields for memory intelligence
    user_id: Optional[str] = Field(
        None, description="User this memory belongs to (for multi-user isolation)"
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence score based on mention frequency"
    )
    mention_count: int = Field(
        default=1, ge=1, description="Number of times this memory has been mentioned"
    )
    first_seen: Optional[datetime] = Field(
        default=None, description="When this memory was first extracted"
    )
    last_seen: Optional[datetime] = Field(
        default=None, description="Most recent mention of this memory"
    )
    archived: bool = Field(default=False, description="Whether this memory has been archived")
    archived_at: Optional[datetime] = Field(
        default=None, description="When this memory was archived"
    )
    superseded_by: Optional[str] = Field(
        default=None, description="ID of the memory that replaced this one"
    )


class MemoryBlock(BaseModel):
    type: Literal["mcp/context/v1"]
    domain: Literal["memory"]
    name: str
    content: list[MemoryFact]


class MemoryPointPayload(BaseModel):
    # Existing fields (unchanged)
    text: str
    type: Literal["fact", "preference", "event", "goal", "weather"]
    tags: list[str]
    importance: Optional[float] = 0.5  # Default if not included
    session_id: str | None
    source: str | None
    timestamp: str
    valid_until: Optional[str | None] = None

    # NEW fields for memory intelligence
    user_id: Optional[str] = None
    confidence: float = 0.5
    mention_count: int = 1
    first_seen: Optional[str] = None  # ISO format timestamp
    last_seen: Optional[str] = None  # ISO format timestamp
    archived: bool = False
    archived_at: Optional[str] = None  # ISO format timestamp
    superseded_by: Optional[str] = None


class MemoryPoint(BaseModel):
    id: str
    vector: list[float]
    payload: MemoryPointPayload


class MemoryConflict(BaseModel):
    """Model for tracking conflicts between contradictory memories"""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique conflict identifier"
    )
    user_id: str = Field(..., description="User this conflict belongs to")
    memory_a_id: str = Field(..., description="ID of first conflicting memory (Qdrant point ID)")
    memory_b_id: str = Field(..., description="ID of second conflicting memory (Qdrant point ID)")
    category: str = Field(..., description="Category of conflict (e.g., location, job, preference)")
    is_singleton_category: bool = Field(
        default=False, description="Whether only one memory should exist in this category"
    )
    similarity_score: float = Field(
        ..., ge=0.0, le=1.0, description="Vector similarity score between the two memories"
    )
    avg_importance: float = Field(
        ..., ge=0.0, le=1.0, description="Average importance of both memories"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="When conflict was detected"
    )
    status: Literal["pending", "resolved", "escalated"] = Field(
        default="pending", description="Current status of conflict"
    )
    resolution_attempts: int = Field(
        default=0, ge=0, description="Number of resolution attempts made"
    )
    resolved_at: Optional[datetime] = Field(default=None, description="When conflict was resolved")
    resolution_type: Optional[Literal["conversational", "manual", "automated"]] = Field(
        default=None, description="How the conflict was resolved"
    )
    winning_memory_id: Optional[str] = Field(
        default=None, description="ID of memory that was kept after resolution"
    )
    clarification_hint: Optional[str] = Field(
        default=None, description="Suggested question to ask user for clarification"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for conflict tracking"
    )


class ConflictResolution(BaseModel):
    """Model for resolving a memory conflict"""

    conflict_id: str = Field(..., description="ID of the conflict being resolved")
    decision: Literal["keep_a", "keep_b", "merge", "both_valid"] = Field(
        ..., description="Resolution decision"
    )
    resolution_type: Literal["conversational", "manual", "automated"] = Field(
        ..., description="How the conflict was resolved"
    )
    confirming_memory_text: Optional[str] = Field(
        default=None,
        description="Text of memory that confirmed the resolution (for conversational)",
    )
    resolved_by: str = Field(..., description="User ID or 'system' for automated resolutions")
    notes: Optional[str] = Field(default=None, description="Additional notes about the resolution")


class ShortTermMemory(BaseModel):
    """Model for short-term conversation memory (last N messages).

    Wraps ChatMessage from casual-llm with a timestamp for storage.
    Supports all message types:
    - UserMessage: User messages (supports text and multimodal content)
    - AssistantMessage: Assistant responses (content and/or tool_calls)
    - ToolResultMessage: Tool/function call results
    - SystemMessage: System prompts (typically not stored, but supported)

    Example:
        from casual_llm.messages import UserMessage, AssistantMessage

        # User message
        ShortTermMemory(
            message=UserMessage(content="Hello!"),
            timestamp="2024-01-01T10:00:00"
        )

        # Assistant message with tool call
        ShortTermMemory(
            message=AssistantMessage(
                content=None,
                tool_calls=[AssistantToolCall(...)]
            ),
            timestamp="2024-01-01T10:00:05"
        )
    """

    message: ChatMessage
    timestamp: str


class MemoryQueryFilter(BaseModel):
    type: Optional[list[str] | None] = None
    min_importance: Optional[float | None] = None
    user_id: Optional[str | None] = None
