import uuid
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class MemoryFact(BaseModel):
    # Existing fields (unchanged)
    text: str
    type: Literal["fact", "preference", "event", "goal", "weather"]
    tags: List[str]
    importance: Optional[float] = 0.5
    source: Optional[Literal["assistant", "tool", "user"]] = None
    valid_until: Optional[str | None] = None

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
    content: List[MemoryFact]


class MemoryPointPayload(BaseModel):
    # Existing fields (unchanged)
    text: str
    type: Literal["fact", "preference", "event", "goal", "weather"]
    tags: List[str]
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
    vector: List[float]
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
    clarification_hint: str = Field(
        ..., description="Suggested question to ask user for clarification"
    )
    metadata: dict = Field(
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
    """Model for short-term conversation memory (last N messages)"""

    content: str
    role: Literal["user", "assistant"]
    timestamp: str


class MemoryQueryFilter(BaseModel):
    type: Optional[List[str] | None] = None
    min_importance: Optional[float | None] = None
    user_id: Optional[str | None] = None
