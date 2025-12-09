"""
Models for vector storage.

Defines the data structures used by vector storage implementations
for storing memory points with embeddings and payloads.
"""

from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime


class MemoryPointPayload(BaseModel):
    """
    Payload for a memory point in vector storage.

    Contains all the metadata and fields for a stored memory.
    """

    text: str
    tags: List[str] = []
    importance: float = 0.5
    type: str = "fact"
    source: Optional[str] = None
    valid_until: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: str

    # Intelligence fields
    user_id: Optional[str] = None
    confidence: float = 0.5
    mention_count: int = 1
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    archived: bool = False
    archived_at: Optional[str] = None
    superseded_by: Optional[str] = None


class MemoryPoint(BaseModel):
    """
    A memory point in vector storage.

    Combines a vector embedding with its associated payload.
    """

    id: str
    vector: List[float]
    payload: MemoryPointPayload
