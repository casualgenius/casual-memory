# Data Models

This document describes all the key data models used in casual-memory.

## Core Models

Located in [src/casual_memory/models.py](../src/casual_memory/models.py):

### MemoryFact

Core memory unit with text content and metadata.

```python
class MemoryFact(BaseModel):
    text: str                    # Memory content (first-person perspective)
    type: str                    # "fact", "preference", "event", "goal"
    tags: list[str]             # Semantic tags for filtering
    importance: float           # 0.0-1.0 (≥0.5 threshold for storage)
    confidence: float = 0.5     # Calculated confidence score
    entity_id: Optional[str]    # Entity this memory belongs to (e.g., user ID)
    namespace: str = "default"  # Namespace for memory isolation
    mention_count: int = 1      # How many times mentioned
    first_seen: Optional[str]   # ISO timestamp of first mention
    last_seen: Optional[str]    # ISO timestamp of last mention
    source: str = "user"        # "user" or "assistant"
    valid_until: Optional[str]  # Temporal validity (ISO timestamp)
    archived: bool = False      # Soft-delete flag
    archived_at: Optional[str]  # When archived
    superseded_by: Optional[str] # ID of memory that replaced this one
```

> **Note:** `user_id` is accepted as a deprecated alias for `entity_id` (emits `DeprecationWarning`).

**Memory Types:**
- `fact` - Factual information (name, location, job, etc.)
- `preference` - User preferences (likes, dislikes, habits)
- `goal` - User goals and aspirations
- `event` - Events (past or future)

### MemoryFactExtraction

Simplified model for LLM extraction (fewer fields = cleaner JSON schema).

```python
class MemoryFactExtraction(BaseModel):
    text: str           # Memory content
    type: str           # Memory type
    tags: list[str]     # Semantic tags
    importance: float   # 0.0-1.0
```

### MemoryConflict

Tracks contradictory memories pending resolution.

```python
class MemoryConflict(BaseModel):
    id: str                     # Unique conflict identifier
    entity_id: str              # Entity this conflict belongs to
    namespace: str = "default"  # Namespace for conflict isolation
    memory_a_id: str            # ID of first conflicting memory
    memory_b_id: str            # ID of second conflicting memory
    category: str               # "location", "job", "preference", "temporal", "factual"
    is_singleton_category: bool # Whether only one memory allowed
    similarity_score: float     # Vector similarity (0.0-1.0)
    status: str = "pending"     # "pending", "resolved", "escalated"
    avg_importance: float       # Average importance of both memories
    clarification_hint: str     # Suggested question for user
    resolution_type: Optional[str]     # "manual", "automated", "conversational"
    winning_memory_id: Optional[str]   # ID of kept memory
    resolution_attempts: int = 0       # Number of resolution attempts
    created_at: str             # ISO timestamp
    resolved_at: Optional[str]  # ISO timestamp when resolved
    metadata: dict = {}         # Additional context
```

### ConflictResolution

Resolution decision for a conflict.

```python
class ConflictResolution(BaseModel):
    conflict_id: str            # ID of conflict being resolved
    decision: str               # "keep_a", "keep_b", "merge", "both_valid"
    resolved_memory: MemoryFact # The memory to keep
    resolution_note: Optional[str]  # User's explanation
```

### ShortTermMemory

Conversation message for short-term storage.

```python
class ShortTermMemory(BaseModel):
    content: str        # Message content
    role: str           # "user" or "assistant"
    timestamp: str      # ISO timestamp
    metadata: dict = {} # Additional context
```

### MemoryQueryFilter

Filter criteria for memory queries.

```python
class MemoryQueryFilter(BaseModel):
    type: Optional[list[str]] = None      # Filter by memory types
    tags: Optional[list[str]] = None      # Filter by tags
    min_importance: Optional[float] = None # Minimum importance
    entity_id: Optional[str] = None        # Filter by entity
    namespace: Optional[str] = None        # Filter by namespace
    exclude_archived: bool = True          # Exclude archived memories
```

## Classification Models

Located in [src/casual_memory/classifiers/models.py](../src/casual_memory/classifiers/models.py):

### SimilarMemory

Wrapper for a memory similar to the new memory being added.

```python
@dataclass
class SimilarMemory:
    memory_id: str          # ID of the memory in vector store
    memory: MemoryFact      # The memory fact itself
    similarity_score: float # Cosine similarity to new memory (0.0-1.0)
```

### SimilarityResult

Result of classifying new memory against one similar memory.

```python
@dataclass
class SimilarityResult:
    similar_memory: SimilarMemory  # The similar memory being compared
    outcome: SimilarityOutcome     # "conflict", "superseded", "same", "neutral"
    confidence: float              # Confidence score (0.0-1.0)
    classifier_name: str           # Name of classifier that made decision
    metadata: dict[str, Any]       # Additional context (NLI scores, category, etc.)
```

**Outcome Values:**
- `conflict` - Contradictory memories requiring user resolution
- `superseded` - Similar memory should be archived (new one is better)
- `same` - Duplicate memory (update existing metadata)
- `neutral` - Distinct facts that can coexist

### MemoryClassificationResult

Overall result for classifying a new memory.

```python
@dataclass
class MemoryClassificationResult:
    new_memory: MemoryFact              # The new memory being classified
    overall_outcome: MemoryOutcome      # "add", "skip", "conflict"
    similarity_results: list[SimilarityResult]  # Results for each similar memory

    @property
    def conflicts_with(self) -> list[str]:
        """IDs of memories this conflicts with."""

    @property
    def supersedes(self) -> list[str]:
        """IDs of memories to archive."""

    @property
    def same_as(self) -> Optional[str]:
        """ID of memory to update metadata."""
```

**Overall Outcome Values:**
- `add` - Insert new memory to vector store
- `skip` - Update existing memory (increment mention_count)
- `conflict` - Create conflict record for user resolution

## Action Models

Located in [src/casual_memory/execution/models.py](../src/casual_memory/execution/models.py):

### MemoryActionResult

Result of executing a classification decision.

```python
@dataclass
class MemoryActionResult:
    action: str                    # "added", "updated", "conflict"
    memory_id: Optional[str]       # ID of memory (added/updated) or None
    conflict_ids: list[str]        # List of conflict IDs created
    superseded_ids: list[str]      # List of archived memory IDs
    metadata: dict[str, Any]       # Additional context
```

## Type Aliases

```python
# Similarity outcomes for individual comparisons
SimilarityOutcome = Literal["conflict", "superseded", "same", "neutral"]

# Overall memory outcomes
MemoryOutcome = Literal["add", "conflict", "skip"]

# Check types for tiered strategy
CheckType = Literal["primary", "secondary"]
```
