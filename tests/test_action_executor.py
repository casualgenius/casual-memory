"""
Unit tests for MemoryActionExecutor.

The MemoryActionExecutor is responsible for translating classification decisions
into concrete storage operations. It receives MemoryClassificationResult objects
from the classification pipeline and executes the appropriate actions:

- **add**: Insert new memory to vector store, optionally archiving superseded memories
- **skip**: Update existing memory's mention count and last_seen timestamp
- **conflict**: Create conflict record(s) without adding the new memory

These tests verify:
1. Correct storage method calls for each outcome type
2. Proper payload construction with all required fields
3. Handling of edge cases (multiple supersedings, multiple conflicts)
4. Return of structured MemoryActionResult with appropriate IDs
"""

import pytest
from unittest.mock import Mock
from datetime import datetime

from casual_memory.execution.action_executor import MemoryActionExecutor
from casual_memory.execution.models import MemoryActionResult
from casual_memory.models import MemoryFact
from casual_memory.classifiers.models import (
    MemoryClassificationResult,
    SimilarMemory,
    SimilarityResult,
)


@pytest.fixture
def mock_vector_store():
    """Mock vector store."""
    store = Mock()
    store.add = Mock(return_value="new_mem_123")
    store.update_memory = Mock(return_value=True)
    store.archive_memory = Mock(return_value=True)
    return store


@pytest.fixture
def mock_conflict_store():
    """Mock conflict store."""
    store = Mock()
    store.add_conflict = Mock(return_value="conflict_123")
    return store


@pytest.fixture
def action_executor(mock_vector_store, mock_conflict_store):
    """Create action executor with mocked stores."""
    return MemoryActionExecutor(
        vector_store=mock_vector_store,
        conflict_store=mock_conflict_store,
    )


@pytest.mark.asyncio
async def test_execute_add_simple(action_executor, mock_vector_store):
    """
    Test executing add action without superseding any old memories.

    This is the simplest case: a completely new memory with no similar
    memories found. The executor should:
    1. Add the new memory to the vector store
    2. Return a MemoryActionResult with action="added"
    3. Include the new memory ID in the result
    4. Not supersede or create conflicts
    """
    # ARRANGE: Create a new memory with no similar memories
    new_memory = MemoryFact(
        text="I live in Paris",
        type="fact",
        tags=["location"],
        user_id="user_123",
        importance=0.8,
        confidence=0.7,
    )

    # Classification result with "add" outcome and empty similarity results
    classification_result = MemoryClassificationResult(
        new_memory=new_memory,
        overall_outcome="add",
        similarity_results=[],  # No similar memories found
    )

    embedding = [0.1, 0.2, 0.3]

    # ACT: Execute the action
    result = await action_executor.execute(classification_result, embedding)

    # ASSERT: Verify the result structure is correct
    assert result.action == "added"
    assert result.memory_id == "new_mem_123"  # ID returned from mock
    assert result.conflict_ids == []  # No conflicts
    assert result.superseded_ids == []  # No memories superseded
    assert "text" in result.metadata  # Metadata includes text snippet

    # Verify vector store.add() was called once with correct parameters
    mock_vector_store.add.assert_called_once()
    call_args = mock_vector_store.add.call_args
    assert call_args.kwargs["vector"] == embedding

    # Verify the payload sent to vector store contains all memory fields
    payload = call_args.kwargs["payload"]
    assert payload["text"] == "I live in Paris"
    assert payload["type"] == "fact"
    assert payload["tags"] == ["location"]
    assert payload["user_id"] == "user_123"
    assert payload["confidence"] == 0.7
    assert payload["archived"] is False
    assert "timestamp" in payload  # Auto-generated timestamp field


@pytest.mark.asyncio
async def test_execute_add_with_superseding(action_executor, mock_vector_store):
    """
    Test executing add action with superseding (replacing) an old memory.

    This tests the refinement case: when a new memory is more detailed/accurate
    than an existing one (e.g., "I work as a senior engineer at Google" vs
    "I work as an engineer"). The executor should:
    1. Add the new memory to the vector store
    2. Archive the old memory, marking it as superseded
    3. Return superseded_ids in the result

    This is important for maintaining memory quality without accumulating
    redundant or outdated information.
    """
    # ARRANGE: Create a refined (more detailed) version of an existing memory
    new_memory = MemoryFact(
        text="I work as a senior engineer at Google",  # More specific
        type="fact",
        tags=[],
        user_id="user_123",
        confidence=0.9,  # Higher confidence
    )

    # The old, less detailed memory that will be superseded
    old_memory = MemoryFact(
        text="I work as an engineer",  # Less specific
        type="fact",
        tags=[],
        user_id="user_123",
        confidence=0.6,  # Lower confidence
    )

    similar_memory = SimilarMemory(
        memory_id="old_mem_100",
        memory=old_memory,
        similarity_score=0.92,  # High similarity indicates same topic
    )

    # Classification result with "add" outcome but with a superseded memory
    classification_result = MemoryClassificationResult(
        new_memory=new_memory,
        overall_outcome="add",
        similarity_results=[
            SimilarityResult(
                similar_memory=similar_memory,
                outcome="superseded",  # Old memory should be archived
                confidence=0.9,
                classifier_name="duplicate",
                metadata={},
            )
        ],
    )

    embedding = [0.1, 0.2, 0.3]

    # ACT: Execute the action
    result = await action_executor.execute(classification_result, embedding)

    # ASSERT: Verify both add and archive operations occurred
    assert result.action == "added"
    assert result.memory_id == "new_mem_123"
    assert result.superseded_ids == ["old_mem_100"]  # Old memory was archived
    assert result.conflict_ids == []

    # Verify vector store.add() was called to insert new memory
    mock_vector_store.add.assert_called_once()

    # Verify vector store.archive_memory() was called to archive old memory
    mock_vector_store.archive_memory.assert_called_once_with(
        memory_id="old_mem_100",
        superseded_by="new_mem_123",  # Link to the new memory
    )


@pytest.mark.asyncio
async def test_execute_add_with_multiple_superseding(action_executor, mock_vector_store):
    """
    Test superseding multiple old memories with a single new memory.

    This edge case can occur when:
    - A user told us partial information multiple times before
    - Now they provide comprehensive information that supersedes all partials
    - Example: "I like pizza" + "I like pasta" → "I love Italian food"

    The executor should:
    1. Add the new comprehensive memory
    2. Archive ALL superseded memories
    3. Link each archived memory to the new one via superseded_by
    4. Return all superseded IDs in the result
    """
    # ARRANGE: One new comprehensive memory that supersedes 3 old ones
    new_memory = MemoryFact(
        text="Detailed fact",  # Comprehensive information
        type="fact",
        tags=[],
        user_id="user_123",
    )

    # Create 3 similarity results, each marking an old memory as superseded
    classification_result = MemoryClassificationResult(
        new_memory=new_memory,
        overall_outcome="add",
        similarity_results=[
            SimilarityResult(
                similar_memory=SimilarMemory(
                    memory_id=f"old_mem_{i}",
                    memory=MemoryFact(text=f"Old fact {i}", type="fact", tags=[], user_id="user_123"),
                    similarity_score=0.9,
                ),
                outcome="superseded",  # Each is superseded
                confidence=0.9,
                classifier_name="duplicate",
                metadata={},
            )
            for i in range(3)  # 3 old memories to supersede
        ],
    )

    embedding = [0.1, 0.2, 0.3]

    # ACT: Execute the action
    result = await action_executor.execute(classification_result, embedding)

    # ASSERT: Verify all 3 memories were superseded
    assert result.action == "added"
    assert len(result.superseded_ids) == 3
    assert result.superseded_ids == ["old_mem_0", "old_mem_1", "old_mem_2"]

    # Verify archive_memory() was called 3 times (once per old memory)
    assert mock_vector_store.archive_memory.call_count == 3


@pytest.mark.asyncio
async def test_execute_skip(action_executor, mock_vector_store):
    """
    Test executing skip action when new memory is a duplicate.

    When the user mentions something they've already told us (e.g., "I live in Paris"
    said twice), we don't want to store it again. Instead, we:
    1. Skip adding the duplicate memory
    2. Update the existing memory's mention_count (increment by 1)
    3. Update the last_seen timestamp
    4. Return action="updated" with the existing memory's ID

    This helps track how frequently information is mentioned, which can
    boost confidence scores over time.
    """
    # ARRANGE: User says something they've already told us
    new_memory = MemoryFact(
        text="I live in Paris",  # Same as existing memory
        type="fact",
        tags=[],
        user_id="user_123",
    )

    # The existing memory that was already stored
    existing_memory = MemoryFact(
        text="I live in Paris",  # Exact same content
        type="fact",
        tags=[],
        user_id="user_123",
        confidence=0.7,
        mention_count=2,  # Already mentioned twice before
    )

    similar_memory = SimilarMemory(
        memory_id="existing_mem_100",
        memory=existing_memory,
        similarity_score=0.98,  # Very high similarity (nearly identical)
    )

    # Classification result with "skip" outcome (don't add duplicate)
    classification_result = MemoryClassificationResult(
        new_memory=new_memory,
        overall_outcome="skip",  # Don't add this duplicate
        similarity_results=[
            SimilarityResult(
                similar_memory=similar_memory,
                outcome="same",  # It's the same memory
                confidence=0.95,
                classifier_name="nli",
                metadata={},
            )
        ],
    )

    embedding = [0.1, 0.2, 0.3]

    # ACT: Execute the action
    result = await action_executor.execute(classification_result, embedding)

    # ASSERT: Verify existing memory was updated, not re-added
    assert result.action == "updated"  # Updated, not added
    assert result.memory_id == "existing_mem_100"  # Points to existing memory
    assert result.conflict_ids == []
    assert result.superseded_ids == []

    # Verify update_memory() was called to increment mention count
    mock_vector_store.update_memory.assert_called_once()
    call_args = mock_vector_store.update_memory.call_args
    assert call_args.kwargs["memory_id"] == "existing_mem_100"

    # Verify the updates include mention count increment and timestamp update
    payload_updates = call_args.kwargs["payload_updates"]
    assert payload_updates["mention_count"] == "+1"  # Increment by 1
    assert "last_seen" in payload_updates  # Update timestamp

    # Critically: verify add() was NOT called (no duplicate created)
    mock_vector_store.add.assert_not_called()


@pytest.mark.asyncio
async def test_execute_conflict_single(action_executor, mock_conflict_store):
    """
    Test executing conflict action when memories contradict each other.

    When the user says something that contradicts existing information
    (e.g., "I live in Paris" vs stored "I live in London"), we can't
    automatically choose which is correct. Instead, we:
    1. Do NOT add the new memory to the vector store
    2. Create a conflict record in the conflict store
    3. Store the new memory's text and embedding in the conflict metadata
    4. Mark the conflict with a category and clarification hint
    5. Return action="conflict" with conflict_ids

    The conflict can later be resolved through user clarification.
    """
    # ARRANGE: User says something contradictory
    new_memory = MemoryFact(
        text="I live in Paris",  # Contradicts existing location
        type="fact",
        tags=[],
        user_id="user_123",
        confidence=0.8,
    )

    # Existing memory that contradicts the new one
    existing_memory = MemoryFact(
        text="I live in London",  # Different location
        type="fact",
        tags=[],
        user_id="user_123",
        confidence=0.8,  # Same confidence - unclear which is right
    )

    similar_memory = SimilarMemory(
        memory_id="existing_mem_100",
        memory=existing_memory,
        similarity_score=0.89,  # Similar topic but contradictory content
    )

    # Classification result with "conflict" outcome
    classification_result = MemoryClassificationResult(
        new_memory=new_memory,
        overall_outcome="conflict",
        similarity_results=[
            SimilarityResult(
                similar_memory=similar_memory,
                outcome="conflict",  # These memories contradict
                confidence=0.9,
                classifier_name="conflict",
                metadata={
                    "category": "location",  # Type of conflict
                    "clarification_hint": "Where do you currently live?",
                    "avg_importance": 0.75,
                    "detection_method": "llm",
                },
            )
        ],
    )

    embedding = [0.1, 0.2, 0.3]

    # ACT: Execute the action
    result = await action_executor.execute(classification_result, embedding)

    # ASSERT: Verify conflict was created, NOT a new memory
    assert result.action == "conflict"
    assert result.memory_id is None  # No new memory was added
    assert len(result.conflict_ids) == 1
    assert result.conflict_ids[0] == "conflict_123"
    assert result.superseded_ids == []
    assert result.metadata["conflict_count"] == 1

    # Verify add_conflict() was called with proper MemoryConflict object
    mock_conflict_store.add_conflict.assert_called_once()
    conflict = mock_conflict_store.add_conflict.call_args.args[0]

    # Verify conflict has correct structure
    assert conflict.user_id == "user_123"
    assert conflict.memory_a_id == "existing_mem_100"  # The stored memory
    assert conflict.memory_b_id.startswith("pending_")  # Temporary ID (not yet stored)
    assert conflict.category == "location"
    assert conflict.similarity_score == 0.89
    assert conflict.avg_importance == 0.75
    assert conflict.clarification_hint == "Where do you currently live?"
    assert conflict.status == "pending"  # Awaiting resolution

    # Verify metadata stores both memories' text for later resolution
    assert conflict.metadata["memory_a_text"] == "I live in London"
    assert conflict.metadata["memory_b_text"] == "I live in Paris"
    assert conflict.metadata["memory_b_pending"] is True  # Flag: not in vector store
    assert conflict.metadata["detection_method"] == "llm"


@pytest.mark.asyncio
async def test_execute_conflict_multiple(action_executor, mock_conflict_store):
    """
    Test creating multiple conflict records for a single new memory.

    This rare edge case occurs when a new memory contradicts multiple
    existing memories. For example:
    - Stored: "I live in London" and "I work in London"
    - New: "I moved to Paris for work"
    - Result: 2 conflicts (location + job location)

    The executor should:
    1. Create a separate conflict record for each contradiction
    2. Return all conflict IDs
    3. NOT add the new memory to vector store
    """
    # ARRANGE: One new memory that conflicts with 2 existing memories
    new_memory = MemoryFact(
        text="New conflicting fact",
        type="fact",
        tags=[],
        user_id="user_123",
    )

    # Classification shows conflicts with 2 different memories
    classification_result = MemoryClassificationResult(
        new_memory=new_memory,
        overall_outcome="conflict",
        similarity_results=[
            SimilarityResult(
                similar_memory=SimilarMemory(
                    memory_id=f"conflict_mem_{i}",
                    memory=MemoryFact(text=f"Old fact {i}", type="fact", tags=[], user_id="user_123"),
                    similarity_score=0.9,
                ),
                outcome="conflict",  # Each is a conflict
                confidence=0.9,
                classifier_name="conflict",
                metadata={
                    "category": "factual",
                    "clarification_hint": f"Which is correct #{i}?",
                    "avg_importance": 0.7,
                },
            )
            for i in range(2)  # 2 separate conflicts
        ],
    )

    embedding = [0.1, 0.2, 0.3]

    # ACT: Execute the action
    result = await action_executor.execute(classification_result, embedding)

    # ASSERT: Verify 2 separate conflicts were created
    assert result.action == "conflict"
    assert len(result.conflict_ids) == 2
    assert result.metadata["conflict_count"] == 2

    # Verify add_conflict() was called twice (once per conflict)
    assert mock_conflict_store.add_conflict.call_count == 2


@pytest.mark.asyncio
async def test_execute_unknown_outcome_raises_error(action_executor):
    """
    Test that invalid/unknown outcomes raise ValueError.

    The executor only supports 3 outcomes: "add", "skip", "conflict".
    If the classification pipeline returns an invalid outcome (e.g., due to
    a bug or misconfiguration), the executor should fail fast with a clear error
    rather than silently ignoring it or producing corrupt data.

    This is a defensive programming test ensuring system integrity.
    """
    # ARRANGE: Create a classification result with an INVALID outcome
    new_memory = MemoryFact(
        text="Test memory",
        type="fact",
        tags=[],
        user_id="user_123",
    )

    classification_result = MemoryClassificationResult(
        new_memory=new_memory,
        overall_outcome="unknown_outcome",  # Invalid outcome (not add/skip/conflict)
        similarity_results=[],
    )

    embedding = [0.1, 0.2, 0.3]

    # ACT & ASSERT: Verify ValueError is raised with descriptive message
    with pytest.raises(ValueError, match="Unknown overall outcome"):
        await action_executor.execute(classification_result, embedding)


@pytest.mark.asyncio
async def test_payload_includes_all_fields(action_executor, mock_vector_store):
    """
    Test that the payload sent to vector store includes ALL MemoryFact fields.

    This is a critical test ensuring no data loss when storing memories.
    The executor must correctly transfer all fields from MemoryFact to the
    storage payload, including optional fields when they're provided.

    This prevents bugs where fields like source, valid_until, or mention_count
    would be silently dropped during storage.
    """
    # ARRANGE: Create a memory with ALL possible fields populated
    new_memory = MemoryFact(
        text="Complete memory",
        type="preference",
        tags=["tag1", "tag2"],
        importance=0.9,
        source="user",  # Optional field
        valid_until="2025-12-31T23:59:59",  # Optional field
        user_id="user_456",
        confidence=0.85,
        mention_count=5,  # Optional field
        first_seen="2024-01-01T00:00:00",  # Optional field
        last_seen="2024-12-01T00:00:00",  # Optional field
    )

    classification_result = MemoryClassificationResult(
        new_memory=new_memory,
        overall_outcome="add",
        similarity_results=[],
    )

    embedding = [0.1, 0.2, 0.3]

    # ACT: Execute the action
    await action_executor.execute(classification_result, embedding)

    # ASSERT: Verify EVERY field made it into the payload
    payload = mock_vector_store.add.call_args.kwargs["payload"]

    # Core fields
    assert payload["text"] == "Complete memory"
    assert payload["type"] == "preference"
    assert payload["tags"] == ["tag1", "tag2"]
    assert payload["importance"] == 0.9

    # Optional fields that were provided
    assert payload["source"] == "user"
    assert payload["valid_until"] == "2025-12-31T23:59:59"
    assert payload["user_id"] == "user_456"
    assert payload["confidence"] == 0.85
    assert payload["mention_count"] == 5
    assert payload["first_seen"] == datetime.fromisoformat("2024-01-01T00:00:00")
    assert payload["last_seen"] == datetime.fromisoformat("2024-12-01T00:00:00")

    # System-managed fields
    assert payload["archived"] is False
    assert payload["superseded_by"] is None
    assert "timestamp" in payload  # Auto-generated by executor


@pytest.mark.asyncio
async def test_payload_defaults_for_optional_fields(action_executor, mock_vector_store):
    """
    Test that the executor sets sensible defaults for optional fields.

    When a MemoryFact is created with only required fields (text, type, tags, user_id),
    the executor should fill in appropriate defaults for:
    - mention_count: 1 (first mention)
    - first_seen: current timestamp
    - last_seen: current timestamp
    - timestamp: current timestamp (when stored)
    - archived: False
    - superseded_by: None

    This ensures memories are always stored with complete, valid metadata.
    """
    # ARRANGE: Create a minimal memory with only required fields
    new_memory = MemoryFact(
        text="Minimal memory",
        type="fact",
        tags=[],
        user_id="user_123",
        # All optional fields left as None/default (not specified)
    )

    classification_result = MemoryClassificationResult(
        new_memory=new_memory,
        overall_outcome="add",
        similarity_results=[],
    )

    embedding = [0.1, 0.2, 0.3]

    # ACT: Execute the action
    await action_executor.execute(classification_result, embedding)

    payload = mock_vector_store.add.call_args.kwargs["payload"]

    # ASSERT: Verify sensible defaults were applied
    assert payload["mention_count"] == 1  # First time mentioned
    assert payload["first_seen"] is not None  # Auto-generated timestamp
    assert payload["last_seen"] is not None  # Auto-generated timestamp
    assert payload["timestamp"] is not None  # Storage timestamp
    assert payload["archived"] is False  # Not archived on creation
    assert payload["superseded_by"] is None  # No superseding relationship yet
