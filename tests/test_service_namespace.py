"""
Tests for namespace flow through MemoryService, ContextService, and ActionExecutor.

Verifies that namespace and entity_id are correctly passed through the
service layer to storage operations.
"""

import warnings
from unittest.mock import AsyncMock, Mock, call

import pytest
from casual_llm.messages import AssistantMessage, UserMessage

from casual_memory.classifiers.models import (
    MemoryClassificationResult,
    SimilarityResult,
    SimilarMemory,
)
from casual_memory.context_service import ContextService
from casual_memory.execution.action_executor import MemoryActionExecutor
from casual_memory.memory_service import MemoryService
from casual_memory.models import MemoryFact, MemoryQueryFilter
from casual_memory.storage.short_term.memory import InMemoryShortTermStore


# --- MemoryService namespace tests ---


@pytest.fixture
def mock_vector_store():
    """Mock vector store."""
    store = Mock()
    store.find_similar_memories = Mock(return_value=[])
    store.search = Mock(return_value=[])
    store.add = Mock(return_value="mem_123")
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
def mock_pipeline():
    """Mock classification pipeline."""
    pipeline = Mock()
    pipeline.classify = AsyncMock()
    return pipeline


@pytest.fixture
def mock_embedding():
    """Mock embedding service."""
    embedding = Mock()
    embedding.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
    embedding.embed_document = AsyncMock(return_value=[0.1, 0.2, 0.3])
    return embedding


@pytest.fixture
def memory_service(mock_vector_store, mock_conflict_store, mock_pipeline, mock_embedding):
    """Create memory service with mocked dependencies."""
    return MemoryService(
        vector_store=mock_vector_store,
        conflict_store=mock_conflict_store,
        pipeline=mock_pipeline,
        embedding=mock_embedding,
    )


@pytest.mark.asyncio
async def test_add_memory_passes_namespace_to_find_similar(
    memory_service, mock_vector_store, mock_pipeline
):
    """add_memory passes namespace from MemoryFact to find_similar_memories."""
    new_memory = MemoryFact(
        text="I live in Paris",
        type="fact",
        tags=[],
        entity_id="user_123",
        namespace="work",
        importance=0.8,
    )

    mock_vector_store.find_similar_memories.return_value = []
    mock_pipeline.classify.return_value = MemoryClassificationResult(
        new_memory=new_memory,
        overall_outcome="add",
        similarity_results=[],
    )

    await memory_service.add_memory(new_memory)

    call_kwargs = mock_vector_store.find_similar_memories.call_args.kwargs
    assert call_kwargs["entity_id"] == "user_123"
    assert call_kwargs["namespace"] == "work"


@pytest.mark.asyncio
async def test_add_memory_default_namespace(memory_service, mock_vector_store, mock_pipeline):
    """add_memory uses default namespace when not specified."""
    new_memory = MemoryFact(
        text="I live in Paris",
        type="fact",
        tags=[],
        entity_id="user_123",
        importance=0.8,
    )

    mock_vector_store.find_similar_memories.return_value = []
    mock_pipeline.classify.return_value = MemoryClassificationResult(
        new_memory=new_memory,
        overall_outcome="add",
        similarity_results=[],
    )

    await memory_service.add_memory(new_memory)

    call_kwargs = mock_vector_store.find_similar_memories.call_args.kwargs
    assert call_kwargs["namespace"] == "default"


@pytest.mark.asyncio
async def test_query_memory_passes_namespace_through_filter(
    memory_service, mock_vector_store
):
    """query_memory passes namespace from filter to vector store search."""
    mock_vector_store.search.return_value = []

    query_filter = MemoryQueryFilter(entity_id="user_123", namespace="work")
    await memory_service.query_memory(
        query="test", filter=query_filter, top_k=5, min_score=0.5
    )

    call_kwargs = mock_vector_store.search.call_args.kwargs
    filters = call_kwargs["filters"]
    assert filters["namespace"] == "work"
    assert filters["entity_id"] == "user_123"


@pytest.mark.asyncio
async def test_query_memory_reconstructs_namespace_in_memoryfact(
    memory_service, mock_vector_store
):
    """query_memory includes namespace and entity_id when constructing MemoryFact."""
    result = Mock()
    result.payload = Mock(
        text="I live in Paris",
        type="fact",
        tags=["location"],
        importance=0.8,
        source=None,
        valid_until=None,
        namespace="work",
        entity_id="user_123",
        confidence=0.8,
        mention_count=2,
        first_seen="2024-01-01T00:00:00",
        last_seen="2024-01-02T00:00:00",
        archived=False,
        archived_at=None,
        superseded_by=None,
    )
    mock_vector_store.search.return_value = [result]

    query_filter = MemoryQueryFilter(entity_id="user_123", namespace="work")
    memories = await memory_service.query_memory(
        query="location", filter=query_filter
    )

    assert len(memories) == 1
    assert memories[0].namespace == "work"
    assert memories[0].entity_id == "user_123"


# --- ActionExecutor namespace tests ---


@pytest.fixture
def action_executor(mock_vector_store, mock_conflict_store):
    """Create action executor with mocked stores."""
    return MemoryActionExecutor(
        vector_store=mock_vector_store,
        conflict_store=mock_conflict_store,
    )


@pytest.mark.asyncio
async def test_execute_add_includes_namespace_in_payload(
    action_executor, mock_vector_store
):
    """_execute_add includes namespace and entity_id in payload."""
    new_memory = MemoryFact(
        text="I live in Paris",
        type="fact",
        tags=[],
        entity_id="user_123",
        namespace="work",
        importance=0.8,
    )

    classification_result = MemoryClassificationResult(
        new_memory=new_memory,
        overall_outcome="add",
        similarity_results=[],
    )

    await action_executor.execute(classification_result, [0.1, 0.2, 0.3])

    payload = mock_vector_store.add.call_args.kwargs["payload"]
    assert payload["namespace"] == "work"
    assert payload["entity_id"] == "user_123"
    # Should NOT have user_id key in the payload
    assert "user_id" not in payload


@pytest.mark.asyncio
async def test_execute_conflict_includes_namespace(
    action_executor, mock_conflict_store
):
    """_execute_conflict creates MemoryConflict with namespace and entity_id."""
    new_memory = MemoryFact(
        text="I live in Paris",
        type="fact",
        tags=[],
        entity_id="user_123",
        namespace="personal",
        importance=0.8,
        confidence=0.8,
    )

    existing_memory = MemoryFact(
        text="I live in London",
        type="fact",
        tags=[],
        entity_id="user_123",
        namespace="personal",
        importance=0.8,
        confidence=0.8,
    )

    similar_memory = SimilarMemory(
        memory_id="existing_mem_100",
        memory=existing_memory,
        similarity_score=0.89,
    )

    classification_result = MemoryClassificationResult(
        new_memory=new_memory,
        overall_outcome="conflict",
        similarity_results=[
            SimilarityResult(
                similar_memory=similar_memory,
                outcome="conflict",
                confidence=0.9,
                classifier_name="conflict",
                metadata={
                    "category": "location",
                    "clarification_hint": "Where do you currently live?",
                    "avg_importance": 0.8,
                },
            )
        ],
    )

    result = await action_executor.execute(classification_result, [0.1, 0.2, 0.3])

    assert result.action == "conflict"
    conflict = mock_conflict_store.add_conflict.call_args.args[0]
    assert conflict.namespace == "personal"
    assert conflict.entity_id == "user_123"


@pytest.mark.asyncio
async def test_execute_conflict_no_default_user_fallback(
    action_executor, mock_conflict_store
):
    """_execute_conflict does not use 'default_user' -- entity_id from MemoryFact is used as-is."""
    new_memory = MemoryFact(
        text="I live in Paris",
        type="fact",
        tags=[],
        entity_id="actual_user",
        importance=0.8,
        confidence=0.8,
    )

    existing_memory = MemoryFact(
        text="I live in London",
        type="fact",
        tags=[],
        entity_id="actual_user",
        importance=0.8,
        confidence=0.8,
    )

    classification_result = MemoryClassificationResult(
        new_memory=new_memory,
        overall_outcome="conflict",
        similarity_results=[
            SimilarityResult(
                similar_memory=SimilarMemory(
                    memory_id="mem_100",
                    memory=existing_memory,
                    similarity_score=0.89,
                ),
                outcome="conflict",
                confidence=0.9,
                classifier_name="conflict",
                metadata={
                    "category": "location",
                    "clarification_hint": "Where?",
                    "avg_importance": 0.8,
                },
            )
        ],
    )

    await action_executor.execute(classification_result, [0.1, 0.2, 0.3])

    conflict = mock_conflict_store.add_conflict.call_args.args[0]
    assert conflict.entity_id == "actual_user"
    assert conflict.entity_id != "default_user"


# --- ContextService namespace tests ---


@pytest.fixture
def short_term_store():
    return InMemoryShortTermStore(max_messages=100)


@pytest.fixture
def context_service(short_term_store):
    return ContextService(short_term_store=short_term_store, short_term_limit=50)


def test_context_add_with_namespace(context_service, short_term_store):
    """add() passes namespace to short-term store."""
    messages = [UserMessage(content="hello")]
    context_service.add("user1", "sess1", messages, namespace="work")

    # Verify messages are stored under namespace
    count = short_term_store.get_message_count("user1:sess1", namespace="work")
    assert count == 1

    # Verify messages are NOT under default namespace
    count_default = short_term_store.get_message_count("user1:sess1", namespace="default")
    assert count_default == 0


def test_context_get_with_namespace(context_service, short_term_store):
    """get() retrieves messages from the correct namespace."""
    messages = [UserMessage(content="work message")]
    context_service.add("user1", "sess1", messages, namespace="work")

    messages2 = [UserMessage(content="personal message")]
    context_service.add("user1", "sess1", messages2, namespace="personal")

    work_result = context_service.get("user1", "sess1", namespace="work")
    assert len(work_result) == 1
    assert work_result[0].message.content == "work message"

    personal_result = context_service.get("user1", "sess1", namespace="personal")
    assert len(personal_result) == 1
    assert personal_result[0].message.content == "personal message"


def test_context_clear_with_namespace(context_service, short_term_store):
    """clear() only clears messages in the specified namespace."""
    messages = [UserMessage(content="hello")]
    context_service.add("user1", "sess1", messages, namespace="work")
    context_service.add("user1", "sess1", messages, namespace="personal")

    count = context_service.clear("user1", "sess1", namespace="work")
    assert count == 1

    # Personal namespace still has messages
    personal_count = short_term_store.get_message_count("user1:sess1", namespace="personal")
    assert personal_count == 1


def test_context_default_namespace(context_service, short_term_store):
    """Methods default to 'default' namespace when not specified."""
    messages = [UserMessage(content="hello")]
    context_service.add("user1", "sess1", messages)

    # Verify stored under default namespace
    count = short_term_store.get_message_count("user1:sess1", namespace="default")
    assert count == 1


def test_context_user_id_deprecation_warning(context_service):
    """Passing user_id keyword argument emits deprecation warning."""
    messages = [UserMessage(content="hello")]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        context_service.add("user1", "sess1", messages, user_id="user1")

        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1
        assert "deprecated" in str(deprecation_warnings[0].message).lower()


def test_context_user_id_deprecation_on_get(context_service):
    """Passing user_id keyword argument to get() emits deprecation warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        context_service.get("user1", "sess1", user_id="user1")

        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1


def test_context_user_id_deprecation_on_clear(context_service):
    """Passing user_id keyword argument to clear() emits deprecation warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        context_service.clear("user1", "sess1", user_id="user1")

        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1


def test_context_backward_compatible_positional_args(context_service, short_term_store):
    """Existing code using positional args continues to work."""
    messages = [UserMessage(content="hello"), AssistantMessage(content="hi")]

    # Old-style call: add(user_id, session_id, messages) - positional args
    result = context_service.add("user1", "sess1", messages)
    assert len(result) == 2

    # Old-style call: get(user_id, session_id)
    retrieved = context_service.get("user1", "sess1")
    assert len(retrieved) == 2

    # Old-style call: clear(user_id, session_id)
    count = context_service.clear("user1", "sess1")
    assert count == 2


def test_context_namespace_isolation(context_service, short_term_store):
    """Messages in different namespaces are fully isolated."""
    context_service.add(
        "user1", "sess1", [UserMessage(content="work msg")], namespace="work"
    )
    context_service.add(
        "user1", "sess1", [UserMessage(content="personal msg")], namespace="personal"
    )
    context_service.add(
        "user1", "sess1", [UserMessage(content="default msg")]
    )

    work = context_service.get("user1", "sess1", namespace="work")
    personal = context_service.get("user1", "sess1", namespace="personal")
    default = context_service.get("user1", "sess1")

    assert len(work) == 1
    assert work[0].message.content == "work msg"
    assert len(personal) == 1
    assert personal[0].message.content == "personal msg"
    assert len(default) == 1
    assert default[0].message.content == "default msg"


def test_compose_key_uses_entity_id(context_service):
    """_compose_key creates entity_id:session_id format."""
    assert context_service._compose_key("alice", "chat1") == "alice:chat1"
