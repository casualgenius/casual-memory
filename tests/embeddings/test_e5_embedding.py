"""Tests for E5 embedding adapter."""

import pytest


@pytest.fixture
def e5_embedder():
    """Create E5 embedder for testing."""
    pytest.importorskip("sentence_transformers")

    from casual_memory.embeddings import E5Embedding

    # Use small model for faster tests
    return E5Embedding(model_name="intfloat/e5-small-v2")


@pytest.mark.asyncio
async def test_model_loading(e5_embedder):
    """Test that E5 model loads correctly."""
    assert e5_embedder.model_name == "intfloat/e5-small-v2"
    assert e5_embedder.dimension == 384  # e5-small-v2 has 384 dimensions


@pytest.mark.asyncio
async def test_embed_document(e5_embedder):
    """Test embed_document with automatic prefix."""
    text = "I live in London"
    vector = await e5_embedder.embed_document(text)

    assert isinstance(vector, list)
    assert len(vector) == e5_embedder.dimension
    assert all(isinstance(v, float) for v in vector)


@pytest.mark.asyncio
async def test_embed_query(e5_embedder):
    """Test embed_query with automatic prefix."""
    text = "Where do I live?"
    vector = await e5_embedder.embed_query(text)

    assert isinstance(vector, list)
    assert len(vector) == e5_embedder.dimension
    assert all(isinstance(v, float) for v in vector)


@pytest.mark.asyncio
async def test_document_query_different_vectors(e5_embedder):
    """Test that document and query embeddings are different for same text."""
    text = "I like pizza"

    doc_vector = await e5_embedder.embed_document(text)
    query_vector = await e5_embedder.embed_query(text)

    # Vectors should be different due to different prefixes
    assert doc_vector != query_vector


@pytest.mark.asyncio
async def test_embed_documents_batch(e5_embedder):
    """Test batch embedding of documents."""
    texts = ["I like pizza", "I enjoy hiking", "Python is great"]
    vectors = await e5_embedder.embed_documents(texts)

    assert len(vectors) == len(texts)
    assert all(len(v) == e5_embedder.dimension for v in vectors)
    assert all(isinstance(v, list) for v in vectors)


@pytest.mark.asyncio
async def test_embed_queries_batch(e5_embedder):
    """Test batch embedding of queries."""
    texts = ["What do I like?", "What do I enjoy?", "What language?"]
    vectors = await e5_embedder.embed_queries(texts)

    assert len(vectors) == len(texts)
    assert all(len(v) == e5_embedder.dimension for v in vectors)
    assert all(isinstance(v, list) for v in vectors)


@pytest.mark.asyncio
async def test_embed_document_empty_text_raises(e5_embedder):
    """Test that embedding empty text raises ValueError."""
    with pytest.raises(ValueError, match="Cannot embed empty text"):
        await e5_embedder.embed_document("")

    with pytest.raises(ValueError, match="Cannot embed empty text"):
        await e5_embedder.embed_document("   ")


@pytest.mark.asyncio
async def test_embed_query_empty_text_raises(e5_embedder):
    """Test that embedding empty query raises ValueError."""
    with pytest.raises(ValueError, match="Cannot embed empty text"):
        await e5_embedder.embed_query("")

    with pytest.raises(ValueError, match="Cannot embed empty text"):
        await e5_embedder.embed_query("   ")


@pytest.mark.asyncio
async def test_embed_documents_empty_list(e5_embedder):
    """Test that embedding empty list returns empty list."""
    vectors = await e5_embedder.embed_documents([])
    assert vectors == []


@pytest.mark.asyncio
async def test_embed_documents_with_empty_text_raises(e5_embedder):
    """Test that batch with empty text raises ValueError."""
    with pytest.raises(ValueError, match="Cannot embed empty texts"):
        await e5_embedder.embed_documents(["valid", "", "also valid"])


@pytest.mark.asyncio
async def test_normalization_enabled():
    """Test that vectors are normalized when enabled."""
    pytest.importorskip("sentence_transformers")

    import math

    from casual_memory.embeddings import E5Embedding

    embedder = E5Embedding(model_name="intfloat/e5-small-v2", normalize_embeddings=True)

    vector = await embedder.embed_document("test text")

    # Check if vector is normalized (L2 norm should be ~1.0)
    l2_norm = math.sqrt(sum(v**2 for v in vector))
    assert abs(l2_norm - 1.0) < 0.01  # Allow small floating point error


@pytest.mark.asyncio
async def test_batch_size_parameter(e5_embedder):
    """Test that batch_size parameter is accepted."""
    texts = ["text 1", "text 2", "text 3"]

    # Should work with custom batch size
    vectors = await e5_embedder.embed_documents(texts, batch_size=2)
    assert len(vectors) == 3


@pytest.mark.asyncio
async def test_deterministic_embeddings(e5_embedder):
    """Test that same text produces same embedding."""
    text = "consistent text"

    vector1 = await e5_embedder.embed_document(text)
    vector2 = await e5_embedder.embed_document(text)

    # Vectors should be identical
    assert vector1 == vector2
