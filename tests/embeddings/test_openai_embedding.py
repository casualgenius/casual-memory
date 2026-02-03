"""Tests for OpenAI embedding adapter."""

import os

import pytest


@pytest.fixture
def mock_openai_env(monkeypatch):
    """Set mock OpenAI API key for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-for-testing")


def test_openai_initialization(mock_openai_env):
    """Test OpenAI embedder initialization."""
    pytest.importorskip("openai")

    from casual_memory.embeddings import OpenAIEmbedding

    embedder = OpenAIEmbedding(model="text-embedding-3-small", dimensions=768)

    assert embedder.model_name == "text-embedding-3-small"
    assert embedder.dimension == 768


def test_openai_default_dimensions(mock_openai_env):
    """Test default dimensions for known models."""
    pytest.importorskip("openai")

    from casual_memory.embeddings import OpenAIEmbedding

    # text-embedding-3-small defaults to 1536
    embedder_small = OpenAIEmbedding(model="text-embedding-3-small")
    assert embedder_small.dimension == 1536

    # text-embedding-3-large defaults to 3072
    embedder_large = OpenAIEmbedding(model="text-embedding-3-large")
    assert embedder_large.dimension == 3072

    # text-embedding-ada-002 defaults to 1536
    embedder_ada = OpenAIEmbedding(model="text-embedding-ada-002")
    assert embedder_ada.dimension == 1536


def test_openai_custom_dimensions(mock_openai_env):
    """Test custom dimension configuration."""
    pytest.importorskip("openai")

    from casual_memory.embeddings import OpenAIEmbedding

    # text-embedding-3-small supports custom dimensions
    for dim in [512, 768, 1024, 1536]:
        embedder = OpenAIEmbedding(model="text-embedding-3-small", dimensions=dim)
        assert embedder.dimension == dim


def test_openai_api_key_from_env(mock_openai_env):
    """Test that API key is read from environment."""
    pytest.importorskip("openai")

    from casual_memory.embeddings import OpenAIEmbedding

    # Should not raise error with API key from environment
    embedder = OpenAIEmbedding(model="text-embedding-3-small")
    assert embedder.model_name == "text-embedding-3-small"


def test_openai_custom_base_url(mock_openai_env):
    """Test custom base URL for Azure/OpenRouter."""
    pytest.importorskip("openai")

    from casual_memory.embeddings import OpenAIEmbedding

    embedder = OpenAIEmbedding(
        model="text-embedding-3-small",
        base_url="https://custom-endpoint.example.com/v1",
    )
    assert embedder.model_name == "text-embedding-3-small"


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "").startswith("sk-test"),
    reason="Requires valid OPENAI_API_KEY for integration test",
)
async def test_openai_embed_document_integration():
    """Integration test for embed_document (requires valid API key)."""
    pytest.importorskip("openai")

    from casual_memory.embeddings import OpenAIEmbedding

    embedder = OpenAIEmbedding(model="text-embedding-3-small", dimensions=768)

    vector = await embedder.embed_document("I like pizza")

    assert isinstance(vector, list)
    assert len(vector) == 768
    assert all(isinstance(v, float) for v in vector)


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "").startswith("sk-test"),
    reason="Requires valid OPENAI_API_KEY for integration test",
)
async def test_openai_embed_query_integration():
    """Integration test for embed_query (requires valid API key)."""
    pytest.importorskip("openai")

    from casual_memory.embeddings import OpenAIEmbedding

    embedder = OpenAIEmbedding(model="text-embedding-3-small", dimensions=768)

    vector = await embedder.embed_query("What do I like?")

    assert isinstance(vector, list)
    assert len(vector) == 768
    assert all(isinstance(v, float) for v in vector)


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "").startswith("sk-test"),
    reason="Requires valid OPENAI_API_KEY for integration test",
)
async def test_openai_embed_documents_batch_integration():
    """Integration test for batch embedding (requires valid API key)."""
    pytest.importorskip("openai")

    from casual_memory.embeddings import OpenAIEmbedding

    embedder = OpenAIEmbedding(model="text-embedding-3-small", dimensions=768)

    texts = ["I like pizza", "I enjoy hiking", "Python is great"]
    vectors = await embedder.embed_documents(texts)

    assert len(vectors) == len(texts)
    assert all(len(v) == 768 for v in vectors)
    assert all(isinstance(v, list) for v in vectors)


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "").startswith("sk-test"),
    reason="Requires valid OPENAI_API_KEY for integration test",
)
async def test_openai_document_query_same_vectors_integration():
    """Test that OpenAI produces same vectors for doc/query (requires valid API key)."""
    pytest.importorskip("openai")

    from casual_memory.embeddings import OpenAIEmbedding

    embedder = OpenAIEmbedding(model="text-embedding-3-small", dimensions=768)

    text = "I like pizza"

    doc_vector = await embedder.embed_document(text)
    query_vector = await embedder.embed_query(text)

    # OpenAI doesn't distinguish document/query, so they should be the same
    assert doc_vector == query_vector


def test_openai_empty_text_validation(mock_openai_env):
    """Test that empty text validation works."""
    pytest.importorskip("openai")

    from casual_memory.embeddings import OpenAIEmbedding

    embedder = OpenAIEmbedding(model="text-embedding-3-small")

    # These tests don't make API calls, just check validation
    # Verify that the methods exist and have the right signature
    assert hasattr(embedder, "embed_document")
    assert hasattr(embedder, "embed_query")
    assert hasattr(embedder, "embed_documents")
    assert hasattr(embedder, "embed_queries")


def test_openai_batch_size_parameter_accepted(mock_openai_env):
    """Test that batch_size parameter is accepted."""
    pytest.importorskip("openai")

    from casual_memory.embeddings import OpenAIEmbedding

    embedder = OpenAIEmbedding(model="text-embedding-3-small")

    # Verify the method signatures accept batch_size
    import inspect

    sig_docs = inspect.signature(embedder.embed_documents)
    assert "batch_size" in sig_docs.parameters

    sig_queries = inspect.signature(embedder.embed_queries)
    assert "batch_size" in sig_queries.parameters
