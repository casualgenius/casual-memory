"""Test that adapters satisfy the TextEmbedding protocol."""

import pytest

from casual_memory.embeddings import TextEmbedding


@pytest.mark.asyncio
async def test_e5_embedding_is_protocol():
    """E5Embedding implements TextEmbedding protocol."""
    pytest.importorskip("sentence_transformers")

    from casual_memory.embeddings import E5Embedding

    embedder = E5Embedding(model_name="intfloat/e5-small-v2")
    assert isinstance(embedder, TextEmbedding)

    # Verify properties
    assert hasattr(embedder, "dimension")
    assert hasattr(embedder, "model_name")
    assert embedder.dimension > 0
    assert embedder.model_name == "intfloat/e5-small-v2"


@pytest.mark.asyncio
async def test_openai_is_protocol(monkeypatch):
    """OpenAIEmbedding implements TextEmbedding protocol."""
    pytest.importorskip("openai")

    # Set mock API key
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-for-testing")

    from casual_memory.embeddings import OpenAIEmbedding

    # Mock mode - just test that it implements the protocol
    embedder = OpenAIEmbedding(model="text-embedding-3-small", dimensions=768)
    assert isinstance(embedder, TextEmbedding)

    # Verify properties
    assert hasattr(embedder, "dimension")
    assert hasattr(embedder, "model_name")
    assert embedder.dimension == 768
    assert embedder.model_name == "text-embedding-3-small"
