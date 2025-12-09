"""
Example: Using Text Embeddings with casual-memory

Demonstrates:
1. E5Embedding (local) with automatic prefix handling
2. OpenAIEmbedding (API) embeddings
3. Dimension compatibility
4. Integration with vector stores
5. Async usage patterns

Install required dependencies:
    pip install casual-memory[embeddings-transformers]  # For E5
    pip install casual-memory[embeddings-openai]       # For OpenAI
    pip install casual-memory[embeddings-all]          # For both
"""

import os
import asyncio


async def example_e5_embedding():
    """Example: Local embedding with E5 models."""
    try:
        from casual_memory.embeddings import E5Embedding
    except ImportError:
        print("⚠️  Skipping E5 example - install with: pip install casual-memory[embeddings-transformers]")
        return

    print("\n=== E5 Embedding ===")

    # Initialize embedder
    embedder = E5Embedding(
        model_name="intfloat/e5-small-v2",  # Use small model for faster demo
        device="cpu",  # or "cuda" for GPU
        normalize_embeddings=True,  # Required for cosine similarity
    )

    print(f"Model: {embedder.model_name}")
    print(f"Dimension: {embedder.dimension}")

    # Embed a document (automatic "passage:" prefix)
    document = "I live in London and work as a software engineer."
    doc_vector = await embedder.embed_document(document)
    print(f"Document vector length: {len(doc_vector)}")

    # Embed a query (automatic "query:" prefix)
    query = "Where does the user live?"
    query_vector = await embedder.embed_query(query)
    print(f"Query vector length: {len(query_vector)}")

    # Batch embedding documents (automatic prefixes)
    documents = [
        "I like pizza",
        "I enjoy hiking",
        "Python is my favorite language"
    ]
    doc_vectors = await embedder.embed_documents(documents)
    print(f"Batch embedded {len(doc_vectors)} documents")


async def example_openai():
    """Example: Cloud embedding with OpenAI."""
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  Skipping OpenAI example - set OPENAI_API_KEY environment variable")
        return

    try:
        from casual_memory.embeddings import OpenAIEmbedding
    except ImportError:
        print("⚠️  Skipping OpenAI example - install with: pip install casual-memory[embeddings-openai]")
        return

    print("\n=== OpenAI Embedding ===")

    # Initialize embedder with custom dimension for compatibility
    embedder = OpenAIEmbedding(
        model="text-embedding-3-small",
        dimensions=384,  # Match E5-small dimension!
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    print(f"Model: {embedder.model_name}")
    print(f"Dimension: {embedder.dimension}")

    # Embed document (no prefix needed for OpenAI)
    document = "I live in London and work as a software engineer."
    doc_vector = await embedder.embed_document(document)
    print(f"Document vector length: {len(doc_vector)}")

    # Batch embedding documents
    documents = [
        "I like pizza",
        "I enjoy hiking",
        "Python is my favorite language"
    ]
    doc_vectors = await embedder.embed_documents(documents)
    print(f"Batch embedded {len(doc_vectors)} documents")


async def example_with_vector_store():
    """Example: Using embedder with vector store."""
    try:
        from casual_memory.embeddings import E5Embedding
        from casual_memory.storage.vector import InMemoryVectorStore
    except ImportError as e:
        print(f"⚠️  Skipping vector store example - {e}")
        return

    print("\n=== Embedding + Vector Store Integration ===")

    # Initialize embedder and vector store
    embedder = E5Embedding(model_name="intfloat/e5-small-v2")
    vector_store = InMemoryVectorStore(dimension=embedder.dimension)

    # Add documents
    documents = [
        "I live in London",
        "I work as a software engineer",
        "I like pizza and pasta",
    ]

    for i, doc in enumerate(documents):
        # Automatic "passage:" prefix added by embed_document()
        vector = await embedder.embed_document(doc)
        payload = {
            "text": doc,
            "type": "fact",
            "user_id": "user_123",
            "timestamp": "2024-01-01T00:00:00Z",
        }
        memory_id = vector_store.add(
            memory_id=f"mem_{i}",
            vector=vector,
            payload=payload,
        )
        print(f"Added: {doc} (ID: {memory_id})")

    # Search
    query = "Where do I live?"
    # Automatic "query:" prefix added by embed_query()
    query_vector = await embedder.embed_query(query)
    results = vector_store.search(
        query_embedding=query_vector,
        top_k=2,
        min_score=0.3,
    )

    print(f"\nSearch: '{query}'")
    for memory, score in results:
        print(f"  [{score:.3f}] {memory.payload['text']}")


async def example_dimension_compatibility():
    """Example: Ensuring dimension compatibility."""
    try:
        from casual_memory.embeddings import E5Embedding
    except ImportError:
        print("⚠️  Skipping compatibility example - E5Embedding not available")
        return

    try:
        from casual_memory.embeddings import OpenAIEmbedding
    except ImportError:
        print("⚠️  OpenAI not available, showing E5 dimensions only")
        e5_embedder = E5Embedding(model_name="intfloat/e5-small-v2")
        print(f"\n=== Dimension Information ===")
        print(f"E5-small dimension: {e5_embedder.dimension}")
        return

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Set OPENAI_API_KEY to test dimension compatibility")
        return

    print("\n=== Dimension Compatibility ===")

    # Both embedders configured to 384 dimensions
    e5_embedder = E5Embedding(
        model_name="intfloat/e5-small-v2"  # Native 384 dims
    )

    openai_embedder = OpenAIEmbedding(
        model="text-embedding-3-small",
        dimensions=384  # Configured to match
    )

    print(f"E5-small dimension: {e5_embedder.dimension}")
    print(f"OpenAI dimension: {openai_embedder.dimension}")
    print(f"Dimensions match: {e5_embedder.dimension == openai_embedder.dimension}")

    print("\n✅ Both embedders can be used with the same vector store (dimension=384)")


async def main():
    """Run all examples."""
    print("🚀 casual-memory Embedding Examples\n")
    print("=" * 60)

    await example_e5_embedding()
    await example_openai()
    await example_dimension_compatibility()
    await example_with_vector_store()

    print("\n" + "=" * 60)
    print("✅ All examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
