"""
Text embedding protocol for casual-memory.

Provides a unified interface for embedding text into dense vectors
for semantic similarity search.
"""

from typing import List, Protocol

from typing_extensions import runtime_checkable


@runtime_checkable
class TextEmbedding(Protocol):
    """
    Protocol for text embedding providers.

    Text embeddings convert strings into dense vector representations,
    enabling semantic similarity search. All implementations must:

    1. Return deterministic vectors for the same input
    2. Normalize vectors if using cosine similarity (recommended)
    3. Expose their output dimension for compatibility checking
    4. Implement async methods for consistency

    Example:
        >>> embedder = E5Embedding()
        >>> vector = await embedder.embed_document("Hello world")
        >>> len(vector) == embedder.dimension
        True
    """

    @property
    def dimension(self) -> int:
        """
        Vector dimension produced by this embedder.

        Critical for Qdrant collection initialization - all vectors
        in a collection must have the same dimension.

        Returns:
            Number of elements in each embedding vector

        Example:
            >>> embedder.dimension
            768
        """
        ...

    @property
    def model_name(self) -> str:
        """
        Identifier of the embedding model.

        Returns:
            Model name or identifier (e.g., "intfloat/e5-base-v2")
        """
        ...

    async def embed_document(self, text: str) -> List[float]:
        """
        Generate embedding for a document to be stored.

        Some models (e.g., E5, Instructor) distinguish between documents
        and queries. This method handles document-specific preprocessing.

        Args:
            text: Document text to embed

        Returns:
            Embedding vector (normalized if using cosine similarity)

        Raises:
            ValueError: If text is empty or too long for the model
        """
        ...

    async def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a search query.

        Some models (e.g., E5, Instructor) distinguish between documents
        and queries. This method handles query-specific preprocessing.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector (normalized if using cosine similarity)

        Raises:
            ValueError: If text is empty or too long for the model
        """
        ...

    async def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple documents efficiently.

        Implementations should override this for batch optimization.

        Args:
            texts: List of document texts
            batch_size: Number of texts to process per batch (default: 32)

        Returns:
            List of embedding vectors (same order as input)

        Raises:
            ValueError: If any text is empty or too long for the model
        """
        ...

    async def embed_queries(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple queries efficiently.

        Implementations should override this for batch optimization.

        Args:
            texts: List of query texts
            batch_size: Number of texts to process per batch (default: 32)

        Returns:
            List of embedding vectors (same order as input)

        Raises:
            ValueError: If any text is empty or too long for the model
        """
        ...
