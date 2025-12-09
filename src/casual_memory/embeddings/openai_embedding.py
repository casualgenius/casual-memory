"""OpenAI embedding adapter for casual-memory."""

import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)


class OpenAIEmbedding:
    """
    Embedding adapter using OpenAI's embedding API.

    Supports OpenAI's embedding models via API:
    - text-embedding-3-small (1536 dims, configurable 512-1536)
    - text-embedding-3-large (3072 dims)
    - text-embedding-ada-002 (1536 dims, legacy)

    Also compatible with OpenAI-compatible APIs (Azure, OpenRouter, etc.)

    Example:
        >>> embedder = OpenAIEmbedding(
        ...     model="text-embedding-3-small",
        ...     dimensions=768,  # Match other embedders
        ...     api_key="sk-..."
        ... )
        >>> vector = await embedder.embed_document("I like pizza")
        >>> len(vector)
        768
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dimensions: Optional[int] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """
        Initialize OpenAI embedder.

        Args:
            model: OpenAI model name (default: text-embedding-3-small)
            api_key: OpenAI API key (None = use OPENAI_API_KEY env var)
            base_url: Custom endpoint (None = official OpenAI, or Azure/OpenRouter)
            dimensions: Output dimension (only for 3-small: 512, 768, 1024, 1536)
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts for failed requests
        """
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai is required for OpenAIEmbedding. "
                "Install with: pip install casual-memory[embeddings-openai]"
            ) from e

        self._model = model
        self._dimensions = dimensions

        # Initialize OpenAI client
        self._client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

        # Determine actual output dimension
        if dimensions is not None:
            self._dimension = dimensions
        else:
            # Default dimensions for each model
            if model == "text-embedding-3-small":
                self._dimension = 1536
            elif model == "text-embedding-3-large":
                self._dimension = 3072
            elif model == "text-embedding-ada-002":
                self._dimension = 1536
            else:
                # Unknown model - make test call to determine
                logger.warning(f"Unknown model {model}, testing dimension...")
                test_vector = self._embed_single("test")
                self._dimension = len(test_vector)

        logger.info(f"OpenAI embedder initialized: {model} ({self._dimension} dimensions)")

    @property
    def dimension(self) -> int:
        """Vector dimension produced by this model."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Identifier of the OpenAI model."""
        return self._model

    def _embed_single(self, text: str) -> List[float]:
        """Internal method to embed a single text."""
        kwargs = {"model": self._model, "input": text}
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions

        response = self._client.embeddings.create(**kwargs)
        return response.data[0].embedding

    async def embed_document(self, text: str) -> List[float]:
        """
        Generate embedding for a document to be stored.

        OpenAI models don't require document/query distinction, so this
        is identical to embed_query() for compatibility with the protocol.

        Args:
            text: Document text to embed

        Returns:
            Embedding vector (length = self.dimension)

        Raises:
            ValueError: If text is empty
            openai.OpenAIError: If API request fails
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        return self._embed_single(text)

    async def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a search query.

        OpenAI models don't require document/query distinction, so this
        is identical to embed_document() for compatibility with the protocol.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector (length = self.dimension)

        Raises:
            ValueError: If text is empty
            openai.OpenAIError: If API request fails
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        return self._embed_single(text)

    async def embed_documents(
        self, texts: List[str], batch_size: int = 32
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple documents efficiently.

        OpenAI API supports batch encoding (up to ~8k texts per request).
        batch_size parameter is accepted for protocol compatibility but
        not used (OpenAI handles batching internally).

        Args:
            texts: List of document texts
            batch_size: Ignored (for protocol compatibility)

        Returns:
            List of embedding vectors (same order as input)

        Raises:
            ValueError: If any text is empty
            openai.OpenAIError: If API request fails
        """
        if not texts:
            return []

        if any(not text or not text.strip() for text in texts):
            raise ValueError("Cannot embed empty texts in batch")

        kwargs = {"model": self._model, "input": texts}
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions

        response = self._client.embeddings.create(**kwargs)

        # Ensure correct ordering (API preserves order)
        return [item.embedding for item in response.data]

    async def embed_queries(
        self, texts: List[str], batch_size: int = 32
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple queries efficiently.

        OpenAI API supports batch encoding (up to ~8k texts per request).
        batch_size parameter is accepted for protocol compatibility but
        not used (OpenAI handles batching internally).

        Args:
            texts: List of query texts
            batch_size: Ignored (for protocol compatibility)

        Returns:
            List of embedding vectors (same order as input)

        Raises:
            ValueError: If any text is empty
            openai.OpenAIError: If API request fails
        """
        if not texts:
            return []

        if any(not text or not text.strip() for text in texts):
            raise ValueError("Cannot embed empty texts in batch")

        kwargs = {"model": self._model, "input": texts}
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions

        response = self._client.embeddings.create(**kwargs)

        # Ensure correct ordering (API preserves order)
        return [item.embedding for item in response.data]
