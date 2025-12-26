"""E5 embedding adapter for casual-memory."""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class E5Embedding:
    """
    E5 model family embedding adapter.

    E5 models are instruction-tuned embeddings that require specific prefixes:
    - "passage: " for documents to be stored
    - "query: " for search queries

    This adapter automatically handles prefix injection via embed_document()
    and embed_query() methods, so users don't need to remember model-specific
    requirements.

    Supported E5 models:
    - intfloat/e5-base-v2 (768 dims) - Default, good balance
    - intfloat/e5-large-v2 (1024 dims) - Higher quality, larger
    - intfloat/e5-small-v2 (384 dims) - Faster, smaller dimension

    Example:
        >>> embedder = E5Embedding(
        ...     model_name="intfloat/e5-base-v2",
        ...     device="cpu"
        ... )
        >>> # Automatic prefix handling
        >>> doc_vector = await embedder.embed_document("I live in London")
        >>> query_vector = await embedder.embed_query("Where do I live?")
        >>> len(doc_vector)
        768
    """

    def __init__(
        self,
        model_name: str = "intfloat/e5-base-v2",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        cache_folder: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        """
        Initialize E5 embedder.

        Args:
            model_name: HuggingFace model identifier (default: e5-base-v2)
            device: Device for computation ("cuda", "cpu", or None for auto)
            normalize_embeddings: L2 normalize vectors (required for cosine similarity)
            show_progress_bar: Show encoding progress (disable in production)
            cache_folder: Directory for model cache (None = default ~/.cache)
            trust_remote_code: Allow custom model code execution
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for E5Embedding. "
                "Install with: pip install casual-memory[embeddings-transformers]"
            ) from e

        self._model_name = model_name
        self._normalize = normalize_embeddings
        self._show_progress = show_progress_bar

        logger.info(f"Loading E5 model: {model_name}")
        self._model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=cache_folder,
            trust_remote_code=trust_remote_code,
        )
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded: {model_name} ({self._dimension} dimensions)")

    @property
    def dimension(self) -> int:
        """Vector dimension produced by this model."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Identifier of the loaded model."""
        return self._model_name

    async def embed_document(self, text: str) -> List[float]:
        """
        Generate embedding for a document to be stored.

        Automatically adds "passage: " prefix for E5 models.

        Args:
            text: Document text to embed (prefix added automatically)

        Returns:
            Embedding vector (normalized if normalize_embeddings=True)

        Raises:
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        # Add E5 document prefix
        prefixed_text = f"passage: {text}"

        embedding = self._model.encode(
            prefixed_text,
            normalize_embeddings=self._normalize,
            show_progress_bar=self._show_progress,
        )
        return embedding.tolist()

    async def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a search query.

        Automatically adds "query: " prefix for E5 models.

        Args:
            text: Query text to embed (prefix added automatically)

        Returns:
            Embedding vector (normalized if normalize_embeddings=True)

        Raises:
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        # Add E5 query prefix
        prefixed_text = f"query: {text}"

        embedding = self._model.encode(
            prefixed_text,
            normalize_embeddings=self._normalize,
            show_progress_bar=self._show_progress,
        )
        return embedding.tolist()

    async def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple documents efficiently.

        Automatically adds "passage: " prefix to each document.

        Args:
            texts: List of document texts
            batch_size: Number of texts to process per batch

        Returns:
            List of embedding vectors (same order as input)

        Raises:
            ValueError: If any text is empty
        """
        if not texts:
            return []

        if any(not text or not text.strip() for text in texts):
            raise ValueError("Cannot embed empty texts in batch")

        # Add E5 document prefix to all texts
        prefixed_texts = [f"passage: {text}" for text in texts]

        embeddings = self._model.encode(
            prefixed_texts,
            normalize_embeddings=self._normalize,
            show_progress_bar=self._show_progress,
            batch_size=batch_size,
        )
        return embeddings.tolist()

    async def embed_queries(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple queries efficiently.

        Automatically adds "query: " prefix to each query.

        Args:
            texts: List of query texts
            batch_size: Number of texts to process per batch

        Returns:
            List of embedding vectors (same order as input)

        Raises:
            ValueError: If any text is empty
        """
        if not texts:
            return []

        if any(not text or not text.strip() for text in texts):
            raise ValueError("Cannot embed empty texts in batch")

        # Add E5 query prefix to all texts
        prefixed_texts = [f"query: {text}" for text in texts]

        embeddings = self._model.encode(
            prefixed_texts,
            normalize_embeddings=self._normalize,
            show_progress_bar=self._show_progress,
            batch_size=batch_size,
        )
        return embeddings.tolist()
