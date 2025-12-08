"""
NLI pre-filter for conflict detection using DeBERTa-v3 cross-encoder.

Uses cross-encoder/nli-deberta-v3-base to quickly filter out obvious
non-contradictions before expensive LLM calls. Achieves:
- 92.38% accuracy on SNLI
- 90.04% accuracy on MNLI
- ~200-500ms inference (CPU) or ~50-100ms (GPU)

Architecture: Cross-encoder processes both sentences jointly, optimized
for pairwise sentence comparison tasks like contradiction detection.
"""

import logging
from typing import Literal, Optional
import importlib.util

logger = logging.getLogger(__name__)

NLILabel = Literal["contradiction", "entailment", "neutral"]


class NLIPreFilter:
    """
    DeBERTa-v3 cross-encoder for fast contradiction pre-filtering.

    Uses sentence-transformers cross-encoder API for efficient inference.
    Model is lazy-loaded on first use to avoid slowing down service startup.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        device: Optional[str] = None,
        enable_caching: bool = True,
    ):
        """
        Initialize the NLI pre-filter.

        Args:
            model_name: Hugging Face model name (default: cross-encoder/nli-deberta-v3-base)
            device: Device to run on ("cuda", "cpu", or None for auto-detect)
            enable_caching: Enable LRU caching for repeated predictions
        """
        self.model_name = model_name
        self.device = device
        self.enable_caching = enable_caching
        self._model = None
        self._prediction_count = 0
        self._cache_hits = 0

        logger.info(
            f"NLIPreFilter initialized (lazy-loading): "
            f"model={model_name}, device={device or 'auto'}, "
            f"caching={enable_caching}"
        )

    def _load_model(self):
        """Lazy-load the cross-encoder model on first use."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading NLI model: {self.model_name}")
            self._model = CrossEncoder(self.model_name, device=self.device)
            logger.info(
                f"NLI model loaded successfully on device: "
                f"{self._model.device if hasattr(self._model, 'device') else 'unknown'}"
            )

        except ImportError as e:
            logger.error(
                "sentence-transformers library not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise ImportError(
                "sentence-transformers required for NLI pre-filter. "
                "Install with: pip install sentence-transformers"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load NLI model: {e}")
            raise

    def predict(self, premise: str, hypothesis: str) -> tuple[NLILabel, list[float]]:
        """
        Predict NLI relationship between two statements.

        Args:
            premise: First statement (memory A)
            hypothesis: Second statement (memory B)

        Returns:
            Tuple of (label, scores) where:
                - label: "contradiction" | "entailment" | "neutral"
                - scores: [contradiction_score, entailment_score, neutral_score]

        Raises:
            ImportError: If sentence-transformers not installed
            Exception: If model inference fails
        """
        # Lazy-load model on first use
        self._load_model()

        # Check cache if enabled
        if self.enable_caching:
            cache_key = (premise, hypothesis)
            cached_result = self._get_cached_prediction(cache_key)
            if cached_result is not None:
                logger.debug("Found cached NLI prediction")
                self._cache_hits += 1
                return cached_result

        # Run inference
        self._prediction_count += 1

        try:
            # CrossEncoder.predict returns raw logits (pre-softmax scores)
            logits = self._model.predict([(premise, hypothesis)])[0]

            # Convert logits to probabilities using softmax
            import numpy as np

            logits_array = np.array(logits)
            exp_logits = np.exp(
                logits_array - np.max(logits_array)
            )  # Subtract max for numerical stability
            probabilities = exp_logits / exp_logits.sum()

            # Convert to list and find max
            scores_list = probabilities.tolist()
            label_idx = scores_list.index(max(scores_list))

            # Map index to label: 0=contradiction, 1=entailment, 2=neutral
            labels: list[NLILabel] = ["contradiction", "entailment", "neutral"]
            label = labels[label_idx]

            result = (label, scores_list)

            # Cache result if enabled
            if self.enable_caching:
                self._cache_prediction(cache_key, result)

            logger.debug(
                f"NLI prediction: {label} "
                f"(scores: C={scores_list[0]:.3f}, E={scores_list[1]:.3f}, N={scores_list[2]:.3f})\n"
                f"  Premise: {premise}\n"
                f"  Hypothesis: {hypothesis}"
            )

            return result

        except Exception as e:
            logger.error(f"NLI prediction failed: {e}")
            raise

    def _get_cached_prediction(
        self, cache_key: tuple[str, str]
    ) -> Optional[tuple[NLILabel, list[float]]]:
        """Get cached prediction result."""
        # Simple dict-based cache (could be upgraded to LRU cache)
        if not hasattr(self, "_cache"):
            self._cache = {}
        return self._cache.get(cache_key)

    def _cache_prediction(self, cache_key: tuple[str, str], result: tuple[NLILabel, list[float]]):
        """Cache prediction result."""
        if not hasattr(self, "_cache"):
            self._cache = {}

        # Limit cache size to prevent memory issues
        if len(self._cache) >= 1000:
            # Remove oldest 20% of entries (simple FIFO)
            items = list(self._cache.items())
            self._cache = dict(items[200:])

        self._cache[cache_key] = result

    def get_metrics(self) -> dict:
        """
        Get NLI filter usage metrics.

        Returns:
            Dictionary with prediction counts and cache statistics
        """
        metrics = {
            "nli_prediction_count": self._prediction_count,
            "nli_cache_hits": self._cache_hits,
            "nli_cache_size": len(self._cache) if hasattr(self, "_cache") else 0,
            "nli_model_loaded": self._model is not None,
        }

        if self._prediction_count > 0:
            cache_hit_rate = (self._cache_hits / (self._prediction_count + self._cache_hits)) * 100
            metrics["nli_cache_hit_rate_percent"] = round(cache_hit_rate, 2)

        return metrics

    def is_available(self) -> bool:
        """
        Check if NLI model is available without loading it.

        Returns:
            True if model is loaded or can be loaded, False otherwise
        """
        if self._model is not None:
            return True

        if importlib.util.find_spec("sentence_transformers") is None:
            return False

        return True
