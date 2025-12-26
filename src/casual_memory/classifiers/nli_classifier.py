"""
NLI-based classifier for fast pre-filtering.

Uses DeBERTa-v3 cross-encoder to quickly classify obvious cases:
- High entailment → same (duplicate/refinement)
- High neutral + low entailment → neutral (distinct facts)
- Uncertain → Pass to next classifier

Performance: ~50-200ms on CPU, ~10-50ms on GPU
Accuracy: 92.38% on SNLI, 90.04% on MNLI
"""

import logging
from typing import Optional

from casual_memory.classifiers.models import (
    CheckType,
    SimilarityResult,
    SimilarMemory,
)
from casual_memory.intelligence.nli_filter import NLIPreFilter
from casual_memory.models import MemoryFact

logger = logging.getLogger(__name__)

# Configuration constants
NLI_ENTAILMENT_THRESHOLD = 0.85
NLI_NEUTRAL_THRESHOLD = 0.5


class NLIClassifier:
    """
    Fast NLI-based classifier for obvious cases.

    Uses cross-encoder NLI model to classify pairs that have clear
    entailment or neutral relationships, passing uncertain cases to
    the next classifier in the pipeline.
    """

    def __init__(
        self,
        nli_filter: Optional[NLIPreFilter] = None,
        entailment_threshold: Optional[float] = None,
        neutral_threshold: Optional[float] = None,
    ):
        """
        Initialize the NLI classifier.

        Args:
            nli_filter: NLI pre-filter instance (default: creates new instance)
            entailment_threshold: Minimum entailment score for same (default: 0.85)
            neutral_threshold: Minimum neutral score for neutral (default: 0.5)
        """
        self.name = "nli"
        self.nli_filter = nli_filter or NLIPreFilter()
        self.entailment_threshold = entailment_threshold or NLI_ENTAILMENT_THRESHOLD
        self.neutral_threshold = neutral_threshold or NLI_NEUTRAL_THRESHOLD

        logger.info(
            f"NLI classifier initialized: "
            f"entailment_threshold={self.entailment_threshold}, "
            f"neutral_threshold={self.neutral_threshold}"
        )

    async def classify_pair(
        self,
        new_memory: MemoryFact,
        similar_memory: SimilarMemory,
        check_type: CheckType = "primary",
        existing_result: Optional[SimilarityResult] = None,
    ) -> Optional[SimilarityResult]:
        """
        Classify a memory pair using NLI model.

        Classification logic:
        1. If existing_result is provided → pass through (NLI only does initial classification)
        2. High entailment (≥ entailment_threshold) → same
           - Indicates paraphrase or refinement (memories are the same)
        3. High neutral (≥ neutral_threshold) + low entailment (< 0.3) → neutral
           - Indicates distinct but related facts (can coexist)
        4. Otherwise → None (pass to next classifier)
           - Uncertain or potential contradiction

        Args:
            new_memory: New memory being added
            similar_memory: Similar memory to compare against
            check_type: Type of check ("primary" or "secondary")
                       NLI checks both types as it's fast and useful for pre-filtering
            existing_result: Result from previous classifier (if any)

        Returns:
            SimilarityResult if confident classification, None otherwise
        """
        # If another classifier already classified, pass through
        if existing_result is not None:
            return existing_result
        try:
            # Run NLI prediction
            label, scores = self.nli_filter.predict(
                premise=similar_memory.memory.text,
                hypothesis=new_memory.text,
            )

            # Extract scores: [contradiction, entailment, neutral]
            contradiction_score = scores[0]
            entailment_score = scores[1]
            neutral_score = scores[2]

            # High entailment = paraphrase/refinement → same
            if entailment_score >= self.entailment_threshold:
                logger.debug(
                    f"NLI same (entailment={entailment_score:.3f}): "
                    f"{similar_memory.memory.text[:50]}... ↔ {new_memory.text[:50]}..."
                )

                return SimilarityResult(
                    similar_memory=similar_memory,
                    outcome="same",
                    confidence=entailment_score,
                    classifier_name=self.name,
                    metadata={
                        "nli_label": label,
                        "nli_scores": {
                            "contradiction": contradiction_score,
                            "entailment": entailment_score,
                            "neutral": neutral_score,
                        },
                    },
                )

            # High neutral + low entailment = distinct facts → neutral
            if neutral_score >= self.neutral_threshold and entailment_score < 0.3:
                logger.debug(
                    f"NLI neutral (neutral={neutral_score:.3f}, entailment={entailment_score:.3f}): "
                    f"{similar_memory.memory.text[:50]}... ↔ {new_memory.text[:50]}..."
                )

                return SimilarityResult(
                    similar_memory=similar_memory,
                    outcome="neutral",
                    confidence=neutral_score,
                    classifier_name=self.name,
                    metadata={
                        "nli_label": label,
                        "nli_scores": {
                            "contradiction": contradiction_score,
                            "entailment": entailment_score,
                            "neutral": neutral_score,
                        },
                    },
                )

            # Uncertain - pass to next classifier
            logger.debug(
                f"NLI pass (C={contradiction_score:.3f}, E={entailment_score:.3f}, N={neutral_score:.3f}): "
                f"{similar_memory.memory.text[:50]}... ↔ {new_memory.text[:50]}..."
            )
            return None

        except Exception as e:
            logger.error(
                f"NLI classifier failed (passing to next classifier): {e}",
                exc_info=True,
            )
            # On error, pass to next classifier
            return None

    def get_metrics(self) -> dict:
        """
        Get NLI classifier metrics.

        Returns:
            Dictionary with NLI filter metrics (prediction counts, cache stats)
        """
        return self.nli_filter.get_metrics()
