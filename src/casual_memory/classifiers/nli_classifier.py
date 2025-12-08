"""
NLI-based classifier for fast pre-filtering of memory pairs.

Uses DeBERTa-v3 cross-encoder to quickly classify obvious cases:
- High entailment → MERGE (duplicate/refinement)
- High neutral + low entailment → ADD (distinct facts)
- Uncertain → Pass to next classifier

Performance: ~50-200ms on CPU, ~10-50ms on GPU
Accuracy: 92.38% on SNLI, 90.04% on MNLI
"""

import logging
from typing import Optional

from casual_memory.classifiers.models import ClassificationRequest, ClassificationResult
from casual_memory.intelligence.nli_filter import NLIPreFilter

logger = logging.getLogger(__name__)

# Configuration constants (previously from app.config)
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
            entailment_threshold: Minimum entailment score for MERGE (default: from config)
            neutral_threshold: Minimum neutral score for ADD (default: from config)
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

    async def classify(self, request: ClassificationRequest) -> ClassificationRequest:
        """
        Classify pairs using NLI model.

        Classification logic:
        1. High entailment (≥ entailment_threshold) → MERGE
           - Indicates paraphrase or refinement
        2. High neutral (≥ neutral_threshold) + low entailment (< 0.3) → ADD
           - Indicates distinct but related facts
        3. Otherwise → Pass to next classifier
           - Uncertain or potential contradiction

        Args:
            request: Classification request with pairs to classify

        Returns:
            Updated request with classified pairs moved to results
        """
        unclassified_pairs = []

        for pair in request.pairs:
            try:
                # Run NLI prediction
                label, scores = self.nli_filter.predict(
                    premise=pair.existing_memory.text,
                    hypothesis=pair.new_memory.text,
                )

                # Extract scores: [contradiction, entailment, neutral]
                contradiction_score = scores[0]
                entailment_score = scores[1]
                neutral_score = scores[2]

                # High entailment = paraphrase/refinement → MERGE
                if entailment_score >= self.entailment_threshold:
                    logger.debug(
                        f"NLI MERGE (entailment={entailment_score:.3f}): "
                        f"{pair.existing_memory.text[:50]}... ↔ {pair.new_memory.text[:50]}..."
                    )

                    request.results.append(
                        ClassificationResult(
                            pair=pair,
                            classification="MERGE",
                            classifier_name=self.name,
                            confidence=entailment_score,
                            metadata={
                                "nli_label": label,
                                "nli_scores": {
                                    "contradiction": contradiction_score,
                                    "entailment": entailment_score,
                                    "neutral": neutral_score,
                                },
                            },
                        )
                    )
                    continue

                # High neutral + low entailment = distinct facts → ADD
                if neutral_score >= self.neutral_threshold and entailment_score < 0.3:
                    logger.debug(
                        f"NLI ADD (neutral={neutral_score:.3f}, entailment={entailment_score:.3f}): "
                        f"{pair.existing_memory.text[:50]}... ↔ {pair.new_memory.text[:50]}..."
                    )

                    request.results.append(
                        ClassificationResult(
                            pair=pair,
                            classification="ADD",
                            classifier_name=self.name,
                            confidence=neutral_score,
                            metadata={
                                "nli_label": label,
                                "nli_scores": {
                                    "contradiction": contradiction_score,
                                    "entailment": entailment_score,
                                    "neutral": neutral_score,
                                },
                            },
                        )
                    )
                    continue

                # Uncertain - pass to next classifier
                logger.debug(
                    f"NLI PASS (C={contradiction_score:.3f}, E={entailment_score:.3f}, N={neutral_score:.3f}): "
                    f"{pair.existing_memory.text[:50]}... ↔ {pair.new_memory.text[:50]}..."
                )
                unclassified_pairs.append(pair)

            except Exception as e:
                logger.error(
                    f"NLI classifier failed for pair (passing to next classifier): {e}",
                    exc_info=True,
                )
                # On error, pass to next classifier
                unclassified_pairs.append(pair)

        # Update request with unclassified pairs
        request.pairs = unclassified_pairs

        logger.info(
            f"NLI classifier: classified {len(request.results)} pairs, "
            f"{len(unclassified_pairs)} remaining"
        )

        return request

    def get_metrics(self) -> dict:
        """
        Get NLI classifier metrics.

        Returns:
            Dictionary with NLI filter metrics (prediction counts, cache stats)
        """
        return self.nli_filter.get_metrics()
