"""
LLM-based duplicate/refinement detection classifier.

Determines if memory pairs are duplicates/refinements or distinct facts.
Always classifies pairs (does not pass to next classifier).

Performance: ~500-2000ms per LLM call
Fallback: High similarity threshold (≥0.95) if LLM unavailable
"""

import logging

from casual_memory.classifiers.models import ClassificationRequest, ClassificationResult
from casual_memory.intelligence.duplicate_detector import LLMDuplicateDetector

logger = logging.getLogger(__name__)


class DuplicateClassifier:
    """
    LLM-based duplicate/refinement detection classifier.

    Uses LLM to distinguish between:
    - MERGE: Duplicates or refinements (same fact, more detail)
    - ADD: Distinct facts (different information)

    Unlike other classifiers, this one always classifies pairs
    (does not pass uncertain cases to next classifier).
    """

    def __init__(self, llm_duplicate_detector: LLMDuplicateDetector):
        """
        Initialize the duplicate classifier.

        Args:
            llm_duplicate_detector: LLM-based duplicate detector instance
        """
        self.name = "duplicate"
        self.detector = llm_duplicate_detector

        logger.info("Duplicate classifier initialized")

    async def classify(self, request: ClassificationRequest) -> ClassificationRequest:
        """
        Classify pairs using LLM duplicate detection.

        Classification logic:
        1. LLM detects same fact/refinement → MERGE
        2. LLM detects distinct facts → ADD

        This classifier always classifies (does not pass pairs).

        Args:
            request: Classification request with pairs to classify

        Returns:
            Updated request with all pairs classified (pairs list will be empty)
        """
        for pair in request.pairs:
            try:
                # Check if duplicate/refinement
                is_duplicate = await self.detector.is_duplicate_or_refinement(
                    memory_a=pair.existing_memory,
                    memory_b=pair.new_memory,
                    similarity_score=pair.similarity_score,
                )

                if is_duplicate:
                    # Same fact or refinement - merge
                    logger.debug(
                        f"MERGE (duplicate/refinement): "
                        f"{pair.existing_memory.text[:50]}... ↔ {pair.new_memory.text[:50]}..."
                    )

                    request.results.append(
                        ClassificationResult(
                            pair=pair,
                            classification="MERGE",
                            classifier_name=self.name,
                            metadata={"duplicate_type": "duplicate_or_refinement"},
                        )
                    )
                else:
                    # Distinct facts - add separately
                    logger.debug(
                        f"ADD (distinct facts): "
                        f"{pair.existing_memory.text[:50]}... ↔ {pair.new_memory.text[:50]}..."
                    )

                    request.results.append(
                        ClassificationResult(
                            pair=pair,
                            classification="ADD",
                            classifier_name=self.name,
                            metadata={"duplicate_type": "distinct_facts"},
                        )
                    )

            except Exception as e:
                # On error, default to ADD (conservative - don't merge if unsure)
                logger.error(
                    f"Duplicate classifier failed for pair (defaulting to ADD): {e}",
                    exc_info=True,
                )

                request.results.append(
                    ClassificationResult(
                        pair=pair,
                        classification="ADD",
                        classifier_name=self.name,
                        metadata={
                            "duplicate_type": "error_fallback",
                            "error": str(e),
                        },
                    )
                )

        # All pairs classified - clear the list
        request.pairs = []

        logger.info(
            f"Duplicate classifier: classified all remaining pairs "
            f"(total results: {len(request.results)})"
        )

        return request

    def get_metrics(self) -> dict:
        """
        Get duplicate classifier metrics.

        Returns:
            Dictionary with detector metrics (LLM call counts, success rates)
        """
        return self.detector.get_metrics()
