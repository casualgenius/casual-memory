"""
LLM-based duplicate/refinement detection classifier.

Determines if memories are duplicates/refinements or distinct facts.
Always returns a classification (does not pass to next classifier).

Performance: ~500-2000ms per LLM call
Fallback: Defaults to neutral if LLM unavailable
"""

import logging
from typing import Optional

from casual_memory.classifiers.models import (
    CheckType,
    SimilarMemory,
    SimilarityResult,
)
from casual_memory.intelligence.duplicate_detector import LLMDuplicateDetector
from casual_memory.models import MemoryFact

logger = logging.getLogger(__name__)


class DuplicateClassifier:
    """
    LLM-based duplicate/refinement detection classifier.

    Uses LLM to distinguish between:
    - same: Duplicates (same fact, update existing metadata)
    - superseded: Refinements (new memory replaces old, more detailed)
    - neutral: Distinct facts (different information, can coexist)

    Unlike other classifiers, this one always returns a result.
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

    async def classify_pair(
        self,
        new_memory: MemoryFact,
        similar_memory: SimilarMemory,
        check_type: CheckType = "primary",
        existing_result: Optional[SimilarityResult] = None,
    ) -> Optional[SimilarityResult]:
        """
        Classify a memory pair using LLM duplicate detection.

        Classification logic:
        1. If existing_result is provided → pass through (duplicate only does initial classification)
        2. For secondary checks → pass through (expensive LLM call, skip on secondary)
        3. LLM detects same fact/refinement:
           - If new memory is longer (>20% more text) → superseded
           - Otherwise → same
        4. LLM detects distinct facts → neutral

        Args:
            new_memory: New memory being added
            similar_memory: Similar memory to compare against
            check_type: Type of check ("primary" or "secondary")
                       Only performs classification on "primary" checks
            existing_result: Result from previous classifier (if any)

        Returns:
            SimilarityResult on primary checks if no existing result, otherwise passes through
        """
        # If another classifier already classified, pass through
        if existing_result is not None:
            return existing_result

        # Skip duplicate detection on secondary checks (expensive LLM call)
        if check_type == "secondary":
            logger.debug(
                "Skipping duplicate detection on secondary check: "
                f"{similar_memory.memory.text[:50]}... ↔ {new_memory.text[:50]}..."
            )
            return None

        try:
            # Check if duplicate/refinement
            is_duplicate = await self.detector.is_duplicate_or_refinement(
                memory_a=similar_memory.memory,
                memory_b=new_memory,
                similarity_score=similar_memory.similarity_score,
            )

            if is_duplicate:
                # Same fact or refinement - determine if superseded or same
                # If new memory is significantly longer, it's a refinement that supersedes
                new_len = len(new_memory.text)
                old_len = len(similar_memory.memory.text)
                length_ratio = new_len / old_len if old_len > 0 else 1.0

                if length_ratio > 1.2:
                    # New memory is 20%+ longer - it's a refinement that supersedes
                    logger.debug(
                        f"SUPERSEDED (refinement, new is {length_ratio:.1f}x longer): "
                        f"{similar_memory.memory.text[:50]}... → {new_memory.text[:50]}..."
                    )

                    return SimilarityResult(
                        similar_memory=similar_memory,
                        outcome="superseded",
                        confidence=0.9,
                        classifier_name=self.name,
                        metadata={
                            "duplicate_type": "refinement",
                            "length_ratio": length_ratio,
                        },
                    )
                else:
                    # Same fact, similar length - it's a duplicate
                    logger.debug(
                        f"SAME (duplicate): "
                        f"{similar_memory.memory.text[:50]}... ↔ {new_memory.text[:50]}..."
                    )

                    return SimilarityResult(
                        similar_memory=similar_memory,
                        outcome="same",
                        confidence=0.9,
                        classifier_name=self.name,
                        metadata={
                            "duplicate_type": "duplicate",
                            "length_ratio": length_ratio,
                        },
                    )
            else:
                # Distinct facts - can coexist
                logger.debug(
                    f"NEUTRAL (distinct facts): "
                    f"{similar_memory.memory.text[:50]}... ↔ {new_memory.text[:50]}..."
                )

                return SimilarityResult(
                    similar_memory=similar_memory,
                    outcome="neutral",
                    confidence=0.8,
                    classifier_name=self.name,
                    metadata={"duplicate_type": "distinct_facts"},
                )

        except Exception as e:
            # On error, default to neutral (conservative - don't merge if unsure)
            logger.error(
                f"Duplicate classifier failed (defaulting to neutral): {e}",
                exc_info=True,
            )

            return SimilarityResult(
                similar_memory=similar_memory,
                outcome="neutral",
                confidence=0.5,
                classifier_name=self.name,
                metadata={
                    "duplicate_type": "error_fallback",
                    "error": str(e),
                },
            )

    def get_metrics(self) -> dict:
        """
        Get duplicate classifier metrics.

        Returns:
            Dictionary with detector metrics (LLM call counts, success rates)
        """
        return self.detector.get_metrics()
