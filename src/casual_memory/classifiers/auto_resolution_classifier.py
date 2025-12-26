"""
Auto-resolution classifier for conflicts based on confidence ratios.

Post-processes conflict results and reclassifies them to superseded/same if one memory
has significantly higher confidence than the other, indicating it's more reliable.

This classifier examines existing conflict outcomes and can override them.
"""

import logging
from typing import Optional

from casual_memory.classifiers.models import (
    CheckType,
    SimilarityResult,
    SimilarMemory,
)
from casual_memory.models import MemoryFact

logger = logging.getLogger(__name__)

# Configuration constants (previously from app.config)
CONFLICT_CONFIDENCE_RATIO_SUPERSEDE = 1.3
CONFLICT_CONFIDENCE_RATIO_KEEP = 0.7


class AutoResolutionClassifier:
    """
    Auto-resolution classifier for confidence-based conflict resolution.

    Examines conflict results and reclassifies them to superseded/same if one memory
    has significantly higher confidence, indicating automatic resolution is appropriate.

    Confidence ratio thresholds:
    - ≥ supersede_threshold (default 1.3): New memory supersedes old
    - ≤ keep_threshold (default 0.7): Old memory kept, new rejected (same)
    - Between thresholds: Keep as conflict (requires manual resolution)
    """

    def __init__(
        self,
        supersede_threshold: Optional[float] = None,
        keep_threshold: Optional[float] = None,
    ):
        """
        Initialize the auto-resolution classifier.

        Args:
            supersede_threshold: Min confidence ratio to supersede old memory (default: 1.3)
            keep_threshold: Max confidence ratio to keep old memory (default: 0.7)
        """
        self.name = "auto_resolution"
        self.supersede_threshold = supersede_threshold or CONFLICT_CONFIDENCE_RATIO_SUPERSEDE
        self.keep_threshold = keep_threshold or CONFLICT_CONFIDENCE_RATIO_KEEP

        logger.info(
            f"Auto-resolution classifier initialized: "
            f"supersede_threshold={self.supersede_threshold}, "
            f"keep_threshold={self.keep_threshold}"
        )

    async def classify_pair(
        self,
        new_memory: MemoryFact,
        similar_memory: SimilarMemory,
        check_type: CheckType = "primary",
        existing_result: Optional[SimilarityResult] = None,
    ) -> Optional[SimilarityResult]:
        """
        Post-process conflict results for auto-resolution.

        Classification logic:
        1. If existing_result is None → pass through (no classification to override)
        2. If existing_result.outcome != "conflict" → pass through (not a conflict)
        3. If outcome == "conflict" → check confidence ratio:
           - ratio ≥ supersede_threshold → override to "superseded" (new memory wins)
           - ratio ≤ keep_threshold → override to "same" (old memory wins)
           - between thresholds → pass through as "conflict" (manual resolution needed)

        Args:
            new_memory: New memory being added
            similar_memory: Similar memory to compare against
            check_type: Type of check ("primary" or "secondary")
            existing_result: Result from previous classifier (typically conflict)

        Returns:
            Modified result if auto-resolved, otherwise passes through existing_result
        """
        # If no previous result, pass through
        if existing_result is None:
            return None

        # Only process conflict outcomes
        if existing_result.outcome != "conflict":
            return existing_result

        try:
            # Calculate confidence ratio
            old_conf = similar_memory.memory.confidence
            new_conf = new_memory.confidence

            # Cannot calculate ratio if old confidence is 0
            if old_conf == 0:
                logger.debug(
                    f"Cannot auto-resolve (old confidence is 0): "
                    f"{similar_memory.memory.text[:50]}..."
                )
                return existing_result

            ratio = new_conf / old_conf

            # Check if auto-resolvable based on confidence ratio
            if ratio >= self.supersede_threshold:
                # New memory is significantly more confident - supersede old
                logger.info(
                    f"Auto-resolved CONFLICT → SUPERSEDED (ratio={ratio:.2f} ≥ {self.supersede_threshold})\n"
                    f"  Old [{old_conf:.2f}]: {similar_memory.memory.text[:80]}\n"
                    f"  New [{new_conf:.2f}]: {new_memory.text[:80]}"
                )

                return SimilarityResult(
                    similar_memory=similar_memory,
                    outcome="superseded",
                    confidence=0.9,  # High confidence in auto-resolution
                    classifier_name=self.name,
                    metadata={
                        "auto_resolved": True,
                        "resolution_decision": "keep_new",
                        "confidence_ratio": ratio,
                        "old_confidence": old_conf,
                        "new_confidence": new_conf,
                        "original_outcome": "conflict",
                    },
                )

            elif ratio <= self.keep_threshold:
                # Old memory is significantly more confident - keep old
                logger.info(
                    f"Auto-resolved CONFLICT → SAME (ratio={ratio:.2f} ≤ {self.keep_threshold})\n"
                    f"  Old [{old_conf:.2f}]: {similar_memory.memory.text[:80]}\n"
                    f"  New [{new_conf:.2f}]: {new_memory.text[:80]}"
                )

                return SimilarityResult(
                    similar_memory=similar_memory,
                    outcome="same",
                    confidence=0.9,  # High confidence in auto-resolution
                    classifier_name=self.name,
                    metadata={
                        "auto_resolved": True,
                        "resolution_decision": "keep_old",
                        "confidence_ratio": ratio,
                        "old_confidence": old_conf,
                        "new_confidence": new_conf,
                        "original_outcome": "conflict",
                    },
                )

            else:
                # Confidence ratio inconclusive - keep as CONFLICT
                logger.debug(
                    f"Cannot auto-resolve (ratio={ratio:.2f} between thresholds): "
                    f"{similar_memory.memory.text[:50]}..."
                )

                # Pass through but add metadata
                existing_result.metadata["confidence_ratio"] = ratio
                existing_result.metadata["auto_resolved"] = False
                existing_result.metadata["old_confidence"] = old_conf
                existing_result.metadata["new_confidence"] = new_conf

                return existing_result

        except Exception as e:
            logger.error(
                f"Auto-resolution failed for conflict (keeping as conflict): {e}",
                exc_info=True,
            )
            return existing_result

    def get_metrics(self) -> dict:
        """
        Get auto-resolution classifier metrics.

        Returns:
            Dictionary with auto-resolution configuration
        """
        return {
            "auto_resolution_supersede_threshold": self.supersede_threshold,
            "auto_resolution_keep_threshold": self.keep_threshold,
        }
