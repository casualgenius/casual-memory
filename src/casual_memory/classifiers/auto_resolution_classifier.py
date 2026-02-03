"""
Auto-resolution classifier for conflicts based on confidence ratios.

Post-processes conflict results and reclassifies them to superseded/same if one memory
has significantly higher confidence than the other, indicating it's more reliable.

This classifier examines existing conflict outcomes and can override them.
"""

import logging
from typing import Any, Optional

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
    Auto-resolution classifier for confidence-based conflict resolution with importance weighting.

    Examines conflict results and reclassifies them to superseded/same if one memory
    has significantly higher confidence, indicating automatic resolution is appropriate.

    **Importance-Weighted Thresholds**:
    Thresholds are adjusted based on the average importance of the two memories:
    - High importance (1.0): Requires stronger evidence (higher ratio) to auto-resolve
    - Low importance (0.6): Easier to auto-resolve (lower ratio required)

    This makes the system safer for critical information (allergies, names) while
    being more permissive for less critical data (preferences).

    Base thresholds (at importance=0.6):
    - ≥ supersede_threshold (default 1.3): New memory supersedes old
    - ≤ keep_threshold (default 0.7): Old memory kept, new rejected (same)
    - Between thresholds: Keep as conflict (requires manual resolution)
    """

    def __init__(
        self,
        base_supersede_threshold: Optional[float] = None,
        base_keep_threshold: Optional[float] = None,
        use_importance_weighting: bool = True,
    ):
        """
        Initialize the auto-resolution classifier.

        Args:
            base_supersede_threshold: Base threshold for superseding (default: 1.3)
                At low importance (0.6), this is used directly.
                At high importance (1.0), this is scaled up to ~2.6
            base_keep_threshold: Base threshold for keeping old (default: 0.7)
                At low importance (0.6), this is used directly.
                At high importance (1.0), this is scaled down to ~0.35
            use_importance_weighting: Whether to adjust thresholds based on importance (default: True)
        """
        self.name = "auto_resolution"
        self.base_supersede_threshold = (
            base_supersede_threshold or CONFLICT_CONFIDENCE_RATIO_SUPERSEDE
        )
        self.base_keep_threshold = base_keep_threshold or CONFLICT_CONFIDENCE_RATIO_KEEP
        self.use_importance_weighting = use_importance_weighting

        logger.info(
            f"Auto-resolution classifier initialized: "
            f"base_supersede={self.base_supersede_threshold}, "
            f"base_keep={self.base_keep_threshold}, "
            f"importance_weighting={self.use_importance_weighting}"
        )

    def _calculate_importance_factor(self, avg_importance: float) -> float:
        """
        Calculate importance scaling factor for thresholds.

        Uses exponential scaling to make thresholds stricter for high-importance memories:
        - importance 1.0 → factor = 2.0 (capped - very strict)
        - importance 0.8 → factor = 2.0 (capped - very strict)
        - importance 0.7 → factor ≈ 1.41 (moderately strict)
        - importance 0.6 → factor = 1.0 (base thresholds)
        - importance 0.5 → factor ≈ 0.71 (more permissive)

        Formula: factor = min(2^((importance - 0.6) / 0.2), 2.0)
        Cap at 2.0 ensures thresholds don't become impossibly strict

        Args:
            avg_importance: Average importance of the two memories (0.0-1.0)

        Returns:
            Scaling factor for threshold adjustment (0.0-2.0)
        """
        # Normalize to 0.6 baseline (where factor = 1.0)
        normalized = (avg_importance - 0.6) / 0.2

        # Exponential scaling: 2^normalized
        # At 0.6: 2^0 = 1.0
        # At 1.0: 2^2 = 4.0 (but we'll cap it at 2.0 for usability)
        factor = 2**normalized

        # Cap at 2.0 to avoid overly strict thresholds
        return min(factor, 2.0)

    def _get_adaptive_thresholds(
        self, new_memory: MemoryFact, old_memory: MemoryFact
    ) -> tuple[float, float]:
        """
        Calculate adaptive thresholds based on memory importance.

        If importance_weighting is disabled, returns base thresholds.
        Otherwise, scales thresholds based on average importance:
        - Higher importance → stricter thresholds (harder to auto-resolve)
        - Lower importance → looser thresholds (easier to auto-resolve)

        Args:
            new_memory: New memory being added
            old_memory: Existing memory being compared

        Returns:
            Tuple of (supersede_threshold, keep_threshold)
        """
        if not self.use_importance_weighting:
            return (self.base_supersede_threshold, self.base_keep_threshold)

        # Calculate average importance
        avg_importance = (new_memory.importance + old_memory.importance) / 2.0

        # Get scaling factor
        factor = self._calculate_importance_factor(avg_importance)

        # Scale thresholds:
        # - supersede_threshold: multiply by factor (higher importance = need higher ratio)
        # - keep_threshold: divide by factor (higher importance = lower threshold to keep old)
        supersede = self.base_supersede_threshold * factor
        keep = self.base_keep_threshold / factor

        return (supersede, keep)

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

            # Get adaptive thresholds based on importance
            supersede_threshold, keep_threshold = self._get_adaptive_thresholds(
                new_memory, similar_memory.memory
            )

            # Log adaptive thresholds if importance weighting is enabled
            if self.use_importance_weighting:
                avg_importance = (new_memory.importance + similar_memory.memory.importance) / 2.0
                logger.debug(
                    f"Adaptive thresholds (avg_importance={avg_importance:.2f}): "
                    f"supersede={supersede_threshold:.2f}, keep={keep_threshold:.2f}"
                )

            # Check if auto-resolvable based on confidence ratio
            if ratio >= supersede_threshold:
                # New memory is significantly more confident - supersede old
                logger.info(
                    f"Auto-resolved CONFLICT → SUPERSEDED (ratio={ratio:.2f} ≥ {supersede_threshold:.2f})\n"
                    f"  Old [{old_conf:.2f}, imp={similar_memory.memory.importance:.1f}]: {similar_memory.memory.text[:80]}\n"
                    f"  New [{new_conf:.2f}, imp={new_memory.importance:.1f}]: {new_memory.text[:80]}"
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
                        "old_importance": similar_memory.memory.importance,
                        "new_importance": new_memory.importance,
                        "supersede_threshold": supersede_threshold,
                        "keep_threshold": keep_threshold,
                        "original_outcome": "conflict",
                    },
                )

            elif ratio <= keep_threshold:
                # Old memory is significantly more confident - keep old
                logger.info(
                    f"Auto-resolved CONFLICT → SAME (ratio={ratio:.2f} ≤ {keep_threshold:.2f})\n"
                    f"  Old [{old_conf:.2f}, imp={similar_memory.memory.importance:.1f}]: {similar_memory.memory.text[:80]}\n"
                    f"  New [{new_conf:.2f}, imp={new_memory.importance:.1f}]: {new_memory.text[:80]}"
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
                        "old_importance": similar_memory.memory.importance,
                        "new_importance": new_memory.importance,
                        "supersede_threshold": supersede_threshold,
                        "keep_threshold": keep_threshold,
                        "original_outcome": "conflict",
                    },
                )

            else:
                # Confidence ratio inconclusive - keep as CONFLICT
                avg_importance = (new_memory.importance + similar_memory.memory.importance) / 2.0
                logger.debug(
                    f"Cannot auto-resolve (ratio={ratio:.2f} between thresholds "
                    f"{keep_threshold:.2f}-{supersede_threshold:.2f}, avg_imp={avg_importance:.2f}): "
                    f"{similar_memory.memory.text[:50]}..."
                )

                # Pass through but add metadata
                existing_result.metadata["confidence_ratio"] = ratio
                existing_result.metadata["auto_resolved"] = False
                existing_result.metadata["old_confidence"] = old_conf
                existing_result.metadata["new_confidence"] = new_conf
                existing_result.metadata["old_importance"] = similar_memory.memory.importance
                existing_result.metadata["new_importance"] = new_memory.importance
                existing_result.metadata["supersede_threshold"] = supersede_threshold
                existing_result.metadata["keep_threshold"] = keep_threshold

                return existing_result

        except Exception as e:
            logger.error(
                f"Auto-resolution failed for conflict (keeping as conflict): {e}",
                exc_info=True,
            )
            return existing_result

    def get_metrics(self) -> dict[str, Any]:
        """
        Get auto-resolution classifier metrics.

        Returns:
            Dictionary with auto-resolution configuration
        """
        return {
            "auto_resolution_base_supersede_threshold": self.base_supersede_threshold,
            "auto_resolution_base_keep_threshold": self.base_keep_threshold,
            "auto_resolution_importance_weighting": self.use_importance_weighting,
        }
