"""
Auto-resolution classifier for conflicts based on confidence ratios.

Post-processes CONFLICT results and reclassifies them to MERGE if one memory
has significantly higher confidence than the other, indicating it's more reliable.

This classifier acts as a post-processor and does not examine pairs.
"""

import logging
from typing import Optional

from casual_memory.classifiers.models import ClassificationRequest

logger = logging.getLogger(__name__)

# Configuration constants (previously from app.config)
CONFLICT_CONFIDENCE_RATIO_SUPERSEDE = 1.3
CONFLICT_CONFIDENCE_RATIO_KEEP = 0.7


class AutoResolutionClassifier:
    """
    Auto-resolution classifier for confidence-based conflict resolution.

    Scans classification results for CONFLICT outcomes and reclassifies
    them to MERGE if one memory has significantly higher confidence,
    indicating automatic resolution is appropriate.

    Confidence ratio thresholds:
    - ≥ supersede_threshold (default 1.3): New memory supersedes old
    - ≤ keep_threshold (default 0.7): Old memory kept, new rejected
    - Between thresholds: Keep as CONFLICT (requires manual resolution)
    """

    def __init__(
        self,
        supersede_threshold: Optional[float] = None,
        keep_threshold: Optional[float] = None,
    ):
        """
        Initialize the auto-resolution classifier.

        Args:
            supersede_threshold: Min confidence ratio to supersede old memory (default: from config)
            keep_threshold: Max confidence ratio to keep old memory (default: from config)
        """
        self.name = "auto_resolution"
        self.supersede_threshold = supersede_threshold or CONFLICT_CONFIDENCE_RATIO_SUPERSEDE
        self.keep_threshold = keep_threshold or CONFLICT_CONFIDENCE_RATIO_KEEP

        logger.info(
            f"Auto-resolution classifier initialized: "
            f"supersede_threshold={self.supersede_threshold}, "
            f"keep_threshold={self.keep_threshold}"
        )

    async def classify(self, request: ClassificationRequest) -> ClassificationRequest:
        """
        Post-process CONFLICT results for auto-resolution.

        This classifier:
        1. Scans request.results for CONFLICT classifications
        2. Calculates confidence ratio (new_conf / old_conf)
        3. Reclassifies to MERGE if ratio meets thresholds
        4. Does NOT modify request.pairs (acts on results only)

        Args:
            request: Classification request with results to post-process

        Returns:
            Updated request with auto-resolved conflicts reclassified to MERGE
        """
        updated_results = []
        auto_resolved_count = 0

        for result in request.results:
            # Skip non-conflict results
            if result.classification != "CONFLICT":
                updated_results.append(result)
                continue

            try:
                # Calculate confidence ratio
                old_conf = result.pair.existing_memory.confidence
                new_conf = result.pair.new_memory.confidence

                # Cannot calculate ratio if old confidence is 0
                if old_conf == 0:
                    logger.debug(
                        f"Cannot auto-resolve (old confidence is 0): "
                        f"{result.pair.existing_memory.text[:50]}..."
                    )
                    updated_results.append(result)
                    continue

                ratio = new_conf / old_conf

                # Check if auto-resolvable
                if ratio >= self.supersede_threshold:
                    # New memory is significantly more confident - supersede old
                    logger.info(
                        f"Auto-resolved: KEEP NEW (ratio={ratio:.2f} ≥ {self.supersede_threshold})\n"
                        f"  Old [{old_conf:.2f}]: {result.pair.existing_memory.text}\n"
                        f"  New [{new_conf:.2f}]: {result.pair.new_memory.text}"
                    )

                    result.classification = "MERGE"
                    result.metadata["auto_resolved"] = True
                    result.metadata["resolution_decision"] = "keep_new"
                    result.metadata["confidence_ratio"] = ratio
                    auto_resolved_count += 1

                elif ratio <= self.keep_threshold:
                    # Old memory is significantly more confident - keep old
                    logger.info(
                        f"Auto-resolved: KEEP OLD (ratio={ratio:.2f} ≤ {self.keep_threshold})\n"
                        f"  Old [{old_conf:.2f}]: {result.pair.existing_memory.text}\n"
                        f"  New [{new_conf:.2f}]: {result.pair.new_memory.text}"
                    )

                    result.classification = "MERGE"
                    result.metadata["auto_resolved"] = True
                    result.metadata["resolution_decision"] = "keep_old"
                    result.metadata["confidence_ratio"] = ratio
                    auto_resolved_count += 1

                else:
                    # Confidence ratio inconclusive - keep as CONFLICT
                    logger.debug(
                        f"Cannot auto-resolve (ratio={ratio:.2f} between thresholds): "
                        f"{result.pair.existing_memory.text[:50]}..."
                    )
                    result.metadata["confidence_ratio"] = ratio
                    result.metadata["auto_resolved"] = False

            except Exception as e:
                logger.error(
                    f"Auto-resolution failed for result (keeping as CONFLICT): {e}",
                    exc_info=True,
                )

            updated_results.append(result)

        # Update results
        request.results = updated_results

        logger.info(
            f"Auto-resolution classifier: auto-resolved {auto_resolved_count} conflicts "
            f"(total results: {len(request.results)})"
        )

        return request

    def get_metrics(self) -> dict:
        """
        Get auto-resolution classifier metrics.

        Returns:
            Dictionary with auto-resolution statistics
        """
        return {
            "auto_resolution_supersede_threshold": self.supersede_threshold,
            "auto_resolution_keep_threshold": self.keep_threshold,
        }
