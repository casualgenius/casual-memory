"""
Classification pipeline orchestrator.

Coordinates multiple classifiers to process memory pairs and determine
the appropriate action (MERGE, CONFLICT, or ADD) for each pair.
"""

import logging

from casual_memory.classifiers.models import (
    ClassificationRequest,
    ClassificationResult,
    MemoryClassifier,
)

logger = logging.getLogger(__name__)


class ClassificationPipeline:
    """
    Orchestrates the classifier pipeline.

    Runs memory pairs through a sequence of classifiers, each examining
    pairs and classifying what they are confident about. Unclassified
    pairs are passed to the next classifier or defaulted to ADD.

    Pipeline flow:
    1. NLI Classifier (fast, filters obvious cases)
    2. Conflict Classifier (LLM-based contradiction detection)
    3. Duplicate Classifier (LLM-based same/distinct detection)
    4. Auto-Resolution Classifier (post-processor for conflicts)
    5. Default Handler (unclassified pairs → ADD)
    """

    def __init__(self, classifiers: list[MemoryClassifier]):
        """
        Initialize the classification pipeline.

        Args:
            classifiers: List of classifiers to run in sequence
        """
        self.classifiers = classifiers

        logger.info(
            f"Classification pipeline initialized with {len(classifiers)} classifiers: "
            f"{[c.name for c in classifiers]}"
        )

    async def classify(self, request: ClassificationRequest) -> ClassificationRequest:
        """
        Run request through all classifiers in sequence.

        Flow:
        1. Each classifier examines pairs in request.pairs
        2. Classifiers classify what they're confident about
        3. Classified pairs move from pairs to results
        4. Remaining pairs pass to next classifier
        5. Final unclassified pairs default to ADD

        Args:
            request: Classification request containing pairs to classify

        Returns:
            Updated request with all pairs classified (in request.results)
        """
        logger.info(f"Pipeline starting with {len(request.pairs)} pairs")

        # Run through all classifiers
        for classifier in self.classifiers:
            pairs_before = len(request.pairs)

            # Run classifier
            request = await classifier.classify(request)

            pairs_after = len(request.pairs)
            classified = pairs_before - pairs_after

            logger.info(
                f"Classifier '{classifier.name}' classified {classified} pairs, "
                f"{pairs_after} remaining"
            )

        # Default handler: unclassified pairs → ADD
        if request.pairs:
            logger.info(f"Default handler classifying {len(request.pairs)} remaining pairs as ADD")

            for pair in request.pairs:
                request.results.append(
                    ClassificationResult(
                        pair=pair,
                        classification="ADD",
                        classifier_name="default_handler",
                        metadata={"reason": "no_classifier_confident"},
                    )
                )

            request.pairs = []  # All pairs now classified

        # Log final summary
        merge_count = self._count_classification(request, "MERGE")
        conflict_count = self._count_classification(request, "CONFLICT")
        add_count = self._count_classification(request, "ADD")

        logger.info(
            f"Pipeline complete: {len(request.results)} results "
            f"(MERGE: {merge_count}, CONFLICT: {conflict_count}, ADD: {add_count})"
        )

        # Log breakdown by classifier
        classifier_breakdown = self._get_classifier_breakdown(request)
        logger.debug(f"Classifier breakdown: {classifier_breakdown}")

        return request

    def _count_classification(self, request: ClassificationRequest, classification: str) -> int:
        """
        Count results with a specific classification.

        Args:
            request: Classification request
            classification: Classification to count (MERGE, CONFLICT, or ADD)

        Returns:
            Count of results with that classification
        """
        return sum(1 for r in request.results if r.classification == classification)

    def _get_classifier_breakdown(self, request: ClassificationRequest) -> dict:
        """
        Get breakdown of results by classifier.

        Args:
            request: Classification request

        Returns:
            Dictionary mapping classifier names to counts
        """
        breakdown = {}
        for result in request.results:
            classifier_name = result.classifier_name
            breakdown[classifier_name] = breakdown.get(classifier_name, 0) + 1

        return breakdown

    def get_metrics(self) -> dict:
        """
        Get metrics from all classifiers.

        Returns:
            Dictionary with metrics from all classifiers
        """
        metrics = {"pipeline_classifier_count": len(self.classifiers)}

        for classifier in self.classifiers:
            # Get classifier metrics if available
            if hasattr(classifier, "get_metrics"):
                classifier_metrics = classifier.get_metrics()
                # Prefix with classifier name to avoid conflicts
                for key, value in classifier_metrics.items():
                    metrics[f"{classifier.name}_{key}"] = value

        return metrics
