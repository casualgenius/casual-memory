"""
Custom Classifier Example

Demonstrates how to create a custom classifier that implements
the MemoryClassifier protocol.
"""

import asyncio

from casual_memory.classifiers import ClassificationPipeline
from casual_memory.classifiers.models import ClassificationRequest, ClassificationResult


class SimpleRuleBasedClassifier:
    """
    Custom classifier using simple rules.

    Implements MemoryClassifier protocol via duck typing (no inheritance).
    """

    def __init__(self, similarity_threshold: float = 0.95):
        self.similarity_threshold = similarity_threshold

    async def classify(self, request: ClassificationRequest) -> ClassificationRequest:
        """
        Classify memory pairs using simple similarity threshold.

        - Similarity >= 0.95 → MERGE
        - Similarity < 0.95 → ADD
        """
        for pair in request.pairs:
            # Skip already classified pairs
            if any(r.pair_id == pair.existing_memory_id for r in request.results):
                continue

            # Simple rule: high similarity = duplicate
            if pair.similarity_score >= self.similarity_threshold:
                classification = "MERGE"
            else:
                classification = "ADD"

            result = ClassificationResult(
                pair_id=pair.existing_memory_id,
                classification=classification,
                classifier_name="SimpleRuleBasedClassifier",
                confidence=pair.similarity_score,
                metadata={"threshold": self.similarity_threshold},
            )

            request.results.append(result)

        return request


async def main():
    print("=== Custom Classifier Example ===\n")

    # Create custom classifier
    custom_classifier = SimpleRuleBasedClassifier(similarity_threshold=0.95)

    # Build pipeline with custom classifier
    _pipeline = ClassificationPipeline(classifiers=[custom_classifier])

    print("Pipeline created with custom classifier")
    print("Ready to classify memory pairs!\n")


if __name__ == "__main__":
    asyncio.run(main())
