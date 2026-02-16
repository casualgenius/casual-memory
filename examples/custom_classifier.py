"""
Custom Classifier Example

Demonstrates how to create a custom classifier that implements
the MemoryClassifier protocol using duck typing (no inheritance required).
"""

import asyncio
from dataclasses import dataclass
from typing import Optional

from casual_memory.classifiers import MemoryClassificationPipeline, SimilarityResult, SimilarMemory
from casual_memory.models import MemoryFact


@dataclass
class SimpleRuleClassificationContext:
    """Context for tracking classification state."""

    new_memory: MemoryFact
    similar_memories: list[SimilarMemory]
    results: list[SimilarityResult]


class SimpleRuleBasedClassifier:
    """
    Custom classifier using simple rules.

    Implements MemoryClassifier protocol via duck typing (no inheritance).

    This classifier uses a simple similarity threshold:
    - Similarity >= 0.95 → same (duplicate)
    - Similarity >= 0.90 → neutral (conservative: let other classifiers decide)
    - Similarity < 0.90 → neutral (distinct)
    """

    def __init__(self, duplicate_threshold: float = 0.95, conflict_threshold: float = 0.90):
        self.duplicate_threshold = duplicate_threshold
        self.conflict_threshold = conflict_threshold
        self.name = "SimpleRuleBasedClassifier"

    async def classify(
        self,
        new_memory: MemoryFact,
        similar_memories: list[SimilarMemory],
        results: Optional[list[SimilarityResult]] = None,
    ) -> list[SimilarityResult]:
        """
        Classify similar memories using simple threshold rules.

        Args:
            new_memory: The new memory being added
            similar_memories: List of similar memories to classify against
            results: Optional existing results (for chaining classifiers)

        Returns:
            List of SimilarityResult for each similar memory
        """
        if results is None:
            results = []

        # Get already-classified memory IDs
        classified_ids = {r.similar_memory.memory_id for r in results}

        for similar in similar_memories:
            # Skip already classified
            if similar.memory_id in classified_ids:
                continue

            # Apply simple rules
            if similar.similarity_score >= self.duplicate_threshold:
                outcome = "same"
                confidence = similar.similarity_score
            elif similar.similarity_score >= self.conflict_threshold:
                # High similarity but not duplicate - might be a conflict
                # In a real classifier, you'd do more analysis here
                outcome = "neutral"  # Conservative: let other classifiers decide
                confidence = 0.5
            else:
                outcome = "neutral"
                confidence = 1.0 - similar.similarity_score

            results.append(
                SimilarityResult(
                    similar_memory=similar,
                    outcome=outcome,
                    confidence=confidence,
                    classifier_name=self.name,
                    metadata={
                        "duplicate_threshold": self.duplicate_threshold,
                        "conflict_threshold": self.conflict_threshold,
                    },
                )
            )

        return results


class TagMatchingClassifier:
    """
    Another custom classifier that focuses on tag matching.

    Demonstrates a different classification strategy based on semantic tags.
    """

    def __init__(self, singleton_tags: Optional[list[str]] = None):
        """
        Args:
            singleton_tags: Tags where only one memory should exist (e.g., ["location", "job"])
        """
        self.singleton_tags = singleton_tags or ["location", "job", "name"]
        self.name = "TagMatchingClassifier"

    async def classify(
        self,
        new_memory: MemoryFact,
        similar_memories: list[SimilarMemory],
        results: Optional[list[SimilarityResult]] = None,
    ) -> list[SimilarityResult]:
        """Classify based on tag overlap and singleton rules."""
        if results is None:
            results = []

        classified_ids = {r.similar_memory.memory_id for r in results}
        new_tags = set(new_memory.tags)

        for similar in similar_memories:
            if similar.memory_id in classified_ids:
                continue

            existing_tags = set(similar.memory.tags)
            common_tags = new_tags & existing_tags

            # Check if any singleton tags overlap
            singleton_overlap = common_tags & set(self.singleton_tags)

            if singleton_overlap and similar.similarity_score >= 0.85:
                # Same singleton category with high similarity - potential conflict
                outcome = "conflict"
                confidence = 0.8
                metadata = {
                    "singleton_tags": list(singleton_overlap),
                    "reason": "singleton_category_conflict",
                }
            elif common_tags and similar.similarity_score >= 0.90:
                # Same tags, very high similarity - likely same fact
                outcome = "same"
                confidence = similar.similarity_score
                metadata = {"common_tags": list(common_tags)}
            else:
                outcome = "neutral"
                confidence = 0.7
                metadata = {"tag_overlap": list(common_tags)}

            results.append(
                SimilarityResult(
                    similar_memory=similar,
                    outcome=outcome,
                    confidence=confidence,
                    classifier_name=self.name,
                    metadata=metadata,
                )
            )

        return results


async def main():
    print("=== Custom Classifier Example ===\n")

    # Create custom classifiers
    rule_classifier = SimpleRuleBasedClassifier(duplicate_threshold=0.95, conflict_threshold=0.90)
    tag_classifier = TagMatchingClassifier(singleton_tags=["location", "job"])

    # Build pipeline with custom classifiers
    pipeline = MemoryClassificationPipeline(
        classifiers=[rule_classifier, tag_classifier],
        strategy="single",  # Only check highest-similarity memory
    )

    print("Pipeline created with custom classifiers:")
    print(f"  - {rule_classifier.name}")
    print(f"  - {tag_classifier.name}")
    print()

    # Create test memory and similar memories
    new_memory = MemoryFact(
        text="I work as a data scientist",
        type="fact",
        tags=["job"],
        importance=0.8,
        confidence=0.7,
        entity_id="user-123",
    )

    similar_memories = [
        SimilarMemory(
            memory_id="mem_001",
            memory=MemoryFact(
                text="I work as a software engineer",
                type="fact",
                tags=["job"],
                importance=0.8,
                confidence=0.8,
                entity_id="user-123",
            ),
            similarity_score=0.87,
        ),
    ]

    print(f"New memory: {new_memory.text}")
    print(f"Existing memory: {similar_memories[0].memory.text}")
    print(f"Similarity: {similar_memories[0].similarity_score}")
    print()

    # Classify
    result = await pipeline.classify(new_memory, similar_memories)

    print(f"Overall outcome: {result.overall_outcome}")
    print()

    for sim_result in result.similarity_results:
        print(f"Classification by: {sim_result.classifier_name}")
        print(f"  Outcome: {sim_result.outcome}")
        print(f"  Confidence: {sim_result.confidence:.2f}")
        print(f"  Metadata: {sim_result.metadata}")


if __name__ == "__main__":
    asyncio.run(main())
