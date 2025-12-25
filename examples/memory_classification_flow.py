"""
End-to-end memory classification pipeline demonstration.

This example shows:
1. Creating a custom classifier pipeline with composable classifiers
2. Using the tiered strategy for performance optimization
3. Auto-resolution in action (confidence-based conflict resolution)
4. Full classification flow from memory extraction to action execution
5. How to customize the pipeline for specific use cases
"""

import asyncio
from typing import List

from casual_memory.models import MemoryFact
from casual_memory.classifiers import (
    MemoryClassificationPipeline,
    NLIClassifier,
    ConflictClassifier,
    DuplicateClassifier,
    AutoResolutionClassifier,
    SimilarMemory,
)


# Mock implementations for demonstration
class MockNLIFilter:
    """Mock NLI filter for demonstration."""

    def predict(self, premise: str, hypothesis: str):
        """Mock NLI prediction using keyword arguments."""
        # Simplified heuristic for demo
        if premise.lower() in hypothesis.lower() or hypothesis.lower() in premise.lower():
            return "entailment", [0.1, 0.9, 0.0]
        return "neutral", [0.1, 0.3, 0.6]

    def get_metrics(self):
        return {"nli_prediction_count": 0}


class MockLLMConflictVerifier:
    """Mock conflict verifier for demonstration."""

    async def verify_conflict(self, memory_a, memory_b, similarity_score):
        """Mock conflict verification."""
        # Simplified heuristic for demo
        a_text = memory_a.text.lower()
        b_text = memory_b.text.lower()

        # Check for location conflicts
        if "live in" in a_text and "live in" in b_text:
            # Extract locations
            a_location = a_text.split("live in")[-1].strip().split()[0]
            b_location = b_text.split("live in")[-1].strip().split()[0]
            if a_location != b_location:
                return True, "llm"

        return False, "llm"

    def get_metrics(self):
        return {"conflict_verifier_llm_call_count": 0}


class MockLLMDuplicateDetector:
    """Mock duplicate detector for demonstration."""

    async def is_duplicate_or_refinement(self, memory_a, memory_b, similarity_score):
        """Mock duplicate detection."""
        # Simplified heuristic for demo
        a_text = memory_a.text.lower()
        b_text = memory_b.text.lower()

        # Check if one contains the other (refinement)
        if a_text in b_text or b_text in a_text:
            return True

        return False

    def get_metrics(self):
        return {"duplicate_detector_llm_call_count": 0}


async def example_1_basic_classification():
    """Example 1: Basic classification with default pipeline."""
    print("=" * 70)
    print("Example 1: Basic Memory Classification")
    print("=" * 70)

    # Create pipeline with default classifiers
    pipeline = MemoryClassificationPipeline(
        classifiers=[
            NLIClassifier(nli_filter=MockNLIFilter()),
            ConflictClassifier(llm_conflict_verifier=MockLLMConflictVerifier()),
            DuplicateClassifier(llm_duplicate_detector=MockLLMDuplicateDetector()),
            AutoResolutionClassifier(),
        ],
        strategy="tiered",  # Use tiered strategy for performance
    )

    # New memory to classify
    new_memory = MemoryFact(
        text="I live in Paris",
        type="fact",
        tags=["location"],
        importance=0.8,
        confidence=0.9,  # High confidence
        user_id="user_123",
    )

    # Similar memories from vector search
    similar_memories = [
        SimilarMemory(
            memory_id="mem_001",
            memory=MemoryFact(
                text="I live in London",
                type="fact",
                tags=["location"],
                importance=0.8,
                confidence=0.6,  # Lower confidence
                user_id="user_123",
            ),
            similarity_score=0.92,  # High similarity
        ),
        SimilarMemory(
            memory_id="mem_002",
            memory=MemoryFact(
                text="I work in Paris",
                type="fact",
                tags=["location", "work"],
                importance=0.7,
                confidence=0.8,
                user_id="user_123",
            ),
            similarity_score=0.88,  # Medium-high similarity
        ),
    ]

    # Classify the new memory
    result = await pipeline.classify(new_memory, similar_memories)

    # Print results
    print(f"\nNew memory: {new_memory.text}")
    print(f"Overall outcome: {result.overall_outcome}")
    print(f"\nSimilarity results ({len(result.similarity_results)}):")

    for sim_result in result.similarity_results:
        print(f"\n  Similar memory: {sim_result.similar_memory.memory.text}")
        print(f"  Outcome: {sim_result.outcome}")
        print(f"  Classifier: {sim_result.classifier_name}")
        print(f"  Confidence: {sim_result.confidence:.2f}")

        if sim_result.outcome == "conflict":
            print(f"  Category: {sim_result.metadata.get('category')}")
            print(f"  Auto-resolved: {sim_result.metadata.get('auto_resolved', False)}")

    # Derived properties
    print(f"\nConflicts with: {result.conflicts_with}")
    print(f"Supersedes: {result.supersedes}")
    print(f"Same as: {result.same_as}")

    print("\n" + "=" * 70 + "\n")


async def example_2_auto_resolution():
    """Example 2: Auto-resolution in action."""
    print("=" * 70)
    print("Example 2: Auto-Resolution (Confidence-Based Conflict Resolution)")
    print("=" * 70)

    # Create pipeline with custom auto-resolution thresholds
    pipeline = MemoryClassificationPipeline(
        classifiers=[
            ConflictClassifier(llm_conflict_verifier=MockLLMConflictVerifier()),
            AutoResolutionClassifier(
                supersede_threshold=1.3,  # New confidence / old confidence >= 1.3 → supersede
                keep_threshold=0.7,  # New confidence / old confidence <= 0.7 → keep old
            ),
        ],
        strategy="single",  # Only check highest-scoring memory
    )

    # Case 1: High new confidence → auto-resolve to superseded
    print("\nCase 1: High new confidence (ratio = 0.9 / 0.5 = 1.8)")

    new_memory_high = MemoryFact(
        text="I live in Berlin",
        type="fact",
        tags=["location"],
        importance=0.8,
        confidence=0.9,  # Very high confidence
        user_id="user_123",
    )

    similar_low = SimilarMemory(
        memory_id="mem_003",
        memory=MemoryFact(
            text="I live in Munich",
            type="fact",
            tags=["location"],
            importance=0.8,
            confidence=0.5,  # Low confidence
            user_id="user_123",
        ),
        similarity_score=0.91,
    )

    result = await pipeline.classify(new_memory_high, [similar_low])
    sim_result = result.similarity_results[0]

    print(f"  Outcome: {sim_result.outcome}")
    print(f"  Auto-resolved: {sim_result.metadata.get('auto_resolved')}")
    print(f"  Resolution: {sim_result.metadata.get('resolution_decision')}")
    print(f"  Confidence ratio: {sim_result.metadata.get('confidence_ratio', 0):.2f}")

    # Case 2: High old confidence → auto-resolve to same (keep old)
    print("\nCase 2: High old confidence (ratio = 0.4 / 0.9 = 0.44)")

    new_memory_low = MemoryFact(
        text="I live in Hamburg",
        type="fact",
        tags=["location"],
        importance=0.8,
        confidence=0.4,  # Low confidence
        user_id="user_123",
    )

    similar_high = SimilarMemory(
        memory_id="mem_004",
        memory=MemoryFact(
            text="I live in Berlin",
            type="fact",
            tags=["location"],
            importance=0.8,
            confidence=0.9,  # Very high confidence
            user_id="user_123",
        ),
        similarity_score=0.90,
    )

    result = await pipeline.classify(new_memory_low, [similar_high])
    sim_result = result.similarity_results[0]

    print(f"  Outcome: {sim_result.outcome}")
    print(f"  Auto-resolved: {sim_result.metadata.get('auto_resolved')}")
    print(f"  Resolution: {sim_result.metadata.get('resolution_decision')}")
    print(f"  Confidence ratio: {sim_result.metadata.get('confidence_ratio', 0):.2f}")

    # Case 3: Similar confidence → keep as conflict
    print("\nCase 3: Similar confidence (ratio = 0.8 / 0.8 = 1.0)")

    new_memory_mid = MemoryFact(
        text="I live in Frankfurt",
        type="fact",
        tags=["location"],
        importance=0.8,
        confidence=0.8,  # Medium confidence
        user_id="user_123",
    )

    similar_mid = SimilarMemory(
        memory_id="mem_005",
        memory=MemoryFact(
            text="I live in Stuttgart",
            type="fact",
            tags=["location"],
            importance=0.8,
            confidence=0.8,  # Same confidence
            user_id="user_123",
        ),
        similarity_score=0.89,
    )

    result = await pipeline.classify(new_memory_mid, [similar_mid])
    sim_result = result.similarity_results[0]

    print(f"  Outcome: {sim_result.outcome}")
    print(f"  Auto-resolved: {sim_result.metadata.get('auto_resolved')}")
    print(f"  Confidence ratio: {sim_result.metadata.get('confidence_ratio', 0):.2f}")

    print("\n" + "=" * 70 + "\n")


async def example_3_custom_pipeline():
    """Example 3: Custom classifier pipeline for specific use case."""
    print("=" * 70)
    print("Example 3: Custom Classifier Pipeline (Conflict-Only)")
    print("=" * 70)

    # Custom pipeline: Only detect conflicts, skip duplicates
    # Useful for scenarios where you only care about contradictions
    pipeline = MemoryClassificationPipeline(
        classifiers=[
            ConflictClassifier(llm_conflict_verifier=MockLLMConflictVerifier()),
            AutoResolutionClassifier(),
        ],
        strategy="all",  # Check all similar memories for conflicts
    )

    new_memory = MemoryFact(
        text="I work as a teacher",
        type="fact",
        tags=["job"],
        importance=0.7,
        confidence=0.8,
        user_id="user_123",
    )

    similar_memories = [
        SimilarMemory(
            memory_id="mem_006",
            memory=MemoryFact(
                text="I work as a doctor",
                type="fact",
                tags=["job"],
                importance=0.7,
                confidence=0.8,
                user_id="user_123",
            ),
            similarity_score=0.88,
        ),
        SimilarMemory(
            memory_id="mem_007",
            memory=MemoryFact(
                text="I like teaching",
                type="preference",
                tags=["job"],
                importance=0.5,
                confidence=0.7,
                user_id="user_123",
            ),
            similarity_score=0.82,
        ),
    ]

    result = await pipeline.classify(new_memory, similar_memories)

    print(f"\nNew memory: {new_memory.text}")
    print(f"Overall outcome: {result.overall_outcome}")
    print(f"\nClassification results:")

    for sim_result in result.similarity_results:
        print(f"\n  Similar: {sim_result.similar_memory.memory.text}")
        print(f"  Outcome: {sim_result.outcome}")
        print(f"  Classifier: {sim_result.classifier_name}")

    print("\n" + "=" * 70 + "\n")


async def example_4_strategy_comparison():
    """Example 4: Comparing different strategies."""
    print("=" * 70)
    print("Example 4: Strategy Comparison (Single vs Tiered vs All)")
    print("=" * 70)

    new_memory = MemoryFact(
        text="I enjoy hiking",
        type="preference",
        tags=["hobby"],
        importance=0.6,
        confidence=0.7,
        user_id="user_123",
    )

    similar_memories = [
        SimilarMemory(
            memory_id="mem_008",
            memory=MemoryFact(
                text="I enjoy hiking in the mountains",
                type="preference",
                tags=["hobby"],
                importance=0.6,
                confidence=0.8,
                user_id="user_123",
            ),
            similarity_score=0.95,
        ),
        SimilarMemory(
            memory_id="mem_009",
            memory=MemoryFact(
                text="I like outdoor activities",
                type="preference",
                tags=["hobby"],
                importance=0.6,
                confidence=0.7,
                user_id="user_123",
            ),
            similarity_score=0.91,
        ),
        SimilarMemory(
            memory_id="mem_010",
            memory=MemoryFact(
                text="I go hiking every weekend",
                type="fact",
                tags=["hobby"],
                importance=0.5,
                confidence=0.6,
                user_id="user_123",
            ),
            similarity_score=0.87,
        ),
    ]

    classifiers = [
        NLIClassifier(nli_filter=MockNLIFilter()),
        DuplicateClassifier(llm_duplicate_detector=MockLLMDuplicateDetector()),
    ]

    for strategy in ["single", "tiered", "all"]:
        pipeline = MemoryClassificationPipeline(
            classifiers=classifiers, strategy=strategy
        )

        result = await pipeline.classify(new_memory, similar_memories)

        print(f"\nStrategy: {strategy}")
        print(f"  Checked: {len(result.similarity_results)} memories")
        print(f"  Overall outcome: {result.overall_outcome}")

        for sim_result in result.similarity_results:
            print(
                f"    - {sim_result.similar_memory.memory.text[:40]}... → {sim_result.outcome}"
            )

    print("\n" + "=" * 70 + "\n")


async def example_5_full_flow():
    """Example 5: Complete end-to-end flow."""
    print("=" * 70)
    print("Example 5: Complete End-to-End Memory Classification Flow")
    print("=" * 70)

    print("\n1. Create classification pipeline")
    pipeline = MemoryClassificationPipeline(
        classifiers=[
            NLIClassifier(nli_filter=MockNLIFilter()),
            ConflictClassifier(llm_conflict_verifier=MockLLMConflictVerifier()),
            DuplicateClassifier(llm_duplicate_detector=MockLLMDuplicateDetector()),
            AutoResolutionClassifier(),
        ],
        strategy="tiered",
        secondary_conflict_threshold=0.90,
        max_secondary_checks=3,
    )

    print("\n2. Extract new memory from conversation")
    new_memory = MemoryFact(
        text="I live in Tokyo",
        type="fact",
        tags=["location"],
        importance=0.8,
        confidence=0.85,
        user_id="user_123",
    )
    print(f"   New memory: {new_memory.text}")

    print("\n3. Get embedding and search for similar memories")
    # In production: embedding = embedding_service.get_embedding(new_memory.text)
    # In production: similar_memories = vector_store.search(embedding, limit=5, threshold=0.85)
    similar_memories = [
        SimilarMemory(
            memory_id="mem_011",
            memory=MemoryFact(
                text="I live in Kyoto",
                type="fact",
                tags=["location"],
                importance=0.8,
                confidence=0.7,
                user_id="user_123",
            ),
            similarity_score=0.93,
        )
    ]
    print(f"   Found {len(similar_memories)} similar memories")

    print("\n4. Classify new memory against similar memories")
    result = await pipeline.classify(new_memory, similar_memories)
    print(f"   Overall outcome: {result.overall_outcome}")

    print("\n5. Execute actions based on classification result")
    if result.overall_outcome == "add":
        print("   → Add new memory to vector store")
        if result.supersedes:
            print(f"   → Archive memories: {result.supersedes}")
    elif result.overall_outcome == "skip":
        print(f"   → Update existing memory: {result.same_as}")
        print("   → Increment mention_count and update last_seen")
    elif result.overall_outcome == "conflict":
        print(f"   → Create conflict record with memories: {result.conflicts_with}")
        print("   → Flag for user resolution")

    print("\n6. Get pipeline metrics")
    metrics = pipeline.get_metrics()
    print(f"   Strategy: {metrics['strategy']}")
    print(f"   Classifiers: {metrics['classifiers']}")

    print("\n" + "=" * 70 + "\n")


async def main():
    """Run all examples."""
    await example_1_basic_classification()
    await example_2_auto_resolution()
    await example_3_custom_pipeline()
    await example_4_strategy_comparison()
    await example_5_full_flow()


if __name__ == "__main__":
    asyncio.run(main())
