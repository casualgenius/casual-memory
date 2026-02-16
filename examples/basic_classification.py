"""
Basic Classification Example

Demonstrates how to use the classification pipeline to classify
a new memory against similar memories and determine the outcome.
"""

import asyncio

from casual_llm import ModelConfig, Provider, create_provider

from casual_memory.classifiers import (
    AutoResolutionClassifier,
    ConflictClassifier,
    DuplicateClassifier,
    MemoryClassificationPipeline,
    NLIClassifier,
    SimilarMemory,
)
from casual_memory.intelligence import LLMConflictVerifier, LLMDuplicateDetector, NLIPreFilter
from casual_memory.models import MemoryFact


async def main():
    print("=== Basic Classification Example ===\n")

    # Initialize LLM provider
    llm_provider = create_provider(
        ModelConfig(
            name="qwen2.5:7b-instruct", provider=Provider.OLLAMA, base_url="http://localhost:11434"
        )
    )

    # Initialize intelligence components
    nli_filter = NLIPreFilter()
    conflict_verifier = LLMConflictVerifier(llm_provider, "qwen2.5:7b-instruct")
    duplicate_detector = LLMDuplicateDetector(llm_provider, "qwen2.5:7b-instruct")

    # Build pipeline with memory-centric classifiers
    pipeline = MemoryClassificationPipeline(
        classifiers=[
            NLIClassifier(nli_filter=nli_filter),
            ConflictClassifier(llm_conflict_verifier=conflict_verifier),
            DuplicateClassifier(llm_duplicate_detector=duplicate_detector),
            AutoResolutionClassifier(base_supersede_threshold=1.3, base_keep_threshold=0.7),
        ],
        strategy="tiered",  # Check primary memory fully, then secondary conflicts only
    )

    # Create new memory to classify
    new_memory = MemoryFact(
        text="I live in Paris",
        type="fact",
        tags=["location"],
        importance=0.9,
        confidence=0.8,
        entity_id="user-123",
    )

    # Similar memories from vector search (these would come from your vector store)
    similar_memories = [
        SimilarMemory(
            memory_id="mem_001",
            memory=MemoryFact(
                text="I live in London",
                type="fact",
                tags=["location"],
                importance=0.8,
                confidence=0.6,
                entity_id="user-123",
            ),
            similarity_score=0.88,
        )
    ]

    print(f"New memory: {new_memory.text}")
    print(f"Similar memories: {len(similar_memories)}")
    print()

    # Classify the new memory against similar memories
    result = await pipeline.classify(new_memory, similar_memories)

    # Display results
    print(f"Overall outcome: {result.overall_outcome}")
    print()

    # Check individual similarity results
    for sim_result in result.similarity_results:
        print(f"Similar memory: {sim_result.similar_memory.memory.text}")
        print(f"  Outcome: {sim_result.outcome}")
        print(f"  Classifier: {sim_result.classifier_name}")
        print(f"  Confidence: {sim_result.confidence:.2f}")

        # If conflict, show additional metadata
        if sim_result.outcome == "conflict":
            print(f"  Category: {sim_result.metadata.get('category', 'unknown')}")
            print(f"  Hint: {sim_result.metadata.get('clarification_hint', '')}")

    print()

    # Derived properties for action execution
    print("Action summary:")
    print(f"  Conflicts with: {result.conflicts_with}")
    print(f"  Supersedes: {result.supersedes}")
    print(f"  Same as: {result.same_as}")


if __name__ == "__main__":
    asyncio.run(main())
