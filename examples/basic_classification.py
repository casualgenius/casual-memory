"""
Basic Classification Example

Demonstrates how to use the classification pipeline to classify
similar memory pairs into MERGE, CONFLICT, or ADD outcomes.
"""

import asyncio
from casual_memory.classifiers import (
    ClassificationPipeline,
    NLIClassifier,
    ConflictClassifier,
    DuplicateClassifier,
)
from casual_memory.classifiers.models import ClassificationRequest, MemoryPair
from casual_memory.intelligence import NLIPreFilter, LLMConflictVerifier, LLMDuplicateDetector
from casual_memory.models import MemoryFact
from casual_llm import create_provider, ModelConfig, Provider


async def main():
    print("=== Basic Classification Example ===\n")

    # Initialize LLM provider
    llm_provider = create_provider(ModelConfig(
        name="qwen2.5:7b-instruct",
        provider=Provider.OLLAMA,
        base_url="http://localhost:11434"
    ))

    # Initialize intelligence components
    nli_filter = NLIPreFilter()
    conflict_verifier = LLMConflictVerifier(llm_provider, "qwen2.5:7b")
    duplicate_detector = LLMDuplicateDetector(llm_provider, "qwen2.5:7b")

    # Build pipeline
    pipeline = ClassificationPipeline(classifiers=[
        NLIClassifier(nli_filter=nli_filter),
        ConflictClassifier(llm_conflict_verifier=conflict_verifier),
        DuplicateClassifier(llm_duplicate_detector=duplicate_detector),
    ])

    # Create test memory pair (conflict case)
    request = ClassificationRequest(
        pairs=[
            MemoryPair(
                existing_memory=MemoryFact(
                    text="I live in London",
                    type="fact",
                    tags=["location"],
                    importance=0.8,
                    source="user"
                ),
                new_memory=MemoryFact(
                    text="I live in Paris",
                    type="fact",
                    tags=["location"],
                    importance=0.9,
                    source="user"
                ),
                similarity_score=0.88,
                existing_memory_id="mem_001"
            )
        ],
        results=[],
        user_id="user_123"
    )

    # Classify
    result = await pipeline.classify(request)

    # Display result
    for classification in result.results:
        print(f"Classification: {classification.classification}")
        print(f"Classifier: {classification.classifier_name}")
        if classification.metadata:
            print(f"Metadata: {classification.metadata}")


if __name__ == "__main__":
    asyncio.run(main())
