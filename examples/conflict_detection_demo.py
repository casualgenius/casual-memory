"""
Conflict Detection Demo

Demonstrates conflict detection with LLM verifier and heuristic fallback.
"""

import asyncio

from casual_llm import ModelConfig, Provider, create_provider

from casual_memory.intelligence import LLMConflictVerifier
from casual_memory.models import MemoryFact


async def main():
    print("=== Conflict Detection Demo ===\n")

    # Initialize LLM provider
    llm_provider = create_provider(
        ModelConfig(
            name="qwen2.5:7b-instruct",
            provider=Provider.OLLAMA,
            base_url="http://localhost:11434",
            temperature=0.1,
        )
    )

    # Create conflict verifier with fallback enabled
    verifier = LLMConflictVerifier(
        llm_provider=llm_provider, model_name="qwen2.5:7b-instruct", enable_fallback=True
    )

    # Test cases
    test_cases = [
        {
            "name": "Location Conflict",
            "memory_a": MemoryFact(
                text="I live in London",
                type="fact",
                tags=["location"],
                importance=0.8,
                source="user",
            ),
            "memory_b": MemoryFact(
                text="I live in Paris",
                type="fact",
                tags=["location"],
                importance=0.9,
                source="user",
            ),
            "similarity": 0.88,
        },
        {
            "name": "Job Refinement (No Conflict)",
            "memory_a": MemoryFact(
                text="I work as an engineer",
                type="fact",
                tags=["job"],
                importance=0.7,
                source="user",
            ),
            "memory_b": MemoryFact(
                text="I work as a senior software engineer at Google",
                type="fact",
                tags=["job"],
                importance=0.9,
                source="user",
            ),
            "similarity": 0.85,
        },
        {
            "name": "Preference Negation",
            "memory_a": MemoryFact(
                text="I like coffee",
                type="preference",
                tags=["drink"],
                importance=0.6,
                source="user",
            ),
            "memory_b": MemoryFact(
                text="I don't like coffee",
                type="preference",
                tags=["drink"],
                importance=0.7,
                source="user",
            ),
            "similarity": 0.92,
        },
    ]

    # Test each case
    for i, case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {case['name']}")
        print(f"  Memory A: {case['memory_a'].text}")
        print(f"  Memory B: {case['memory_b'].text}")
        print(f"  Similarity: {case['similarity']:.2f}")

        is_conflict, method = await verifier.verify_conflict(
            case["memory_a"], case["memory_b"], case["similarity"]
        )

        print(f"  Result: {'CONFLICT' if is_conflict else 'NO CONFLICT'}")
        print(f"  Method: {method}")
        print()

    # Show metrics
    metrics = verifier.get_metrics()
    print("Metrics:")
    print(f"  Total calls: {metrics['conflict_verifier_llm_call_count']}")
    print(f"  Success rate: {metrics.get('conflict_verifier_llm_success_rate_percent', 0):.1f}%")
    print(f"  Fallback count: {metrics['conflict_verifier_fallback_count']}")


if __name__ == "__main__":
    asyncio.run(main())
