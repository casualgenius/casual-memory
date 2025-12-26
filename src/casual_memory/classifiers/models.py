"""
Data structures for the memory-centric classifier pipeline.

Defines the core data models used throughout the classification pipeline:
- SimilarMemory: A memory similar to the new memory being added
- SimilarityResult: Result of classifying new memory against one similar memory
- MemoryClassificationResult: Overall result for a new memory
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

from casual_memory.models import MemoryFact

# Type aliases for memory-centric classification
SimilarityOutcome = Literal["conflict", "superseded", "same", "neutral"]
MemoryOutcome = Literal["add", "conflict", "skip"]
CheckType = Literal["primary", "secondary"]


@dataclass
class SimilarMemory:
    """
    A memory similar to the new memory being added.

    Used in the memory-centric classification pipeline to represent
    existing memories that might be related to a new memory.

    Attributes:
        memory_id: ID of the memory in vector store
        memory: The memory fact itself
        similarity_score: Cosine similarity to new memory (0.0-1.0)
    """

    memory_id: str
    memory: MemoryFact
    similarity_score: float


@dataclass
class SimilarityResult:
    """
    Result of classifying a new memory against one similar memory.

    Defines what should happen to the existing similar memory.

    Attributes:
        similar_memory: The similar memory being compared
        outcome: What to do with this similar memory
            - "conflict": New memory conflicts with this one (reference in conflict record)
            - "superseded": This memory is superseded by new one (archive it)
            - "same": Memories are the same (update metadata)
            - "neutral": Memories can coexist (no action)
        confidence: Confidence score (0.0-1.0)
        classifier_name: Name of classifier that made the decision
        metadata: Additional classifier-specific info (e.g., conflict details)
    """

    similar_memory: SimilarMemory
    outcome: SimilarityOutcome
    confidence: float
    classifier_name: str
    metadata: dict = field(default_factory=dict)


@dataclass
class MemoryClassificationResult:
    """
    Overall result for classifying a new memory.

    Defines what should happen to the new memory and provides
    derived properties for action execution.

    Attributes:
        new_memory: The new memory being classified
        overall_outcome: What to do with the new memory
            - "add": Insert to vector store (may archive similar memories as side effect)
            - "conflict": Insert to conflict store, don't add to vector store
            - "skip": Do nothing (memory already exists, update existing metadata)
        similarity_results: Individual results for each similar memory
    """

    new_memory: MemoryFact
    overall_outcome: MemoryOutcome
    similarity_results: list[SimilarityResult]

    @property
    def conflicts_with(self) -> list[str]:
        """IDs of memories this conflicts with (for conflict records)."""
        return [
            r.similar_memory.memory_id for r in self.similarity_results if r.outcome == "conflict"
        ]

    @property
    def supersedes(self) -> list[str]:
        """IDs of memories to archive (when overall_outcome = "add")."""
        return [
            r.similar_memory.memory_id for r in self.similarity_results if r.outcome == "superseded"
        ]

    @property
    def same_as(self) -> Optional[str]:
        """ID of memory to update metadata (when overall_outcome = "skip")."""
        for r in self.similarity_results:
            if r.outcome == "same":
                return r.similar_memory.memory_id
        return None
