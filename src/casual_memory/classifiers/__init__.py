"""
Memory-centric classifier pipeline for memory classification.

Provides a pipeline architecture for classifying new memories against similar memories,
determining both overall outcomes (add/conflict/skip) and individual similarity outcomes
(conflict/superseded/same/neutral) using multiple specialized classifiers.
"""

from casual_memory.classifiers.auto_resolution_classifier import (
    AutoResolutionClassifier,
)
from casual_memory.classifiers.conflict_classifier import ConflictClassifier
from casual_memory.classifiers.duplicate_classifier import DuplicateClassifier
from casual_memory.classifiers.models import (
    CheckType,
    MemoryClassificationResult,
    MemoryOutcome,
    SimilarityOutcome,
    SimilarityResult,
    SimilarMemory,
)
from casual_memory.classifiers.nli_classifier import NLIClassifier
from casual_memory.classifiers.pipeline import MemoryClassificationPipeline

__all__ = [
    # Data structures
    "CheckType",
    "MemoryClassificationResult",
    "SimilarMemory",
    "SimilarityOutcome",
    "MemoryOutcome",
    "SimilarityResult",
    # Pipeline
    "MemoryClassificationPipeline",
    # Classifiers
    "NLIClassifier",
    "ConflictClassifier",
    "DuplicateClassifier",
    "AutoResolutionClassifier",
]
