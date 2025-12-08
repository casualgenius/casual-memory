"""
Classifier pipeline for memory classification.

Provides a pipeline architecture for classifying memory pairs into
MERGE, CONFLICT, or ADD outcomes using multiple specialized classifiers.
"""

from casual_memory.classifiers.models import (
    MemoryPair,
    ClassificationResult,
    ClassificationRequest,
    MemoryClassifier,
)

# Import classifiers as they are extracted
# from casual_memory.classifiers.pipeline import ClassificationPipeline
# from casual_memory.classifiers.nli_classifier import NLIClassifier
# from casual_memory.classifiers.conflict_classifier import ConflictClassifier
# from casual_memory.classifiers.duplicate_classifier import DuplicateClassifier
# from casual_memory.classifiers.auto_resolution_classifier import AutoResolutionClassifier

__all__ = [
    # Data structures
    "MemoryPair",
    "ClassificationResult",
    "ClassificationRequest",
    "MemoryClassifier",
    # Pipeline (will be added as extracted)
    # "ClassificationPipeline",
    # Classifiers (will be added as extracted)
    # "NLIClassifier",
    # "ConflictClassifier",
    # "DuplicateClassifier",
    # "AutoResolutionClassifier",
]
