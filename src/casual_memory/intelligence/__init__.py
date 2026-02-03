"""
Intelligence components for memory classification.

Provides NLI filtering, LLM-based verification, and confidence scoring utilities.
"""

from casual_memory.intelligence.confidence import (
    calculate_confidence,
    calculate_days_since,
    calculate_days_span,
)
from casual_memory.intelligence.conflict_verifier import LLMConflictVerifier
from casual_memory.intelligence.duplicate_detector import LLMDuplicateDetector
from casual_memory.intelligence.nli_filter import NLIPreFilter

__all__ = [
    # NLI
    "NLIPreFilter",
    # LLM verifiers
    "LLMConflictVerifier",
    "LLMDuplicateDetector",
    # Confidence utilities
    "calculate_confidence",
    "calculate_days_span",
    "calculate_days_since",
]
