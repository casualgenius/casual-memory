"""
LLM-based conflict detection classifier.

Detects contradictions between memories using LLM verification.
Returns conflict outcome if detected, None otherwise.

Performance: ~500-2000ms per LLM call
Accuracy: 96.2% (with qwen3-next-80b)
"""

import logging
from typing import Optional

from casual_memory.classifiers.models import (
    CheckType,
    SimilarityResult,
    SimilarMemory,
)
from casual_memory.intelligence.conflict_verifier import LLMConflictVerifier
from casual_memory.models import MemoryFact

logger = logging.getLogger(__name__)


class ConflictClassifier:
    """
    LLM-based conflict detection classifier.

    Uses LLM to detect contradictions between memories.
    Returns conflict outcome if detected, None to pass to next classifier.
    """

    def __init__(self, llm_conflict_verifier: LLMConflictVerifier):
        """
        Initialize the conflict classifier.

        Args:
            llm_conflict_verifier: LLM-based conflict verifier instance
        """
        self.name = "conflict"
        self.verifier = llm_conflict_verifier

        logger.info("Conflict classifier initialized")

    async def classify_pair(
        self,
        new_memory: MemoryFact,
        similar_memory: SimilarMemory,
        check_type: CheckType = "primary",
        existing_result: Optional[SimilarityResult] = None,
    ) -> Optional[SimilarityResult]:
        """
        Classify a memory pair using LLM conflict detection.

        Classification logic:
        1. If existing_result is provided → pass through (conflict only does initial classification)
        2. LLM detects contradiction → conflict outcome
        3. LLM confirms no contradiction → None (pass to next classifier)

        Args:
            new_memory: New memory being added
            similar_memory: Similar memory to compare against
            check_type: Type of check ("primary" or "secondary")
                       Conflict checks both types as conflicts are important to detect
            existing_result: Result from previous classifier (if any)

        Returns:
            SimilarityResult with conflict outcome if detected, None otherwise
        """
        # If another classifier already classified, pass through
        if existing_result is not None:
            return existing_result
        try:
            # Verify conflict using LLM
            is_conflicting, detection_method = await self.verifier.verify_conflict(
                memory_a=similar_memory.memory,
                memory_b=new_memory,
                similarity_score=similar_memory.similarity_score,
            )

            if is_conflicting:
                # Conflict detected - determine category and hint
                category = self._categorize_conflict(similar_memory.memory.text, new_memory.text)
                clarification_hint = self._generate_clarification_hint(
                    similar_memory.memory.text, new_memory.text, category
                )

                # Calculate average importance
                avg_importance = (similar_memory.memory.importance + new_memory.importance) / 2

                logger.debug(
                    f"CONFLICT detected ({detection_method}, category={category}): "
                    f"{similar_memory.memory.text[:50]}... ↔ {new_memory.text[:50]}..."
                )

                return SimilarityResult(
                    similar_memory=similar_memory,
                    outcome="conflict",
                    confidence=0.9,  # High confidence from LLM
                    classifier_name=self.name,
                    metadata={
                        "detection_method": detection_method,
                        "category": category,
                        "clarification_hint": clarification_hint,
                        "avg_importance": avg_importance,
                    },
                )
            else:
                # Not a conflict - pass to next classifier
                logger.debug(
                    f"NO CONFLICT ({detection_method}): "
                    f"{similar_memory.memory.text[:50]}... ↔ {new_memory.text[:50]}..."
                )
                return None

        except Exception as e:
            logger.error(
                f"Conflict classifier failed (passing to next classifier): {e}",
                exc_info=True,
            )
            # On error, pass to next classifier
            return None

    def _categorize_conflict(self, text_a: str, text_b: str) -> str:
        """
        Determine conflict category using keyword matching.

        Args:
            text_a: First memory text
            text_b: Second memory text

        Returns:
            Category string (e.g., "location", "job", "preference")
        """
        text_combined = (text_a + " " + text_b).lower()

        # Location keywords
        if any(word in text_combined for word in ["live", "reside", "located", "city", "country"]):
            return "location"

        # Job/career keywords
        if any(word in text_combined for word in ["work", "job", "career", "employed", "position"]):
            return "job"

        # Preference keywords
        if any(word in text_combined for word in ["like", "love", "hate", "prefer", "favorite"]):
            return "preference"

        # Temporal keywords
        if any(
            word in text_combined
            for word in ["used to", "previously", "now", "currently", "before"]
        ):
            return "temporal"

        # Default category
        return "factual"

    def _generate_clarification_hint(self, text_a: str, text_b: str, category: str) -> str:
        """
        Generate a hint for user clarification.

        Args:
            text_a: First memory text
            text_b: Second memory text
            category: Conflict category

        Returns:
            Clarification hint as a question
        """
        # Category-specific hints
        category_hints = {
            "location": "Where do you currently live?",
            "job": "What is your current job or position?",
            "preference": "How do you feel about this now?",
            "temporal": "Is this still the case, or has it changed?",
        }

        return category_hints.get(
            category, "Which of these statements is more accurate or current?"
        )

    def get_metrics(self) -> dict:
        """
        Get conflict classifier metrics.

        Returns:
            Dictionary with verifier metrics (LLM call counts, success rates)
        """
        return self.verifier.get_metrics()
