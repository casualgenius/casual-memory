"""
LLM-based conflict detection classifier.

Detects contradictions between memory pairs using LLM verification.
Passes non-conflicts to the next classifier in the pipeline.

Performance: ~500-2000ms per LLM call
Accuracy: 96.2% (with qwen3-next-80b)
"""

import logging

from casual_memory.classifiers.models import ClassificationRequest, ClassificationResult, MemoryPair
from casual_memory.intelligence.conflict_verifier import LLMConflictVerifier
from casual_memory.models import MemoryConflict

logger = logging.getLogger(__name__)


class ConflictClassifier:
    """
    LLM-based conflict detection classifier.

    Uses LLM to detect contradictions between memory pairs.
    Classifies contradictions as CONFLICT and passes non-conflicts
    to the next classifier.
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

    async def classify(self, request: ClassificationRequest) -> ClassificationRequest:
        """
        Classify pairs using LLM conflict detection.

        Classification logic:
        1. LLM detects contradiction → CONFLICT
        2. LLM confirms no contradiction → Pass to next classifier

        Args:
            request: Classification request with pairs to classify

        Returns:
            Updated request with classified pairs moved to results
        """
        unclassified_pairs = []

        for pair in request.pairs:
            try:
                # Verify conflict using LLM
                is_conflicting, detection_method = await self.verifier.verify_conflict(
                    memory_a=pair.existing_memory,
                    memory_b=pair.new_memory,
                    similarity_score=pair.similarity_score,
                )

                if is_conflicting:
                    # Conflict detected - create conflict object
                    conflict = self._create_conflict_object(pair, detection_method)

                    logger.debug(
                        f"CONFLICT detected ({detection_method}): "
                        f"{pair.existing_memory.text[:50]}... ↔ {pair.new_memory.text[:50]}..."
                    )

                    request.results.append(
                        ClassificationResult(
                            pair=pair,
                            classification="CONFLICT",
                            classifier_name=self.name,
                            metadata={
                                "detection_method": detection_method,
                                "conflict": conflict,  # Full MemoryConflict object
                            },
                        )
                    )
                else:
                    # Not a conflict - pass to duplicate classifier
                    logger.debug(
                        f"NO CONFLICT ({detection_method}): "
                        f"{pair.existing_memory.text[:50]}... ↔ {pair.new_memory.text[:50]}..."
                    )
                    unclassified_pairs.append(pair)

            except Exception as e:
                logger.error(
                    f"Conflict classifier failed for pair (passing to next classifier): {e}",
                    exc_info=True,
                )
                # On error, pass to next classifier
                unclassified_pairs.append(pair)

        # Update request with unclassified pairs
        request.pairs = unclassified_pairs

        logger.info(
            f"Conflict classifier: classified {len(request.results) - len([r for r in request.results if r.classifier_name != self.name])} pairs, "
            f"{len(unclassified_pairs)} remaining"
        )

        return request

    def _create_conflict_object(self, pair: MemoryPair, detection_method: str) -> MemoryConflict:
        """
        Create a MemoryConflict object for a conflicting pair.

        Args:
            pair: The memory pair with conflict
            detection_method: Method used to detect conflict ("llm" or "heuristic_fallback")

        Returns:
            MemoryConflict object with all metadata
        """
        memory_a = pair.existing_memory
        memory_b = pair.new_memory

        # Determine conflict category
        category = self._categorize_conflict(memory_a.text, memory_b.text)

        # Generate clarification hint
        clarification_hint = self._generate_clarification_hint(
            memory_a.text, memory_b.text, category
        )

        # Calculate average importance
        avg_importance = (memory_a.importance + memory_b.importance) / 2

        # Build metadata
        metadata = {
            "memory_a_text": memory_a.text,
            "memory_b_text": memory_b.text,
            "memory_a_type": memory_a.type,
            "memory_b_type": memory_b.type,
            "detection_method": detection_method,
        }

        return MemoryConflict(
            user_id=memory_a.user_id or "default_user",
            memory_a_id=pair.existing_memory_id,
            memory_b_id="pending",  # Will be set after insertion
            category=category,
            similarity_score=pair.similarity_score,
            avg_importance=avg_importance,
            clarification_hint=clarification_hint,
            metadata=metadata,
        )

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
