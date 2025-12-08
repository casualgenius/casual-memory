"""
LLM-based duplicate/refinement detection.

Determines if two similar memories are the same fact (duplicate/refinement)
or distinct facts that should both be stored.
"""

import logging

from casual_memory.models import MemoryFact
from casual_llm import LLMProvider, UserMessage

logger = logging.getLogger(__name__)


# Duplicate/refinement detection prompt
DUPLICATE_DETECTION_PROMPT = """Are these two statements describing the same fact (one being a refinement or duplicate of the other) or are they distinct pieces of information?

Statement A: "{statement_a}"
Statement B: "{statement_b}"

Consider:

SAME FACT (duplicate/refinement):
- "I live in London" vs "I live in Central London" → SAME
- "I work as engineer" vs "I work as software engineer" → SAME
- "I like pizza" vs "I really enjoy pizza" → SAME
- "My girlfriend is Bow" vs "My girlfriend's name is Bow" → SAME

DISTINCT FACTS (different information):
- "I live in Bangkok" vs "I work in Bangkok" → DISTINCT (residence vs employment)
- "I like coffee" vs "I like tea" → DISTINCT (different preferences)
- "My name is Alex" vs "My age is 30" → DISTINCT (different attributes)
- "I know Python" vs "I know JavaScript" → DISTINCT (different skills)

Respond with ONLY one word: SAME or DISTINCT

Answer:"""


class LLMDuplicateDetector:
    """
    LLM-based duplicate/refinement detection.

    Determines if two memories are:
    - SAME: duplicate or refinement (should merge)
    - DISTINCT: separate facts (should store both)
    """

    def __init__(self, llm_provider: LLMProvider, model_name: str):
        """
        Initialize the duplicate detector.

        Args:
            llm_provider: LLM provider instance
            model_name: Name of the model (for logging)
        """
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.llm_call_count = 0
        self.llm_success_count = 0
        self.llm_failure_count = 0
        self.heuristic_fallback_count = 0

        logger.info(f"LLMDuplicateDetector initialized: model={model_name}")

    async def _call_llm(self, prompt: str):
        """
        Call the LLM for duplicate detection.

        Args:
            prompt: The prompt to send

        Returns:
            LLM response

        Raises:
            Exception: If LLM call fails
        """
        self.llm_call_count += 1
        try:
            messages = [UserMessage(content=prompt)]
            response = await self.llm_provider.chat(
                messages,
                response_format="text",
                temperature=0.1,
                max_tokens=10,  # We only need SAME or DISTINCT
            )
            self.llm_success_count += 1
            return response
        except Exception:
            self.llm_failure_count += 1
            raise

    async def is_duplicate_or_refinement(
        self, memory_a: MemoryFact, memory_b: MemoryFact, similarity_score: float
    ) -> bool:
        """
        Determine if two memories are duplicates/refinements.

        Args:
            memory_a: First memory (existing)
            memory_b: Second memory (new)
            similarity_score: Vector similarity (0-1)

        Returns:
            True if duplicates/refinements (should merge)
            False if distinct facts (should store separately)
        """
        prompt = DUPLICATE_DETECTION_PROMPT.format(
            statement_a=memory_a.text, statement_b=memory_b.text
        )

        try:
            llm_response = await self._call_llm(prompt)
            response_upper = llm_response.content.upper()
            is_same = "SAME" in response_upper

            logger.debug(
                f"Duplicate check: {'DUPLICATE' if is_same else 'DISTINCT'}\n"
                f"  A: {memory_a.text}\n"
                f"  B: {memory_b.text}\n"
                f"  Response: {llm_response.content}"
            )

            return is_same

        except Exception as e:
            # If LLM fails, use conservative heuristic:
            # Only merge if similarity is very high (0.95+)
            logger.warning(
                f"LLM duplicate detection failed: {e}. "
                f"Using conservative heuristic (merge only if similarity >= 0.95)"
            )

            self.heuristic_fallback_count += 1
            is_duplicate = similarity_score >= 0.95

            logger.info(
                f"Heuristic duplicate check: similarity={similarity_score:.3f}, "
                f"result={'DUPLICATE' if is_duplicate else 'DISTINCT'}"
            )

            return is_duplicate

    def get_metrics(self) -> dict:
        """
        Get metrics about duplicate detection.

        Returns:
            Dictionary with call counts and success rates
        """
        metrics = {
            "duplicate_detector_llm_call_count": self.llm_call_count,
            "duplicate_detector_llm_success_count": self.llm_success_count,
            "duplicate_detector_llm_failure_count": self.llm_failure_count,
            "duplicate_detector_heuristic_fallback_count": self.heuristic_fallback_count,
        }

        if self.llm_call_count > 0:
            success_rate = (self.llm_success_count / self.llm_call_count) * 100
            metrics["duplicate_detector_llm_success_rate_percent"] = round(success_rate, 2)

        return metrics
