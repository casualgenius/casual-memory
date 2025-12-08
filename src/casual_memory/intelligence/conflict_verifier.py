"""
LLM-based conflict verification with heuristic fallback.

Handles final verification of potential conflicts using LLM,
with graceful degradation to heuristic-based detection.
"""

import logging

from casual_memory.models import MemoryFact
from casual_llm import LLMProvider, UserMessage

logger = logging.getLogger(__name__)


# Conflict detection prompt (based on our research with 96.2% accuracy)
CONFLICT_DETECTION_PROMPT = """Do these two statements contradict each other?

Statement A: "{statement_a}"
Statement B: "{statement_b}"

Consider:
- Direct contradictions: "I live in X" vs "I live in Y" → YES
- Refinements: "I work as engineer" vs "I work as software engineer at Google" → NO
- Temporal changes: "I used to X" vs "I quit X 5 years ago" → NO (both true at different times)
- Synonyms: "software developer" vs "software engineer" → NO
- Unrelated facts: Different topics entirely → NO

Respond with ONLY one word: YES or NO

Answer:"""


class LLMConflictVerifier:
    """
    LLM-based conflict verification with heuristic fallback.

    Handles final verification of potential conflicts using LLM,
    with graceful degradation to heuristic-based detection.
    """

    def __init__(self, llm_provider: LLMProvider, model_name: str, enable_fallback: bool = True):
        """
        Initialize the LLM conflict verifier.

        Args:
            llm_provider: LLM provider instance (OpenAI, Ollama, etc.)
            model_name: Name of the model (for logging)
            enable_fallback: Enable heuristic fallback when LLM fails
        """
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.enable_fallback = enable_fallback
        self.llm_call_count = 0
        self.llm_success_count = 0
        self.llm_failure_count = 0
        self.fallback_count = 0

        logger.info(
            f"LLMConflictVerifier initialized: model={model_name}, "
            f"enable_fallback={enable_fallback}"
        )

    async def _call_llm(self, prompt: str):
        """
        Call the LLM for conflict verification.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            The LLM's response

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
                max_tokens=10,  # We only need YES or NO
            )
            self.llm_success_count += 1
            return response
        except Exception:
            self.llm_failure_count += 1
            raise

    async def verify_conflict(
        self, memory_a: MemoryFact, memory_b: MemoryFact, similarity_score: float
    ) -> tuple[bool, str]:
        """
        Verify if two memories conflict using LLM.

        Args:
            memory_a: First memory
            memory_b: Second memory
            similarity_score: Vector similarity score (for fallback)

        Returns:
            Tuple of (is_conflicting, detection_method)
            - is_conflicting: True if memories conflict
            - detection_method: "llm" or "heuristic_fallback"
        """
        prompt = CONFLICT_DETECTION_PROMPT.format(
            statement_a=memory_a.text, statement_b=memory_b.text
        )

        try:
            # Try LLM-based detection
            llm_response = await self._call_llm(prompt)
            response_upper = llm_response.content.upper()
            is_conflicting = "YES" in response_upper

            logger.debug(
                f"LLM conflict verification: {'CONFLICT' if is_conflicting else 'NO CONFLICT'}\n"
                f"  A: {memory_a.text}\n"
                f"  B: {memory_b.text}\n"
                f"  Response: {llm_response.content}"
            )

            return is_conflicting, "llm"

        except Exception as e:
            # LLM failed - use fallback if enabled
            logger.warning(
                f"LLM conflict verification failed: {e}. "
                f"Fallback={'enabled' if self.enable_fallback else 'disabled'}"
            )

            if self.enable_fallback:
                is_conflicting = self._heuristic_conflict_detection(
                    memory_a, memory_b, similarity_score
                )
                self.fallback_count += 1

                if is_conflicting:
                    logger.info(
                        f"Fallback heuristic detected conflict:\n"
                        f"  A: {memory_a.text}\n"
                        f"  B: {memory_b.text}"
                    )

                return is_conflicting, "heuristic_fallback"
            else:
                # Re-raise if fallback disabled
                raise

    def _heuristic_conflict_detection(
        self, memory_a: MemoryFact, memory_b: MemoryFact, similarity_score: float
    ) -> bool:
        """
        Fallback heuristic-based conflict detection when LLM unavailable.

        Uses simple keyword-based rules to detect likely conflicts:
        - High similarity (0.90+) + conflicting keywords = likely conflict
        - Location changes, job changes, preference negations, etc.

        Args:
            memory_a: First memory
            memory_b: Second memory
            similarity_score: Vector similarity score

        Returns:
            True if heuristic suggests conflict, False otherwise
        """
        # Require very high similarity for heuristic approach
        if similarity_score < 0.90:
            return False

        text_a = memory_a.text.lower()
        text_b = memory_b.text.lower()

        # Check for explicit negation patterns
        negation_patterns = [
            ("like", "don't like"),
            ("like", "hate"),
            ("love", "hate"),
            ("can", "can't"),
            ("can", "cannot"),
            ("will", "won't"),
            ("is", "isn't"),
            ("am", "am not"),
        ]

        for pos, neg in negation_patterns:
            if (pos in text_a and neg in text_b) or (neg in text_a and pos in text_b):
                logger.debug(f"Heuristic: negation pattern {pos}/{neg} detected")
                return True

        # Check for location conflicts
        location_indicators = ["live in", "reside in", "located in", "based in", "from"]
        if any(loc in text_a for loc in location_indicators) and any(
            loc in text_b for loc in location_indicators
        ):
            if similarity_score >= 0.92:
                logger.debug(
                    f"Heuristic: location conflict detected " f"(similarity={similarity_score:.3f})"
                )
                return True

        # Check for job/role conflicts
        job_indicators = ["work as", "job as", "employed as", "position as", "role as"]
        if any(job in text_a for job in job_indicators) and any(
            job in text_b for job in job_indicators
        ):
            if similarity_score >= 0.92:
                logger.debug(
                    f"Heuristic: job conflict detected " f"(similarity={similarity_score:.3f})"
                )
                return True

        return False

    def get_metrics(self) -> dict:
        """
        Get metrics about conflict verification.

        Returns:
            Dictionary with call counts and success rates
        """
        metrics = {
            "conflict_verifier_llm_call_count": self.llm_call_count,
            "conflict_verifier_llm_success_count": self.llm_success_count,
            "conflict_verifier_llm_failure_count": self.llm_failure_count,
            "conflict_verifier_fallback_count": self.fallback_count,
        }

        if self.llm_call_count > 0:
            success_rate = (self.llm_success_count / self.llm_call_count) * 100
            metrics["conflict_verifier_llm_success_rate_percent"] = round(success_rate, 2)

        return metrics
