"""
Memory-centric classification pipeline.

Classifies a new memory against similar memories using a tiered strategy.
Determines both the overall outcome for the new memory and individual
outcomes for each similar memory.
"""

import logging
from typing import Literal

from casual_memory.classifiers.models import (
    CheckType,
    MemoryClassificationResult,
    SimilarMemory,
    SimilarityResult,
)
from casual_memory.models import MemoryFact

logger = logging.getLogger(__name__)


class MemoryClassificationPipeline:
    """
    Memory-centric classification pipeline.

    Classifies a new memory against similar memories using a configurable strategy:
    - "single": Only check highest-scoring similar memory (fastest)
    - "tiered": Check highest fully, then check secondary for conflicts only (recommended)
    - "all": Check all similar memories fully (most thorough, slowest)

    The tiered strategy balances accuracy and performance:
    1. Primary (highest similarity): Full classification (conflict + duplicate check)
    2. Secondary (score ≥ secondary_threshold): Conflict check only
    3. Early stopping: Stop when finding conflict or same
    """

    def __init__(
        self,
        classifiers: list,
        strategy: Literal["single", "tiered", "all"] = "tiered",
        secondary_conflict_threshold: float = 0.90,
        max_secondary_checks: int = 3,
    ):
        """
        Initialize the memory classification pipeline.

        Args:
            classifiers: List of classifiers to run sequentially (e.g., [NLIClassifier, ConflictClassifier, DuplicateClassifier])
            strategy: Checking strategy ("single", "tiered", or "all")
            secondary_conflict_threshold: Minimum similarity score for secondary conflict checks (tiered mode)
            max_secondary_checks: Maximum number of secondary memories to check (tiered mode)
        """
        self.classifiers = classifiers
        self.strategy = strategy
        self.secondary_conflict_threshold = secondary_conflict_threshold
        self.max_secondary_checks = max_secondary_checks

        logger.info(
            f"MemoryClassificationPipeline initialized with {len(classifiers)} classifiers, "
            f"strategy={strategy}, secondary_threshold={secondary_conflict_threshold}, "
            f"max_secondary={max_secondary_checks}"
        )

    async def classify(
        self,
        new_memory: MemoryFact,
        similar_memories: list[SimilarMemory],
    ) -> MemoryClassificationResult:
        """
        Classify a new memory against similar memories.

        Args:
            new_memory: The new memory being added
            similar_memories: List of similar memories (sorted by score, highest first)

        Returns:
            MemoryClassificationResult with overall outcome and individual similarity results
        """
        if not similar_memories:
            # No similar memories - default to add
            logger.info("No similar memories found, defaulting to ADD")
            return MemoryClassificationResult(
                new_memory=new_memory,
                overall_outcome="add",
                similarity_results=[],
            )

        logger.info(
            f"Classifying new memory against {len(similar_memories)} similar memories "
            f"(strategy={self.strategy})"
        )

        # Filter similar memories based on strategy
        memories_to_check = self._filter_by_strategy(similar_memories)

        # Classify each similar memory
        similarity_results = []
        for i, similar_mem in enumerate(memories_to_check):
            # Determine check type based on position
            check_type: CheckType = "primary" if i == 0 else "secondary"

            # Run classifiers sequentially until one returns a result
            result = await self._classify_with_pipeline(
                new_memory, similar_mem, check_type
            )

            similarity_results.append(result)

            # Early stopping: Stop if we found a conflict or same
            if result.outcome in ["conflict", "same"]:
                logger.info(
                    f"Early stopping after {i+1} checks - found {result.outcome}"
                )
                break

        # Derive overall outcome from similarity results
        overall_outcome = self._derive_overall_outcome(similarity_results)

        logger.info(
            f"Classification complete: overall={overall_outcome}, "
            f"checked {len(similarity_results)}/{len(similar_memories)} similar memories"
        )

        return MemoryClassificationResult(
            new_memory=new_memory,
            overall_outcome=overall_outcome,
            similarity_results=similarity_results,
        )

    def _filter_by_strategy(
        self, similar_memories: list[SimilarMemory]
    ) -> list[SimilarMemory]:
        """
        Filter similar memories based on checking strategy.

        Args:
            similar_memories: List of similar memories (sorted by score)

        Returns:
            Filtered list based on strategy
        """
        if self.strategy == "single":
            # Only check highest similarity
            return similar_memories[:1]

        elif self.strategy == "tiered":
            # Check highest + secondary high-similarity memories
            if not similar_memories:
                return []

            # Always include highest
            filtered = [similar_memories[0]]

            # Add secondary memories above threshold
            for mem in similar_memories[1 : self.max_secondary_checks + 1]:
                if mem.similarity_score >= self.secondary_conflict_threshold:
                    filtered.append(mem)
                else:
                    break  # Sorted, so can stop once below threshold

            return filtered

        else:  # strategy == "all"
            # Check all memories
            return similar_memories

    async def _classify_with_pipeline(
        self,
        new_memory: MemoryFact,
        similar_memory: SimilarMemory,
        check_type: CheckType,
    ) -> SimilarityResult:
        """
        Run classifiers sequentially, passing results through the chain.

        Each classifier receives the current result and can:
        1. Pass it through unchanged (return existing_result)
        2. Override it with a new classification (return new result)
        3. Make initial classification if existing_result is None

        Example flow:
        - NLIClassifier: existing_result=None → returns "same" or None
        - ConflictClassifier: existing_result="same" → passes through OR existing_result=None → returns "conflict"
        - AutoResolutionClassifier: existing_result="conflict" → checks confidence → returns "superseded" or passes through
        - DuplicateClassifier: existing_result="superseded" → passes through OR existing_result=None → classifies

        Args:
            new_memory: New memory being added
            similar_memory: Similar memory to compare against
            check_type: "primary" for highest-scoring memory, "secondary" for others

        Returns:
            SimilarityResult from classifier chain or default neutral result
        """
        # Run classifiers sequentially, passing result through chain
        result = None
        for classifier in self.classifiers:
            result = await classifier.classify_pair(
                new_memory, similar_memory, check_type, result
            )

        # If no classifier provided a result, return default neutral
        if result is None:
            result = SimilarityResult(
                similar_memory=similar_memory,
                outcome="neutral",
                confidence=0.5,
                classifier_name="default",
                metadata={
                    "reason": "no_classifier_confident",
                    "check_type": check_type,
                },
            )

        return result

    def _derive_overall_outcome(
        self, results: list[SimilarityResult]
    ) -> Literal["add", "conflict", "skip"]:
        """
        Derive overall outcome from similarity results.

        Rules (priority order):
        1. If any conflict → overall = "conflict" (don't add new memory)
        2. If any same → overall = "skip" (memory already exists)
        3. Otherwise → overall = "add" (add new memory, may archive old ones)

        Args:
            results: List of similarity results

        Returns:
            Overall outcome: "add", "conflict", or "skip"
        """
        if not results:
            return "add"

        # Check for conflicts (highest priority)
        if any(r.outcome == "conflict" for r in results):
            logger.debug("Overall outcome: conflict (found conflicting memory)")
            return "conflict"

        # Check for same (second priority)
        if any(r.outcome == "same" for r in results):
            logger.debug("Overall outcome: skip (memory already exists)")
            return "skip"

        # Default: add (may have superseded or neutral memories)
        superseded_count = sum(1 for r in results if r.outcome == "superseded")
        if superseded_count > 0:
            logger.debug(
                f"Overall outcome: add (will supersede {superseded_count} memories)"
            )
        else:
            logger.debug("Overall outcome: add (all similarities neutral)")

        return "add"

    def get_metrics(self) -> dict:
        """
        Get metrics from the pipeline.

        Returns:
            Dictionary with pipeline configuration
        """
        return {
            "strategy": self.strategy,
            "secondary_conflict_threshold": self.secondary_conflict_threshold,
            "max_secondary_checks": self.max_secondary_checks,
            "classifier_count": len(self.classifiers),
            "classifiers": [
                classifier.__class__.__name__ for classifier in self.classifiers
            ],
        }
