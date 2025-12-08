"""
Data structures for the classifier pipeline.

Defines the core data models used throughout the classification pipeline:
- MemoryPair: A pair of memories being compared
- ClassificationResult: The outcome of classifying a pair
- ClassificationRequest: Request object passed through the pipeline
- MemoryClassifier: Protocol defining the classifier interface
"""

from dataclasses import dataclass, field
from typing import Protocol, Literal, Optional

from casual_memory.models import MemoryFact


# Type alias for classification outcomes
ClassificationOutcome = Literal["MERGE", "CONFLICT", "ADD"]


@dataclass
class MemoryPair:
    """
    A pair of memories being compared for classification.

    Attributes:
        existing_memory: The memory already stored in the system
        new_memory: The new memory being added
        similarity_score: Vector similarity score (0.0-1.0)
        existing_memory_id: Qdrant point ID of the existing memory
    """

    existing_memory: MemoryFact
    new_memory: MemoryFact
    similarity_score: float
    existing_memory_id: str  # Qdrant point ID


@dataclass
class ClassificationResult:
    """
    Result of classifying a memory pair.

    Attributes:
        pair: The memory pair that was classified
        classification: The classification outcome (MERGE, CONFLICT, or ADD)
        classifier_name: Name of the classifier that made the decision
        confidence: Optional confidence score (0.0-1.0)
        metadata: Additional classifier-specific metadata (e.g., NLI scores, detection method)
    """

    pair: MemoryPair
    classification: ClassificationOutcome
    classifier_name: str
    confidence: Optional[float] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ClassificationRequest:
    """
    Request object passed through the classifier pipeline.

    As the request flows through the pipeline:
    - Classifiers examine pairs in the `pairs` list
    - For pairs they confidently classify, they:
      1. Create a ClassificationResult
      2. Add it to the `results` list
      3. Remove the pair from `pairs`
    - Unclassified pairs remain in `pairs` for the next classifier

    Attributes:
        pairs: List of memory pairs awaiting classification
        results: List of classification results
        user_id: User ID for logging and context
    """

    pairs: list[MemoryPair]
    results: list[ClassificationResult]
    user_id: str


class MemoryClassifier(Protocol):
    """
    Protocol defining the interface for memory classifiers.

    All classifiers must implement this protocol to participate in the pipeline.
    Classifiers should only classify pairs they are confident about, passing
    uncertain pairs to the next classifier in the pipeline.

    Attributes:
        name: Classifier name for logging and result tracking
    """

    name: str

    async def classify(self, request: ClassificationRequest) -> ClassificationRequest:
        """
        Classify pairs in the request.

        This method should:
        1. Examine pairs in request.pairs
        2. For pairs it can confidently classify:
           - Create a ClassificationResult
           - Add it to request.results
           - Remove the pair from request.pairs
        3. Return the updated request

        Error Handling:
        - On error: Log the error, don't modify the request, let next classifier handle
        - Errors should not break the pipeline

        Args:
            request: The classification request containing pairs to classify

        Returns:
            Updated request with classified pairs moved from pairs to results
        """
        ...
