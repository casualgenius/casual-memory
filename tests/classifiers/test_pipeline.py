"""
Unit tests for Classification Pipeline.

Tests the pipeline orchestration logic including:
- Sequential classifier execution
- Default handler for unclassified pairs
- Metrics aggregation
- Error propagation
"""

import pytest
from unittest.mock import Mock, AsyncMock
from casual_memory.models import MemoryFact
from casual_memory.classifiers.models import (
    MemoryPair,
    ClassificationRequest,
    ClassificationResult,
)
from casual_memory.classifiers.pipeline import ClassificationPipeline


@pytest.fixture
def sample_memory_pairs():
    """Create sample memory pairs for testing."""
    pair1 = MemoryPair(
        existing_memory=MemoryFact(
            text="I live in London",
            type="fact",
            tags=[],
            importance=0.8,
            user_id="user123",
        ),
        new_memory=MemoryFact(
            text="I live in Central London",
            type="fact",
            tags=[],
            importance=0.8,
            user_id="user123",
        ),
        similarity_score=0.92,
        existing_memory_id="mem_1",
    )

    pair2 = MemoryPair(
        existing_memory=MemoryFact(
            text="I like coffee",
            type="preference",
            tags=[],
            importance=0.5,
            user_id="user123",
        ),
        new_memory=MemoryFact(
            text="I like tea",
            type="preference",
            tags=[],
            importance=0.5,
            user_id="user123",
        ),
        similarity_score=0.87,
        existing_memory_id="mem_2",
    )

    pair3 = MemoryPair(
        existing_memory=MemoryFact(
            text="I work as a teacher",
            type="fact",
            tags=[],
            importance=0.7,
            user_id="user123",
        ),
        new_memory=MemoryFact(
            text="I work as a doctor",
            type="fact",
            tags=[],
            importance=0.7,
            user_id="user123",
        ),
        similarity_score=0.91,
        existing_memory_id="mem_3",
    )

    return [pair1, pair2, pair3]


def create_mock_classifier(name: str, classify_behavior):
    """
    Create a mock classifier.

    Args:
        name: Classifier name
        classify_behavior: Callable that takes request and returns modified request
    """
    mock = Mock()
    mock.name = name
    mock.classify = AsyncMock(side_effect=classify_behavior)
    return mock


@pytest.mark.asyncio
async def test_pipeline_sequential_execution(sample_memory_pairs):
    """Test that classifiers execute sequentially and pass pairs down."""

    # Create mock classifiers with specific behaviors
    def classifier1_behavior(request):
        # Classifies first pair as MERGE
        if request.pairs:
            pair = request.pairs[0]
            request.results.append(
                ClassificationResult(
                    pair=pair,
                    classification="MERGE",
                    classifier_name="classifier1",
                )
            )
            request.pairs = request.pairs[1:]
        return request

    def classifier2_behavior(request):
        # Classifies first remaining pair as CONFLICT
        if request.pairs:
            pair = request.pairs[0]
            request.results.append(
                ClassificationResult(
                    pair=pair,
                    classification="CONFLICT",
                    classifier_name="classifier2",
                )
            )
            request.pairs = request.pairs[1:]
        return request

    def classifier3_behavior(request):
        # Classifies remaining pairs as ADD
        for pair in request.pairs:
            request.results.append(
                ClassificationResult(
                    pair=pair,
                    classification="ADD",
                    classifier_name="classifier3",
                )
            )
        request.pairs = []
        return request

    classifier1 = create_mock_classifier("classifier1", classifier1_behavior)
    classifier2 = create_mock_classifier("classifier2", classifier2_behavior)
    classifier3 = create_mock_classifier("classifier3", classifier3_behavior)

    pipeline = ClassificationPipeline(classifiers=[classifier1, classifier2, classifier3])

    request = ClassificationRequest(
        pairs=sample_memory_pairs,
        results=[],
        user_id="user123",
    )

    result = await pipeline.classify(request)

    # All pairs should be classified
    assert len(result.results) == 3
    assert len(result.pairs) == 0

    # Check correct classifiers made decisions
    assert result.results[0].classifier_name == "classifier1"
    assert result.results[0].classification == "MERGE"

    assert result.results[1].classifier_name == "classifier2"
    assert result.results[1].classification == "CONFLICT"

    assert result.results[2].classifier_name == "classifier3"
    assert result.results[2].classification == "ADD"


@pytest.mark.asyncio
async def test_pipeline_default_handler(sample_memory_pairs):
    """Test that unclassified pairs are handled by default handler."""

    # Create classifier that doesn't classify anything
    def pass_through_behavior(request):
        # Pass all pairs through
        return request

    classifier = create_mock_classifier("pass_through", pass_through_behavior)

    pipeline = ClassificationPipeline(classifiers=[classifier])

    request = ClassificationRequest(
        pairs=sample_memory_pairs,
        results=[],
        user_id="user123",
    )

    result = await pipeline.classify(request)

    # All pairs should be classified as ADD by default handler
    assert len(result.results) == 3
    assert len(result.pairs) == 0

    for res in result.results:
        assert res.classification == "ADD"
        assert res.classifier_name == "default_handler"
        assert res.metadata["reason"] == "no_classifier_confident"


@pytest.mark.asyncio
async def test_pipeline_empty_pairs():
    """Test pipeline with no pairs to classify."""
    classifier = create_mock_classifier("classifier", lambda req: req)

    pipeline = ClassificationPipeline(classifiers=[classifier])

    request = ClassificationRequest(
        pairs=[],
        results=[],
        user_id="user123",
    )

    result = await pipeline.classify(request)

    # Should return empty results
    assert len(result.results) == 0
    assert len(result.pairs) == 0


@pytest.mark.asyncio
async def test_pipeline_already_classified():
    """Test pipeline with pairs already classified."""
    existing_result = ClassificationResult(
        pair=MemoryPair(
            existing_memory=MemoryFact(
                text="I live in London",
                type="fact",
                tags=[],
                importance=0.8,
                user_id="user123",
            ),
            new_memory=MemoryFact(
                text="I live in Paris",
                type="fact",
                tags=[],
                importance=0.8,
                user_id="user123",
            ),
            similarity_score=0.91,
            existing_memory_id="mem_1",
        ),
        classification="CONFLICT",
        classifier_name="previous_classifier",
    )

    classifier = create_mock_classifier("classifier", lambda req: req)

    pipeline = ClassificationPipeline(classifiers=[classifier])

    request = ClassificationRequest(
        pairs=[],
        results=[existing_result],
        user_id="user123",
    )

    result = await pipeline.classify(request)

    # Should keep existing results unchanged
    assert len(result.results) == 1
    assert result.results[0].classification == "CONFLICT"
    assert result.results[0].classifier_name == "previous_classifier"


@pytest.mark.asyncio
async def test_pipeline_partial_classification(sample_memory_pairs):
    """Test pipeline where some classifiers classify and others pass."""

    def classifier1_behavior(request):
        # Classify only first pair
        if request.pairs:
            pair = request.pairs[0]
            request.results.append(
                ClassificationResult(
                    pair=pair,
                    classification="MERGE",
                    classifier_name="classifier1",
                )
            )
            request.pairs = request.pairs[1:]
        return request

    def classifier2_behavior(request):
        # Pass everything through
        return request

    classifier1 = create_mock_classifier("classifier1", classifier1_behavior)
    classifier2 = create_mock_classifier("classifier2", classifier2_behavior)

    pipeline = ClassificationPipeline(classifiers=[classifier1, classifier2])

    request = ClassificationRequest(
        pairs=sample_memory_pairs,
        results=[],
        user_id="user123",
    )

    result = await pipeline.classify(request)

    # First pair classified by classifier1, rest by default handler
    assert len(result.results) == 3
    assert result.results[0].classifier_name == "classifier1"
    assert result.results[0].classification == "MERGE"

    assert result.results[1].classifier_name == "default_handler"
    assert result.results[1].classification == "ADD"

    assert result.results[2].classifier_name == "default_handler"
    assert result.results[2].classification == "ADD"


def test_pipeline_get_metrics():
    """Test that pipeline aggregates metrics from classifiers."""

    # Create mock classifiers with metrics
    classifier1 = Mock()
    classifier1.name = "classifier1"
    classifier1.get_metrics = Mock(return_value={"metric1": 10, "metric2": 20})

    classifier2 = Mock()
    classifier2.name = "classifier2"
    classifier2.get_metrics = Mock(return_value={"metric3": 30})

    classifier3 = Mock()
    classifier3.name = "classifier3"
    # No get_metrics method - simulate hasattr returning False
    del classifier3.get_metrics

    pipeline = ClassificationPipeline(classifiers=[classifier1, classifier2, classifier3])

    metrics = pipeline.get_metrics()

    # Should aggregate metrics with classifier name prefixes
    assert metrics["pipeline_classifier_count"] == 3
    assert metrics["classifier1_metric1"] == 10
    assert metrics["classifier1_metric2"] == 20
    assert metrics["classifier2_metric3"] == 30


@pytest.mark.asyncio
async def test_pipeline_classifier_order():
    """Test that classifiers execute in the correct order."""
    execution_order = []

    def create_tracking_classifier(name):
        def behavior(request):
            execution_order.append(name)
            return request

        return create_mock_classifier(name, behavior)

    classifier_a = create_tracking_classifier("a")
    classifier_b = create_tracking_classifier("b")
    classifier_c = create_tracking_classifier("c")

    pipeline = ClassificationPipeline(classifiers=[classifier_a, classifier_b, classifier_c])

    request = ClassificationRequest(
        pairs=[],
        results=[],
        user_id="user123",
    )

    await pipeline.classify(request)

    # Should execute in order
    assert execution_order == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_pipeline_breakdown_logging(sample_memory_pairs):
    """Test that pipeline calculates correct breakdown by classifier."""

    def classifier1_behavior(request):
        if request.pairs:
            pair = request.pairs[0]
            request.results.append(
                ClassificationResult(
                    pair=pair,
                    classification="MERGE",
                    classifier_name="classifier1",
                )
            )
            request.pairs = request.pairs[1:]
        return request

    def classifier2_behavior(request):
        if request.pairs:
            pair = request.pairs[0]
            request.results.append(
                ClassificationResult(
                    pair=pair,
                    classification="CONFLICT",
                    classifier_name="classifier2",
                )
            )
            request.pairs = request.pairs[1:]
        return request

    classifier1 = create_mock_classifier("classifier1", classifier1_behavior)
    classifier2 = create_mock_classifier("classifier2", classifier2_behavior)

    pipeline = ClassificationPipeline(classifiers=[classifier1, classifier2])

    request = ClassificationRequest(
        pairs=sample_memory_pairs,
        results=[],
        user_id="user123",
    )

    result = await pipeline.classify(request)

    # Get breakdown (private method, but testing behavior)
    breakdown = pipeline._get_classifier_breakdown(result)

    assert breakdown["classifier1"] == 1
    assert breakdown["classifier2"] == 1
    assert breakdown["default_handler"] == 1


@pytest.mark.asyncio
async def test_pipeline_counts_by_classification(sample_memory_pairs):
    """Test that pipeline correctly counts classifications."""

    def classifier_behavior(request):
        # Classify as different types
        if request.pairs:
            request.results.append(
                ClassificationResult(
                    pair=request.pairs[0],
                    classification="MERGE",
                    classifier_name="classifier",
                )
            )
            request.results.append(
                ClassificationResult(
                    pair=request.pairs[1],
                    classification="CONFLICT",
                    classifier_name="classifier",
                )
            )
            request.results.append(
                ClassificationResult(
                    pair=request.pairs[2],
                    classification="ADD",
                    classifier_name="classifier",
                )
            )
            request.pairs = []
        return request

    classifier = create_mock_classifier("classifier", classifier_behavior)

    pipeline = ClassificationPipeline(classifiers=[classifier])

    request = ClassificationRequest(
        pairs=sample_memory_pairs,
        results=[],
        user_id="user123",
    )

    result = await pipeline.classify(request)

    # Count each classification type
    merge_count = pipeline._count_classification(result, "MERGE")
    conflict_count = pipeline._count_classification(result, "CONFLICT")
    add_count = pipeline._count_classification(result, "ADD")

    assert merge_count == 1
    assert conflict_count == 1
    assert add_count == 1
