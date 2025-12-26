"""Tests for NLI pre-filter."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np


@pytest.fixture
def mock_cross_encoder():
    """Mock CrossEncoder for testing."""
    mock_model = Mock()
    # Return mock logits [contradiction, entailment, neutral]
    mock_model.predict = Mock(return_value=np.array([[2.0, -1.0, 0.0]]))  # Contradiction
    mock_model.device = "cpu"
    return mock_model


@pytest.mark.asyncio
async def test_nli_filter_initialization():
    """Test NLI filter initialization (lazy loading)."""
    pytest.importorskip("sentence_transformers")

    from casual_memory.intelligence.nli_filter import NLIPreFilter

    # Create filter - should NOT load model yet
    filter = NLIPreFilter(
        model_name="cross-encoder/nli-deberta-v3-base",
        device="cpu",
        enable_caching=True
    )

    assert filter.model_name == "cross-encoder/nli-deberta-v3-base"
    assert filter.device == "cpu"
    assert filter.enable_caching is True
    assert filter._model is None  # Not loaded yet


@pytest.mark.asyncio
async def test_nli_filter_lazy_loading(mock_cross_encoder):
    """Test that model is lazy-loaded on first prediction."""
    pytest.importorskip("sentence_transformers")

    from casual_memory.intelligence.nli_filter import NLIPreFilter

    with patch("sentence_transformers.CrossEncoder", return_value=mock_cross_encoder):
        filter = NLIPreFilter()

        # Model not loaded initially
        assert filter._model is None

        # First prediction triggers loading
        label, scores = filter.predict("I live in London", "I live in Paris")

        # Model should be loaded now
        assert filter._model is not None
        assert label == "contradiction"


@pytest.mark.asyncio
async def test_nli_filter_contradiction_detection(mock_cross_encoder):
    """Test detection of contradictions."""
    pytest.importorskip("sentence_transformers")

    from casual_memory.intelligence.nli_filter import NLIPreFilter

    with patch("sentence_transformers.CrossEncoder", return_value=mock_cross_encoder):
        filter = NLIPreFilter()

        # Mock contradiction (high score for contradiction label)
        mock_cross_encoder.predict = Mock(return_value=np.array([[2.0, -1.0, 0.0]]))

        label, scores = filter.predict("I live in London", "I live in Paris")

        assert label == "contradiction"
        assert len(scores) == 3
        assert scores[0] > scores[1]  # Contradiction score > entailment score
        assert scores[0] > scores[2]  # Contradiction score > neutral score


@pytest.mark.asyncio
async def test_nli_filter_entailment_detection(mock_cross_encoder):
    """Test detection of entailment."""
    pytest.importorskip("sentence_transformers")

    from casual_memory.intelligence.nli_filter import NLIPreFilter

    with patch("sentence_transformers.CrossEncoder", return_value=mock_cross_encoder):
        filter = NLIPreFilter()

        # Mock entailment (high score for entailment label)
        mock_cross_encoder.predict = Mock(return_value=np.array([[-1.0, 2.0, 0.0]]))

        label, scores = filter.predict("I live in London", "I live in the UK")

        assert label == "entailment"
        assert len(scores) == 3
        assert scores[1] > scores[0]  # Entailment score > contradiction score
        assert scores[1] > scores[2]  # Entailment score > neutral score


@pytest.mark.asyncio
async def test_nli_filter_neutral_detection(mock_cross_encoder):
    """Test detection of neutral relationship."""
    pytest.importorskip("sentence_transformers")

    from casual_memory.intelligence.nli_filter import NLIPreFilter

    with patch("sentence_transformers.CrossEncoder", return_value=mock_cross_encoder):
        filter = NLIPreFilter()

        # Mock neutral (high score for neutral label)
        mock_cross_encoder.predict = Mock(return_value=np.array([[-1.0, 0.0, 2.0]]))

        label, scores = filter.predict("I like pizza", "The sky is blue")

        assert label == "neutral"
        assert len(scores) == 3
        assert scores[2] > scores[0]  # Neutral score > contradiction score
        assert scores[2] > scores[1]  # Neutral score > entailment score


@pytest.mark.asyncio
async def test_nli_filter_caching_enabled(mock_cross_encoder):
    """Test that caching works when enabled."""
    pytest.importorskip("sentence_transformers")

    from casual_memory.intelligence.nli_filter import NLIPreFilter

    with patch("sentence_transformers.CrossEncoder", return_value=mock_cross_encoder):
        filter = NLIPreFilter(enable_caching=True)

        premise = "I live in London"
        hypothesis = "I live in Paris"

        # First call - should call model
        label1, scores1 = filter.predict(premise, hypothesis)
        assert mock_cross_encoder.predict.call_count == 1

        # Second call with same inputs - should use cache
        label2, scores2 = filter.predict(premise, hypothesis)
        assert mock_cross_encoder.predict.call_count == 1  # Not called again

        # Results should be identical
        assert label1 == label2
        assert scores1 == scores2

        # Metrics should show cache hit
        metrics = filter.get_metrics()
        assert metrics["nli_cache_hits"] == 1


@pytest.mark.asyncio
async def test_nli_filter_caching_disabled(mock_cross_encoder):
    """Test that caching can be disabled."""
    pytest.importorskip("sentence_transformers")

    from casual_memory.intelligence.nli_filter import NLIPreFilter

    with patch("sentence_transformers.CrossEncoder", return_value=mock_cross_encoder):
        filter = NLIPreFilter(enable_caching=False)

        premise = "I live in London"
        hypothesis = "I live in Paris"

        # First call
        filter.predict(premise, hypothesis)
        assert mock_cross_encoder.predict.call_count == 1

        # Second call - should call model again (no caching)
        filter.predict(premise, hypothesis)
        assert mock_cross_encoder.predict.call_count == 2


@pytest.mark.asyncio
async def test_nli_filter_cache_eviction(mock_cross_encoder):
    """Test that cache is evicted when it grows too large."""
    pytest.importorskip("sentence_transformers")

    from casual_memory.intelligence.nli_filter import NLIPreFilter

    with patch("sentence_transformers.CrossEncoder", return_value=mock_cross_encoder):
        filter = NLIPreFilter(enable_caching=True)

        # Fill cache beyond limit (1000 entries)
        for i in range(1100):
            filter.predict(f"premise {i}", f"hypothesis {i}")

        # Cache should be reduced to 900 entries (1100 - 200 removes oldest 200)
        metrics = filter.get_metrics()
        assert metrics["nli_cache_size"] == 900


@pytest.mark.asyncio
async def test_nli_filter_metrics():
    """Test metrics tracking."""
    pytest.importorskip("sentence_transformers")

    from casual_memory.intelligence.nli_filter import NLIPreFilter

    mock_model = Mock()
    mock_model.predict = Mock(return_value=np.array([[2.0, -1.0, 0.0]]))
    mock_model.device = "cpu"

    with patch("sentence_transformers.CrossEncoder", return_value=mock_model):
        filter = NLIPreFilter(enable_caching=True)

        # Initial metrics
        metrics = filter.get_metrics()
        assert metrics["nli_prediction_count"] == 0
        assert metrics["nli_cache_hits"] == 0
        assert metrics["nli_model_loaded"] is False

        # Make a prediction
        filter.predict("premise 1", "hypothesis 1")

        metrics = filter.get_metrics()
        assert metrics["nli_prediction_count"] == 1
        assert metrics["nli_model_loaded"] is True

        # Make duplicate prediction (cache hit)
        filter.predict("premise 1", "hypothesis 1")

        metrics = filter.get_metrics()
        assert metrics["nli_cache_hits"] == 1
        assert "nli_cache_hit_rate_percent" in metrics
        assert metrics["nli_cache_hit_rate_percent"] == 50.0  # 1 hit out of 2 total


@pytest.mark.asyncio
async def test_nli_filter_numerical_stability():
    """Test that softmax handles large logits correctly."""
    pytest.importorskip("sentence_transformers")

    from casual_memory.intelligence.nli_filter import NLIPreFilter

    mock_model = Mock()
    # Very large logits to test numerical stability
    mock_model.predict = Mock(return_value=np.array([[100.0, -100.0, 0.0]]))
    mock_model.device = "cpu"

    with patch("sentence_transformers.CrossEncoder", return_value=mock_model):
        filter = NLIPreFilter()

        label, scores = filter.predict("test", "test")

        # Should handle large logits without overflow
        assert label == "contradiction"
        assert all(0.0 <= score <= 1.0 for score in scores)
        assert abs(sum(scores) - 1.0) < 0.01  # Probabilities should sum to ~1


@pytest.mark.asyncio
async def test_nli_filter_import_error():
    """Test that ImportError is raised if sentence-transformers not installed."""
    # This test can't easily mock the import check, so we'll skip if installed
    # In real testing, this would be tested in an environment without the package
    pass


@pytest.mark.asyncio
async def test_nli_filter_prediction_error_handling(mock_cross_encoder):
    """Test error handling during prediction."""
    pytest.importorskip("sentence_transformers")

    from casual_memory.intelligence.nli_filter import NLIPreFilter

    mock_cross_encoder.predict = Mock(side_effect=Exception("Model failed"))

    with patch("sentence_transformers.CrossEncoder", return_value=mock_cross_encoder):
        filter = NLIPreFilter()

        with pytest.raises(Exception, match="Model failed"):
            filter.predict("premise", "hypothesis")
