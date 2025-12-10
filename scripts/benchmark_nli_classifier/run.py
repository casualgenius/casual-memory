#!/usr/bin/env python3
"""
NLI Classifier Benchmarking Tool

Tests the NLI Classifier with real DeBERTa-v3 model to evaluate threshold settings
and classifier performance on various memory pair scenarios.

Usage:
    # Use defaults
    python run.py

    # Custom thresholds
    python run.py --entailment-threshold 0.80 --neutral-threshold 0.55

    # Custom test cases
    python run.py --test-cases custom_cases.json

    # Custom output directory
    python run.py --output-dir results
"""

import asyncio
import json
import logging
import os
import sys
import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/')))

from casual_memory.classifiers.nli_classifier import NLIClassifier
from casual_memory.classifiers.models import SimilarMemory
from casual_memory.models import MemoryFact

# Configure logging
logger = logging.getLogger("nli-benchmark")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@dataclass
class TestCase:
    """A single test case for NLI classifier."""
    name: str
    existing_memory: str
    new_memory: str
    expected_outcome: str  # "same", "neutral", or "pass"
    category: str
    description: str = ""


@dataclass
class BenchmarkResult:
    """Result from testing a single memory pair."""
    test_name: str
    existing_memory: str
    new_memory: str
    contradiction_score: float
    entailment_score: float
    neutral_score: float
    expected_outcome: str
    actual_outcome: str  # "same", "neutral", or "pass"
    passed: bool
    duration_ms: float
    category: str
    description: str


def load_test_cases(config_path: Optional[str] = None) -> List[TestCase]:
    """
    Load test cases from JSON file.

    Args:
        config_path: Path to test cases JSON file (default: test_cases.json in same dir)

    Returns:
        List of TestCase objects
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "test_cases.json")

    logger.info(f"Loading test cases from: {config_path}")

    with open(config_path, 'r') as f:
        data = json.load(f)

    test_cases = [
        TestCase(
            name=tc["name"],
            existing_memory=tc["existing_memory"],
            new_memory=tc["new_memory"],
            expected_outcome=tc["expected_outcome"],
            category=tc["category"],
            description=tc.get("description", "")
        )
        for tc in data["test_cases"]
    ]

    logger.info(f"Loaded {len(test_cases)} test cases")
    return test_cases


async def run_benchmark(
    test_cases: List[TestCase],
    entailment_threshold: float,
    neutral_threshold: float,
) -> List[BenchmarkResult]:
    """
    Run NLI classifier benchmark on all test cases.

    Args:
        test_cases: List of test cases to run
        entailment_threshold: Threshold for "same" classification
        neutral_threshold: Threshold for "neutral" classification

    Returns:
        List of benchmark results
    """
    logger.info(f"Initializing NLI classifier (entailment={entailment_threshold}, neutral={neutral_threshold})")

    # Initialize NLI classifier with custom thresholds
    classifier = NLIClassifier(
        entailment_threshold=entailment_threshold,
        neutral_threshold=neutral_threshold
    )

    results = []

    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"Running test {i}/{len(test_cases)}: {test_case.name}")

        # Create memory objects
        new_memory = MemoryFact(
            text=test_case.new_memory,
            type="fact",
            tags=[],
            user_id="test_user"
        )

        existing_memory_fact = MemoryFact(
            text=test_case.existing_memory,
            type="fact",
            tags=[],
            user_id="test_user"
        )

        similar_memory = SimilarMemory(
            memory_id="test_memory_1",
            memory=existing_memory_fact,
            similarity_score=0.9  # Assume high vector similarity
        )

        # Run classification with timing
        start_time = time.time()

        try:
            result = await classifier.classify_pair(
                new_memory=new_memory,
                similar_memory=similar_memory
            )

            duration_ms = (time.time() - start_time) * 1000

            # Extract scores and outcome
            if result is None:
                # Classifier passed - uncertain case
                actual_outcome = "pass"
                # Get raw scores for display
                nli_scores = {"contradiction": 0.0, "entailment": 0.0, "neutral": 0.0}
                # Need to call NLI filter directly to get scores when classifier passes
                label, scores = classifier.nli_filter.predict(
                    premise=test_case.existing_memory,
                    hypothesis=test_case.new_memory
                )
                nli_scores = {
                    "contradiction": scores[0],
                    "entailment": scores[1],
                    "neutral": scores[2]
                }
            else:
                actual_outcome = result.outcome
                nli_scores = result.metadata.get("nli_scores", {
                    "contradiction": 0.0,
                    "entailment": 0.0,
                    "neutral": 0.0
                })

            # Check if result matches expectation
            passed = (actual_outcome == test_case.expected_outcome)

            results.append(BenchmarkResult(
                test_name=test_case.name,
                existing_memory=test_case.existing_memory,
                new_memory=test_case.new_memory,
                contradiction_score=nli_scores["contradiction"],
                entailment_score=nli_scores["entailment"],
                neutral_score=nli_scores["neutral"],
                expected_outcome=test_case.expected_outcome,
                actual_outcome=actual_outcome,
                passed=passed,
                duration_ms=duration_ms,
                category=test_case.category,
                description=test_case.description
            ))

            status = "✓ PASS" if passed else "✗ FAIL"
            logger.info(
                f"  {status} - Expected: {test_case.expected_outcome}, "
                f"Actual: {actual_outcome}, "
                f"Scores: C={nli_scores['contradiction']:.3f}, "
                f"E={nli_scores['entailment']:.3f}, "
                f"N={nli_scores['neutral']:.3f}"
            )

        except Exception as e:
            logger.error(f"  Error running test {test_case.name}: {e}", exc_info=True)
            # Add failed result
            results.append(BenchmarkResult(
                test_name=test_case.name,
                existing_memory=test_case.existing_memory,
                new_memory=test_case.new_memory,
                contradiction_score=0.0,
                entailment_score=0.0,
                neutral_score=0.0,
                expected_outcome=test_case.expected_outcome,
                actual_outcome="error",
                passed=False,
                duration_ms=0.0,
                category=test_case.category,
                description=f"Error: {str(e)}"
            ))

    # Get metrics
    metrics = classifier.get_metrics()
    logger.info(f"NLI Classifier Metrics: {metrics}")

    return results


def generate_report(
    results: List[BenchmarkResult],
    entailment_threshold: float,
    neutral_threshold: float,
    output_path: str
):
    """
    Generate markdown report of benchmark results.

    Args:
        results: List of benchmark results
        entailment_threshold: Entailment threshold used
        neutral_threshold: Neutral threshold used
        output_path: Path to write report
    """
    with open(output_path, 'w') as f:
        # Header
        f.write("# NLI Classifier Benchmark Results\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Configuration
        f.write("## Configuration\n\n")
        f.write(f"- **Entailment Threshold:** {entailment_threshold}\n")
        f.write(f"- **Neutral Threshold:** {neutral_threshold}\n")
        f.write(f"- **Model:** cross-encoder/nli-deberta-v3-base\n")
        f.write(f"- **Total Test Cases:** {len(results)}\n\n")

        # Results table
        f.write("## Detailed Results\n\n")
        f.write("| Test Case | Existing Memory | New Memory | C Score | E Score | N Score | Expected | Actual | Status | Time (ms) |\n")
        f.write("|-----------|----------------|------------|---------|---------|---------|----------|--------|--------|----------|\n")

        for result in results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            f.write(
                f"| {result.test_name} | "
                f"{result.existing_memory[:30]}... | "
                f"{result.new_memory[:30]}... | "
                f"{result.contradiction_score:.3f} | "
                f"{result.entailment_score:.3f} | "
                f"{result.neutral_score:.3f} | "
                f"{result.expected_outcome} | "
                f"{result.actual_outcome} | "
                f"{status} | "
                f"{result.duration_ms:.1f} |\n"
            )

        f.write("\n")

        # Breakdown by category
        f.write("## Results by Category\n\n")
        categories = {}
        for result in results:
            if result.category not in categories:
                categories[result.category] = {"total": 0, "passed": 0}
            categories[result.category]["total"] += 1
            if result.passed:
                categories[result.category]["passed"] += 1

        for category, stats in sorted(categories.items()):
            pass_rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
            f.write(f"- **{category}**: {stats['passed']}/{stats['total']} ({pass_rate:.1f}%)\n")

        f.write("\n")

        # Summary
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        pass_rate = (passed / total * 100) if total > 0 else 0
        avg_time = sum(r.duration_ms for r in results) / total if total > 0 else 0

        f.write("## Summary\n\n")
        f.write(f"- **Total Cases:** {total}\n")
        f.write(f"- **Passed:** {passed} ({pass_rate:.1f}%)\n")
        f.write(f"- **Failed:** {failed} ({100 - pass_rate:.1f}%)\n")
        f.write(f"- **Average Time:** {avg_time:.1f}ms\n\n")

        # Failed cases detail
        if failed > 0:
            f.write("## Failed Cases\n\n")
            for result in results:
                if not result.passed:
                    f.write(f"### {result.test_name} ({result.category})\n\n")
                    f.write(f"**Description:** {result.description}\n\n")
                    f.write(f"- **Existing:** {result.existing_memory}\n")
                    f.write(f"- **New:** {result.new_memory}\n")
                    f.write(f"- **Scores:** C={result.contradiction_score:.3f}, ")
                    f.write(f"E={result.entailment_score:.3f}, ")
                    f.write(f"N={result.neutral_score:.3f}\n")
                    f.write(f"- **Expected:** {result.expected_outcome}\n")
                    f.write(f"- **Actual:** {result.actual_outcome}\n\n")

        # Recommendations
        f.write("## Threshold Recommendations\n\n")
        f.write("Based on the results above, consider:\n\n")

        # Analyze false positives/negatives
        false_same = [r for r in results if r.expected_outcome != "same" and r.actual_outcome == "same"]
        false_neutral = [r for r in results if r.expected_outcome != "neutral" and r.actual_outcome == "neutral"]
        missed_same = [r for r in results if r.expected_outcome == "same" and r.actual_outcome != "same"]
        missed_neutral = [r for r in results if r.expected_outcome == "neutral" and r.actual_outcome != "neutral"]

        if false_same:
            avg_e = sum(r.entailment_score for r in false_same) / len(false_same)
            f.write(f"- **Lower entailment threshold** if too many false 'same' classifications "
                   f"(avg E score: {avg_e:.3f})\n")

        if missed_same:
            avg_e = sum(r.entailment_score for r in missed_same) / len(missed_same)
            f.write(f"- **Raise entailment threshold** if missing 'same' classifications "
                   f"(avg E score: {avg_e:.3f})\n")

        if false_neutral:
            avg_n = sum(r.neutral_score for r in false_neutral) / len(false_neutral)
            f.write(f"- **Lower neutral threshold** if too many false 'neutral' classifications "
                   f"(avg N score: {avg_n:.3f})\n")

        if missed_neutral:
            avg_n = sum(r.neutral_score for r in missed_neutral) / len(missed_neutral)
            f.write(f"- **Raise neutral threshold** if missing 'neutral' classifications "
                   f"(avg N score: {avg_n:.3f})\n")

        if not (false_same or false_neutral or missed_same or missed_neutral):
            f.write("- Current thresholds appear to be working well! ✓\n")

        f.write("\n")

    logger.info(f"Report written to: {output_path}")


def main():
    """Main entry point for the NLI classifier benchmark tool."""
    parser = argparse.ArgumentParser(
        description="Benchmark NLI Classifier with real DeBERTa-v3 model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use defaults
  python run.py

  # Custom thresholds
  python run.py --entailment-threshold 0.80 --neutral-threshold 0.55

  # Custom test cases
  python run.py --test-cases custom_cases.json

  # Custom output directory
  python run.py --output-dir results
        """
    )

    parser.add_argument(
        "--test-cases",
        type=str,
        default=None,
        help="Path to test cases JSON file (default: test_cases.json in script dir)"
    )

    parser.add_argument(
        "--entailment-threshold",
        type=float,
        default=0.85,
        help="Entailment threshold for 'same' classification (default: 0.85)"
    )

    parser.add_argument(
        "--neutral-threshold",
        type=float,
        default=0.5,
        help="Neutral threshold for 'neutral' classification (default: 0.5)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results (default: results)"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )

    args = parser.parse_args()
    logger.setLevel(args.log_level)

    # Load test cases
    try:
        test_cases = load_test_cases(args.test_cases)
    except Exception as e:
        logger.error(f"Failed to load test cases: {e}")
        return 1

    # Run benchmark
    try:
        logger.info("Starting NLI classifier benchmark...")
        results = asyncio.run(
            run_benchmark(
                test_cases=test_cases,
                entailment_threshold=args.entailment_threshold,
                neutral_threshold=args.neutral_threshold
            )
        )
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        return 1

    # Generate report
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(args.output_dir, f"nli_benchmark_{timestamp}.md")

        generate_report(
            results=results,
            entailment_threshold=args.entailment_threshold,
            neutral_threshold=args.neutral_threshold,
            output_path=output_path
        )

        # Print summary to console
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        pass_rate = (passed / total * 100) if total > 0 else 0

        logger.info("=" * 60)
        logger.info(f"Benchmark Complete!")
        logger.info(f"Total: {total}, Passed: {passed}, Failed: {total - passed}")
        logger.info(f"Pass Rate: {pass_rate:.1f}%")
        logger.info(f"Report: {output_path}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Failed to generate report: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
