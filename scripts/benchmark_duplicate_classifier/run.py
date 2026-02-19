#!/usr/bin/env python3
"""
Duplicate Classifier Benchmarking Tool

Tests the LLM Duplicate Classifier with various models to evaluate prompt effectiveness
and classifier performance on duplicate/refinement detection scenarios.

Usage:
    # Run all enabled models from configs/models.json
    python run.py

    # Override with a single model
    python run.py --provider ollama --model llama3.2

    # Custom prompt
    python run.py --prompt custom_prompt.txt

    # Custom output directory
    python run.py --output-dir results
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

from casual_llm import ClientConfig, ModelConfig, create_client, create_model

from casual_memory.intelligence.duplicate_detector import LLMDuplicateDetector
from casual_memory.models import MemoryFact

# Import shared config loader (scripts/ must be on sys.path)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from shared.config_loader import load_models

DEFAULT_CONFIG_DIR = Path(__file__).parent / "configs"

# Configure logging
logger = logging.getLogger("duplicate-benchmark")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@dataclass
class TestCase:
    """A single test case for duplicate classifier."""

    name: str
    memory_a: str
    memory_b: str
    expected_same: bool
    category: str
    description: str = ""
    acceptable_either_way: bool = False  # If true, both SAME and DISTINCT are valid


@dataclass
class BenchmarkResult:
    """Result from testing a single memory pair."""

    test_name: str
    memory_a: str
    memory_b: str
    expected_same: bool
    actual_same: bool
    passed: bool
    duration_ms: float
    category: str
    description: str
    llm_response: str = ""


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

    with open(config_path, "r") as f:
        data = json.load(f)

    test_cases = [
        TestCase(
            name=tc["name"],
            memory_a=tc["memory_a"],
            memory_b=tc["memory_b"],
            expected_same=tc["expected_same"],
            category=tc["category"],
            description=tc.get("description", ""),
            acceptable_either_way=tc.get("acceptable_either_way", False),
        )
        for tc in data["test_cases"]
    ]

    logger.info(f"Loaded {len(test_cases)} test cases")
    return test_cases


def load_custom_prompt(prompt_path: str) -> str:
    """Load custom prompt from file."""
    logger.info(f"Loading custom prompt from: {prompt_path}")
    with open(prompt_path, "r") as f:
        return f.read()


async def run_benchmark(
    test_cases: List[TestCase],
    client_config: ClientConfig,
    model_config: ModelConfig,
    custom_prompt: Optional[str] = None,
) -> List[BenchmarkResult]:
    """
    Run duplicate classifier benchmark on all test cases.

    Args:
        test_cases: List of test cases to run
        client_config: Client configuration for LLM connection
        model_config: Model configuration
        custom_prompt: Optional custom prompt template

    Returns:
        List of benchmark results
    """
    logger.info(f"Initializing LLM: {client_config.provider}/{model_config.name}")

    # Create LLM client and model
    client = create_client(client_config)
    model = create_model(client, model_config)

    # Initialize duplicate detector
    detector = LLMDuplicateDetector(model=model, system_prompt=custom_prompt)

    results = []

    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"Running test {i}/{len(test_cases)}: {test_case.name}")

        # Create memory objects
        memory_a = MemoryFact(
            text=test_case.memory_a, type="fact", tags=[], importance=0.5, entity_id="test_user"
        )

        memory_b = MemoryFact(
            text=test_case.memory_b, type="fact", tags=[], importance=0.5, entity_id="test_user"
        )

        # Run classification with timing
        start_time = time.time()
        llm_response = ""

        try:
            is_same = await detector.is_duplicate_or_refinement(
                memory_a=memory_a,
                memory_b=memory_b,
                similarity_score=0.85,  # Mid-range similarity to avoid heuristic fallback
            )

            duration_ms = (time.time() - start_time) * 1000

            # Check if result matches expectation
            # If acceptable_either_way is true, both SAME and DISTINCT are valid
            if test_case.acceptable_either_way:
                passed = True  # Always pass when both outcomes are acceptable
            else:
                passed = is_same == test_case.expected_same

            results.append(
                BenchmarkResult(
                    test_name=test_case.name,
                    memory_a=test_case.memory_a,
                    memory_b=test_case.memory_b,
                    expected_same=test_case.expected_same,
                    actual_same=is_same,
                    passed=passed,
                    duration_ms=duration_ms,
                    category=test_case.category,
                    description=test_case.description,
                    llm_response=llm_response,
                )
            )

            status = "✓ PASS" if passed else "✗ FAIL"
            note = " (either outcome acceptable)" if test_case.acceptable_either_way else ""
            logger.info(
                f"  {status} - Expected: {test_case.expected_same}, " f"Actual: {is_same}{note}"
            )

        except Exception as e:
            logger.error(f"  Error running test {test_case.name}: {e}", exc_info=True)
            # Add failed result
            results.append(
                BenchmarkResult(
                    test_name=test_case.name,
                    memory_a=test_case.memory_a,
                    memory_b=test_case.memory_b,
                    expected_same=test_case.expected_same,
                    actual_same=False,
                    passed=False,
                    duration_ms=0.0,
                    category=test_case.category,
                    description=f"Error: {str(e)}",
                    llm_response="",
                )
            )

    # Get metrics
    metrics = detector.get_metrics()
    logger.info(f"Duplicate Detector Metrics: {metrics}")

    return results


def generate_report(
    results: List[BenchmarkResult],
    client_config: ClientConfig,
    model_config: ModelConfig,
    custom_prompt_used: bool,
    output_path: str,
):
    """
    Generate markdown report of benchmark results.

    Args:
        results: List of benchmark results
        client_config: Client configuration used
        model_config: Model configuration used
        custom_prompt_used: Whether a custom prompt was used
        output_path: Path to write report
    """
    with open(output_path, "w") as f:
        # Header
        f.write("# Duplicate Classifier Benchmark Results\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Configuration
        f.write("## Configuration\n\n")
        f.write(f"- **Provider:** {client_config.provider}\n")
        f.write(f"- **Model:** {model_config.name}\n")
        f.write(f"- **Custom Prompt:** {'Yes' if custom_prompt_used else 'No (default)'}\n")
        f.write(f"- **Total Test Cases:** {len(results)}\n\n")

        # Results table
        f.write("## Detailed Results\n\n")
        f.write("| Test Case | Memory A | Memory B | Expected | Actual | Status | Time (ms) |\n")
        f.write("|-----------|----------|----------|----------|--------|--------|-------|\n")

        for result in results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            expected_str = "SAME" if result.expected_same else "DISTINCT"
            actual_str = "SAME" if result.actual_same else "DISTINCT"

            f.write(
                f"| {result.test_name} | "
                f"{result.memory_a[:25]}... | "
                f"{result.memory_b[:25]}... | "
                f"{expected_str} | "
                f"{actual_str} | "
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
                    f.write(f"- **Memory A:** {result.memory_a}\n")
                    f.write(f"- **Memory B:** {result.memory_b}\n")
                    f.write(f"- **Expected:** {'SAME' if result.expected_same else 'DISTINCT'}\n")
                    f.write(f"- **Actual:** {'SAME' if result.actual_same else 'DISTINCT'}\n\n")

        # Analysis
        f.write("## Analysis\n\n")

        false_positives = [r for r in results if not r.expected_same and r.actual_same]
        false_negatives = [r for r in results if r.expected_same and not r.actual_same]

        if false_positives:
            f.write(f"### False Positives ({len(false_positives)})\n\n")
            f.write("Cases incorrectly classified as SAME (should be DISTINCT):\n\n")
            for r in false_positives:
                f.write(f'- **{r.test_name}**: "{r.memory_a}" vs "{r.memory_b}"\n')
            f.write("\n")

        if false_negatives:
            f.write(f"### False Negatives ({len(false_negatives)})\n\n")
            f.write("Cases incorrectly classified as DISTINCT (should be SAME):\n\n")
            for r in false_negatives:
                f.write(f'- **{r.test_name}**: "{r.memory_a}" vs "{r.memory_b}"\n')
            f.write("\n")

        # Recommendations
        f.write("## Recommendations\n\n")

        if false_positives:
            f.write("- **High false positive rate:** Prompt may be too aggressive at merging.\n")
            f.write(
                "  Consider adding examples that distinguish between similar but distinct facts.\n"
            )

        if false_negatives:
            f.write("- **Missing duplicates/refinements:** Prompt may need clearer examples\n")
            f.write("  of what constitutes a refinement vs. a distinct fact.\n")

        if not (false_positives or false_negatives):
            f.write("- Current prompt is working well! ✓\n")

        f.write("\n")

    logger.info(f"Report written to: {output_path}")


def main():
    """Main entry point for the duplicate classifier benchmark tool."""
    parser = argparse.ArgumentParser(
        description="Benchmark Duplicate Classifier with LLM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all enabled models from configs/models.json (default)
  python run.py

  # Use a custom models config
  python run.py --models-config configs/examples/models_ollama_only.json

  # Override with a single model
  python run.py --provider ollama --model llama3.2

  # Custom prompt
  python run.py --prompt custom_prompt.txt

  # Custom output directory
  python run.py --output-dir results
        """,
    )

    parser.add_argument(
        "--test-cases",
        type=str,
        default=None,
        help="Path to test cases JSON file (default: test_cases.json in script dir)",
    )

    parser.add_argument(
        "--models-config",
        type=str,
        default=None,
        help="Path to models config JSON (default: configs/models.json)",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["openai", "ollama"],
        help="LLM provider for single-model mode (overrides --models-config)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for single-model mode (overrides --models-config)",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Path to custom prompt template file (default: uses built-in prompt)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results (default: results)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()
    logger.setLevel(args.log_level)

    # Load test cases
    try:
        test_cases = load_test_cases(args.test_cases)
    except Exception as e:
        logger.error(f"Failed to load test cases: {e}")
        return 1

    # Load custom prompt if specified
    custom_prompt = None
    if args.prompt:
        try:
            custom_prompt = load_custom_prompt(args.prompt)
        except Exception as e:
            logger.error(f"Failed to load custom prompt: {e}")
            return 1

    # Determine model configs: single-model override or config file
    model_configs = []
    if args.provider or args.model:
        # Single model mode (explicit override)
        provider = args.provider or "openai"
        model_name = args.model or "gpt-4o-mini"
        base_url = os.getenv("OLLAMA_ENDPOINT") if provider == "ollama" else None
        client_config = ClientConfig(name=provider, provider=provider, base_url=base_url)
        model_config = ModelConfig(name=model_name)
        model_configs = [(client_config, model_config)]
    else:
        # Multi-model mode from config file (default)
        config_path = args.models_config or str(DEFAULT_CONFIG_DIR / "models.json")
        try:
            logger.info(f"Loading models from config: {config_path}")
            model_configs = load_models(config_path, default_config_dir=DEFAULT_CONFIG_DIR)
            logger.info(f"Loaded {len(model_configs)} model(s) for comparison")
        except Exception as e:
            logger.error(f"Failed to load models config: {e}")
            return 1

    # Run benchmark for each model
    all_results = []
    try:
        logger.info("Starting duplicate classifier benchmark...")
        for client_config, model_config in model_configs:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing model: {client_config.provider.value}/{model_config.name}")
            logger.info(f"{'='*60}\n")

            results = asyncio.run(
                run_benchmark(
                    test_cases=test_cases,
                    client_config=client_config,
                    model_config=model_config,
                    custom_prompt=custom_prompt,
                )
            )
            all_results.append((client_config, model_config, results))
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        return 1

    # Generate reports
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate individual reports for each model
        for client_config, model_config, results in all_results:
            model_safe_name = model_config.name.replace("/", "_").replace(":", "_")
            output_path = os.path.join(
                args.output_dir, f"duplicate_benchmark_{model_safe_name}_{timestamp}.md"
            )

            generate_report(
                results=results,
                client_config=client_config,
                model_config=model_config,
                custom_prompt_used=(custom_prompt is not None),
                output_path=output_path,
            )

            # Print summary for this model
            total = len(results)
            passed = sum(1 for r in results if r.passed)
            pass_rate = (passed / total * 100) if total > 0 else 0

            logger.info(f"\n{'='*60}")
            logger.info(f"Results for {client_config.provider.value}/{model_config.name}:")
            logger.info(f"Total: {total}, Passed: {passed}, Failed: {total - passed}")
            logger.info(f"Pass Rate: {pass_rate:.1f}%")
            logger.info(f"Report: {output_path}")
            logger.info(f"{'='*60}\n")

        # If multiple models, generate comparison report
        if len(all_results) > 1:
            comparison_path = os.path.join(
                args.output_dir, f"duplicate_benchmark_comparison_{timestamp}.md"
            )
            generate_comparison_report(all_results, comparison_path, custom_prompt is not None)
            logger.info(f"\nComparison report: {comparison_path}")

    except Exception as e:
        logger.error(f"Failed to generate report: {e}", exc_info=True)
        return 1

    return 0


def generate_comparison_report(
    all_results: List[tuple[ClientConfig, ModelConfig, List[BenchmarkResult]]],
    output_path: str,
    custom_prompt_used: bool,
):
    """Generate a comparison report across multiple models."""
    with open(output_path, "w") as f:
        f.write("# Duplicate Classifier Model Comparison\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Custom Prompt:** {'Yes' if custom_prompt_used else 'No (default)'}\n\n")

        # Summary table
        f.write("## Summary\n\n")
        f.write("| Model | Provider | Total | Passed | Failed | Pass Rate | Avg Time (ms) |\n")
        f.write("|-------|----------|-------|--------|--------|-----------|---------------|\n")

        for client_config, model_config, results in all_results:
            total = len(results)
            passed = sum(1 for r in results if r.passed)
            failed = total - passed
            pass_rate = (passed / total * 100) if total > 0 else 0
            avg_time = sum(r.duration_ms for r in results) / total if total > 0 else 0

            f.write(
                f"| {model_config.name} | "
                f"{client_config.provider.value} | "
                f"{total} | {passed} | {failed} | "
                f"{pass_rate:.1f}% | {avg_time:.1f} |\n"
            )

        f.write("\n")

        # Detailed comparison by test case
        f.write("## Detailed Results by Test Case\n\n")

        # Get all test case names from first model
        if all_results:
            first_results = all_results[0][2]

            f.write("| Test Case |")
            for _, model_config, _ in all_results:
                f.write(f" {model_config.name} |")
            f.write("\n")

            f.write("|-----------|")
            for _ in all_results:
                f.write("--------|")
            f.write("\n")

            for test_idx, first_result in enumerate(first_results):
                f.write(f"| {first_result.test_name} |")
                for _, model_config, results in all_results:
                    result = results[test_idx]
                    status = "✓" if result.passed else "✗"
                    actual = "SAME" if result.actual_same else "DIST"
                    f.write(f" {status} {actual} |")
                f.write("\n")

        f.write("\n")

    logger.info(f"Comparison report written to: {output_path}")


if __name__ == "__main__":
    sys.exit(main())
