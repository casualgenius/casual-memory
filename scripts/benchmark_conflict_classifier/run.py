#!/usr/bin/env python3
"""
Conflict Classifier Benchmarking Tool

Tests the LLM Conflict Classifier with various models to evaluate prompt effectiveness
and classifier performance on different conflict scenarios.

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

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src/")))

from casual_llm import ClientConfig, ModelConfig, create_client, create_model

from casual_memory.intelligence.conflict_verifier import LLMConflictVerifier
from casual_memory.intelligence.prompts import CONFLICT_DETECTION_SYSTEM_PROMPT_DETAILED
from casual_memory.models import MemoryFact

# Try to import shared config loader
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
    from shared.config_loader import load_models

    HAS_CONFIG_LOADER = True
except ImportError:
    HAS_CONFIG_LOADER = False

DEFAULT_CONFIG_DIR = Path(__file__).parent / "configs"

# Configure logging
logger = logging.getLogger("conflict-benchmark")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@dataclass
class TestCase:
    """A single test case for conflict classifier."""

    name: str
    memory_a: str
    memory_b: str
    expected_conflict: bool
    category: str
    description: str = ""
    type_a: str = "fact"
    type_b: str = "fact"


@dataclass
class BenchmarkResult:
    """Result from testing a single memory pair."""

    test_name: str
    memory_a: str
    memory_b: str
    expected_conflict: bool
    actual_conflict: bool
    passed: bool
    detection_method: str
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
            expected_conflict=tc["expected_conflict"],
            category=tc["category"],
            description=tc.get("description", ""),
            type_a=tc.get("type_a", "fact"),
            type_b=tc.get("type_b", "fact"),
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
    Run conflict classifier benchmark on all test cases.

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

    # Initialize conflict verifier
    verifier = LLMConflictVerifier(
        model=model,
        enable_fallback=False,  # Disable fallback for pure LLM testing
        system_prompt=custom_prompt,
    )

    results = []

    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"Running test {i}/{len(test_cases)}: {test_case.name}")

        # Create memory objects
        memory_a = MemoryFact(text=test_case.memory_a, type=test_case.type_a, tags=[], importance=0.5, entity_id="test_user")

        memory_b = MemoryFact(text=test_case.memory_b, type=test_case.type_b, tags=[], importance=0.5, entity_id="test_user")

        # Run classification with timing
        start_time = time.time()
        llm_response = ""

        try:
            is_conflict, detection_method = await verifier.verify_conflict(
                memory_a=memory_a,
                memory_b=memory_b,
                similarity_score=0.95,  # High similarity to avoid fallback triggers
            )

            duration_ms = (time.time() - start_time) * 1000

            # Check if result matches expectation
            passed = is_conflict == test_case.expected_conflict

            results.append(
                BenchmarkResult(
                    test_name=test_case.name,
                    memory_a=test_case.memory_a,
                    memory_b=test_case.memory_b,
                    expected_conflict=test_case.expected_conflict,
                    actual_conflict=is_conflict,
                    passed=passed,
                    detection_method=detection_method,
                    duration_ms=duration_ms,
                    category=test_case.category,
                    description=test_case.description,
                    llm_response=llm_response,
                )
            )

            status = "✓ PASS" if passed else "✗ FAIL"
            logger.info(
                f"  {status} - Expected: {test_case.expected_conflict}, "
                f"Actual: {is_conflict}, Method: {detection_method}"
            )

        except Exception as e:
            logger.error(f"  Error running test {test_case.name}: {e}", exc_info=True)
            # Add failed result
            results.append(
                BenchmarkResult(
                    test_name=test_case.name,
                    memory_a=test_case.memory_a,
                    memory_b=test_case.memory_b,
                    expected_conflict=test_case.expected_conflict,
                    actual_conflict=False,
                    passed=False,
                    detection_method="error",
                    duration_ms=0.0,
                    category=test_case.category,
                    description=f"Error: {str(e)}",
                    llm_response="",
                )
            )

    # Get metrics
    metrics = verifier.get_metrics()
    logger.info(f"Conflict Verifier Metrics: {metrics}")

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
        f.write("# Conflict Classifier Benchmark Results\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Configuration
        f.write("## Configuration\n\n")
        f.write(f"- **Provider:** {client_config.provider}\n")
        f.write(f"- **Model:** {model_config.name}\n")
        f.write(f"- **Custom Prompt:** {'Yes' if custom_prompt_used else 'No (default)'}\n")
        f.write(f"- **Total Test Cases:** {len(results)}\n\n")

        # Results table
        f.write("## Detailed Results\n\n")
        f.write(
            "| Test Case | Memory A | Memory B | Expected | Actual | Status | Method | Time (ms) |\n"
        )
        f.write(
            "|-----------|----------|----------|----------|--------|--------|--------|----------|\n"
        )

        for result in results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            expected_str = "CONFLICT" if result.expected_conflict else "NO"
            actual_str = "CONFLICT" if result.actual_conflict else "NO"

            f.write(
                f"| {result.test_name} | "
                f"{result.memory_a[:25]}... | "
                f"{result.memory_b[:25]}... | "
                f"{expected_str} | "
                f"{actual_str} | "
                f"{status} | "
                f"{result.detection_method} | "
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
                    f.write(
                        f"- **Expected:** {'CONFLICT' if result.expected_conflict else 'NO CONFLICT'}\n"
                    )
                    f.write(
                        f"- **Actual:** {'CONFLICT' if result.actual_conflict else 'NO CONFLICT'}\n"
                    )
                    f.write(f"- **Detection Method:** {result.detection_method}\n\n")

        # Analysis
        f.write("## Analysis\n\n")

        false_positives = [r for r in results if not r.expected_conflict and r.actual_conflict]
        false_negatives = [r for r in results if r.expected_conflict and not r.actual_conflict]

        if false_positives:
            f.write(f"### False Positives ({len(false_positives)})\n\n")
            f.write("Cases incorrectly classified as conflicts:\n\n")
            for r in false_positives:
                f.write(f'- **{r.test_name}**: "{r.memory_a}" vs "{r.memory_b}"\n')
            f.write("\n")

        if false_negatives:
            f.write(f"### False Negatives ({len(false_negatives)})\n\n")
            f.write("Conflicts that were missed:\n\n")
            for r in false_negatives:
                f.write(f'- **{r.test_name}**: "{r.memory_a}" vs "{r.memory_b}"\n')
            f.write("\n")

        # Recommendations
        f.write("## Recommendations\n\n")

        if false_positives:
            f.write(
                "- **High false positive rate:** Consider refining prompt to be more conservative\n"
            )
            f.write("  about marking refinements and temporal changes as conflicts\n")

        if false_negatives:
            f.write("- **Missing contradictions:** Prompt may need to be more explicit about\n")
            f.write("  detecting incompatible states and values\n")

        if not (false_positives or false_negatives):
            f.write("- Current prompt is working well! ✓\n")

        f.write("\n")

    logger.info(f"Report written to: {output_path}")


def main():
    """Main entry point for the conflict classifier benchmark tool."""
    parser = argparse.ArgumentParser(
        description="Benchmark Conflict Classifier with LLM models",
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
        choices=["openai", "anthropic", "ollama"],
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
        "--use-detailed-prompt", action="store_true", help="Use the more detailed prompt variant"
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
    elif args.use_detailed_prompt:
        custom_prompt = CONFLICT_DETECTION_SYSTEM_PROMPT_DETAILED
        logger.info("Using detailed prompt variant")

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
        if not HAS_CONFIG_LOADER:
            logger.error("Config loader not available. Cannot load models config")
            return 1
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
        logger.info("Starting conflict classifier benchmark...")
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
                args.output_dir, f"conflict_benchmark_{model_safe_name}_{timestamp}.md"
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
                args.output_dir, f"conflict_benchmark_comparison_{timestamp}.md"
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
        f.write("# Conflict Classifier Model Comparison\n\n")
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
                    actual = "CONF" if result.actual_conflict else "NO"
                    f.write(f" {status} {actual} |")
                f.write("\n")

        f.write("\n")

    logger.info(f"Comparison report written to: {output_path}")


if __name__ == "__main__":
    sys.exit(main())
