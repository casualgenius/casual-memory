#!/usr/bin/env python3
"""
Memory Extraction Comparison Tool

Compares memory extraction across different LLM models using configurable
model sets, conversation tests, and system prompts.

Configuration:
    Uses external configuration files for flexibility:
    - configs/models.json: Model configurations
    - configs/conversations.json: Test conversation pairs
    - configs/system_prompt.md: Memory extraction prompt

    All configs can be overridden via CLI arguments.

Usage:
    # Use defaults
    python run.py

    # Use custom configs
    python run.py --models-config custom_models.json

    # Use example configs
    python run.py --models-config configs/examples/models_ollama_only.json
"""

import argparse
import asyncio
import datetime
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

# Make sure the app directory is in the python path
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src/")))

from casual_llm import ChatMessage, ModelConfig, create_provider
from config_loader import ConfigLoader

from casual_memory.extractors import LLMMemoryExtracter

# Configure logging
logger = logging.getLogger("memory-comparison")
logging.basicConfig(level=logging.INFO)

# ============================================================================
# Configuration and Types
# ============================================================================


@dataclass
class ExtractionResult:
    """Result from a single memory extraction attempt"""

    model_name: str
    memories: List[Dict[str, Any]]  # List of extracted memories
    duration: float
    error: str | None = None


# ============================================================================
# Report Generation
# ============================================================================


def format_conversation(conversation: List[ChatMessage]) -> str:
    """Format a conversation for display in markdown."""
    return "\n".join([f"- {msg.role}: {msg.content}" for msg in conversation])


def format_tags(tags: Any) -> str:
    """Format tags field for display in table."""
    if isinstance(tags, list):
        return ", ".join(tags)
    return str(tags)


def write_conversation_header(f, conversation_num: int, conversation: List[ChatMessage]):
    """Write conversation header to output file."""
    f.write(f"## Conversation {conversation_num}\n\n")
    f.write("```\n")
    f.write(format_conversation(conversation))
    f.write("\n```\n\n")


def write_results_table(f, results: List[ExtractionResult]):
    """Write extraction results as a markdown table."""
    f.write("| Model | Memory | Type | Tags | Importance | Source | Valid Until | Duration |\n")
    f.write("|---|---|---|---|---|---|---|---|\n")

    for result in results:
        if result.error:
            # Show error in table
            row = (
                f"| {result.model_name} "
                f"| ERROR: {result.error} "
                f"| - | - | - | - | - "
                f"| {result.duration:.2f}s |"
            )
            f.write(row + "\n")
        elif not result.memories:
            # No memories extracted
            row = (
                f"| {result.model_name} "
                f"| (no memories extracted) "
                f"| - | - | - | - | - "
                f"| {result.duration:.2f}s |"
            )
            f.write(row + "\n")
        else:
            # Write a row for each extracted memory
            for idx, mem in enumerate(result.memories):
                tags_str = format_tags(mem.get("tags", []))
                # Only show duration on first row for this model
                duration_str = f"{result.duration:.2f}s" if idx == 0 else ""

                row = (
                    f"| {result.model_name} "
                    f"| {mem.get('text', '')} "
                    f"| {mem.get('type', '')} "
                    f"| {tags_str} "
                    f"| {mem.get('importance', '')} "
                    f"| {mem.get('source', '')} "
                    f"| {mem.get('valid_until', '')} "
                    f"| {duration_str} |"
                )
                f.write(row + "\n")

    f.write("\n---\n\n")


def generate_report(
    output_path: str,
    title: str,
    system_prompt: str,
    conversations: List[List[ChatMessage]],
    model_results: List[List[ExtractionResult]],
):
    """
    Generate a markdown report of extraction results.

    Args:
        output_path: Path to write the report
        title: Report title
        system_prompt: System prompt used for extraction
        conversations: List of conversations tested
        model_results: List of results per conversation
    """
    with open(output_path, "w", encoding="utf-8") as file:
        # Header
        file.write(f"# {title}\n")
        file.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # System prompt
        file.write("## Extraction System Prompt\n\n")
        file.write(system_prompt)
        file.write("\n\n")

        # Results per conversation
        conversation_number = 0
        for conversation in conversations:
            conversation_number = conversation_number + 1
            write_conversation_header(file, conversation_number, conversation)

            results = [model_result[conversation_number - 1] for model_result in model_results]
            write_results_table(file, results)

    logger.info(f"Report written to {output_path}")


# ============================================================================
# Main Execution
# ============================================================================


async def run_extraction_comparison(
    model_configs: List[ModelConfig],
    system_prompt: str,
    conversations: List[List[ChatMessage]],
    output_dir: str,
):
    """
    Run memory extraction comparison for specified models and conversations.

    Args:
        model_configs: List of model configurations to test
        system_prompt: System prompt for extraction
        conversations: List of test conversations
        output_dir: Directory to write results
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
    results = []

    for model in model_configs:
        logger.info(f"Testing {model.name}")
        logger.info("Creating Provider")
        provider = create_provider(model)
        logger.info("Creating Memory Extractor")
        extractor = LLMMemoryExtracter(llm_provider=provider, prompt=system_prompt)

        model_results = []
        count = 0
        for conversation in conversations:
            count = count + 1
            logger.info(f"Test Conversation {count}")
            start_time = time.time()
            memories = await extractor.extract(conversation)
            duration = time.time() - start_time

            # Return all extracted memories
            model_results.append(
                ExtractionResult(
                    model_name=model.name,
                    memories=[m.model_dump() for m in memories],
                    duration=duration,
                )
            )

        results.append(model_results)

    output_path = os.path.join(output_dir, f"memory_comparison_{timestamp}.md")
    generate_report(
        output_path, "Memory Extraction Comparison", system_prompt, conversations, results
    )


def main():
    """Main entry point for the comparison tool."""
    parser = argparse.ArgumentParser(
        description="Compare memory extraction across different LLM models and providers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default configurations
  python run.py

  # Use custom model configurations
  python run.py --models-config custom_models.json

  # Use example configs
  python run.py --models-config configs/examples/models_ollama_only.json

  # Test with minimal conversations
  python run.py --conversations-config configs/examples/conversations_simple.json

Configuration:
  Default configs are in configs/:
    - models.json: Model configurations
    - conversations.json: Test conversation pairs
    - system_prompt.md: Extraction prompt template
        """,
    )

    parser.add_argument(
        "--models-config",
        type=str,
        default=None,
        help="Path to models config JSON (default: configs/models.json)",
    )

    parser.add_argument(
        "--conversations-config",
        type=str,
        default=None,
        help="Path to conversations config JSON (default: configs/conversations.json)",
    )

    parser.add_argument(
        "--prompt-config",
        type=str,
        default=None,
        help="Path to system prompt file (default: configs/system_prompt.md)",
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

    # Load configurations
    try:
        logger.info("Loading configurations...")
        model_configs = ConfigLoader.load_models(args.models_config)
        conversations = ConfigLoader.load_conversations(args.conversations_config)
        system_prompt = ConfigLoader.load_system_prompt(args.prompt_config)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Configuration error: {e}")
        return 1

    logger.info(f"Loaded {len(model_configs)} model(s): {[m.name for m in model_configs]}")
    logger.info(f"Loaded {len(conversations)} conversation(s)")

    try:
        asyncio.run(
            run_extraction_comparison(
                model_configs,
                system_prompt,
                conversations,
                args.output_dir,
            )
        )

        logger.info("Comparison complete!")

    except Exception as e:
        logger.error(f"Comparison failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
