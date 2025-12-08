#!/usr/bin/env python3
"""
Memory Extraction Comparison Tool

Compares memory extraction across different LLM models (OpenAI and Ollama)
for both user-sourced and assistant-sourced memories.

Configuration:
    Models are defined in DEFAULT_MODELS with explicit provider type and optional
    per-model base URLs. This makes it easy to test different models and providers
    without complex command-line arguments.

Usage:
    python scripts/compare_memory_extraction.py                        # Test all models
    python scripts/compare_memory_extraction.py --mode user            # Test only user extraction
    python scripts/compare_memory_extraction.py --providers ollama     # Test only Ollama models
    python scripts/compare_memory_extraction.py --models qwen2.5:7b-instruct gemma2:9b  # Specific models
"""

import asyncio
import logging
import os
import argparse
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import datetime
import time
from dotenv import load_dotenv

load_dotenv()

# Make sure the app directory is in the python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/')))

from casual_llm import ChatMessage, UserMessage, AssistantMessage, ModelConfig, Provider, create_provider
from casual_memory import LLMMemoryExtracter


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
# Default Model Configurations
# ============================================================================
prompt = """You are an expert memory extraction agent. Your purpose is to identify and extract significant pieces of information from a conversation that should be remembered.

You will be given a conversation as a list of JSON messages.

### Instructions
1.  **Analyze the Conversation**: Carefully read the entire conversation to understand the context.
2.  **Extract Memories**: Identify any facts, preferences, goals, or significant events. Do NOT extract conversational filler (e.g., "Okay, got it", "That's great!").
3.  **Format Output**: Return a single JSON object with a top-level key "memories", containing a list of memory objects. Do not include any other text, explanations, or markdown.

For each memory you find, return a JSON object with the following fields:
- `text` (string): A concise, self-contained statement.
- `type` (string): The category of the memory. Must be one of:
    - `fact`: Objective, verifiable information (e.g., name, location, job, medical conditions, allergies, physical attributes)
    - `preference`: Subjective likes/dislikes or opinions (e.g., favorite food, enjoyment of activities, distaste for crowds)
    - `goal`: Intentions or tasks to accomplish (e.g., reminders, learning objectives, future aspirations)
    - `event`: Specific occurrences with dates (e.g., appointments, trips, birthdays, meetings)
- `tags` (list of strings): A list of relevant lowercase keywords.
- `importance` (float): A score from 0.0 to 1.0 indicating how crucial the memory is. Medical information and safety-critical facts (allergies) should be 0.9-1.0.
- `source` (string): The role of the message originator: "user", "assistant", or "tool".
- `valid_until` (string, optional): An ISO8601 timestamp for temporary memories. Today is {today_natural} (ISO: {isonow}).
    - Use this for reminders, appointments, or information that expires (e.g., "remind me tomorrow morning").

### CRITICAL RULES
- **First-Person Perspective**: For memories about the user (facts, preferences, goals), the `text` MUST be in the first person (e.g., "My name is Alex", "I am learning guitar"). Do NOT use "The user is...".
- **Self-Contained Memories**: The `text` should be understandable without the context of the conversation.
    - BAD: "Yes, that's my name."
    - GOOD: "My name is Alex."
    - BAD: "but I can't stand crowded places." (sentence fragment with "but")
    - GOOD: "I can't stand crowded places."
- **Atomic Facts**: Split compound statements into separate, focused memories:
    - If a sentence contains multiple independent facts, extract each as a separate memory
    - Example: "I'm allergic to peanuts and shellfish" → TWO memories (peanut allergy + shellfish allergy)
    - Example: "I work as a software engineer at Google" → Multiple memories (job title + employer)
    - Medical/safety information (allergies) is especially important to split for precise matching
- **State Changes**: When extracting information about changes over time, focus on the CURRENT state:
    - "I used to smoke but I quit 5 years ago" → Extract BOTH: current state ("I don't smoke") AND the change event ("I quit smoking 5 years ago")
    - The current state is usually more important than the past behavior
- **Sentiment Extraction**: When users express strong feelings (love, hate, enjoy, can't stand), extract these as separate preference memories:
    - "I went to Paris and absolutely loved it" → event ("I visited Paris") + preference ("I loved visiting Paris")
    - "I enjoy hiking but hate crowds" → TWO preferences (positive + negative)
- **Time References**: Extract temporal information as mentioned by the user, preserving times:
    - "I have a dentist appointment tomorrow at 2pm" → "I have a dentist appointment tomorrow at 2pm"
    - "I have a team meeting on Friday at 10am" → "I have a team meeting on Friday at 10am"
    - Keep relative dates ("tomorrow", "Friday", "in 2 days") - the system will normalize them
    - For `valid_until`: Leave as null - the system will calculate expiry times for temporal memories
- **No Redundancy**: Do not extract memories that are already present or are minor variations of other extracted memories in the same turn.
- **Always Return a List**: Even if you find only one memory, it MUST be inside the "memories" list.

### Examples

Example 1 - If the user says, "Pick up my prescription at the pharmacy on Thursday at 3pm.", you should return:
```json
{{
  "memories": [
    {{
      "text": "I need to pick up my prescription at the pharmacy on Thursday at 3pm.",
      "type": "goal",
      "tags": ["reminder", "pharmacy", "prescription"],
      "importance": 0.8,
      "source": "user",
      "valid_until": null
    }}
  ]
}}
```

Example 2 - If the user says, "I have a haircut appointment next Monday at 11am and dinner plans on Saturday evening.", you should return:
```json
{{
  "memories": [
    {{
      "text": "I have a haircut appointment next Monday at 11am.",
      "type": "event",
      "tags": ["appointment", "haircut"],
      "importance": 0.7,
      "source": "user",
      "valid_until": null
    }},
    {{
      "text": "I have dinner plans on Saturday in the evening.",
      "type": "event",
      "tags": ["dinner", "plans"],
      "importance": 0.8,
      "source": "user",
      "valid_until": null
    }}
  ]
}}
```

Example 3 - If the user says, "I graduated from MIT in 2020 with a degree in computer science.", you should return:
```json
{{
  "memories": [
    {{
      "text": "I graduated from MIT in 2020.",
      "type": "fact",
      "tags": ["education", "mit", "graduation"],
      "importance": 0.9,
      "source": "user",
      "valid_until": null
    }},
    {{
      "text": "I have a degree in computer science.",
      "type": "fact",
      "tags": ["education", "degree", "computer science"],
      "importance": 0.8,
      "source": "user",
      "valid_until": null
    }}
  ]
}}
```

Example 4 - If the user says, "I'm allergic to dairy and tree nuts.", you should return:
```json
{{
  "memories": [
    {{
      "text": "I am allergic to dairy.",
      "type": "fact",
      "tags": ["allergy", "dairy", "health"],
      "importance": 1.0,
      "source": "user",
      "valid_until": null
    }},
    {{
      "text": "I am allergic to tree nuts.",
      "type": "fact",
      "tags": ["allergy", "tree nuts", "health"],
      "importance": 1.0,
      "source": "user",
      "valid_until": null
    }}
  ]
}}
```

Example 5 - If the user says, "I used to drink coffee every day but I stopped 3 months ago.", you should return:
```json
{{
  "memories": [
    {{
      "text": "I don't drink coffee.",
      "type": "fact",
      "tags": ["coffee", "beverage", "habit"],
      "importance": 0.7,
      "source": "user",
      "valid_until": null
    }},
    {{
      "text": "I stopped drinking coffee 3 months ago.",
      "type": "event",
      "tags": ["coffee", "stopped", "habit"],
      "importance": 0.5,
      "source": "user",
      "valid_until": null
    }}
  ]
}}
```

Example 6 - If the user says, "I visited Tokyo last year and had an amazing time.", you should return:
```json
{{
  "memories": [
    {{
      "text": "I visited Tokyo last year.",
      "type": "event",
      "tags": ["travel", "tokyo", "vacation"],
      "importance": 0.7,
      "source": "user",
      "valid_until": null
    }},
    {{
      "text": "I had an amazing time in Tokyo.",
      "type": "preference",
      "tags": ["tokyo", "travel", "enjoyment"],
      "importance": 0.6,
      "source": "user",
      "valid_until": null
    }}
  ]
}}
```

Example 7 - If the user says, "My birthday is July 23rd.", you should return:
```json
{{
  "memories": [
    {{
      "text": "My birthday is on July 23rd.",
      "type": "fact",
      "tags": ["birthday", "personal"],
      "importance": 0.9,
      "source": "user",
      "valid_until": null
    }}
  ]
}}
```

Note: Birthdays are recurring annual events, so they don't expire - use `type: "fact"` not `"event"` and `valid_until: null`.

Example 8 - If the user says, "Remind me to call my mom next Tuesday.", you should return:
```json
{{
  "memories": [
    {{
      "text": "I need to call my mom next Tuesday.",
      "type": "goal",
      "tags": ["reminder", "call", "mom"],
      "importance": 0.7,
      "source": "user",
      "valid_until": null
    }}
  ]
}}
```

Even if there is only ONE memory, it MUST be inside the "memories" array.
"""

MODELS = [
    # OpenAI models (api_key from environment variable OPENAI_API_KEY)
    # ModelConfig(
    #     name="gpt-4o-mini",
    #     provider=Provider.OPENAI,
    #     api_key=None  # Will use OPENAI_API_KEY env var
    # ),

    # OpenAI-compatible models via OpenRouter (requires api_key or OPENAI_API_KEY)
    # ModelConfig(
    #     name="qwen/qwen-2.5-72b-instruct:free",
    #     provider=Provider.OPENAI,
    #     base_url="https://openrouter.ai/api/v1",
    #     api_key=None  # Will use OPENAI_API_KEY env var
    # ),
    # ModelConfig(
    #     name="openai/gpt-4.1-nano",
    #     provider=Provider.OPENAI,
    #     base_url="https://openrouter.ai/api/v1",
    #     api_key=config.OPENAI_API_KEY,
    # ),
    # ModelConfig(
    #     name="qwen/qwen3-next-80b-a3b-instruct-2509",
    #     provider=Provider.OPENAI,
    #     base_url="https://openrouter.ai/api/v1",
    #     api_key=os.getenv("OPENAI_API_KEY"),
    # ),
    # ModelConfig(
    #     name="qwen/qwen3-235b-a22b-2507",
    #     provider=Provider.OPENAI,
    #     base_url="https://openrouter.ai/api/v1",
    #     api_key=config.OPENAI_API_KEY,
    # ),
    # ModelConfig(
    #     name="meta-llama/llama-3.3-70b-instruct",
    #     provider=Provider.OPENAI,
    #     base_url="https://openrouter.ai/api/v1",
    #     api_key=config.OPENAI_API_KEY,
    # ),

    # Ollama models
    ModelConfig(
        name="qwen2.5:7b-instruct", 
        provider=Provider.OLLAMA, 
        base_url=os.getenv("OLLAMA_ENDPOINT")
    ),
    # ModelConfig(name="gemma2:9b", provider=Provider.OLLAMA),            # Best instruction following
    # ModelConfig(name="hermes3:latest", provider=Provider.OLLAMA),       # Previous No 2
]

# ============================================================================
# Sample Conversations
# ============================================================================

CONVERSATIONS: List[List[ChatMessage]] = [
    [
        UserMessage(content="I live in London and I prefer to get weather updates in Celsius."),
        AssistantMessage(content="Okay, I'll remember that you live in London and prefer Celsius for weather forecasts.")
    ],
    [
        UserMessage(content="What's my name?"),
        AssistantMessage(content="You haven't told me your name yet."),
        UserMessage(content="Oh, right. My name is Alex."),
        AssistantMessage(content="Got it. I'll remember that your name is Alex.")
    ],
    [
        UserMessage(content="Can you remind me to buy milk tomorrow morning?"),
        AssistantMessage(content="Sure, I'll remind you to buy milk tomorrow morning.")
    ],
    [
        UserMessage(content="I have a doctor's appointment in 2 days"),
        AssistantMessage(content="Oh, I hope everything is ok.")
    ],
    [
        UserMessage(content="I'm trying to learn how to play the guitar."),
        AssistantMessage(content="That's great! It's a wonderful instrument. I'll make a note that you're learning to play the guitar.")
    ],
    [
        UserMessage(content="I went to Paris last summer and absolutely loved it."),
        AssistantMessage(content="That sounds wonderful! I'll remember that you visited Paris.")
    ],
    [
        UserMessage(content="I have a dentist appointment tomorrow at 2pm and a team meeting on Friday at 10am."),
        AssistantMessage(content="Got it, I've noted both appointments.")
    ],
    [
        UserMessage(content="I want to visit Japan someday."),
        AssistantMessage(content="That would be amazing! I'll remember that goal.")
    ],
    [
        UserMessage(content="I work as a software engineer at Google in London."),
        AssistantMessage(content="That's great! I'll make a note of that.")
    ],
    [
        UserMessage(content="I'm allergic to peanuts and shellfish."),
        AssistantMessage(content="Important to know, I'll remember that.")
    ],
    [
        UserMessage(content="I really enjoy hiking on weekends, but I can't stand crowded places."),
        AssistantMessage(content="Got it, I'll keep that in mind.")
    ],
    [
        UserMessage(content="My birthday is next Tuesday."),
        AssistantMessage(content="I'll make a note of that!")
    ],
    [
        UserMessage(content="I used to smoke but I quit 5 years ago."),
        AssistantMessage(content="Good for you! That's an important change.")
    ]
]


# ============================================================================
# Report Generation
# ============================================================================

def format_conversation(conversation: List[ChatMessage]) -> str:
    """Format a conversation for display in markdown."""
    return "\n".join([f"- {msg.role}: {msg.content}" for msg in conversation])


def format_tags(tags: Any) -> str:
    """Format tags field for display in table."""
    if isinstance(tags, list):
        return ', '.join(tags)
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
                tags_str = format_tags(mem.get('tags', []))
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
        all_results: List of results per conversation
    """
    with open(output_path, "w", encoding="utf-8") as file:
        # Header
        file.write(f"# {title}\n")
        file.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # System prompt
        file.write(f"## Extraction System Prompt\n\n")
        file.write(system_prompt)
        file.write("\n\n")

        # Results per conversation
        converstation_number = 0
        for conversation in conversations:
            converstation_number = converstation_number + 1
            write_conversation_header(file, converstation_number, conversation)

            results = [model_result[converstation_number - 1] for model_result in model_results]
            write_results_table(file, results)

    logger.info(f"Report written to {output_path}")


# ============================================================================
# Main Execution
# ============================================================================

async def run_extraction_comparison(
    model_configs: List[ModelConfig],
    output_dir: str,
):
    """
    Run memory extraction comparison for specified mode and models.

    Args:
        model_configs: List of model configurations to test
        output_dir: Directory to write results
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M')
    results = []

    for model in model_configs:
        logger.info(f"Testing {model.name}")
        logger.info(f"Creating Provider")
        provider = create_provider(model)
        logger.info(f"Creating Memory Extractor")
        extractor = LLMMemoryExtracter(llm_provider=provider, prompt=prompt)

        model_results = []
        count = 0
        for conversation in CONVERSATIONS:
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
                    duration=duration
                )
            )

        results.append(model_results)

    output_path = os.path.join(output_dir, f"memory_comparison_{timestamp}.md")
    generate_report(
        output_path,
        "User Memory Extraction Comparison",
        prompt,
        CONVERSATIONS,
        results
    )



def main():
    """Main entry point for the comparison tool."""
    parser = argparse.ArgumentParser(
        description="Compare memory extraction across different LLM models and providers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all configured models
  python scripts/compare_memory_extraction.py

  # Override default OpenAI base URL for all OpenAI models
  OPENAI_BASE_URL=https://api.together.xyz/v1 python scripts/compare_memory_extraction.py

Configuration:
  Models are configured in DEFAULT_MODELS with explicit provider types and optional
  per-model base URLs. To add a new model, edit DEFAULT_MODELS in the script.
        """
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

    # Parse model configurations
    model_configs = MODELS

    logger.info(f"Testing {len(model_configs)} model(s): {[m.name for m in model_configs]}")
    try:
        asyncio.run(
            run_extraction_comparison(
                model_configs,
                args.output_dir,
            )
        )

        logger.info("Comparison complete!")

    except Exception as e:
        logger.error(f"Comparison failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
