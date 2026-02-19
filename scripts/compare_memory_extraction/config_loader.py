"""Configuration loader for memory extraction comparison script."""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

# Import shared config loader (scripts/ must be on sys.path)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from casual_llm import (
    AssistantMessage,
    ChatMessage,
    ClientConfig,
    ModelConfig,
    UserMessage,
)
from shared.config_loader import load_models

# ============================================================================
# Pydantic Models for Conversations
# ============================================================================


class ConversationMessage(BaseModel):
    """Single message in a conversation"""

    role: Literal["user", "assistant"]
    content: str


class ConversationEntry(BaseModel):
    """Single test conversation"""

    id: str = Field(..., description="Unique identifier for this conversation")
    description: str = Field(..., description="What this conversation tests")
    enabled: bool = Field(True, description="Whether to include in tests")
    messages: List[ConversationMessage]
    expected_memories: Optional[int] = Field(None, description="Expected number of memories")
    tags: List[str] = Field(default_factory=list, description="Test case tags")


class ConversationsConfig(BaseModel):
    """Container for all test conversations"""

    conversations: List[ConversationEntry]
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


# ============================================================================
# Configuration Loader
# ============================================================================


class ConfigLoader:
    """Loads and validates configuration files for the comparison script"""

    DEFAULT_CONFIG_DIR = Path(__file__).parent / "configs"

    @classmethod
    def load_models(
        cls, path: Optional[str | Path] = None
    ) -> List[tuple[ClientConfig, ModelConfig]]:
        """
        Load model configurations from JSON file.

        Args:
            path: Path to models.json file. If None, uses default location.

        Returns:
            List of (ClientConfig, ModelConfig) tuples ready for use with casual-llm
        """
        return load_models(path, default_config_dir=cls.DEFAULT_CONFIG_DIR)

    @classmethod
    def load_conversations(cls, path: Optional[str | Path] = None) -> List[List[ChatMessage]]:
        """
        Load test conversations from JSON file.

        Args:
            path: Path to conversations.json file. If None, uses default location.

        Returns:
            List of conversation lists, where each conversation is a list of ChatMessage objects

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid or no enabled conversations found
        """
        config_path = Path(path) if path else cls.DEFAULT_CONFIG_DIR / "conversations.json"

        if not config_path.exists():
            error_msg = f"""
Conversations configuration file not found: {config_path}

To fix this:
  1. Create a conversations.json file with at least one conversation:
     {{
       "conversations": [
         {{
           "id": "test1",
           "description": "Simple test",
           "enabled": true,
           "messages": [
             {{"role": "user", "content": "My name is Alex."}},
             {{"role": "assistant", "content": "Nice to meet you, Alex!"}}
           ]
         }}
       ]
     }}

  2. Or specify a custom path:
     --conversations-config /path/to/conversations.json
            """
            raise FileNotFoundError(error_msg.strip())

        # Load and validate JSON
        try:
            with open(config_path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {config_path}: {e}")

        try:
            config = ConversationsConfig(**data)
        except ValidationError as e:
            raise ValueError(f"Invalid conversations config structure: {e}")

        # Convert to ChatMessage objects
        conversations = []
        for entry in config.conversations:
            if not entry.enabled:
                continue

            messages = []
            for msg in entry.messages:
                if msg.role == "user":
                    messages.append(UserMessage(content=msg.content))
                elif msg.role == "assistant":
                    messages.append(AssistantMessage(content=msg.content))

            conversations.append(messages)

        if not conversations:
            raise ValueError(
                f"No enabled conversations found in {config_path}. Set 'enabled': true for at least one conversation."
            )

        return conversations

    @classmethod
    def load_system_prompt(cls, path: Optional[str | Path] = None) -> str:
        """
        Load system prompt template from file.

        Args:
            path: Path to prompt file. If None, uses default location.

        Returns:
            Raw system prompt template string with {today_natural} and {isonow} placeholders
            (placeholders will be filled by LLMMemoryExtracter)

        Raises:
            FileNotFoundError: If prompt file doesn't exist
        """
        config_path = Path(path) if path else cls.DEFAULT_CONFIG_DIR / "system_prompt.md"

        if not config_path.exists():
            error_msg = f"""
System prompt file not found: {config_path}

To fix this:
  1. Create a prompt file at the default location, or

  2. Specify a custom path:
     --prompt-config /path/to/prompt.md

The prompt should include placeholders {{today_natural}} and {{isonow}} for date formatting.
            """
            raise FileNotFoundError(error_msg.strip())

        with open(config_path, encoding="utf-8") as f:
            template = f.read()

        # Return raw template - LLMMemoryExtracter will fill in the placeholders
        return template
