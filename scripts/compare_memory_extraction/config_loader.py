"""Configuration loader for memory extraction comparison script."""

import json
import os

# Make sure we can import from casual-llm
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src/")))

from casual_llm import (
    AssistantMessage,
    ChatMessage,
    ModelConfig,
    Provider,
    UserMessage,
)

# ============================================================================
# Pydantic Models for Configuration Validation
# ============================================================================


class ModelConfigEntry(BaseModel):
    """Single model configuration entry"""

    name: str = Field(..., description="Model name/identifier")
    provider: Literal["openai", "ollama"] = Field(..., description="LLM provider type")
    base_url: Optional[str] = Field(None, description="Static base URL")
    base_url_env: Optional[str] = Field(None, description="Environment variable for base URL")
    api_key: Optional[str] = Field(None, description="Static API key (NOT recommended)")
    api_key_env: Optional[str] = Field(None, description="Environment variable for API key")
    enabled: bool = Field(True, description="Whether to include this model in tests")
    description: Optional[str] = Field(None, description="Human-readable description")


class ModelsConfig(BaseModel):
    """Container for all model configurations"""

    models: List[ModelConfigEntry]
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


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
    def load_models(cls, path: Optional[str | Path] = None) -> List[ModelConfig]:
        """
        Load model configurations from JSON file.

        Args:
            path: Path to models.json file. If None, uses default location.

        Returns:
            List of ModelConfig objects ready for use with casual-llm

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid or no enabled models found
        """
        config_path = Path(path) if path else cls.DEFAULT_CONFIG_DIR / "models.json"

        if not config_path.exists():
            error_msg = f"""
Models configuration file not found: {config_path}

To fix this:
  1. Create the default config directory:
     mkdir -p {cls.DEFAULT_CONFIG_DIR}

  2. Create a models.json file with at least one model:
     {{
       "models": [
         {{
           "name": "qwen2.5:7b-instruct",
           "provider": "ollama",
           "enabled": true
         }}
       ]
     }}

  3. Or specify a custom path:
     --models-config /path/to/models.json
            """
            raise FileNotFoundError(error_msg.strip())

        # Load and validate JSON
        try:
            with open(config_path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {config_path}: {e}")

        try:
            config = ModelsConfig(**data)
        except ValidationError as e:
            raise ValueError(f"Invalid models config structure: {e}")

        # Convert to casual-llm ModelConfig objects
        model_configs = []
        for entry in config.models:
            if not entry.enabled:
                continue

            # Resolve environment variables
            api_key = None
            if entry.api_key:
                api_key = entry.api_key
            elif entry.api_key_env:
                api_key = os.getenv(entry.api_key_env)
                if not api_key:
                    print(
                        f"Warning: Environment variable {entry.api_key_env} not set for model {entry.name}"
                    )

            base_url = None
            if entry.base_url:
                base_url = entry.base_url
            elif entry.base_url_env:
                base_url = os.getenv(entry.base_url_env)
                if not base_url:
                    print(
                        f"Warning: Environment variable {entry.base_url_env} not set for model {entry.name}"
                    )

            # Convert provider string to Provider enum
            provider_map = {
                "openai": Provider.OPENAI,
                "ollama": Provider.OLLAMA,
            }
            provider = provider_map[entry.provider]

            model_configs.append(
                ModelConfig(name=entry.name, provider=provider, base_url=base_url, api_key=api_key)
            )

        if not model_configs:
            raise ValueError(
                f"No enabled models found in {config_path}. Set 'enabled': true for at least one model."
            )

        return model_configs

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
