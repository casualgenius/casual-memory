"""Configuration loader for classifier benchmark scripts."""

import json
import os
from pathlib import Path
from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field, ValidationError

# Make sure we can import from casual-llm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/')))

from casual_llm import ModelConfig, Provider


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


# ============================================================================
# Configuration Loader
# ============================================================================


class ConfigLoader:
    """Loads and validates configuration files for benchmark scripts"""

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
           "name": "gpt-4o-mini",
           "provider": "openai",
           "api_key_env": "OPENAI_API_KEY",
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
                if not api_key and entry.provider != "ollama":
                    print(f"Warning: Environment variable {entry.api_key_env} not set for model {entry.name}")

            base_url = None
            if entry.base_url:
                base_url = entry.base_url
            elif entry.base_url_env:
                base_url = os.getenv(entry.base_url_env)
                if not base_url and entry.provider == "ollama":
                    print(f"Warning: Environment variable {entry.base_url_env} not set for model {entry.name}")

            # Convert provider string to Provider enum
            provider_map = {
                "openai": Provider.OPENAI,
                "ollama": Provider.OLLAMA,
            }
            provider = provider_map.get(entry.provider.lower())
            if not provider:
                raise ValueError(f"Unknown provider: {entry.provider}. Supported: openai, ollama")

            model_configs.append(
                ModelConfig(
                    name=entry.name,
                    provider=provider,
                    base_url=base_url,
                    api_key=api_key
                )
            )

        if not model_configs:
            raise ValueError(f"No enabled models found in {config_path}. Set 'enabled': true for at least one model.")

        return model_configs
