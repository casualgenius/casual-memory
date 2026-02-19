"""Shared configuration loader for benchmark and comparison scripts."""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src/")))

from casual_llm import ClientConfig, ModelConfig

# ============================================================================
# Pydantic Models for Configuration Validation
# ============================================================================


class ClientConfigEntry(BaseModel):
    """Reusable client connection configuration.

    The key in the 'clients' dict becomes the ClientConfig.name, which enables
    automatic API key resolution from {NAME.upper()}_API_KEY env vars.
    """

    provider: str = Field(..., description="LLM provider: 'openai', 'ollama', or 'anthropic'")
    base_url: Optional[str] = Field(None, description="Static base URL")
    base_url_env: Optional[str] = Field(
        None, description="Environment variable for base URL"
    )
    api_key: Optional[str] = Field(None, description="Static API key (NOT recommended)")
    api_key_env: Optional[str] = Field(
        None,
        description="Explicit env var for API key (overrides name-based auto-resolution)",
    )


class ModelConfigEntry(BaseModel):
    """Single model entry referencing a client by key."""

    name: str = Field(..., description="Model name/identifier")
    client: str = Field(..., description="Key into the 'clients' dict")
    enabled: bool = Field(True, description="Whether to include this model in tests")
    description: Optional[str] = Field(None, description="Human-readable description")


class ModelsConfig(BaseModel):
    """Container for client and model configurations."""

    clients: Dict[str, ClientConfigEntry]
    models: List[ModelConfigEntry]
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


# ============================================================================
# Configuration Loader
# ============================================================================


def load_models(
    path: Optional[str | Path] = None,
    default_config_dir: Optional[Path] = None,
) -> List[tuple[ClientConfig, ModelConfig]]:
    """
    Load model configurations from JSON file.

    The JSON file should have a 'clients' dict and a 'models' list.
    Each model references a client by key. The client key becomes the
    ClientConfig.name, enabling automatic API key resolution from
    {NAME.upper()}_API_KEY environment variables.

    Args:
        path: Path to models.json file. If None, uses default_config_dir.
        default_config_dir: Default directory containing models.json.

    Returns:
        List of (ClientConfig, ModelConfig) tuples ready for use with casual-llm.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config is invalid or no enabled models found.
    """
    config_path = Path(path) if path else None
    if config_path is None and default_config_dir:
        config_path = default_config_dir / "models.json"
    if config_path is None:
        raise ValueError("No config path provided and no default directory set.")

    if not config_path.exists():
        raise FileNotFoundError(
            f"Models configuration file not found: {config_path}\n\n"
            f"Create a models.json file with a 'clients' dict and 'models' list:\n"
            f"  {{\n"
            f'    "clients": {{\n'
            f'      "openai": {{ "provider": "openai" }},\n'
            f'      "ollama": {{ "provider": "ollama", "base_url_env": "OLLAMA_ENDPOINT" }}\n'
            f"    }},\n"
            f'    "models": [\n'
            f'      {{ "name": "gpt-4o-mini", "client": "openai", "enabled": true }}\n'
            f"    ]\n"
            f"  }}\n\n"
            f"Or specify a custom path with --models-config"
        )

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

    # Convert to casual-llm config objects
    model_configs: List[tuple[ClientConfig, ModelConfig]] = []
    for entry in config.models:
        if not entry.enabled:
            continue

        # Look up client config
        if entry.client not in config.clients:
            raise ValueError(
                f"Model '{entry.name}' references unknown client '{entry.client}'. "
                f"Available clients: {list(config.clients.keys())}"
            )
        client_entry = config.clients[entry.client]

        # Resolve base_url from env var if needed
        base_url = client_entry.base_url
        if not base_url and client_entry.base_url_env:
            base_url = os.getenv(client_entry.base_url_env)
            if not base_url:
                print(
                    f"Warning: Environment variable {client_entry.base_url_env} "
                    f"not set for client '{entry.client}'"
                )

        # Resolve API key: explicit > api_key_env > name-based auto-resolution (via ClientConfig.name)
        api_key = client_entry.api_key
        if not api_key and client_entry.api_key_env:
            api_key = os.getenv(client_entry.api_key_env)

        # ClientConfig.name enables {NAME.upper()}_API_KEY auto-resolution in create_client()
        # String provider is coerced to Provider enum by ClientConfig.__post_init__
        client_config = ClientConfig(
            name=entry.client,
            provider=client_entry.provider,
            base_url=base_url,
            api_key=api_key,
        )

        model_configs.append((client_config, ModelConfig(name=entry.name)))

    if not model_configs:
        raise ValueError(
            f"No enabled models found in {config_path}. "
            f"Set 'enabled': true for at least one model."
        )

    return model_configs
