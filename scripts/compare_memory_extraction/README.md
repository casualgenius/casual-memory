# Memory Extraction Comparison Tool

A benchmarking tool for comparing memory extraction performance across different LLM models and providers.

## Overview

This tool allows you to:
- Test different LLM models (OpenAI, Ollama, Anthropic)
- Compare extraction results across conversation test cases
- Experiment with different system prompts
- Generate markdown reports with detailed results

## Quick Start

```bash
# From project root
cd scripts/compare_memory_extraction

# Run with default configurations
python run.py

# Or use custom configs
python run.py --models-config configs/examples/models_ollama_only.json
```

## Configuration Files

All configuration is externalized in JSON/Markdown files for easy modification.

### models.json

Defines which LLM models to test.

**Location**: `configs/models.json`

**Schema**:
```json
{
  "models": [
    {
      "name": "model-name",           // Model identifier
      "provider": "openai|ollama|anthropic",  // Provider type
      "base_url": "https://...",      // Optional: static base URL
      "base_url_env": "ENV_VAR",      // Optional: env var for base URL
      "api_key": "sk-...",            // Optional: static API key (not recommended)
      "api_key_env": "OPENAI_API_KEY",// Optional: env var for API key
      "enabled": true,                // Whether to include in tests
      "description": "..."            // Human-readable description
    }
  ],
  "metadata": {
    "version": "1.0",
    "description": "..."
  }
}
```

**Example**:
```json
{
  "models": [
    {
      "name": "qwen2.5:7b-instruct",
      "provider": "ollama",
      "base_url_env": "OLLAMA_ENDPOINT",
      "enabled": true,
      "description": "Qwen 2.5 7B via Ollama"
    }
  ]
}
```

### conversations.json

Defines test conversation pairs for extraction.

**Location**: `configs/conversations.json`

**Schema**:
```json
{
  "conversations": [
    {
      "id": "unique_id",              // Unique identifier
      "description": "...",           // What this tests
      "enabled": true,                // Whether to include
      "messages": [                   // Message sequence
        {"role": "user|assistant", "content": "..."}
      ],
      "expected_memories": 2,         // Optional: expected count
      "tags": ["fact", "location"]    // Optional: test tags
    }
  ],
  "metadata": {
    "version": "1.0"
  }
}
```

**Example**:
```json
{
  "conversations": [
    {
      "id": "location_fact",
      "description": "Simple location memory",
      "enabled": true,
      "messages": [
        {"role": "user", "content": "I live in London."},
        {"role": "assistant", "content": "Got it!"}
      ],
      "expected_memories": 1,
      "tags": ["fact", "location"]
    }
  ]
}
```

### system_prompt.md

The memory extraction system prompt with template placeholders.

**Location**: `configs/system_prompt.md`

**Placeholders**:
- `{today_natural}`: Replaced with current date (e.g., "Tuesday, December 09, 2025")
- `{isonow}`: Replaced with current ISO timestamp

**Example**:
```markdown
You are an expert memory extraction agent...

Today's date: {today_natural} (ISO: {isonow})
```

## Usage Examples

### Basic Usage

```bash
# Use default configurations
python run.py

# Specify output directory
python run.py --output-dir my_results

# Change log level
python run.py --log-level DEBUG
```

### Testing Different Models

```bash
# Test only Ollama models
python run.py --models-config configs/examples/models_ollama_only.json

# Test with custom model config
python run.py --models-config /path/to/custom_models.json
```

### Testing Different Conversations

```bash
# Quick test with minimal conversations
python run.py --conversations-config configs/examples/conversations_simple.json

# Full test suite (default)
python run.py --conversations-config configs/conversations.json
```

### Testing Different Prompts

```bash
# Use a custom system prompt
python run.py --prompt-config /path/to/custom_prompt.md
```

### Combining Options

```bash
# Test Ollama models with simple conversations
python run.py \
  --models-config configs/examples/models_ollama_only.json \
  --conversations-config configs/examples/conversations_simple.json \
  --output-dir quick_test
```

## Environment Variables

API keys and endpoints should be configured via environment variables:

```bash
# OpenAI / OpenRouter API key
export OPENAI_API_KEY="sk-..."

# Ollama endpoint (if not default)
export OLLAMA_ENDPOINT="http://localhost:11434"

# Anthropic API key
export ANTHROPIC_API_KEY="sk-ant-..."
```

You can also use a `.env` file in the project root.

## Output

The tool generates a markdown report in the output directory with:
- Timestamp
- System prompt used
- Results table for each conversation showing:
  - Model name
  - Extracted memories
  - Memory type, tags, importance
  - Extraction duration

**Example output**: `results/memory_comparison_202512091430.md`

## File Structure

```
compare_memory_extraction/
├── run.py                           # Main script
├── config_loader.py                 # Configuration loading logic
├── README.md                        # This file
└── configs/
    ├── models.json                  # Default model configs
    ├── conversations.json           # Default test conversations
    ├── system_prompt.md             # Default extraction prompt
    └── examples/
        ├── models_ollama_only.json  # Example: Ollama-only testing
        └── conversations_simple.json # Example: Minimal test set
```

## Creating Custom Configurations

### Custom Model Set

1. Copy an example: `cp configs/examples/models_ollama_only.json my_models.json`
2. Edit `my_models.json` to add/remove/configure models
3. Run: `python run.py --models-config my_models.json`

### Custom Conversations

1. Copy the default: `cp configs/conversations.json my_conversations.json`
2. Add/remove/modify conversations
3. Run: `python run.py --conversations-config my_conversations.json`

### Custom Prompt

1. Copy the default: `cp configs/system_prompt.md my_prompt.md`
2. Modify the prompt (keep `{today_natural}` and `{isonow}` placeholders)
3. Run: `python run.py --prompt-config my_prompt.md`

## Tips

### Quick Iteration

Use the simple conversations example for rapid testing:
```bash
python run.py --conversations-config configs/examples/conversations_simple.json
```

### Prompt Development

1. Copy `system_prompt.md` to a new file
2. Make modifications
3. Test with simple conversations first
4. Once satisfied, test with full conversation set

### Model Comparison

Enable/disable models in the JSON config instead of commenting them out:
```json
{
  "name": "expensive-model",
  "enabled": false  // Skip this one for now
}
```

### Debugging

Enable debug logging to see detailed output:
```bash
python run.py --log-level DEBUG
```

## Common Issues

### "Models configuration file not found"

**Solution**: Make sure you're running from the correct directory or use absolute paths:
```bash
cd scripts/compare_memory_extraction
python run.py
```

### "Environment variable OPENAI_API_KEY not set"

**Solution**: Set the required environment variables or create a `.env` file:
```bash
export OPENAI_API_KEY="your-key-here"
```

### "No enabled models found"

**Solution**: Check that at least one model has `"enabled": true` in your models config.

## Contributing

To add new example configs:
1. Create the config file in `configs/examples/`
2. Add documentation to this README
3. Test it works: `python run.py --models-config configs/examples/your_new_config.json`

## Related Tools

This tool is part of the `casual-memory` benchmark suite. Future tools may include:
- `scripts/compare_nli_filter/` - NLI pre-filter comparison
- `scripts/compare_conflict_detection/` - Conflict detection benchmarks
