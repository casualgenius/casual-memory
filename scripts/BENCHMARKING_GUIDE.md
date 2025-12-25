# Classifier Benchmarking Guide

Complete guide to benchmarking the NLI, Conflict, and Duplicate classifiers with multi-model comparison support.

## Overview

This project includes three comprehensive benchmarking tools to evaluate and compare classifier performance across different LLM models:

1. **NLI Classifier** - DeBERTa cross-encoder pre-filter
2. **Conflict Classifier** - LLM-based contradiction detection
3. **Duplicate Classifier** - LLM-based duplicate/refinement detection

All benchmarks support:
- **Single-model mode** (backward compatible)
- **Multi-model comparison** (compare multiple models side-by-side)
- **Custom prompts** (test different prompt strategies)
- **Detailed reports** (markdown reports with analysis)

## Quick Start

### Single Model (Backward Compatible)

```bash
# Conflict classifier with single model
cd scripts/benchmark_conflict_classifier
uv run python run.py --provider ollama --model qwen2.5:7b-instruct

# Duplicate classifier with single model
cd scripts/benchmark_duplicate_classifier
uv run python run.py --provider ollama --model qwen2.5:7b-instruct

# NLI classifier (no multi-model support yet)
cd scripts/benchmark_nli_classifier
uv run python run.py
```

### Multi-Model Comparison

```bash
# Conflict classifier comparing multiple models
cd scripts/benchmark_conflict_classifier
uv run python run.py --models-config configs/models.json

# Duplicate classifier comparing multiple models
cd scripts/benchmark_duplicate_classifier
uv run python run.py --models-config configs/models.json

# Use example configs for Ollama-only
uv run python run.py --models-config configs/examples/models_ollama_only.json
```

## Configuration Files

### Model Configuration Format

Create a `models.json` file to define which models to test:

```json
{
  "models": [
    {
      "name": "gpt-4o-mini",
      "provider": "openai",
      "api_key_env": "OPENAI_API_KEY",
      "enabled": true,
      "description": "OpenAI GPT-4o-mini"
    },
    {
      "name": "qwen2.5:7b-instruct",
      "provider": "ollama",
      "base_url_env": "OLLAMA_ENDPOINT",
      "enabled": true,
      "description": "Qwen 2.5 7B local model"
    }
  ],
  "metadata": {
    "version": "1.0",
    "description": "Model configurations for benchmarking"
  }
}
```

### Configuration Fields

- **name** (required): Model identifier (e.g., "gpt-4o-mini", "qwen2.5:7b-instruct")
- **provider** (required): LLM provider - "openai" or "ollama"
- **enabled** (required): Whether to include this model in tests
- **api_key** (optional): Static API key (NOT recommended)
- **api_key_env** (optional): Environment variable containing API key
- **base_url** (optional): Static base URL
- **base_url_env** (optional): Environment variable containing base URL
- **description** (optional): Human-readable description

### Example Configurations

**Default Config** (`configs/models.json`):
- GPT-4o-mini (disabled by default to avoid costs)
- Qwen 2.5 7B (enabled)

**Ollama Only** (`configs/examples/models_ollama_only.json`):
- Multiple Ollama models for local testing
- No API costs
- Fast iteration

## Environment Variables

```bash
# For OpenAI models
export OPENAI_API_KEY="your-key-here"

# For Ollama (optional, defaults to localhost:11434)
export OLLAMA_ENDPOINT="http://your-ollama-host:11434"
```

## Generated Reports

### Single Model Mode

When running with a single model:
- Individual report: `{classifier}_benchmark_{model}_{timestamp}.md`

### Multi-Model Mode

When running with multiple models:
- Individual reports for each model: `{classifier}_benchmark_{model}_{timestamp}.md`
- Comparison report: `{classifier}_benchmark_comparison_{timestamp}.md`

### Report Contents

**Individual Reports:**
- Configuration details
- Detailed results table
- Results by category
- Summary statistics
- Failed cases analysis
- Recommendations

**Comparison Reports:**
- Summary table comparing all models
- Pass rates and timing comparison
- Side-by-side test case results
- Quick identification of model strengths/weaknesses

## Usage Examples

### Test Multiple Local Models

```bash
# Edit configs/examples/models_ollama_only.json to enable models
# Enable: qwen2.5:7b-instruct, gemma2:9b, llama3.2

uv run python run.py --models-config configs/examples/models_ollama_only.json
```

### Compare OpenAI vs Local

```bash
# Edit configs/models.json
# Enable both gpt-4o-mini and qwen2.5:7b-instruct

uv run python run.py --models-config configs/models.json
```

### Test Custom Prompt

```bash
# Single model with custom prompt
uv run python run.py --prompt my_prompt.txt

# Multi-model with custom prompt
uv run python run.py --models-config configs/models.json --prompt my_prompt.txt
```

### Specify Output Directory

```bash
uv run python run.py --models-config configs/models.json --output-dir my_results
```

## Benchmark-Specific Features

### Conflict Classifier

**Test Cases:** 24 scenarios covering:
- Direct contradictions (location, job, preference)
- Temporal progressions (past → present)
- Refinements (base → detailed)
- Compatible facts
- Edge cases

**Key Metrics:**
- Pass rate (target: >95%)
- False positives (refinements marked as conflicts)
- False negatives (missed contradictions)
- Average processing time

**Special Options:**
```bash
# Use detailed prompt variant
uv run python run.py --use-detailed-prompt
```

### Duplicate Classifier

**Test Cases:** 24 scenarios covering:
- Exact duplicates
- Paraphrases
- Refinements (base → detailed)
- Distinct facts (different attributes)
- Contradictions
- Temporal changes

**Key Metrics:**
- Pass rate (target: >85%)
- False positives (distinct facts marked as same)
- False negatives (missed refinements)
- Average processing time

### NLI Classifier

**Test Cases:** 18 scenarios covering:
- Exact duplicates
- Paraphrases
- Refinements
- Distinct facts
- Contradictions

**Key Metrics:**
- Pass rate (expected: ~60%, it's a pre-filter)
- Pre-filter effectiveness
- Processing speed (target: <200ms)

**Note:** NLI classifier doesn't support multi-model mode (uses fixed DeBERTa model)

## Interpreting Results

### Good Performance Indicators

**Conflict Classifier:**
- High accuracy on direct contradictions (>95%)
- Low false positive rate on refinements (<5%)
- Correct handling of temporal progressions

**Duplicate Classifier:**
- High accuracy on refinements (>80%)
- Low false positive rate on distinct facts (<10%)
- Correct handling of different attributes

### Common Issues

**High False Positive Rate:**
- Prompt may be too aggressive
- Need clearer distinction examples
- Consider adding edge case examples

**High False Negative Rate:**
- Prompt may be too conservative
- Missing key pattern examples
- Need more explicit guidelines

### Model Comparison Insights

The comparison report helps identify:
- **Consistency**: Which models are most reliable?
- **Speed**: Which models are fastest?
- **Edge cases**: Which models handle specific scenarios better?
- **Cost**: OpenAI vs local models performance/cost trade-off

## Best Practices

### 1. Start with Defaults

```bash
# Test with default config first
uv run python run.py --models-config configs/models.json
```

### 2. Analyze Results

- Review individual reports for each model
- Check comparison report for side-by-side analysis
- Identify patterns in failures

### 3. Iterate on Prompts

- Test custom prompts
- Compare results against baseline
- Update prompts in `src/casual_memory/intelligence/prompts.py`

### 4. Regular Testing

- Re-run benchmarks after prompt changes
- Track performance over time
- Maintain historical results for comparison

### 5. Cost Management

- Disable expensive models in default config
- Use Ollama for rapid iteration
- Enable OpenAI only for final validation

## Advanced Usage

### Creating Custom Test Cases

Edit `test_cases.json` in each benchmark directory:

```json
{
  "test_cases": [
    {
      "name": "my_test",
      "memory_a": "First statement",
      "memory_b": "Second statement",
      "expected_conflict": true,  // or expected_same for duplicate
      "category": "my_category",
      "description": "What this tests"
    }
  ]
}
```

### Custom Prompt Format

Prompts must include placeholders:

**Conflict Classifier:**
```
Do these contradict?

A: "{statement_a}"
B: "{statement_b}"

Answer: YES or NO
```

**Duplicate Classifier:**
```
Are these the same?

A: "{statement_a}"
B: "{statement_b}"

Answer: SAME or DISTINCT
```

## Troubleshooting

### "Config loader not available"

The config_loader.py file is missing. Ensure you have:
```
scripts/benchmark_{classifier}/config_loader.py
```

### "No enabled models found"

All models in your config have `"enabled": false`. Enable at least one:
```json
{"enabled": true}
```

### "Environment variable X not set"

Set the required environment variable:
```bash
export OPENAI_API_KEY="your-key"
export OLLAMA_ENDPOINT="http://localhost:11434"
```

### Ollama Connection Failed

```bash
# Start Ollama
ollama serve

# Pull required models
ollama pull qwen2.5:7b-instruct
```

## Performance Benchmarks

**Typical Results (Qwen 2.5 7B on Ollama):**

| Classifier | Accuracy | Avg Time | Notes |
|------------|----------|----------|-------|
| NLI | 61% | 130ms | Pre-filter working as intended |
| Conflict | 95.8% | 120ms | Excellent performance |
| Duplicate | 75% | 230ms | Needs prompt improvement for refinements |

**OpenAI GPT-4o-mini (typical):**
- Conflict: ~98% accuracy, ~500ms
- Duplicate: ~90% accuracy, ~500ms
- Cost: ~$0.02 per benchmark

## Related Documentation

- [Conflict Classifier Benchmark README](benchmark_conflict_classifier/README.md)
- [Duplicate Classifier Benchmark README](benchmark_duplicate_classifier/README.md)
- [NLI Classifier Benchmark README](benchmark_nli_classifier/README.md)
- [Project Documentation](../CLAUDE.md)

## Contributing

To add new test cases or improve benchmarks:

1. Add test cases to `test_cases.json`
2. Run benchmarks to validate
3. Update prompts if needed in `src/casual_memory/intelligence/prompts.py`
4. Re-run benchmarks to verify improvements
5. Submit PR with before/after results
