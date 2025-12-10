# Conflict Classifier Benchmarking Tool

Benchmarking tool for testing the LLM-based Conflict Classifier with various models and prompts. This tool helps evaluate prompt effectiveness and classifier performance on different conflict scenarios.

## Overview

The Conflict Classifier uses an LLM to determine if two memory statements contradict each other. This benchmarking tool allows you to:

- Test different LLM models (OpenAI, Anthropic, Ollama)
- Evaluate custom prompts
- Identify false positives and false negatives
- Compare performance across models

## Prerequisites

### API Keys

Depending on the provider you want to use, set the appropriate API key:

```bash
# For OpenAI
export OPENAI_API_KEY="your-key-here"

# For Anthropic
export ANTHROPIC_API_KEY="your-key-here"

# For Ollama (no key needed, must be running locally)
# Start Ollama: ollama serve
```

### Dependencies

```bash
# Install with LLM provider support
uv sync --all-extras
```

## Quick Start

Run with default settings (OpenAI GPT-4o-mini):

```bash
cd scripts/benchmark_conflict_classifier
uv run python run.py
```

This will:
1. Load default test cases from `test_cases.json`
2. Use OpenAI's GPT-4o-mini model
3. Run all 24 test cases
4. Generate a markdown report in `results/conflict_benchmark_MODEL_TIMESTAMP.md`

## Usage

### Basic Usage

```bash
# Use defaults (OpenAI GPT-4o-mini)
uv run python run.py

# Specify output directory
uv run python run.py --output-dir my_results
```

### Model Selection

Test different models:

```bash
# Use GPT-4o (more expensive but more accurate)
uv run python run.py --model gpt-4o

# Use Anthropic Claude
uv run python run.py --provider anthropic --model claude-3-5-sonnet-20241022

# Use local Ollama model
uv run python run.py --provider ollama --model llama3.2
```

### Prompt Testing

Test different prompt variants:

```bash
# Use the detailed prompt variant (built-in)
uv run python run.py --use-detailed-prompt

# Use custom prompt from file
uv run python run.py --prompt my_custom_prompt.txt
```

**Custom Prompt Format:**
Your custom prompt file must include `{statement_a}` and `{statement_b}` placeholders:

```
Do these two statements contradict?

A: "{statement_a}"
B: "{statement_b}"

Answer YES or NO.
```

### Custom Test Cases

Use your own test cases:

```bash
uv run python run.py --test-cases my_test_cases.json
```

See `test_cases.json` for the expected format.

### Logging

Adjust logging verbosity:

```bash
# Debug mode (shows all LLM responses)
uv run python run.py --log-level DEBUG

# Quiet mode (warnings and errors only)
uv run python run.py --log-level WARNING
```

## Test Case Format

Test cases are defined in JSON format:

```json
{
  "test_cases": [
    {
      "name": "location_contradiction",
      "memory_a": "I live in Paris",
      "memory_b": "I live in London",
      "expected_conflict": true,
      "category": "location",
      "description": "Direct location contradiction"
    }
  ]
}
```

### Fields

- **name** (string): Unique identifier for the test case
- **memory_a** (string): First memory statement
- **memory_b** (string): Second memory statement
- **expected_conflict** (boolean): Whether these should be classified as conflicting
- **category** (string): Category for grouping results
- **description** (string, optional): Human-readable explanation

### Categories

Organize test cases by category for better analysis:

- `location`: Location-based contradictions
- `job`: Job/role contradictions
- `preference`: Preference contradictions
- `temporal`: Temporal progressions (not conflicts)
- `refinement`: Refinements with added detail (not conflicts)
- `duplicate`: Exact duplicates (not conflicts)
- `distinct`: Unrelated facts (not conflicts)
- `compatible`: Compatible facts (not conflicts)

## Understanding the Report

The benchmark generates a markdown report with several sections:

### 1. Configuration

Shows the provider, model, and settings used for the benchmark.

### 2. Detailed Results

Table showing:
- Test case name
- Memory pair texts (truncated)
- Expected vs actual conflict classification
- Pass/fail status
- Detection method
- Processing time

### 3. Results by Category

Pass rate breakdown by test case category.

### 4. Summary

Overall statistics:
- Total test cases
- Pass/fail counts
- Average processing time

### 5. Failed Cases

Detailed breakdown of failed cases with:
- Full memory texts
- Expected vs actual results
- Detection method
- Test description

### 6. Analysis

Breakdown of:
- **False Positives**: Non-conflicts incorrectly classified as conflicts
- **False Negatives**: Actual conflicts that were missed

### 7. Recommendations

Automated suggestions based on failure patterns.

## Interpreting Results

### Good Performance Indicators

- **High accuracy on contradictions** (location, job, preference conflicts)
- **Low false positive rate** (not flagging refinements as conflicts)
- **Correct handling of temporal changes** (past → present not a contradiction)

### Common Issues

**High False Positive Rate:**
- Prompt may be too aggressive
- Consider refinements vs contradictions
- Temporal changes marked as conflicts

**High False Negative Rate:**
- Prompt may be too conservative
- Missing subtle contradictions
- Need clearer examples in prompt

## Prompt Engineering Tips

### What Makes a Good Conflict Detection Prompt?

1. **Clear Examples**: Show both conflicts and non-conflicts
2. **Handle Edge Cases**: Refinements, temporal changes, synonyms
3. **Explicit Output Format**: "Respond with ONLY: YES or NO"
4. **Low Temperature**: Use temperature=0.1 for consistent results

### Built-in Prompts

We provide two built-in prompts:

1. **CONFLICT_DETECTION_PROMPT** (default): Concise, proven 96.2% accuracy
2. **CONFLICT_DETECTION_PROMPT_DETAILED**: More verbose with detailed guidelines

Test both:

```bash
# Default (concise)
uv run python run.py

# Detailed
uv run python run.py --use-detailed-prompt
```

## Performance Notes

### Cost Estimates

- **GPT-4o-mini**: ~$0.02 for 24 test cases
- **GPT-4o**: ~$0.15 for 24 test cases
- **Claude 3.5 Sonnet**: ~$0.10 for 24 test cases
- **Ollama (local)**: Free!

### Speed

- **OpenAI**: ~300-800ms per test
- **Anthropic**: ~400-1000ms per test
- **Ollama (local)**: ~200-500ms per test (depends on hardware)

## Example Workflow

1. **Baseline**: Run with default model and prompt
   ```bash
   uv run python run.py --output-dir baseline
   ```

2. **Analyze**: Review report and identify issues
   - False positives? → Need more conservative prompt
   - False negatives? → Need more aggressive prompt

3. **Test variations**: Try different models and prompts
   ```bash
   uv run python run.py --model gpt-4o --output-dir gpt4o_test
   uv run python run.py --use-detailed-prompt --output-dir detailed_prompt
   uv run python run.py --prompt custom.txt --output-dir custom_test
   ```

4. **Compare**: Review reports side-by-side

5. **Update**: Apply best prompt to production code in `src/casual_memory/intelligence/prompts.py`

## Troubleshooting

### API Key Errors

```bash
# Check if key is set
echo $OPENAI_API_KEY

# Set temporarily
export OPENAI_API_KEY="your-key"
```

### Ollama Connection Failed

```bash
# Start Ollama server
ollama serve

# Pull the model you want to use
ollama pull llama3.2
```

### Import Errors

Ensure dependencies are installed:

```bash
uv sync --all-extras
```

## Contributing

To add more test cases:

1. Edit `test_cases.json`
2. Add test cases with clear expected outcomes
3. Run benchmark to validate
4. Submit PR with report showing results

## Related Files

- [LLMConflictVerifier Source](../../src/casual_memory/intelligence/conflict_verifier.py)
- [Prompts](../../src/casual_memory/intelligence/prompts.py)
- [Conflict Classifier](../../src/casual_memory/classifiers/conflict_classifier.py)
- [CLAUDE.md](../../CLAUDE.md) - Overall project documentation

## Comparison with NLI Benchmark

The NLI benchmark ([scripts/benchmark_nli_classifier](../benchmark_nli_classifier)) tests the fast DeBERTa-based pre-filter, while this tool tests the slower but more accurate LLM-based conflict classifier.

| Tool | What It Tests | Speed | Cost | Accuracy |
|------|---------------|-------|------|----------|
| NLI Benchmark | DeBERTa cross-encoder | ~130ms | Free | Good for obvious cases |
| Conflict Benchmark | LLM (GPT-4, Claude, etc.) | ~500ms | $$ | High accuracy |

Use both to optimize the entire classification pipeline!
