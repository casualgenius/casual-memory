# Duplicate Classifier Benchmarking Tool

Benchmarking tool for testing the LLM-based Duplicate Classifier with various models and prompts. This tool helps evaluate prompt effectiveness and classifier performance on duplicate/refinement detection scenarios.

## Overview

The Duplicate Classifier uses an LLM to determine if two memory statements are:
- **SAME**: Duplicates or refinements (should be merged)
- **DISTINCT**: Separate facts (should be stored separately)

This benchmarking tool allows you to:
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
cd scripts/benchmark_duplicate_classifier
uv run python run.py
```

This will:
1. Load default test cases from `test_cases.json`
2. Use OpenAI's GPT-4o-mini model
3. Run all 24 test cases
4. Generate a markdown report in `results/duplicate_benchmark_MODEL_TIMESTAMP.md`

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
# Use custom prompt from file
uv run python run.py --prompt my_custom_prompt.txt
```

**Custom Prompt Format:**
Your custom prompt file must include `{statement_a}` and `{statement_b}` placeholders and expect "SAME" or "DISTINCT" responses:

```
Are these statements about the same thing?

A: "{statement_a}"
B: "{statement_b}"

Answer: SAME or DISTINCT
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
      "name": "location_refinement",
      "memory_a": "I live in Tokyo",
      "memory_b": "I live in Shibuya, Tokyo",
      "expected_same": true,
      "category": "refinement",
      "description": "Location with added detail (neighborhood)"
    }
  ]
}
```

### Fields

- **name** (string): Unique identifier for the test case
- **memory_a** (string): First memory statement
- **memory_b** (string): Second memory statement
- **expected_same** (boolean): Whether these should be classified as SAME (duplicate/refinement)
- **category** (string): Category for grouping results
- **description** (string, optional): Human-readable explanation

### Categories

Organize test cases by category for better analysis:

**SAME Cases:**
- `duplicate`: Exact duplicates
- `paraphrase`: Same fact, different wording
- `intensity`: Same preference, different intensity
- `refinement`: Base fact + added detail
- `clarification`: Added clarifying information

**DISTINCT Cases:**
- `distinct_attribute`: Different attributes about same subject
- `distinct_preference`: Different preferences
- `distinct_skill`: Different skills
- `distinct_info`: Different types of information
- `contradiction`: Contradictory facts
- `temporal`: Temporal progression
- `unrelated`: Completely unrelated facts

## Understanding the Report

The benchmark generates a markdown report with several sections:

### 1. Configuration

Shows the provider, model, and settings used for the benchmark.

### 2. Detailed Results

Table showing:
- Test case name
- Memory pair texts (truncated)
- Expected vs actual classification
- Pass/fail status
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
- Test description

### 6. Analysis

Breakdown of:
- **False Positives**: DISTINCT cases incorrectly classified as SAME
- **False Negatives**: SAME cases incorrectly classified as DISTINCT

### 7. Recommendations

Automated suggestions based on failure patterns.

## Interpreting Results

### Good Performance Indicators

- **High accuracy on exact duplicates and paraphrases** (should all be SAME)
- **Correct handling of refinements** (base fact + detail = SAME)
- **Low false positive rate** (not merging distinct facts)
- **Correct handling of distinct facts** (different preferences, skills = DISTINCT)

### Common Issues

**High False Positive Rate:**
- Prompt may be too aggressive at merging
- Not distinguishing between similar but distinct facts
- Example: "I like coffee" vs "I like tea" marked as SAME

**High False Negative Rate:**
- Prompt may be too conservative
- Not recognizing refinements properly
- Example: "I live in Tokyo" vs "I live in Shibuya, Tokyo" marked as DISTINCT

## Prompt Engineering Tips

### What Makes a Good Duplicate Detection Prompt?

1. **Clear Examples**: Show both SAME and DISTINCT cases
2. **Handle Refinements**: Distinguish "added detail" from "different fact"
3. **Attribute Awareness**: Same subject + different attribute = DISTINCT
4. **Explicit Output Format**: "Respond with ONLY: SAME or DISTINCT"
5. **Low Temperature**: Use temperature=0.1 for consistent results

### Testing Your Prompt

1. **Start with baseline**: Run with default prompt
2. **Identify issues**: Review false positives/negatives
3. **Refine examples**: Add examples for failing categories
4. **Test again**: Verify improvements
5. **Compare results**: Check if pass rate improved

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
   - False positives? → Need stricter distinction rules
   - False negatives? → Need clearer refinement examples

3. **Test variations**: Try different models and prompts
   ```bash
   uv run python run.py --model gpt-4o --output-dir gpt4o_test
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

- [LLMDuplicateDetector Source](../../src/casual_memory/intelligence/duplicate_detector.py)
- [Prompts](../../src/casual_memory/intelligence/prompts.py)
- [Duplicate Classifier](../../src/casual_memory/classifiers/duplicate_classifier.py)
- [CLAUDE.md](../../CLAUDE.md) - Overall project documentation

## Comparison with Other Benchmarks

| Tool | What It Tests | Speed | Cost | Use Case |
|------|---------------|-------|------|----------|
| NLI Benchmark | DeBERTa cross-encoder | ~130ms | Free | Pre-filter (same/neutral/uncertain) |
| Conflict Benchmark | LLM conflict detection | ~500ms | $$ | Contradiction detection |
| Duplicate Benchmark | LLM duplicate/refinement | ~500ms | $$ | Merge vs. separate decision |

Use all three to optimize the entire classification pipeline:
1. **NLI** filters obvious same/distinct cases quickly
2. **Conflict** detects contradictions
3. **Duplicate** handles subtle same/distinct decisions

## Key Differences from Conflict Classifier

| Aspect | Conflict Classifier | Duplicate Classifier |
|--------|-------------------|---------------------|
| Purpose | Detect contradictions | Detect duplicates/refinements |
| Output | CONFLICT / NO CONFLICT | SAME / DISTINCT |
| Edge Case | Temporal progression (past vs present) | Refinements (base + detail) |
| False Positive Risk | Marking refinements as conflicts | Merging distinct but similar facts |
| False Negative Risk | Missing subtle contradictions | Missing refinements as duplicates |

## Best Practices

1. **Test regularly**: Run benchmark after prompt changes
2. **Use diverse examples**: Cover all categories in test cases
3. **Monitor metrics**: Track pass rates over time
4. **Avoid overfitting**: Use different examples in prompt vs. test cases
5. **Validate with real data**: Supplement synthetic tests with real memory pairs
