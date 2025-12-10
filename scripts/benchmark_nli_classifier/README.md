# NLI Classifier Benchmarking Tool

Benchmarking tool for testing the NLI Classifier with the real DeBERTa-v3 model. This tool helps evaluate threshold settings and classifier performance on various memory pair scenarios.

## Overview

The NLI (Natural Language Inference) Classifier is a fast pre-filter that uses a cross-encoder model to quickly classify obvious cases before expensive LLM calls. This benchmarking tool allows you to:

- Test the classifier on various memory pair scenarios
- Evaluate different threshold settings
- Identify false positives and false negatives
- Get recommendations for threshold tuning

## Prerequisites

Install the required dependencies:

```bash
# Install sentence-transformers for NLI model
uv sync --extra transformers
```

Or if using pip:

```bash
pip install sentence-transformers torch
```

## Quick Start

Run with default settings:

```bash
cd scripts/benchmark_nli_classifier
python run.py
```

This will:
1. Load default test cases from `test_cases.json`
2. Initialize the NLI classifier with default thresholds (entailment=0.85, neutral=0.5)
3. Run all test cases
4. Generate a markdown report in `results/nli_benchmark_YYYYMMDD_HHMMSS.md`

## Usage

### Basic Usage

```bash
# Use default configuration
python run.py

# Specify output directory
python run.py --output-dir my_results
```

### Threshold Tuning

The NLI classifier has two key thresholds:

- **Entailment Threshold** (default: 0.85): Minimum entailment score to classify as "same"
- **Neutral Threshold** (default: 0.5): Minimum neutral score to classify as "neutral" (when entailment < 0.3)

Test different thresholds:

```bash
# Lower entailment threshold (more aggressive "same" classification)
python run.py --entailment-threshold 0.80

# Higher neutral threshold (stricter "neutral" classification)
python run.py --neutral-threshold 0.60

# Combine both
python run.py --entailment-threshold 0.80 --neutral-threshold 0.55
```

### Custom Test Cases

Use your own test cases:

```bash
python run.py --test-cases my_test_cases.json
```

See `test_cases.json` for the expected format.

### Logging

Adjust logging verbosity:

```bash
# Debug mode (shows all NLI predictions)
python run.py --log-level DEBUG

# Quiet mode (warnings and errors only)
python run.py --log-level WARNING
```

## Test Case Format

Test cases are defined in JSON format:

```json
{
  "test_cases": [
    {
      "name": "exact_duplicate",
      "existing_memory": "I live in Paris",
      "new_memory": "I live in Paris",
      "expected_outcome": "same",
      "category": "duplicate",
      "description": "Exact same text - should classify as same"
    }
  ]
}
```

### Expected Outcomes

- **`same`**: Memories should be classified as duplicates/paraphrases (high entailment)
- **`neutral`**: Memories should be classified as distinct facts (high neutral, low entailment)
- **`pass`**: Classifier should be uncertain and pass to next classifier (contradictions, edge cases)

### Categories

Organize test cases by category for better analysis:

- `duplicate`: Exact duplicates
- `paraphrase`: Same meaning, different wording
- `refinement`: More specific version of existing fact
- `distinct`: Completely different facts
- `contradiction`: Contradictory statements
- `edge_case`: Uncertain cases

## Understanding the Report

The benchmark generates a markdown report with several sections:

### 1. Configuration

Shows the thresholds and settings used for the benchmark.

### 2. Detailed Results

Table showing:
- Test case name
- Memory pair texts
- NLI scores (Contradiction, Entailment, Neutral)
- Expected vs actual outcomes
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
- NLI scores
- Expected vs actual outcomes
- Test description

### 6. Threshold Recommendations

Automated suggestions based on failure patterns:
- Lower/raise entailment threshold
- Lower/raise neutral threshold
- Average scores for failed cases

## Interpreting NLI Scores

Each memory pair produces three scores (they sum to ~1.0):

- **Contradiction (C)**: Probability that memories contradict
- **Entailment (E)**: Probability that new memory is entailed by existing (paraphrase/refinement)
- **Neutral (N)**: Probability that memories are unrelated or distinct facts

### Classification Logic

1. If `E >= entailment_threshold` (default 0.85) → Classify as **same**
2. Else if `N >= neutral_threshold` (default 0.5) AND `E < 0.3` → Classify as **neutral**
3. Else → **Pass** to next classifier (uncertain)

### Example Scores

```
Duplicate: C=0.05, E=0.92, N=0.03 → same (E >= 0.85)
Distinct:  C=0.10, E=0.20, N=0.70 → neutral (N >= 0.5, E < 0.3)
Conflict:  C=0.85, E=0.10, N=0.05 → pass (uncertain)
```

## Threshold Tuning Guidelines

### If too many false "same" classifications:

- **Lower entailment threshold** (e.g., 0.85 → 0.80)
- This makes the classifier more conservative about merging memories

### If missing "same" classifications:

- **Raise entailment threshold** (e.g., 0.85 → 0.90)
- This makes the classifier more aggressive about merging

### If too many false "neutral" classifications:

- **Lower neutral threshold** (e.g., 0.5 → 0.45)
- This makes the classifier more conservative about marking as distinct

### If missing "neutral" classifications:

- **Raise neutral threshold** (e.g., 0.5 → 0.55)
- This makes the classifier more aggressive about marking as distinct

## Performance Notes

- **First run**: Model download and loading (~1-2 minutes)
- **Subsequent runs**: Model is cached locally
- **CPU inference**: ~100-200ms per prediction
- **GPU inference**: ~20-50ms per prediction
- **Caching**: Identical pairs are cached for faster re-testing

## Example Workflow

1. **Baseline**: Run with default thresholds
   ```bash
   python run.py --output-dir baseline
   ```

2. **Analyze**: Review report and identify issues
   - Too many false positives? → Lower threshold
   - Too many false negatives? → Raise threshold

3. **Test variations**: Try different thresholds
   ```bash
   python run.py --entailment-threshold 0.80 --output-dir threshold_80
   python run.py --entailment-threshold 0.90 --output-dir threshold_90
   ```

4. **Compare**: Review reports side-by-side

5. **Update**: Apply best thresholds to production code in `src/casual_memory/classifiers/nli_classifier.py`

## Troubleshooting

### Model Download Issues

If the DeBERTa-v3 model fails to download:

```bash
# Pre-download the model
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/nli-deberta-v3-base')"
```

### Out of Memory

For large test suites on CPU:

```bash
# Process fewer cases at a time by creating smaller test case files
python run.py --test-cases small_batch_1.json
python run.py --test-cases small_batch_2.json
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

## Related Documentation

- [NLI Classifier Source](../../src/casual_memory/classifiers/nli_classifier.py)
- [NLI Pre-filter Source](../../src/casual_memory/intelligence/nli_filter.py)
- [Classification Pipeline](../../src/casual_memory/classifiers/pipeline.py)
- [CLAUDE.md](../../CLAUDE.md) - Overall project documentation
