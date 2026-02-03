# Troubleshooting Guide

This document covers common usage scenarios and troubleshooting tips.

## Common Usage Scenarios

### Handling Conflicts

When a conflict is detected, you can retrieve and resolve it:

```python
# Get pending conflicts for a user
conflicts = await conflict_store.get_pending_conflicts(user_id="user_123")

for conflict in conflicts:
    print(f"Conflict ID: {conflict.id}")
    print(f"Category: {conflict.category}")  # location, job, preference, temporal, factual
    print(f"Memory A: {conflict.memory_a.text}")
    print(f"Memory B: {conflict.memory_b.text}")
    print(f"Hints: {conflict.clarification_hint}")

    # Resolve the conflict
    resolution = ConflictResolution(
        conflict_id=conflict.id,
        decision="keep_b",  # or "keep_a", "merge", "both_valid"
        resolved_memory=conflict.memory_b,  # the chosen memory
    )
    await conflict_store.resolve_conflict(conflict.id, resolution)
```

### Extracting Memories from Conversations

```python
from casual_memory.extractors import LLMMemoryExtracter
from casual_llm import create_provider, ModelConfig, Provider

llm_provider = create_provider(ModelConfig(
    name="gpt-4",
    provider=Provider.OPENAI,
))
extracter = LLMMemoryExtracter(llm_provider=llm_provider, source="user")

# Extract from user message
user_message = "I love pizza and I work at Google in San Francisco"
memories = await extracter.extract([UserMessage(content=user_message)])

for memory in memories:
    print(f"Type: {memory.type}, Text: {memory.text}, Tags: {memory.tags}")
    # Type: preference, Text: loves pizza, Tags: [food]
    # Type: fact, Text: works at Google, Tags: [job, company]
    # Type: fact, Text: works in San Francisco, Tags: [location, work]
```

### Tuning Classification Performance

```python
# For speed: Use single strategy (check only highest similarity)
pipeline = MemoryClassificationPipeline(
    classifiers=[...],
    strategy="single",  # Fastest
)

# For thoroughness: Use all strategy (check all similar memories)
pipeline = MemoryClassificationPipeline(
    classifiers=[...],
    strategy="all",  # Most thorough, slowest
)

# For balance: Use tiered strategy with custom thresholds
pipeline = MemoryClassificationPipeline(
    classifiers=[...],
    strategy="tiered",  # Default
    secondary_conflict_threshold=0.90,  # Only check very similar memories
    max_secondary_checks=3,  # Limit secondary checks
)
```

### Adjusting Auto-Resolution Sensitivity

```python
# More aggressive auto-resolution (fewer conflicts escalated to user)
auto_resolver = AutoResolutionClassifier(
    supersede_threshold=1.2,  # Lower threshold = easier to supersede
    keep_threshold=0.8,       # Higher threshold = easier to keep old
)

# More conservative (more conflicts escalated to user)
auto_resolver = AutoResolutionClassifier(
    supersede_threshold=1.5,  # Higher threshold = harder to supersede
    keep_threshold=0.5,       # Lower threshold = harder to keep old
)
```

## Troubleshooting

### NLI Classifier Not Working

**Symptoms:** NLI classifier raises ImportError or doesn't load model.

**Solutions:**
1. Check if `sentence-transformers` is installed:
   ```bash
   uv sync --extra transformers
   ```

2. Verify model downloads work (first run downloads ~400MB):
   ```python
   from casual_memory.intelligence import NLIPreFilter
   nli = NLIPreFilter()
   nli.predict("I live in London", "I live in Paris")  # Triggers download
   ```

3. Check GPU availability:
   - NLI is faster on GPU (~50ms) but works on CPU (~200ms)
   - For CPU-only:
     ```bash
     pip install torch --index-url https://download.pytorch.org/whl/cpu
     ```

### High Memory Usage

**Solutions:**
1. Use lazy loading (already default - models load on first use)

2. Limit `max_similar` in `add_memory()` to reduce classification workload:
   ```python
   result = await service.add_memory(memory, max_similar=3)  # Default is 5
   ```

3. Use `strategy="single"` instead of `"tiered"` or `"all"`

4. Archive old memories periodically

### Slow Classification

**Solutions:**
1. Use NLI classifier first (fast pre-filter, ~50-200ms)

2. Switch to smaller/faster LLM model:
   ```python
   # Instead of gpt-4
   conflict_verifier = LLMConflictVerifier(llm_provider, "gpt-3.5-turbo")
   ```

3. Use `strategy="single"` to check only highest-similarity memory

4. Consider batching memory additions

### Too Many False Conflicts

**Symptoms:** Conflicts detected for memories that aren't really contradictory.

**Solutions:**
1. Increase NLI thresholds (stricter):
   ```python
   nli_classifier = NLIClassifier(
       nli_filter=nli_filter,
       entailment_threshold=0.90,  # Default is 0.85
   )
   ```

2. Adjust auto-resolution thresholds to resolve more automatically

3. Use more powerful LLM model for conflict detection

### Missing Conflicts

**Symptoms:** Contradictory memories not detected as conflicts.

**Solutions:**
1. Lower similarity threshold to find more similar memories:
   ```python
   result = await service.add_memory(memory, similarity_threshold=0.75)  # Default is 0.85
   ```

2. Increase `max_similar` to check more memories:
   ```python
   result = await service.add_memory(memory, max_similar=10)  # Default is 5
   ```

3. Use `strategy="all"` to check all similar memories

### LLM Provider Errors

**Symptoms:** LLM calls fail or timeout.

**Solutions:**
1. Check LLM provider configuration:
   ```python
   # For Ollama
   llm_provider = create_provider(ModelConfig(
       name="qwen2.5:7b-instruct",
       provider=Provider.OLLAMA,
       base_url="http://localhost:11434",  # Verify this is correct
   ))
   ```

2. Verify model is available:
   ```bash
   # For Ollama
   ollama list
   ollama pull qwen2.5:7b-instruct
   ```

3. Check network connectivity to API endpoints

4. The classifiers have automatic fallback to heuristics when LLM fails

## Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| NLI prediction | ~200ms | CPU (cold) |
| NLI prediction | ~50ms | GPU or cached |
| LLM conflict check | ~1.2s | qwen2.5:7b via Ollama |
| LLM duplicate check | ~1.0s | qwen2.5:7b via Ollama |
| Full pipeline (5 pairs) | ~3.5s | ~60% filtered by NLI |
| Qdrant vector search | ~50ms | 10k memories, top 5 results |

## Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/casualgenius/casual-memory/issues)
2. Review the [Architecture Guide](ARCHITECTURE.md)
3. Look at the [examples/](../examples/) directory
