# Migration Guide

This guide helps you migrate from the existing memory services (`memory-agent-service` and `memory-store-service`) to the `casual-memory` library.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Differences](#key-differences)
3. [Migration Steps](#migration-steps)
4. [Code Examples](#code-examples)
5. [Breaking Changes](#breaking-changes)
6. [Troubleshooting](#troubleshooting)

---

## Overview

### Why Migrate?

The `casual-memory` library extracts the core intelligence and storage logic from the microservices into a reusable, well-tested library:

**Benefits:**
- ✅ **Reusable** - Use in any Python project, not just the AI assistant
- ✅ **Well-tested** - 93 tests with 51% coverage
- ✅ **Type-safe** - Full type hints and protocol-based design
- ✅ **Documented** - Comprehensive docs and examples
- ✅ **Flexible** - Swap LLM providers, storage backends
- ✅ **Standalone** - No service dependencies

---

## Key Differences

### Import Changes

| Old | New |
|-----|-----|
| `from memory_store_service.app.conflict_detector import ConflictDetector` | `from casual_memory.intelligence import LLMConflictVerifier` |
| `from memory_agent_service.app.memory_extractor import MemoryExtractor` | `from casual_memory.extractors import LLMMemoryExtractor` |
| `from shared.models.memory import MemoryFact` | `from casual_memory.models import MemoryFact` |

### Method Changes

| Old | New |
|-----|-----|
| `provider.generate(messages)` | `provider.chat(messages)` (returns AssistantMessage) |
| `extractor.extract(messages, source="user")` | `LLMMemoryExtractor(source="user").extract(messages)` |
| `detector.detect_conflict(...)` | `verifier.verify_conflict(...)` |

---

## Migration Steps

### Step 1: Install casual-memory

```bash
uv add casual-memory
uv add ../casual-llm --path  # If using local development version
```

### Step 2: Update Imports

```python
# Before
from memory_agent_service.app.memory_extractor import MemoryExtractor
from shared.llm_providers import create_provider

# After
from casual_memory.extractors import LLMMemoryExtractor
from casual_llm import create_provider, ModelConfig, Provider
```

### Step 3: Update Configuration

```python
# Before
config = ModelConfig(provider="ollama", base_url="http://localhost:11434/api/chat")

# After
config = ModelConfig(provider=Provider.OLLAMA, base_url="http://localhost:11434")
# Providers append paths internally (/api/chat)
```

### Step 4: Update Code

```python
# Before
provider = create_provider(config)
extractor = MemoryExtractor(provider)
memories = await extractor.extract(messages, source="user")

# After
provider = create_provider(config)
extractor = LLMMemoryExtractor(llm_provider=provider, source="user")
memories = await extractor.extract(messages)
```

---

## Code Examples

### Memory Extraction

```python
from casual_memory.extractors import LLMMemoryExtractor
from casual_llm import create_provider, ModelConfig, Provider

config = ModelConfig(
    name="qwen2.5:7b-instruct",
    provider=Provider.OLLAMA,
    base_url="http://localhost:11434"
)

provider = create_provider(config)
user_extractor = LLMMemoryExtractor(llm_provider=provider, source="user")
assistant_extractor = LLMMemoryExtractor(llm_provider=provider, source="assistant")

# Extract from conversation
user_memories = await user_extractor.extract(messages)
assistant_memories = await assistant_extractor.extract(messages)
```

### Classification Pipeline

```python
from casual_memory.classifiers import (
    ClassificationPipeline,
    NLIClassifier,
    ConflictClassifier
)
from casual_memory.intelligence import NLIPreFilter, LLMConflictVerifier

nli_filter = NLIPreFilter()
conflict_verifier = LLMConflictVerifier(provider, "qwen2.5:7b")

pipeline = ClassificationPipeline(classifiers=[
    NLIClassifier(nli_filter=nli_filter),
    ConflictClassifier(llm_conflict_verifier=conflict_verifier)
])

result = await pipeline.classify(request)
```

---

## Breaking Changes

### 1. Provider Response Type

```python
# Before: String
response = await provider.generate(messages)
content = response

# After: AssistantMessage
response = await provider.chat(messages)
content = response.content
```

### 2. Base URL Format

```python
# Wrong
base_url="http://localhost:11434/api/chat"  # Don't include path

# Correct
base_url="http://localhost:11434"  # Providers add /api/chat automatically
```

### 3. Provider Enum

```python
# Before
provider="ollama"  # String

# After
provider=Provider.OLLAMA  # Enum
```

---

## Troubleshooting

### ImportError: No module named 'casual_llm'

```bash
# Install casual-llm
uv add ../casual-llm --path
```

### Model not loading (NLI)

```bash
# Install transformers extra
uv add "casual-memory[transformers]"
```

### Base URL connection errors

Make sure to use base URL only:
```python
base_url="http://localhost:11434"  # Correct
base_url="http://localhost:11434/api/chat"  # Wrong
```

---

## Migration Checklist

- [ ] Install casual-memory and casual-llm
- [ ] Update imports
- [ ] Replace `generate()` with `chat()`
- [ ] Update ModelConfig (Provider enum, base URLs)
- [ ] Update test mocks (AsyncMock for chat())
- [ ] Run tests and fix type errors

---

## Getting Help

- **Documentation**: [README.md](../README.md) | [ARCHITECTURE.md](ARCHITECTURE.md)
- **Examples**: [examples/](../examples/)
- **Tests**: [tests/](../tests/)

For ai-assistant specific migration, see LIBRARY_EXTRACTION_PLAN.md.
