# casual-memory

**Intelligent semantic memory with conflict detection, classification pipeline, and storage abstraction**

[![PyPI version](https://badge.fury.io/py/casual-memory.svg)](https://badge.fury.io/py/casual-memory)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/casualgenius/casual-memory/workflows/Tests/badge.svg)](https://github.com/casualgenius/casual-memory/actions)

---

## 🚀 Features

### 👑 Classification Pipeline (Core Innovation)
- **Protocol-based architecture** - Composable, extensible classifiers
- **NLI Pre-filtering** - Fast semantic filtering (~50-200ms)
- **LLM Conflict Detection** - High-accuracy contradiction detection (96%+)
- **LLM Duplicate Detection** - Smart deduplication vs distinct facts
- **Auto-Resolution** - Confidence-based conflict resolution
- **Graceful degradation** - Heuristic fallback when LLM unavailable

### 🧠 Memory Intelligence
- **Memory extraction** from conversations (user & assistant messages)
- **Conflict detection** with categorization (location, preference, temporal, factual)
- **Confidence scoring** based on mention frequency and recency
- **Memory archiving** with soft-delete patterns
- **Temporal memory** with date normalization and expiry

### 🔌 Storage Abstraction
- **Protocol-based** - Works with any vector database
- **Optional adapters** - Qdrant, PostgreSQL, Redis
- **In-memory implementations** - For testing
- **Bring your own** - Implement custom backends

---

## 📦 Installation

### Minimal (core only)
```bash
pip install casual-memory
```

### With specific backends
```bash
# With NLI support (sentence-transformers)
pip install casual-memory[transformers]

# With Qdrant vector store
pip install casual-memory[qdrant]

# With PostgreSQL conflict store
pip install casual-memory[postgres]

# With Redis short-term store
pip install casual-memory[redis]

# Full installation (all extras)
pip install casual-memory[all]
```

### CPU-only installation (no CUDA)
By default, PyTorch includes CUDA which is a large download. For CPU-only machines:
```bash
# Install CPU-only PyTorch first
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Then install casual-memory with transformers
pip install casual-memory[transformers]
```

### For development
```bash
git clone https://github.com/yourusername/casual-memory
cd casual-memory
uv sync --all-extras
```

---

## 🎯 Quick Start

### Classification Pipeline

```python
from casual_memory.classifiers import (
    MemoryClassificationPipeline,
    NLIClassifier,
    ConflictClassifier,
    DuplicateClassifier,
    AutoResolutionClassifier,
    SimilarMemory,
)
from casual_memory.intelligence import NLIPreFilter, LLMConflictVerifier, LLMDuplicateDetector
from casual_memory import MemoryFact
from casual_llm import create_client, create_model, ClientConfig, ModelConfig, Provider

# Initialize components
nli_filter = NLIPreFilter()
client = create_client(ClientConfig(
    provider=Provider.OLLAMA,
    base_url="http://localhost:11434"
))
model = create_model(client, ModelConfig(name="qwen2.5:7b-instruct"))

conflict_verifier = LLMConflictVerifier(model, "qwen2.5:7b-instruct")
duplicate_detector = LLMDuplicateDetector(model, "qwen2.5:7b-instruct")

# Build pipeline
pipeline = MemoryClassificationPipeline(
    classifiers=[
        NLIClassifier(nli_filter=nli_filter),
        ConflictClassifier(llm_conflict_verifier=conflict_verifier),
        DuplicateClassifier(llm_duplicate_detector=duplicate_detector),
        AutoResolutionClassifier(supersede_threshold=1.3, keep_threshold=0.7),
    ],
    strategy="tiered",  # "single", "tiered", or "all"
)

# Create new memory and similar memories from vector search
new_memory = MemoryFact(
    text="I live in Paris",
    type="fact",
    tags=["location"],
    importance=0.9,
    confidence=0.8,
    entity_id="user-123",
)

similar_memories = [
    SimilarMemory(
        memory_id="mem_001",
        memory=MemoryFact(
            text="I live in London",
            type="fact",
            tags=["location"],
            importance=0.8,
            confidence=0.6,
            entity_id="user-123",
        ),
        similarity_score=0.91,
    )
]

# Classify the new memory against similar memories
result = await pipeline.classify(new_memory, similar_memories)

# Check overall outcome: "add", "skip", or "conflict"
print(f"Overall outcome: {result.overall_outcome}")

# Check individual similarity results
for sim_result in result.similarity_results:
    print(f"Similar memory: {sim_result.similar_memory.memory.text}")
    print(f"Outcome: {sim_result.outcome}")  # "conflict", "superseded", "same", "neutral"
    print(f"Classifier: {sim_result.classifier_name}")
    if sim_result.outcome == "conflict":
        print(f"Category: {sim_result.metadata.get('category')}")
```

### Memory Extraction

```python
from casual_memory.extractors import LLMMemoryExtracter
from casual_llm import UserMessage, AssistantMessage

# Create extractors for user and assistant memories
user_extractor = LLMMemoryExtracter(model=model, source="user")
assistant_extractor = LLMMemoryExtracter(model=model, source="assistant")

messages = [
    UserMessage(content="My name is Alex and I live in Bangkok"),
    AssistantMessage(content="Nice to meet you, Alex!"),
]

# Extract user-stated memories
user_memories = await user_extractor.extract(messages)
# [MemoryFact(text="My name is Alex", type="fact", importance=0.9, ...),
#  MemoryFact(text="I live in Bangkok", type="fact", importance=0.8, ...)]

# Extract assistant-observed memories
assistant_memories = await assistant_extractor.extract(messages)
```

### Custom Storage Backend

```python
from typing import Any, Optional

class MyVectorStore:
    """Custom vector store implementing VectorMemoryStore protocol"""

    def add(self, embedding: list[float], payload: dict) -> str:
        """Add memory to store, return memory ID."""
        # Your implementation
        return "memory_id"

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        min_score: float = 0.7,
        filters: Optional[Any] = None,
    ) -> list[Any]:
        """Search for similar memories."""
        # Your implementation
        return []

    def update_memory(self, memory_id: str, updates: dict) -> bool:
        """Update memory metadata (mention_count, last_seen, etc.)."""
        # Your implementation
        return True

    def archive_memory(self, memory_id: str, superseded_by: Optional[str] = None) -> bool:
        """Soft-delete memory."""
        # Your implementation
        return True
```

---

## 🏗️ Architecture

### Classification Pipeline Flow

```
Input: New Memory + Similar Memories (from vector search)
  ↓
1. NLI Classifier (~50-200ms)
  ├─ High entailment (≥0.85) → same (duplicate)
  ├─ High neutral (≥0.5) → neutral (distinct)
  └─ Uncertain → Pass to next classifier
  ↓
2. Conflict Classifier (~500-2000ms)
  ├─ LLM detects contradiction → conflict
  └─ No conflict → Pass to next classifier
  ↓
3. Duplicate Classifier (~500-2000ms)
  ├─ Same fact → same
  ├─ Refinement (more detail) → superseded
  └─ Distinct facts → neutral
  ↓
4. Auto-Resolution Classifier
  ├─ Analyze conflict results
  ├─ High new confidence (ratio ≥1.3) → superseded (keep_new)
  ├─ High old confidence (ratio ≤0.7) → same (keep_old)
  └─ Similar confidence → Keep as conflict
  ↓
Output: MemoryClassificationResult
  ├─ overall_outcome: "add" | "skip" | "conflict"
  └─ similarity_results: Individual outcomes for each similar memory
```

### Key Concepts

**Similarity Outcomes** (for each similar memory):
- `conflict` - Contradictory memories requiring user resolution
- `superseded` - Similar memory should be archived (new one is better)
- `same` - Duplicate memory (update existing metadata)
- `neutral` - Distinct facts that can coexist

**Memory Outcomes** (overall action):
- `add` - Insert new memory to vector store
- `skip` - Update existing memory (increment mention_count)
- `conflict` - Create conflict record for user resolution

**Confidence Scoring:**
- Based on mention frequency (1 mention = 0.5, 5+ mentions = 0.95)
- Recency factor (decay after 30 days)
- Spread factor (boost if mentioned over time)

**Memory Types:**
- `fact` - Factual information (name, location, job, etc.)
- `preference` - User preferences (likes, dislikes, habits)
- `goal` - User goals and aspirations
- `event` - Events (past or future)

---

## 📚 Documentation

- [Architecture Guide](docs/ARCHITECTURE.md) - System design and concepts
- [Examples](examples/) - Working example code

---

## 🧪 Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=casual_memory --cov-report=html

# Run specific test file
uv run pytest tests/classifiers/test_pipeline.py -v

# Run specific test
uv run pytest tests/classifiers/test_pipeline.py::test_pipeline_sequential_execution -v
```

---

## 🎯 Benchmarks

Classification pipeline performance on our test dataset:

| Model | Conflict Accuracy | Avg Time |
|-------|-------------------|----------|
| qwen2.5:7b-instruct | 96.2% | 1.2s |
| llama3:8b | 94.5% | 1.5s |
| gpt-3.5-turbo | 97.1% | 0.8s |

NLI Pre-filter performance:
- Accuracy: 92.38% (SNLI), 90.04% (MNLI)
- Speed: ~200ms CPU, ~50ms GPU
- Filters: 70-85% of obvious cases before LLM

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

Built with:
- [casual-llm](https://github.com/yourusername/casual-llm) - LLM provider abstraction
- [sentence-transformers](https://www.sbert.net/) - NLI models
- Inspired by research in semantic memory and conflict detection

---

## 🔗 Links

- [Documentation](https://github.com/yourusername/casual-memory#readme)
- [Issue Tracker](https://github.com/yourusername/casual-memory/issues)
- [Changelog](CHANGELOG.md)
- [Examples](examples/)
