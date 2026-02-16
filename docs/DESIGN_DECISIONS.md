# Design Decisions

This document explains the key design patterns and architectural decisions in casual-memory.

## 1. Protocol-Based Architecture

**Why:** Maximum flexibility without tight coupling

- All storage uses Python `Protocol` (structural subtyping, not inheritance)
- Enables swapping implementations without code changes
- Works with any vector database, SQL dialect, or cache backend
- Runtime type checking with `isinstance()` checks

```python
# ❌ Inheritance-based (rigid)
class MyClassifier(BaseClassifier):
    def classify(self, request):
        return super().classify(request)

# ✅ Protocol-based (flexible)
class MyClassifier:
    async def classify(self, new_memory, similar_memories, results=None):
        # Automatically implements MemoryClassifier protocol
        return results
```

## 2. Sequential Classifier Chain

**Why:** Chain-of-responsibility pattern with graceful fallback

- Each classifier can pass uncertain cases to next classifier
- Early stopping on high-confidence decisions (performance optimization)
- Graceful degradation when components fail (e.g., LLM unavailable)
- Classifiers are stateless and composable

```python
# Classifiers only classify what they're confident about
# Uncertain cases pass to the next classifier
pipeline = MemoryClassificationPipeline(
    classifiers=[
        NLIClassifier(...),      # Fast, filters ~70-85% of cases
        ConflictClassifier(...),  # LLM-based, catches contradictions
        DuplicateClassifier(...), # LLM-based, catches duplicates
        AutoResolutionClassifier(...),  # Resolves by confidence ratio
    ]
)
```

## 3. Lazy Loading

**Why:** Faster startup, lower memory usage

- Models (NLI, embeddings) loaded only on first use
- Avoids loading sentence-transformers if not needed
- Reduces initial memory footprint

```python
class NLIPreFilter:
    def __init__(self):
        self.model = None  # Lazy load

    def predict(self, text_a, text_b):
        if self.model is None:
            self.model = CrossEncoder(...)  # Load on first use
        return self.model.predict(...)
```

## 4. Soft Delete with Versioning

**Why:** Preserve history and enable recovery

- `archived=True` flag instead of deletion
- `superseded_by` field tracks replacement chain
- Enables temporal queries and rollback
- Supports audit trails and debugging

```python
# When memory is superseded, it's archived not deleted
await vector_store.archive_memory(
    memory_id="mem_123",
    superseded_by="mem_456"  # New memory ID
)

# After archiving:
# archived=True
# archived_at="2024-01-15T10:30:00Z"
# superseded_by="mem_456"
```

## 5. Multi-Entity Isolation with Namespace Scoping

**Why:** Built for production multi-tenant systems with flexible grouping

- `entity_id` and `namespace` fields in all memory structures from the start
- All storage operations filter by both `entity_id` and `namespace`
- **Namespaces** provide logical isolation (e.g., `"work"`, `"personal"`)
- **Entity IDs** identify whose memories they are (e.g., a user, agent, or organization)
- Prevents data leakage between entities and across namespaces

```python
# Different entities have separate memory spaces
vector_store.add(embedding, payload={"entity_id": "alice", "namespace": "default", ...})
vector_store.add(embedding, payload={"entity_id": "bob", "namespace": "default", ...})

# Searches scoped by entity_id and namespace
results = vector_store.search(embedding, filters={"entity_id": "alice", "namespace": "work"})
```

> **Deprecation note**: The `user_id` field/parameter is deprecated across all models and services. Use `entity_id` instead. Passing `user_id` still works (with a `DeprecationWarning`) during the migration period.

## 6. Async/Await Throughout

**Why:** Non-blocking I/O for efficiency

- All LLM calls, embeddings, and storage operations are async
- Enables concurrent processing of multiple memories
- Better resource utilization in production
- Natural fit for web frameworks (FastAPI, etc.)

```python
# Sequential (slow)
for memory in memories:
    result = await classifier.classify(memory, similar)

# Concurrent (fast) - when operations are independent
tasks = [classifier.classify(m, s) for m, s in pairs]
results = await asyncio.gather(*tasks)
```

## 7. Metadata-Rich Results

**Why:** Auditability and debugging

- Every decision includes confidence score
- Results include classifier name that made the decision
- Metadata dict carries decision-specific context (NLI scores, conflict category)
- Enables metrics tracking and decision analysis

```python
SimilarityResult(
    similar_memory=...,
    outcome="conflict",
    confidence=0.85,
    classifier_name="ConflictClassifier",
    metadata={
        "category": "location",
        "clarification_hint": "Which city do you currently live in?",
        "detection_method": "llm",
    }
)
```

## 8. Graceful Degradation with Optional Dependencies

**Why:** Core functionality works without all extras

- Core works without transformers/Qdrant/PostgreSQL
- Optional imports with try/except in `__init__.py`
- Fallback to heuristics when LLM unavailable

```python
try:
    from sentence_transformers import CrossEncoder
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class NLIPreFilter:
    def predict(self, text_a, text_b):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Install: pip install casual-memory[transformers]")
        ...
```

## 9. Confidence-Based Decision Making

**Why:** Minimize user interruptions

- Auto-resolution when confidence ratio is clear (≥1.3 or ≤0.7)
- Only escalate to user when truly ambiguous
- Confidence scoring based on mention frequency, recency, and spread
- Reduces cognitive load on users

```python
# New memory confidence = 0.9, existing = 0.5
# Ratio = 0.9 / 0.5 = 1.8 (≥ 1.3 threshold)
# → Auto-resolve: supersede existing with new

# New memory confidence = 0.8, existing = 0.8
# Ratio = 1.0 (between 0.7 and 1.3)
# → Keep as conflict: ask user to decide
```

## 10. Pydantic Models for Validation

**Why:** Type safety and automatic validation

- All data models use Pydantic `BaseModel`
- Automatic validation on construction
- Serialization/deserialization for free
- OpenAPI schema generation for APIs
- Better IDE support and type hints

```python
class MemoryFact(BaseModel):
    text: str
    type: str
    importance: float  # Validated: must be float

# Automatic validation
fact = MemoryFact(text="I live in London", type="fact", importance="invalid")
# Raises: ValidationError
```

## Trade-offs and Alternatives

### Why Protocol-based instead of ABC/inheritance?

**Chosen:** Protocols (structural subtyping)
**Alternative:** Abstract Base Classes (nominal subtyping)

Protocols allow any class with matching methods to be used, without explicit inheritance. This is more flexible for users implementing custom storage backends.

### Why Sequential instead of Parallel classification?

**Chosen:** Sequential with early termination
**Alternative:** Parallel classification with voting

Sequential allows early termination when a confident decision is made (e.g., NLI says "same" with 0.95 confidence). Parallel would require waiting for all classifiers, wasting LLM calls.

### Why Async everywhere?

**Chosen:** Async throughout
**Alternative:** Sync with optional async wrappers

Async from the start avoids the "function coloring" problem where you'd need to maintain both sync and async versions. Most use cases involve I/O (LLM calls, database queries) that benefit from async.

### Why Soft delete instead of hard delete?

**Chosen:** Soft delete with `archived` flag
**Alternative:** Hard delete with separate audit log

Soft delete keeps everything in one place, simplifies queries for "what was this memory replaced by", and allows easy recovery. The trade-off is slightly larger storage.
