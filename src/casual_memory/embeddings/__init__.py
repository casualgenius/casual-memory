"""
Text embedding abstractions for casual-memory.

Provides protocol-based embedding interfaces with model-specific adapters:
- E5Embedding: E5 model family with automatic prefix handling
- OpenAIEmbedding: OpenAI API embeddings
"""

from casual_memory.embeddings.protocol import TextEmbedding

__all__ = [
    "TextEmbedding",
]

# Optional adapters (import only if dependencies available)
try:
    from casual_memory.embeddings.e5_embedding import E5Embedding

    __all__.append("E5Embedding")
except ImportError:
    pass

try:
    from casual_memory.embeddings.openai_embedding import OpenAIEmbedding

    __all__.append("OpenAIEmbedding")
except ImportError:
    pass
