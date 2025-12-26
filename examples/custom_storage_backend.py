"""
Custom Storage Backend Example

Demonstrates how to implement a custom storage backend using
the VectorStore protocol.
"""

import asyncio
from typing import List, Optional

from casual_memory.models import MemoryFact


class InMemoryVectorStore:
    """
    Simple in-memory vector store implementation.

    Implements VectorStore protocol via duck typing.
    """

    def __init__(self):
        self.memories: dict[str, MemoryFact] = {}
        self.next_id = 1

    async def initialize(self):
        """Initialize storage (no-op for in-memory)."""
        pass

    async def add(self, memory: MemoryFact, user_id: str) -> str:
        """Add memory and return ID."""
        memory_id = f"{user_id}_mem_{self.next_id}"
        self.next_id += 1

        # Store with user_id prefix for isolation
        memory.id = memory_id
        self.memories[memory_id] = memory

        return memory_id

    async def search(
        self, query_text: str, user_id: str, limit: int = 5, exclude_archived: bool = True
    ) -> List[MemoryFact]:
        """
        Simple keyword search (no embeddings).

        In production, this would use vector similarity.
        """
        results = []

        for memory_id, memory in self.memories.items():
            # User isolation
            if not memory_id.startswith(f"{user_id}_"):
                continue

            # Skip archived if requested
            if exclude_archived and memory.archived:
                continue

            # Simple keyword matching
            if query_text.lower() in memory.text.lower():
                results.append(memory)

            if len(results) >= limit:
                break

        return results

    async def update(self, memory_id: str, memory: MemoryFact, user_id: str):
        """Update existing memory."""
        if memory_id in self.memories:
            self.memories[memory_id] = memory

    async def archive(self, memory_id: str, user_id: str, superseded_by: Optional[str] = None):
        """Soft-delete memory."""
        if memory_id in self.memories:
            self.memories[memory_id].archived = True
            self.memories[memory_id].superseded_by = superseded_by


async def main():
    print("=== Custom Storage Backend Example ===\n")

    # Create custom storage
    storage = InMemoryVectorStore()
    await storage.initialize()

    # Add memories
    memory1 = MemoryFact(
        text="I live in Bangkok", type="fact", tags=["location"], importance=0.8, source="user"
    )

    memory2 = MemoryFact(
        text="I work in Bangkok",
        type="fact",
        tags=["location", "job"],
        importance=0.7,
        source="user",
    )

    mem_id1 = await storage.add(memory1, user_id="user_123")
    mem_id2 = await storage.add(memory2, user_id="user_123")

    print("Added 2 memories")
    print(f"  Memory 1 ID: {mem_id1}")
    print(f"  Memory 2 ID: {mem_id2}\n")

    # Search
    results = await storage.search("Bangkok", user_id="user_123", limit=5)

    print(f"Search results for 'Bangkok': {len(results)} found")
    for i, memory in enumerate(results, 1):
        print(f"  {i}. {memory.text}")

    # Archive one memory
    await storage.archive(mem_id1, user_id="user_123", superseded_by=mem_id2)
    print(f"\nArchived memory {mem_id1}")

    # Search again (excluding archived)
    results = await storage.search("Bangkok", user_id="user_123", exclude_archived=True)

    print(f"Search results after archiving: {len(results)} found")
    for i, memory in enumerate(results, 1):
        print(f"  {i}. {memory.text}")

    print("\n=== Example Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
