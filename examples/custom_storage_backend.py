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

    def _key(self, namespace: str, entity_id: str) -> str:
        """Build a composite key prefix for namespace + entity isolation."""
        return f"{namespace}:{entity_id}"

    async def add(self, memory: MemoryFact, entity_id: str, namespace: str = "default") -> str:
        """Add memory and return ID."""
        prefix = self._key(namespace, entity_id)
        memory_id = f"{prefix}_mem_{self.next_id}"
        self.next_id += 1

        memory.id = memory_id
        self.memories[memory_id] = memory

        return memory_id

    async def search(
        self,
        query_text: str,
        entity_id: str,
        namespace: str = "default",
        limit: int = 5,
        exclude_archived: bool = True,
    ) -> List[MemoryFact]:
        """
        Simple keyword search (no embeddings).

        In production, this would use vector similarity.
        """
        prefix = self._key(namespace, entity_id)
        results = []

        for memory_id, memory in self.memories.items():
            # Namespace + entity isolation
            if not memory_id.startswith(f"{prefix}_"):
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

    async def update(
        self, memory_id: str, memory: MemoryFact, entity_id: str, namespace: str = "default"
    ):
        """Update existing memory."""
        if memory_id in self.memories:
            self.memories[memory_id] = memory

    async def archive(
        self,
        memory_id: str,
        entity_id: str,
        namespace: str = "default",
        superseded_by: Optional[str] = None,
    ):
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

    mem_id1 = await storage.add(memory1, entity_id="user-123")
    mem_id2 = await storage.add(memory2, entity_id="user-123")

    print("Added 2 memories")
    print(f"  Memory 1 ID: {mem_id1}")
    print(f"  Memory 2 ID: {mem_id2}\n")

    # Search
    results = await storage.search("Bangkok", entity_id="user-123", limit=5)

    print(f"Search results for 'Bangkok': {len(results)} found")
    for i, memory in enumerate(results, 1):
        print(f"  {i}. {memory.text}")

    # Archive one memory
    await storage.archive(mem_id1, entity_id="user-123", superseded_by=mem_id2)
    print(f"\nArchived memory {mem_id1}")

    # Search again (excluding archived)
    results = await storage.search("Bangkok", entity_id="user-123", exclude_archived=True)

    print(f"Search results after archiving: {len(results)} found")
    for i, memory in enumerate(results, 1):
        print(f"  {i}. {memory.text}")

    # --- Namespace isolation ---
    print("\n--- Namespace Isolation ---\n")

    # Same entity, different namespaces → memories are isolated
    work_memory = MemoryFact(
        text="I work at Acme Corp", type="fact", tags=["job"], importance=0.9, source="user"
    )
    personal_memory = MemoryFact(
        text="I work out at the gym every morning",
        type="fact",
        tags=["routine"],
        importance=0.6,
        source="user",
    )

    await storage.add(work_memory, entity_id="user-123", namespace="work")
    await storage.add(personal_memory, entity_id="user-123", namespace="personal")

    # Search "work" in each namespace
    work_results = await storage.search("work", entity_id="user-123", namespace="work")
    personal_results = await storage.search("work", entity_id="user-123", namespace="personal")

    print(f"Search 'work' in namespace='work': {len(work_results)} found")
    for m in work_results:
        print(f"  - {m.text}")

    print(f"Search 'work' in namespace='personal': {len(personal_results)} found")
    for m in personal_results:
        print(f"  - {m.text}")

    # Neither namespace sees the other's memories
    print(
        f"\nNamespaces are isolated: work sees {len(work_results)}, personal sees {len(personal_results)}"
    )

    print("\n=== Example Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
