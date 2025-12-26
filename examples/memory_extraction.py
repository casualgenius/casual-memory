"""
Memory Extraction Example

Demonstrates how to extract memories from conversations using
LLM-based extraction.
"""

import asyncio
from casual_memory.extractors import LLMMemoryExtractor
from casual_llm import create_provider, ModelConfig, Provider, UserMessage, AssistantMessage


async def main():
    print("=== Memory Extraction Example ===\n")

    # Initialize LLM provider
    llm_provider = create_provider(ModelConfig(
        name="qwen2.5:7b-instruct",
        provider=Provider.OLLAMA,
        base_url="http://localhost:11434",
        temperature=0.2  # Low temperature for consistent extraction
    ))

    # Create extractors for user and assistant memories
    user_extractor = LLMMemoryExtractor(llm_provider=llm_provider, source="user")
    assistant_extractor = LLMMemoryExtractor(llm_provider=llm_provider, source="assistant")

    # Sample conversation
    messages = [
        UserMessage(content="My name is Alex and I live in Bangkok. I work as a software engineer."),
        AssistantMessage(content="Nice to meet you, Alex! Bangkok is a great city."),
        UserMessage(content="I really enjoy hiking and photography in my free time."),
        AssistantMessage(content="Those are wonderful hobbies!"),
    ]

    print("Conversation:")
    for msg in messages:
        role = "User" if isinstance(msg, UserMessage) else "Assistant"
        print(f"  {role}: {msg.content}")

    print("\nExtracting memories...\n")

    # Extract user-stated memories
    user_memories = await user_extractor.extract(messages)

    print(f"User Memories Extracted ({len(user_memories)}):")
    for i, memory in enumerate(user_memories, 1):
        print(f"  {i}. {memory.text}")
        print(f"     Type: {memory.type}, Importance: {memory.importance:.2f}")
        print(f"     Tags: {memory.tags}")

    # Extract assistant-observed memories
    assistant_memories = await assistant_extractor.extract(messages)

    print(f"\nAssistant Memories Extracted ({len(assistant_memories)}):")
    for i, memory in enumerate(assistant_memories, 1):
        print(f"  {i}. {memory.text}")
        print(f"     Type: {memory.type}, Importance: {memory.importance:.2f}")

    print("\n=== Example Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
