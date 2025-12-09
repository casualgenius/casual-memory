"""
Example: Using LLMMemoryExtracter with custom prompts

This example demonstrates how to use the LLMMemoryExtracter with different
system prompts for extracting user memories vs. assistant-provided information.
"""

import asyncio
from casual_llm import create_provider, ModelConfig, Provider, UserMessage, AssistantMessage
from casual_memory import LLMMemoryExtracter
from casual_memory.extractors.prompts import USER_MEMORY_PROMPT, ASSISTANT_MEMORY_PROMPT


async def main():
    # Setup LLM provider (using Ollama for this example)
    model = ModelConfig(
        name="qwen2.5:7b-instruct",
        provider=Provider.OLLAMA,
    )
    llm_provider = create_provider(model)

    print("=" * 80)
    print("Example 1: Extracting User Memories")
    print("=" * 80)

    # Create extractor with user memory prompt
    user_extractor = LLMMemoryExtracter(
        llm_provider=llm_provider,
        prompt=USER_MEMORY_PROMPT
    )

    # Example conversation with user memories
    user_conversation = [
        UserMessage(content="My name is Alex and I live in London."),
        AssistantMessage(content="Nice to meet you, Alex! I'll remember that."),
        UserMessage(content="I'm allergic to peanuts and I have a doctor's appointment tomorrow at 2pm."),
        AssistantMessage(content="Important to note about the allergy. I've noted your appointment as well."),
    ]

    # Extract user memories
    user_memories = await user_extractor.extract(user_conversation)

    print(f"\nExtracted {len(user_memories)} user memories:")
    for i, memory in enumerate(user_memories, 1):
        print(f"\n{i}. {memory.text}")
        print(f"   Type: {memory.type}")
        print(f"   Tags: {', '.join(memory.tags)}")
        print(f"   Importance: {memory.importance}")
        print(f"   Source: {memory.source}")

    print("\n" + "=" * 80)
    print("Example 2: Extracting Assistant Memories")
    print("=" * 80)

    # Create extractor with assistant memory prompt
    assistant_extractor = LLMMemoryExtracter(
        llm_provider=llm_provider,
        prompt=ASSISTANT_MEMORY_PROMPT
    )

    # Example conversation with assistant-provided information
    assistant_conversation = [
        UserMessage(content="What's the weather like tomorrow?"),
        AssistantMessage(content="Tomorrow will be partly cloudy with a high of 15°C and a low of 8°C. There's a 20% chance of rain in the afternoon."),
        UserMessage(content="Can you recommend a good Italian restaurant nearby?"),
        AssistantMessage(content="Based on your location in London, I recommend Padella in Borough Market. They're known for excellent fresh pasta and the prices are reasonable."),
    ]

    # Extract assistant memories
    assistant_memories = await assistant_extractor.extract(assistant_conversation)

    print(f"\nExtracted {len(assistant_memories)} assistant memories:")
    for i, memory in enumerate(assistant_memories, 1):
        print(f"\n{i}. {memory.text}")
        print(f"   Type: {memory.type}")
        print(f"   Tags: {', '.join(memory.tags)}")
        print(f"   Importance: {memory.importance}")
        print(f"   Source: {memory.source}")
        if memory.valid_until:
            print(f"   Valid Until: {memory.valid_until}")

    print("\n" + "=" * 80)
    print("Example 3: Using a Custom Prompt")
    print("=" * 80)

    # You can also create your own custom prompts for specific use cases
    custom_prompt = """Extract only location-related information from the conversation.

Return a JSON object with a "memories" array. Each memory should have:
- text: The location fact
- type: Always "fact"
- tags: Location-related tags
- importance: 0.8
- source: "user" or "assistant"
- valid_until: null

Today is {today_natural} (ISO: {isonow}).

Only extract location information. Return {{"memories": []}} if none found.
"""

    location_extractor = LLMMemoryExtracter(
        llm_provider=llm_provider,
        prompt=custom_prompt
    )

    location_conversation = [
        UserMessage(content="I'm traveling to Paris next week and then to Tokyo in December."),
        AssistantMessage(content="Enjoy your trips! Paris is beautiful this time of year."),
    ]

    location_memories = await location_extractor.extract(location_conversation)

    print(f"\nExtracted {len(location_memories)} location memories:")
    for i, memory in enumerate(location_memories, 1):
        print(f"\n{i}. {memory.text}")
        print(f"   Type: {memory.type}")
        print(f"   Tags: {', '.join(memory.tags)}")

    print("\n" + "=" * 80)
    print("Key Takeaways")
    print("=" * 80)
    print("""
1. LLMMemoryExtracter is flexible - use different prompts for different contexts
2. USER_MEMORY_PROMPT: For extracting facts, preferences, goals from user messages
3. ASSISTANT_MEMORY_PROMPT: For extracting tool results, recommendations, calculations
4. Create custom prompts for specialized extraction needs
5. The prompt receives {today_natural} and {isonow} placeholders for date context
    """)


if __name__ == "__main__":
    asyncio.run(main())
