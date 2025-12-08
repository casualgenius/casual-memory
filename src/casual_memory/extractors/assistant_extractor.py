from datetime import datetime
from typing import List
import json
from casual_memory.models import MemoryFact
from casual_llm import ChatMessage, LLMProvider
import logging

logger = logging.getLogger(__name__)

assistant_system_prompt = """You are an expert memory extraction agent for assistant responses. Your purpose is to identify valuable, personalized information from assistant messages that should be remembered.

You will be given a conversation as a list of JSON messages.

### Instructions
1.  **Analyze Assistant Messages**: Look for valuable information provided by the assistant.
2.  **Extract Selectively**: ONLY extract information that is personalized, user-specific, or the result of tool calls. Do NOT extract generic facts or educational content.
3.  **Format Output**: Return a single JSON object with a top-level key "memories", containing a list of memory objects. Do not include any other text, explanations, or markdown.

For each memory you find, return a JSON object with the following fields:
- `text` (string): A concise, self-contained statement.
- `type` (string): Must be EXACTLY one of: `fact`, `preference`, `goal`, or `event`
    - Use `fact` for tool results, calculations, weather forecasts
    - Use `preference` for recommendations (restaurants, products, etc.)
- `tags` (list of strings): A list of relevant lowercase keywords.
- `importance` (float): **CRITICAL** - Score from 0.0 to 1.0. For assistant-provided information that you choose to extract, use **0.9 or 1.0** (not 0.7 or 0.8). If information isn't important enough for 0.9+, don't extract it at all.
- `source` (string): **REQUIRED**. Must be "assistant".
- `valid_until` (string, optional): ISO8601 timestamp for when this memory expires. Today is {today_natural} (ISO: {isonow}).
    - **Weather forecasts**: Set to end of forecast day (e.g., "2025-11-14T23:59:59" for tomorrow's weather)
    - **Restaurant/product recommendations**: 7 days from today
    - **Calculations based on user data**: 30 days from today
    - **If no expiry needed**: use null

### CRITICAL RULES
- **ONLY Extract Personalized Information**:
    - ✅ DO extract: Weather forecasts, calculation results, personalized recommendations, tool results
    - ❌ DO NOT extract: Generic facts ("Paris is the capital of France"), educational content ("Python is a programming language"), conversational filler
- **Key Test**: Ask "Is this specific to THIS user or could it apply to anyone?" If it applies to anyone, DO NOT extract it.
- **Always Return "memories" Array**: Even if empty, return {{"memories": []}}

### Examples

Example 1 - Weather forecast (DO extract):
User: "What's the weather tomorrow?"
Assistant: "Tomorrow (November 13th) will be partly cloudy with a high of 15°C."

Response:
```json
{{
  "memories": [
    {{
      "text": "The weather on November 13th will be partly cloudy with a high of 15°C.",
      "type": "fact",
      "tags": ["weather", "forecast", "temperature"],
      "importance": 0.9,
      "source": "assistant",
      "valid_until": "2025-11-13T23:59:59"
    }}
  ]
}}
```

Example 2 - Restaurant recommendation (DO extract):
User: "Recommend a good Italian restaurant?"
Assistant: "Based on your location in London, try Padella in Borough Market for fresh pasta."

Response:
```json
{{
  "memories": [
    {{
      "text": "Padella in Borough Market is recommended for fresh pasta.",
      "type": "preference",
      "tags": ["restaurant", "italian", "recommendation", "padella"],
      "importance": 0.9,
      "source": "assistant",
      "valid_until": "2025-11-19T23:59:59"
    }}
  ]
}}
```

Example 3 - Calculation based on user data (DO extract):
User: "How much would I save if I reduced my coffee spending from £5 per day to £2 per day?"
Assistant: "You'll save £3 per day. Over a month (30 days), that's £90 in savings. Over a year, you'd save £1,095!"

Response:
```json
{{
  "memories": [
    {{
      "text": "Reducing daily coffee spending from £5 to £2 would save £3 per day, £90 per month, and £1,095 per year.",
      "type": "fact",
      "tags": ["savings", "coffee", "calculation", "budget"],
      "importance": 1.0,
      "source": "assistant",
      "valid_until": "2025-12-13T23:59:59"
    }}
  ]
}}
```

Note: Calculations expire in 30 days since they're based on current spending patterns that might change.

Example 4 - Generic fact (DO NOT extract):
User: "What's the capital of France?"
Assistant: "The capital of France is Paris."

Response:
```json
{{
  "memories": []
}}
```

Today's date is {today_natural} (ISO: {isonow}). Extract only personalized, valuable information.
"""


class AssistantMemoryExtracter:
    """Extracts memories from assistant messages - tool results, recommendations, calculations."""

    def __init__(self, memory_llm: LLMProvider):
        self.memory_llm = memory_llm

    async def extract(self, messages: List[ChatMessage]) -> List[MemoryFact]:
        memories: List[MemoryFact] = []
        now = datetime.now()

        system_prompt = assistant_system_prompt.format(
            today_natural = now.strftime("%A, %B %d, %Y"),
            isonow = now.isoformat()
        )

        try:
            response = await self.memory_llm.chat(messages=messages, system_prompt=system_prompt, response_format="json")
            response_data = json.loads(response.content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse assistant memory extraction JSON: {e}")
            return memories
        except Exception as e:
            logger.error(f"Assistant Memory LLM Failed: {e}")
            return memories

        for result in response_data["memories"]:
            # Filter using raw importance BEFORE weighting
            # Weighting should be applied later during retrieval/ranking if needed
            raw_importance = result.get("importance", 0.5)
            if raw_importance >= 0.5:
                memory = MemoryFact(
                    text=result["text"],
                    type=result.get("type", "fact"),
                    tags=result.get("tags", []),
                    importance=raw_importance,  # Store raw importance
                    source="assistant",  # Always assistant
                    valid_until=result.get("valid_until", None)
                )
                memories.append(memory)

        logger.info(f"Extracted {len(memories)} assistant memories")

        return memories
