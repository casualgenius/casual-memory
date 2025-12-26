import json
import logging
from datetime import datetime
from typing import List

from casual_llm import ChatMessage, LLMProvider, SystemMessage, UserMessage

from casual_memory.models import MemoryFact

logger = logging.getLogger(__name__)


class LLMMemoryExtracter:
    """Extracts memories from messages in conversations."""

    def __init__(self, llm_provider: LLMProvider, prompt: str):
        self.prompt = prompt
        self.llm_provider = llm_provider

    async def extract(self, messages: List[ChatMessage]) -> List[MemoryFact]:
        from casual_memory.utils.date_normalizer import normalize_memory_dates

        memories: List[MemoryFact] = []
        now = datetime.now()

        # Simplified prompt - no date calculation needed
        system_prompt = self.prompt.format(
            today_natural=now.strftime("%A, %B %d, %Y"), isonow=now.isoformat()
        )

        prompt = "\n".join([message.model_dump_json() for message in messages])

        # Build LLM messages using casual-llm format
        llm_messages = [
            SystemMessage(content=system_prompt),
            UserMessage(content=prompt),
        ]

        try:
            logger.debug("Extracting memories")
            response = await self.llm_provider.chat(
                messages=llm_messages, response_format="json", temperature=0.2
            )
            response_data = json.loads(response.content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse memory extraction JSON: {e}")
            return memories
        except Exception as e:
            logger.error(f"Memory LLM Failed: {e}")
            return memories

        for result in response_data["memories"]:
            # Normalize dates in the memory before creating MemoryFact
            result = normalize_memory_dates(result, now)

            # Filter using raw importance BEFORE weighting
            # Weighting should be applied later during retrieval/ranking if needed
            raw_importance = result.get("importance", 0.5)
            if raw_importance >= 0.5:
                memory = MemoryFact(
                    text=result["text"],
                    type=result.get("type", "fact"),
                    tags=result.get("tags", []),
                    importance=raw_importance,  # Store raw importance
                    source=result["source"],
                    valid_until=result.get("valid_until", None),
                )
                memories.append(memory)

        logger.info(f"Extracted {len(memories)} user memories")

        return memories
