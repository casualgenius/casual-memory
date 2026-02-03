import logging
from datetime import datetime
from typing import Sequence

from casual_llm import ChatMessage, LLMProvider, SystemMessage, UserMessage
from pydantic import ValidationError

from casual_memory.extractors.models import MemoryExtractionResponse
from casual_memory.models import MemoryFact

logger = logging.getLogger(__name__)


class LLMMemoryExtracter:
    """Extracts memories from messages in conversations using JSON schema."""

    def __init__(self, llm_provider: LLMProvider, prompt: str):
        self.prompt = prompt
        self.llm_provider = llm_provider

    async def extract(self, messages: list[ChatMessage]) -> list[MemoryFact]:
        from casual_memory.utils.date_normalizer import normalize_memory_dates

        now = datetime.now()

        # Format prompt with current date/time
        system_prompt = self.prompt.format(
            today_natural=now.strftime("%A, %B %d, %Y"), isonow=now.isoformat()
        )

        prompt = "\n".join([message.model_dump_json() for message in messages])

        # Build LLM messages using casual-llm format
        llm_messages: Sequence[SystemMessage | UserMessage] = [
            SystemMessage(content=system_prompt),
            UserMessage(content=prompt),
        ]

        try:
            logger.debug("Extracting memories with JSON schema")
            response = await self.llm_provider.chat(
                messages=llm_messages,  # type: ignore[arg-type]
                response_format=MemoryExtractionResponse,  # Pass Pydantic model
                temperature=0.2,
            )

            # Parse JSON response to Pydantic model
            content = response.content
            if content is None:
                raise ValueError("LLM response content is None")

            # DEBUG: Log raw LLM response to check if importance field is included
            logger.debug(f"Raw LLM response: {content[:500]}...")  # First 500 chars

            extraction_response = MemoryExtractionResponse.model_validate_json(content)
            memories = extraction_response.memories

            logger.debug(f"LLM returned {len(memories)} memories")

        except ValidationError as e:
            logger.error(f"Failed to validate memory extraction response: {e}")
            raise ValueError(f"LLM response did not match expected schema: {e}") from e
        except Exception as e:
            logger.error(f"Memory extraction failed: {e}")
            raise

        # Convert extraction results to full MemoryFact instances
        # MemoryFactExtraction -> normalize dates -> MemoryFact with system fields
        filtered_memories: list[MemoryFact] = []
        for memory_extraction in memories:
            # Convert extraction model to dict and normalize dates
            memory_dict = memory_extraction.model_dump()
            normalized_dict = normalize_memory_dates(memory_dict, now)

            # Convert to full MemoryFact (adds system-managed fields with defaults)
            # System fields (user_id, confidence, mention_count, etc.) will be set
            # to their defaults and can be updated by the calling code
            normalized_memory = MemoryFact(**normalized_dict)

            # Filter by importance threshold
            if normalized_memory.importance >= 0.5:
                filtered_memories.append(normalized_memory)

        logger.info(f"Extracted {len(filtered_memories)} memories (filtered from {len(memories)})")

        return filtered_memories
