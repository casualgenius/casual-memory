import logging
from datetime import datetime
from typing import Sequence

from casual_llm import ChatMessage, LLMProvider, SystemMessage, UserMessage
from pydantic import BaseModel, ValidationError

from casual_memory.extractors.models import MemoryExtractionResponse
from casual_memory.models import MemoryFact

logger = logging.getLogger(__name__)


class LLMMemoryExtracter:
    """Extracts memories from messages in conversations using JSON schema.

    The extractor supports custom extraction models and prompts for flexibility:
    - extraction_model: Pydantic model defining the JSON schema for LLM response.
      Must have a 'memories' attribute returning a list of memory-like objects.
    - prompt: System prompt for the LLM with {today_natural} and {isonow} placeholders.

    Example with defaults (standard memory extraction):
        extractor = LLMMemoryExtracter(llm_provider, USER_MEMORY_PROMPT)

    Example with custom model and prompt (e.g., Moltbook):
        class CustomMemory(BaseModel):
            text: str
            type: str  # Custom types like 'insight', 'reflection'
            importance: float

        class CustomExtractionResponse(BaseModel):
            memories: list[CustomMemory]

        extractor = LLMMemoryExtracter(
            llm_provider=provider,
            prompt=CUSTOM_PROMPT,
            extraction_model=CustomExtractionResponse,
        )
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt: str,
        extraction_model: type[BaseModel] = MemoryExtractionResponse,
    ):
        """Initialize the memory extractor.

        Args:
            llm_provider: LLM provider for chat completions (casual-llm compatible).
            prompt: System prompt with {today_natural} and {isonow} placeholders.
            extraction_model: Pydantic model for structured LLM output.
                Must have a 'memories' attribute. Defaults to MemoryExtractionResponse.
        """
        self.prompt = prompt
        self.llm_provider = llm_provider
        self.extraction_model = extraction_model

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
                response_format=self.extraction_model,  # Pass configurable Pydantic model
                temperature=0.2,
            )

            # Parse JSON response to Pydantic model
            content = response.content
            if content is None:
                raise ValueError("LLM response content is None")

            # DEBUG: Log raw LLM response to check if importance field is included
            logger.debug(f"Raw LLM response: {content[:500]}...")  # First 500 chars

            extraction_response = self.extraction_model.model_validate_json(content)

            # Validate that the extraction model exposes a 'memories' list
            if not hasattr(extraction_response, "memories"):
                raise ValueError(
                    f"Extraction model {self.extraction_model.__name__} must have a "
                    f"'memories' attribute, but {type(extraction_response).__name__} "
                    f"has: {list(extraction_response.model_fields.keys())}"
                )

            memories = extraction_response.memories
            if not isinstance(memories, list):
                raise ValueError(
                    f"Expected 'memories' to be a list, got {type(memories).__name__} "
                    f"from {self.extraction_model.__name__}"
                )

            for i, item in enumerate(memories):
                if not isinstance(item, BaseModel):
                    raise ValueError(
                        f"Expected memories[{i}] to be a BaseModel instance, "
                        f"got {type(item).__name__} from {self.extraction_model.__name__}"
                    )

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
