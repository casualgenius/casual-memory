"""Shared test helpers."""

from unittest.mock import AsyncMock, Mock


class MockModel:
    """Mock casual-llm Model for testing.

    Simulates Model.chat() by returning a mock AssistantMessage with
    the given response_content as its content attribute.
    """

    def __init__(self, response_content: str, name: str = "test-model"):
        self.name = name
        self.response_content = response_content
        self.chat = AsyncMock(return_value=Mock(content=response_content))
