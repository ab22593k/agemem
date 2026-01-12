"""Shared pytest fixtures for agemem tests."""

import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

# Add project root to sys.path for easier imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agentic import AgenticMemoryProcessor
from src.ltm import LTMManager
from src.stm import STMManager


@pytest.fixture
def mock_llm_response():
    """Helper to create a mocked LLM response."""

    def _create_response(content):
        mock = MagicMock()
        mock.content = content
        return mock

    return _create_response


@pytest.fixture
def mock_llm():
    """Mocked LangChain LLM."""
    llm = AsyncMock()
    return llm


@pytest.fixture
def agentic_processor(mock_llm):
    """AgenticMemoryProcessor with a mocked LLM."""
    proc = AgenticMemoryProcessor()
    proc.llm = mock_llm
    return proc


@pytest.fixture
def stm_manager():
    """STMManager instance."""
    return STMManager(max_tokens=1000)


@pytest.fixture
def weaviate_host():
    """Get Weaviate host from environment or use default."""
    return os.getenv("WEAVIATE_HOST", "localhost:8080")


@pytest.fixture
async def ltm_manager(weaviate_host):
    """LTMManager instance for integration tests."""
    manager = LTMManager(host=weaviate_host, use_vector_search=False)
    await manager.initialize()
    yield manager
    await manager.close()
