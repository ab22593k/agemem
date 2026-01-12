"""Tests for Agentic Memory Processor.

Standardized to use shared fixtures and mocks.
Run with: pytest tests/test_agentic.py
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.ltm import MemoryEntry


@pytest.mark.asyncio
async def test_form_memory(agentic_processor, mock_llm_response):
    """Test memory formation with mocked LLM response."""
    mock_response = mock_llm_response(
        '```json\n{"keywords": ["test"], "tags": ["unit"], "context_description": "A test context."}\n```'
    )

    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = mock_response

    with patch("src.agentic.ChatPromptTemplate.from_messages") as mock_from:
        mock_from.return_value.__or__.return_value = mock_chain

        result = await agentic_processor.form_memory("Raw content")

        assert result["keywords"] == ["test"]
        assert result["tags"] == ["unit"]
        assert result["context_description"] == "A test context."


@pytest.mark.asyncio
async def test_evolve_memory(agentic_processor, mock_llm_response):
    """Test memory evolution (linking + evolution)."""
    mock_response = mock_llm_response(
        '```json\n{"links": ["neighbor1"], "evolutions": [{"id": "neighbor1", "context_description": "Evolved context"}]}\n```'
    )

    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = mock_response

    with patch("src.agentic.ChatPromptTemplate.from_messages") as mock_from:
        mock_from.return_value.__or__.return_value = mock_chain

        new_mem = MemoryEntry(id="new", content="New content")
        neighbors = [MemoryEntry(id="neighbor1", content="Old content")]

        result = await agentic_processor.evolve_memory(new_mem, neighbors)

        assert result["links"] == ["neighbor1"]
        assert result["evolutions"][0]["id"] == "neighbor1"
        assert result["evolutions"][0]["context_description"] == "Evolved context"


@pytest.mark.asyncio
async def test_plan_merge(agentic_processor, mock_llm_response):
    """Test planning a merge of redundant memories."""
    mock_response = mock_llm_response(
        '```json\n{"survivor_id": "mem1", "new_content": "Merged content", "new_context": "Merged context", "redundant_ids": ["mem2"]}\n```'
    )

    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = mock_response

    with patch("src.agentic.ChatPromptTemplate.from_messages") as mock_from:
        mock_from.return_value.__or__.return_value = mock_chain

        m1 = MemoryEntry(id="mem1", content="Content 1")
        m2 = MemoryEntry(id="mem2", content="Content 2")

        result = await agentic_processor.plan_merge([m1, m2])

        assert result["survivor_id"] == "mem1"
        assert result["new_content"] == "Merged content"
        assert "mem2" in result["redundant_ids"]
