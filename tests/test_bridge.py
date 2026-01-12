"""Tests for Bridge CLI.

Run with: pytest tests/test_bridge.py
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.bridge_cli import main
from src.ltm import MemoryFunction


@pytest.mark.asyncio
async def test_bridge_retrieve_links():
    """Test the bridge retrieve command with the --links flag."""
    mock_entry = MagicMock()
    mock_entry.id = "test-id"
    mock_entry.content = "Test content"
    mock_entry.context_description = "Context"
    mock_entry.keywords = ["kw1"]
    mock_entry.memory_function = MemoryFunction.FACTUAL

    mock_ltm = AsyncMock()
    mock_ltm.retrieve.return_value = [mock_entry]
    mock_ltm.__aenter__.return_value = mock_ltm

    with (
        patch("sys.argv", ["bridge.py", "retrieve", "query", "--links"]),
        patch("src.bridge_cli.LTMManager", return_value=mock_ltm),
        patch("src.bridge_cli.AgenticMemoryProcessor"),
        patch("builtins.print") as mock_print,
    ):
        await main()

        mock_ltm.retrieve.assert_called_once_with(
            query="query", top_k=5, include_links=True, memory_function=None
        )

        assert mock_print.call_count >= 1
        args, _ = mock_print.call_args_list[0]
        assert "ID: test-id" in args[0]
        assert "Significance: Context" in args[0]


@pytest.mark.asyncio
async def test_bridge_memorize_flow():
    """Test the memorize command triggers the updated agentic pipeline."""
    mock_entry = MagicMock()
    mock_entry.id = "new-id"

    mock_ltm = AsyncMock()
    mock_ltm.add.return_value = mock_entry
    mock_ltm.__aenter__.return_value = mock_ltm

    mock_proc = MagicMock()
    mock_proc.llm = MagicMock()
    mock_proc.form_memory = AsyncMock(
        return_value={"keywords": ["k"], "tags": ["t"], "context_description": "c"}
    )
    mock_proc.orchestrate_lifecycle = AsyncMock()

    with (
        patch("sys.argv", ["bridge.py", "memorize", "New content"]),
        patch("src.bridge_cli.LTMManager", return_value=mock_ltm),
        patch("src.bridge_cli.AgenticMemoryProcessor", return_value=mock_proc),
        patch("builtins.print"),
    ):
        await main()

        # Verify new Dynamics methods are called
        mock_proc.form_memory.assert_called_once_with(
            "New content", function=MemoryFunction.FACTUAL
        )
        mock_ltm.add.assert_called_once()
        _, kwargs = mock_ltm.add.call_args
        assert kwargs["keywords"] == ["k"]
        assert kwargs["context_description"] == "c"
        mock_proc.orchestrate_lifecycle.assert_called_once()
