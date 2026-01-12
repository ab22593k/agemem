"""Tests for Short-Term Memory Manager.

Tests for context tracking, summarization, and filtering.
Run with: pytest tests/test_stm.py
"""

import pytest


class TestSTMManagerBasicOperations:
    """Test basic STM operations."""

    def test_init_default_params(self, stm_manager):
        """Test initialization with default parameters."""
        assert stm_manager.max_tokens == 1000  # From fixture
        assert stm_manager.current_tokens == 0

    def test_estimate_tokens(self, stm_manager):
        """Test token estimation."""
        text = "Hello world, this is a test."
        estimated = stm_manager.estimate_tokens(text)

        assert isinstance(estimated, int)
        assert estimated > 0

    def test_track_context(self, stm_manager):
        """Test context tracking update."""
        stm_manager.track_context("Short content")
        first_count = stm_manager.current_tokens

        stm_manager.track_context("Much longer content that should have more tokens")
        second_count = stm_manager.current_tokens

        assert second_count > first_count

    def test_get_stats(self, stm_manager):
        """Test getting stats after tracking content."""
        stm_manager.track_context("x" * 800)  # ~200 tokens
        stats = stm_manager.get_stats()

        assert stats["current_tokens"] > 0
        assert stats["usage_percent"] > 0
        assert "should_summarize" in stats

    @pytest.mark.asyncio
    async def test_filter_with_keywords(self, stm_manager):
        """Test filtering with keywords."""
        content = """Line 1: important information
Line 2: some other text
Line 3: urgent deadline"""

        filtered = await stm_manager.filter(content, criteria="important,urgent", semantic=False)

        assert "important information" in filtered
        assert "urgent deadline" in filtered
        assert "some other text" not in filtered

    @pytest.mark.asyncio
    async def test_summary_with_llm_skip(self, stm_manager):
        """Test summary skips LLM when API key missing or forces heuristic."""
        stm_manager.llm = None  # Force heuristic
        content = "\n".join([f"Line {i}" for i in range(1, 21)])

        summary = await stm_manager.summary(content, aggressive=False)

        assert "lines omitted" in summary

    @pytest.mark.asyncio
    async def test_summary_with_span(self, stm_manager):
        """Test summary with integer span (last N lines)."""
        stm_manager.llm = None  # Force heuristic
        content = ["Msg 1", "Msg 2", "Msg 3", "Msg 4", "Msg 5"]

        summary = await stm_manager.summary(content, span=2)

        assert "Msg 4" in summary
        assert "Msg 5" in summary
        assert "Msg 1" not in summary


class TestSTMIntegrationScenarios:
    """Integration scenarios for STM."""

    @pytest.mark.asyncio
    async def test_full_context_management_cycle(self, stm_manager):
        """Test full cycle of context tracking and management."""
        long_content = "\n".join([f"Line {i}: dummy text content" for i in range(100)])
        stm_manager.track_context(long_content)

        _ = stm_manager.get_stats()
        # Heuristic: 100 lines * approx 25 chars / 4 = 625 tokens.
        # With max_tokens=1000 and threshold 0.7, it might be close.

        summary = await stm_manager.summary(long_content, aggressive=True)
        assert len(summary) < len(long_content)
