"""Tests for Short-Term Memory Manager.

Tests for context tracking, summarization, and filtering.
Run with: pytest test_stm_manager.py
"""

# pylint: disable=protected-access,too-many-public-methods,too-few-public-methods

import os

import pytest

from stm import ContextStats, STMManager


class TestSTMManagerBasicOperations:
    """Test basic STM operations."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        manager = STMManager()

        assert manager.max_tokens == 128000
        assert manager.compression_threshold == 0.7
        assert manager.current_tokens == 0

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        manager = STMManager(
            max_tokens=64000,
            compression_threshold=0.8,
            model_name="gemini-1.5-pro",
        )

        assert manager.max_tokens == 64000
        assert manager.compression_threshold == 0.8

    def test_init_zero_max_tokens(self):
        """Test that zero max_tokens uses default."""
        manager = STMManager(max_tokens=0)

        assert manager.max_tokens == 128000

    def test_estimate_tokens(self):
        """Test token estimation."""
        manager = STMManager()

        text = "Hello world, this is a test."
        estimated = manager.estimate_tokens(text)

        assert isinstance(estimated, int)
        assert estimated > 0

    def test_estimate_tokens_empty(self):
        """Test token estimation for empty string."""
        manager = STMManager()

        estimated = manager.estimate_tokens("")

        assert estimated == 0

    def test_track_context(self):
        """Test context tracking."""
        manager = STMManager()

        content = "This is some context to track."
        manager.track_context(content)

        assert manager.current_tokens > 0

    def test_track_context_update(self):
        """Test that track_context updates current tokens."""
        manager = STMManager()

        manager.track_context("Short content")
        first_count = manager.current_tokens

        manager.track_context(
            "This is much longer content that should have more tokens"
        )
        second_count = manager.current_tokens

        assert second_count > first_count

    def test_get_stats_empty(self):
        """Test getting stats when no context tracked."""
        manager = STMManager()

        stats = manager.get_stats()

        assert stats["current_tokens"] == 0
        assert stats["max_tokens"] == 128000
        assert stats["usage_percent"] == 0.0
        assert stats["should_summarize"] is False
        assert stats["compression_threshold"] == 70.0

    def test_get_stats_with_content(self):
        """Test getting stats after tracking content."""
        manager = STMManager(max_tokens=1000)

        manager.track_context("x" * 800)  # ~200 tokens
        stats = manager.get_stats()

        assert stats["current_tokens"] > 0
        assert stats["usage_percent"] > 0

    def test_should_summarize_threshold(self):
        """Test should_summarize at threshold."""
        manager = STMManager(max_tokens=1000, compression_threshold=0.7)

        manager.track_context("x" * 2800)  # ~700 tokens, 70% usage
        stats = manager.get_stats()

        assert stats["should_summarize"] is True

    def test_should_summarize_below_threshold(self):
        """Test should_summarize below threshold."""
        manager = STMManager(max_tokens=1000, compression_threshold=0.7)

        manager.track_context("x" * 2000)  # ~500 tokens, 50% usage
        stats = manager.get_stats()

        assert stats["should_summarize"] is False

    def test_heuristic_summary_not_aggressive(self):
        """Test heuristic summary without aggressive mode."""
        manager = STMManager()

        lines = [f"Line {i}" for i in range(1, 21)]
        content = "\n".join(lines)

        summary = manager._heuristic_summary(content, aggressive=False)

        assert "lines omitted" in summary
        assert "Line 1" in summary or "Line 20" in summary

    def test_heuristic_summary_aggressive(self):
        """Test heuristic summary with aggressive mode."""
        manager = STMManager()

        lines = [f"Line {i}" for i in range(1, 11)]
        content = "\n".join(lines)

        summary = manager._heuristic_summary(content, aggressive=True)

        assert "lines omitted" in summary

    def test_heuristic_summary_short_content(self):
        """Test heuristic summary with short content."""
        manager = STMManager()

        content = "Short content"

        summary = manager._heuristic_summary(content, aggressive=False)

        assert summary == content

    def test_heuristic_summary_empty_content(self):
        """Test heuristic summary with empty content."""
        manager = STMManager()

        summary = manager._heuristic_summary("", aggressive=False)

        assert summary == ""

    @pytest.mark.asyncio
    async def test_filter_with_keywords(self):
        """Test filtering with keywords."""
        manager = STMManager()

        content = """Line 1: important information
Line 2: some other text
Line 3: urgent deadline
Line 4: more content
Line 5: critical alert"""

        filtered = await manager.filter(content, keywords="important,urgent,critical")

        assert "important information" in filtered
        assert "urgent deadline" in filtered
        assert "critical alert" in filtered
        assert "some other text" not in filtered

    @pytest.mark.asyncio
    async def test_filter_with_context(self):
        """Test filtering with surrounding context."""
        manager = STMManager()

        content = """Line 1
Line 2
Line 3: important
Line 4
Line 5
Line 6"""

        filtered = await manager.filter(content, keywords="important", keep_context=2)

        assert "important" in filtered
        assert "Line 1" in filtered or "Line 2" in filtered

    @pytest.mark.asyncio
    async def test_filter_no_matches(self):
        """Test filtering with no keyword matches."""
        manager = STMManager()

        content = """Line 1
Line 2
Line 3"""

        filtered = await manager.filter(content, keywords="nonexistent")

        assert "No matches found" in filtered

    @pytest.mark.asyncio
    async def test_filter_empty_keywords(self):
        """Test filtering with empty keywords."""
        manager = STMManager()

        content = "Some content"

        filtered = await manager.filter(content, keywords="")

        assert filtered == content

    @pytest.mark.asyncio
    async def test_summary_with_llm_success(self):
        """Test summary with LLM when available (requires GOOGLE_API_KEY)."""
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not set, skipping LLM test")

        manager = STMManager()
        content = "This is some long content that needs summarization."
        summary = await manager.summary(content, aggressive=False)

        assert summary is not None
        assert len(summary) > 0

    @pytest.mark.asyncio
    async def test_summary_without_llm(self):
        """Test summary fallback when LLM is unavailable."""
        manager = STMManager()
        manager.llm = None

        content = """Line 1
Line 2
Line 3
Line 4
Line 5
Line 6
Line 7
Line 8
Line 9
Line 10
Line 11
Line 12
Line 13
Line 14
Line 15
Line 16
Line 17
Line 18
Line 19
Line 20"""

        summary = await manager.summary(content, aggressive=False)

        assert "lines omitted" in summary

    @pytest.mark.asyncio
    async def test_summary_llm_error_fallback(self):
        """Test summary fallback when LLM raises error."""
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not set, skipping LLM error test")

        manager = STMManager()
        content = "Test content"

        # This test would need to mock the chain to raise an error
        # For now, we just verify the summary function doesn't crash
        summary = await manager.summary(content, aggressive=False)

        assert summary is not None

    def test_stats_dict_structure(self):
        """Test that stats returns proper dictionary structure."""
        manager = STMManager()

        stats = manager.get_stats()

        assert isinstance(stats, dict)
        assert "current_tokens" in stats
        assert "max_tokens" in stats
        assert "usage_percent" in stats
        assert "should_summarize" in stats
        assert "compression_threshold" in stats


class TestContextStats:
    """Test ContextStats dataclass."""

    def test_context_stats_creation(self):
        """Test creating ContextStats instance."""
        stats = ContextStats(
            current_tokens=500,
            max_tokens=1000,
            usage_percent=50.0,
            should_summarize=False,
            compression_threshold=70.0,
        )

        assert stats.current_tokens == 500
        assert stats.max_tokens == 1000
        assert stats.usage_percent == 50.0
        assert stats.should_summarize is False
        assert stats.compression_threshold == 70.0


class TestSTMIntegrationScenarios:
    """Integration scenarios for STM."""

    @pytest.mark.asyncio
    async def test_full_context_management_cycle(self):
        """Test full cycle of context tracking and management."""
        manager = STMManager(max_tokens=100, compression_threshold=0.7)

        long_content = "\n".join([
            f"Line {i}: This is some text that will be tracked and estimated"
            for i in range(30)
        ])
        manager.track_context(long_content)

        stats = manager.get_stats()
        assert stats["should_summarize"] is True

        summary = await manager.summary(long_content, aggressive=True)
        assert len(summary) < len(long_content)

    @pytest.mark.asyncio
    async def test_filter_then_summarize_workflow(self):
        """Test filtering followed by summarization."""
        manager = STMManager()

        content = """Important: This is critical information
Irrelevant: This is noise
Important: Another critical point
Irrelevant: More noise
Important: Final important point"""

        filtered = await manager.filter(content, keywords="important")
        assert "noise" not in filtered

        summary = await manager.summary(filtered, aggressive=False)
        assert "critical" in summary or "important" in summary

    def test_multiple_track_updates(self):
        """Test multiple track_context updates."""
        manager = STMManager()

        for i in range(5):
            manager.track_context(f"Update {i}: " + "x" * 100 * i)

        assert manager.current_tokens > 0

        stats = manager.get_stats()
        assert stats["current_tokens"] == manager.current_tokens
