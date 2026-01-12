"""Tests for Long-Term Memory Manager.

Integration tests require a running Weaviate instance.
Run with: pytest tests/test_ltm.py
"""

# pylint: disable=redefined-outer-name,unused-argument,broad-exception-caught

import asyncio
import os
from datetime import datetime

import pytest

from src.ltm import LTMManager


@pytest.fixture
def weaviate_host():
    """Get Weaviate host from environment or use default."""
    return os.getenv("WEAVIATE_HOST", "localhost:8080")


@pytest.fixture
def manager(weaviate_host):
    """Create and initialize LTM Manager for testing."""
    manager = LTMManager(host=weaviate_host, use_vector_search=False)

    async def setup():
        await manager.initialize()
        return manager

    return asyncio.run(setup())


@pytest.fixture
async def cleanup(manager):
    """Cleanup after each test."""
    yield manager

    try:
        await manager.close()
    except Exception:
        pass


class TestLTMManagerBasicOperations:
    """Test basic LTM operations."""

    @pytest.mark.asyncio
    async def test_add_memory(self, manager):
        """Test adding a new memory."""
        content = "Test memory entry at " + datetime.now().isoformat()
        metadata = {"tags": ["test", "unit"], "session_id": "test-session"}

        entry = await manager.add(
            content=content,
            metadata=metadata,
            memory_type="context",
            quality=0.7,
        )

        assert entry.id is not None
        assert entry.id != ""
        assert entry.content == content
        assert entry.metadata == metadata
        assert entry.memory_type == "context"
        assert entry.quality == 0.7
        assert entry.usage_count == 0

    @pytest.mark.asyncio
    async def test_add_memory_default_params(self, manager):
        """Test adding memory with default parameters."""
        content = "Test memory with defaults"

        entry = await manager.add(content=content)

        assert entry.id is not None
        assert entry.content == content
        assert entry.metadata == {}
        assert entry.memory_type is None
        assert entry.quality == 0.5

    @pytest.mark.asyncio
    async def test_add_memory_invalid_quality(self, manager):
        """Test that invalid quality is normalized to default."""
        content = "Test memory with invalid quality"

        entry = await manager.add(content=content, quality=-1)

        assert entry.quality == 0.5

    @pytest.mark.asyncio
    async def test_add_memory_too_long(self, manager):
        """Test that content exceeding max length raises error."""
        content = "x" * (manager.MAX_CONTENT_LENGTH + 1)

        with pytest.raises(ValueError, match="exceeds maximum length"):
            await manager.add(content=content)

    @pytest.mark.asyncio
    async def test_retrieve_memory(self, manager):
        """Test retrieving memories."""
        content = "Unique test content for retrieval: " + datetime.now().isoformat()

        await manager.add(content=content, memory_type="retrieval-test")

        results = await manager.retrieve(query="retrieval-test", top_k=5)

        assert len(results) > 0
        assert any("Unique test content for retrieval" in r.content for r in results)

    @pytest.mark.asyncio
    async def test_retrieve_memory_with_filters(self, manager):
        """Test retrieving memories with filters."""
        memory_type = "filter-test"

        await manager.add(
            content="Memory with session and user",
            memory_type=memory_type,
            quality=0.6,
        )

        results = await manager.retrieve(
            query="filter-test",
            memory_type=memory_type,
            min_quality=0.4,
        )

        assert len(results) > 0
        assert all(r.memory_type == memory_type for r in results)

    @pytest.mark.asyncio
    async def test_retrieve_memory_update_usage(self, manager):
        """Test that retrieval updates usage count."""
        content = "Test usage tracking: " + datetime.now().isoformat()

        entry = await manager.add(content=content, memory_type="usage-test")
        initial_count = int(entry.usage_count or 0)

        await manager.retrieve(query="usage-test", update_usage=True)

        results = await manager.retrieve(query="usage-test", update_usage=False)
        assert len(results) > 0
        # Wait a bit for Weaviate to update
        await asyncio.sleep(0.5)
        # Fetch again to be sure
        results = await manager.retrieve(query="usage-test", update_usage=False)
        assert int(results[0].usage_count or 0) > initial_count

    @pytest.mark.asyncio
    async def test_retrieve_memory_limit(self, manager):
        """Test that retrieval respects top_k parameter."""
        for i in range(10):
            await manager.add(
                content=f"Test memory {i}",
                memory_type="limit-test",
                quality=0.5 + (i * 0.04),
            )

        results = await manager.retrieve(query="limit-test", top_k=3)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_update_memory(self, manager):
        """Test updating an existing memory."""
        content = "Original content"
        updated_content = "Updated content"

        entry = await manager.add(content=content)
        original_id = entry.id

        updated_entry = await manager.update(
            entry_id=original_id, content=updated_content
        )

        assert updated_entry.id == original_id
        assert updated_entry.content == updated_content

    @pytest.mark.asyncio
    async def test_update_memory_not_found(self, manager):
        """Test updating non-existent memory."""
        with pytest.raises(ValueError, match="not found"):
            await manager.update(entry_id="non-existent-id", content="content")

    @pytest.mark.asyncio
    async def test_delete_memory(self, manager):
        """Test deleting a memory."""
        content = "To be deleted"
        entry = await manager.add(content=content)
        memory_id = entry.id

        await manager.delete(entry_id=memory_id)

        results = await manager.retrieve(query="To be deleted")
        assert not any(r.id == memory_id for r in results)

    @pytest.mark.asyncio
    async def test_delete_memory_not_found(self, manager):
        """Test deleting non-existent memory."""
        with pytest.raises(ValueError, match="not found"):
            await manager.delete(entry_id="non-existent-id")

    @pytest.mark.asyncio
    async def test_update_quality(self, manager):
        """Test updating memory quality."""
        content = "Test quality update"
        entry = await manager.add(content=content, quality=0.5)
        memory_id = entry.id

        await manager.update_quality(entry_id=memory_id, quality=0.9)

        results = await manager.retrieve(query="quality update")
        assert len(results) > 0
        assert results[0].quality == 0.9

    @pytest.mark.asyncio
    async def test_update_quality_invalid_range(self, manager):
        """Test that invalid quality values raise error."""
        content = "Test invalid quality"
        entry = await manager.add(content=content)

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            await manager.update_quality(entry_id=entry.id, quality=1.5)

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            await manager.update_quality(entry_id=entry.id, quality=-0.1)

    @pytest.mark.asyncio
    async def test_get_stats(self, manager):
        """Test getting memory statistics."""
        await manager.add(content="Memory 1", memory_type="stats", quality=0.8)
        await manager.add(content="Memory 2", memory_type="stats", quality=0.6)
        await manager.add(content="Memory 3", memory_type="stats", quality=0.4)

        stats = await manager.get_stats()

        assert "total_memories" in stats
        assert "average_quality" in stats
        assert "total_retrievals" in stats
        assert stats["total_memories"] >= 3

    @pytest.mark.asyncio
    async def test_memory_metadata_persistence(self, manager):
        """Test that metadata is properly persisted and retrieved."""
        metadata = {"important": True, "project": "alpha"}
        content = "Test metadata persistence"

        await manager.add(content=content, metadata=metadata)

        results = await manager.retrieve(query="metadata persistence")
        assert len(results) > 0
        found = False
        for r in results:
            if r.content == content:
                assert r.metadata == metadata
                found = True
                break
        assert found
