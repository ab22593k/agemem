"""Tests for Long-Term Memory Manager.

Integration tests require a running Weaviate instance.
Run with: pytest test_ltm_manager.py
"""

# pylint: disable=redefined-outer-name,unused-argument,broad-exception-caught

import asyncio
import os
import uuid
from datetime import datetime

import pytest

from src.ltm import LTMManager


@pytest.fixture
def weaviate_host():
    """Get Weaviate host from environment or use default."""
    return os.getenv("WEAVIATE_HOST", "localhost:8080")


@pytest.fixture
def ltm_manager(weaviate_host):
    """Create and initialize LTM Manager for testing."""
    manager = LTMManager(host=weaviate_host, use_vector_search=False)

    async def setup():
        await manager.initialize()
        return manager

    return asyncio.run(setup())


@pytest.fixture
async def cleanup(ltm_manager):
    """Cleanup after each test."""
    yield ltm_manager

    try:
        await ltm_manager.close()
    except Exception:
        pass


class TestLTMManagerBasicOperations:
    """Test basic LTM operations."""

    @pytest.mark.asyncio
    async def test_add_memory(self, ltm_manager):
        """Test adding a new memory."""
        content = "Test memory entry at " + datetime.now().isoformat()
        tags = ["test", "unit"]

        entry = await ltm_manager.add(
            content=content,
            tags=tags,
            session_id="test-session",
            user_id="test-user",
            quality=0.7,
        )

        assert entry.id is not None
        assert entry.id != ""
        assert entry.content == content
        assert entry.tags == tags
        assert entry.session_id == "test-session"
        assert entry.user_id == "test-user"
        assert entry.quality == 0.7
        assert entry.usage_count == 0

    @pytest.mark.asyncio
    async def test_add_memory_default_params(self, ltm_manager):
        """Test adding memory with default parameters."""
        content = "Test memory with defaults"

        entry = await ltm_manager.add(content=content, tags=[])

        assert entry.id is not None
        assert entry.content == content
        assert entry.tags == []
        assert entry.session_id is None
        assert entry.user_id is None
        assert entry.quality == 0.5

    @pytest.mark.asyncio
    async def test_add_memory_invalid_quality(self, ltm_manager):
        """Test that invalid quality is normalized to default."""
        content = "Test memory with invalid quality"

        entry = await ltm_manager.add(content=content, tags=[], quality=-1)

        assert entry.quality == 0.5

    @pytest.mark.asyncio
    async def test_add_memory_too_long(self, ltm_manager):
        """Test that content exceeding max length raises error."""
        content = "x" * (ltm_manager.MAX_CONTENT_LENGTH + 1)

        with pytest.raises(ValueError, match="exceeds maximum length"):
            await ltm_manager.add(content=content, tags=[])

    @pytest.mark.asyncio
    async def test_retrieve_memory(self, ltm_manager):
        """Test retrieving memories."""
        content = "Unique test content for retrieval: " + datetime.now().isoformat()

        await ltm_manager.add(content=content, tags=["retrieval-test"])

        results = await ltm_manager.retrieve(query="retrieval-test", limit=5)

        assert len(results) > 0
        assert any("Unique test content for retrieval" in r.content for r in results)

    @pytest.mark.asyncio
    async def test_retrieve_memory_with_filters(self, ltm_manager):
        """Test retrieving memories with filters."""
        session_id = "test-session-123"
        user_id = "test-user-456"

        await ltm_manager.add(
            content="Memory with session and user",
            tags=["filter-test"],
            session_id=session_id,
            user_id=user_id,
        )

        results = await ltm_manager.retrieve(
            query="filter-test",
            session_id=session_id,
            user_id=user_id,
            min_quality=0.4,
        )

        assert len(results) > 0
        assert all(r.session_id == session_id for r in results)
        assert all(r.user_id == user_id for r in results)

    @pytest.mark.asyncio
    async def test_retrieve_memory_update_usage(self, ltm_manager):
        """Test that retrieval updates usage count."""
        content = "Test usage tracking: " + datetime.now().isoformat()

        entry = await ltm_manager.add(content=content, tags=["usage-test"])
        initial_count = int(entry.usage_count or 0)

        await ltm_manager.retrieve(query="usage-test", update_usage=True)

        results = await ltm_manager.retrieve(query="usage-test", update_usage=False)
        assert len(results) > 0
        assert int(results[0].usage_count or 0) > initial_count

    @pytest.mark.asyncio
    async def test_retrieve_memory_limit(self, ltm_manager):
        """Test that retrieval respects limit parameter."""
        for i in range(10):
            await ltm_manager.add(
                content=f"Test memory {i}",
                tags=["limit-test"],
                quality=0.5 + (i * 0.04),
            )

        results = await ltm_manager.retrieve(query="limit-test", limit=3)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_update_memory(self, ltm_manager):
        """Test updating an existing memory."""
        content = "Original content"
        updated_content = "Updated content"

        entry = await ltm_manager.add(content=content, tags=[])
        original_id = entry.id

        updated_entry = await ltm_manager.update(
            entry_id=original_id, content=updated_content
        )

        assert updated_entry.id == original_id
        assert updated_entry.content == updated_content

    @pytest.mark.asyncio
    async def test_update_memory_not_found(self, ltm_manager):
        """Test updating non-existent memory."""
        with pytest.raises(ValueError, match="not found"):
            await ltm_manager.update(entry_id="non-existent-id", content="content")

    @pytest.mark.asyncio
    async def test_delete_memory(self, ltm_manager):
        """Test deleting a memory."""
        content = "To be deleted"
        entry = await ltm_manager.add(content=content, tags=[])
        memory_id = entry.id

        await ltm_manager.delete(entry_id=memory_id)

        results = await ltm_manager.retrieve(query="To be deleted")
        assert not any(r.id == memory_id for r in results)

    @pytest.mark.asyncio
    async def test_delete_memory_not_found(self, ltm_manager):
        """Test deleting non-existent memory."""
        with pytest.raises(ValueError, match="not found"):
            await ltm_manager.delete(entry_id="non-existent-id")

    @pytest.mark.asyncio
    async def test_update_quality(self, ltm_manager):
        """Test updating memory quality."""
        content = "Test quality update"
        entry = await ltm_manager.add(content=content, tags=[], quality=0.5)
        memory_id = entry.id

        await ltm_manager.update_quality(entry_id=memory_id, quality=0.9)

        results = await ltm_manager.retrieve(query="quality update")
        assert len(results) > 0
        assert results[0].quality == 0.9

    @pytest.mark.asyncio
    async def test_update_quality_invalid_range(self, ltm_manager):
        """Test that invalid quality values raise error."""
        content = "Test invalid quality"
        entry = await ltm_manager.add(content=content, tags=[])

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            await ltm_manager.update_quality(entry_id=entry.id, quality=1.5)

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            await ltm_manager.update_quality(entry_id=entry.id, quality=-0.1)

    @pytest.mark.asyncio
    async def test_get_stats(self, ltm_manager):
        """Test getting memory statistics."""
        await ltm_manager.add(content="Memory 1", tags=["stats"], quality=0.8)
        await ltm_manager.add(content="Memory 2", tags=["stats"], quality=0.6)
        await ltm_manager.add(content="Memory 3", tags=["stats"], quality=0.4)

        stats = await ltm_manager.get_stats()

        assert "total_memories" in stats
        assert "average_quality" in stats
        assert "total_retrievals" in stats
        assert stats["total_memories"] >= 3

    @pytest.mark.asyncio
    async def test_memory_tags_persistence(self, ltm_manager):
        """Test that tags are properly persisted and retrieved."""
        tags = ["important", "user-preference", "project-alpha"]
        content = "Test tag persistence"

        await ltm_manager.add(content=content, tags=tags)

        results = await ltm_manager.retrieve(query="tag persistence")
        assert len(results) > 0
        # Weaviate might return more results, check if our entry is there
        found = False
        for r in results:
            if r.content == content:
                assert set(r.tags) == set(tags)
                found = True
                break
        assert found

    @pytest.mark.asyncio
    async def test_session_scoping(self, ltm_manager):
        """Test that session scoping works correctly."""
        session_a = f"session-a-{uuid.uuid4().hex[:8]}"
        session_b = f"session-b-{uuid.uuid4().hex[:8]}"

        await ltm_manager.add(
            content="Session A memory", tags=["session"], session_id=session_a
        )
        await ltm_manager.add(
            content="Session B memory", tags=["session"], session_id=session_b
        )

        await asyncio.sleep(1)  # Extra wait for consistency

        results_a = await ltm_manager.retrieve(query="Session A", session_id=session_a)
        results_b = await ltm_manager.retrieve(query="Session B", session_id=session_b)

        assert len(results_a) > 0
        for r in results_a:
            print(
                f"Result A: id={r.id}, session_id={r.session_id}, content={r.content}"
            )
        assert all(r.session_id == session_a for r in results_a)
        assert len(results_b) > 0
        for r in results_b:
            print(
                f"Result B: id={r.id}, session_id={r.session_id}, content={r.content}"
            )
        assert all(r.session_id == session_b for r in results_b)

    @pytest.mark.asyncio
    async def test_user_scoping(self, ltm_manager):
        """Test that user scoping works correctly."""
        user_x = f"user-x-{uuid.uuid4().hex[:8]}"
        user_y = f"user-y-{uuid.uuid4().hex[:8]}"

        await ltm_manager.add(content="User X memory", tags=["user"], user_id=user_x)
        await ltm_manager.add(content="User Y memory", tags=["user"], user_id=user_y)

        await asyncio.sleep(1)

        results_x = await ltm_manager.retrieve(query="user", user_id=user_x)
        results_y = await ltm_manager.retrieve(query="user", user_id=user_y)

        assert len(results_x) > 0
        assert all(r.user_id == user_x for r in results_x)
        assert len(results_y) > 0
        assert all(r.user_id == user_y for r in results_y)
