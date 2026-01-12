"""Tests for Long-Term Memory Manager.

Integration tests require a running Weaviate instance.
Run with: pytest tests/test_ltm.py
"""

import asyncio
import uuid
from datetime import datetime

import pytest

from src.ltm import MemoryFunction


class TestLTMManagerBasicOperations:
    """Test basic LTM operations."""

    @pytest.mark.asyncio
    async def test_add_memory(self, ltm_manager):
        """Test adding a new memory."""
        content = "Test memory entry at " + datetime.now().isoformat()
        metadata = {"tags": ["test", "unit"], "session_id": "test-session"}

        entry = await ltm_manager.add(
            content=content,
            metadata=metadata,
            memory_function=MemoryFunction.FACTUAL,
            quality=0.7,
        )

        assert entry.id is not None
        assert entry.content == content
        assert entry.metadata == metadata
        assert entry.memory_function == MemoryFunction.FACTUAL
        assert entry.quality == 0.7

    @pytest.mark.asyncio
    async def test_add_memory_default_params(self, ltm_manager):
        """Test adding memory with default parameters."""
        content = "Test memory with defaults"

        entry = await ltm_manager.add(content=content)

        assert entry.id is not None
        assert entry.memory_function == MemoryFunction.FACTUAL
        assert entry.quality == 0.5

    @pytest.mark.asyncio
    async def test_retrieve_memory(self, ltm_manager):
        """Test retrieving memories with robust search."""
        content = "Unique test content for retrieval: " + uuid.uuid4().hex

        await ltm_manager.add(content=content, memory_function=MemoryFunction.FACTUAL)

        results = await ltm_manager.retrieve(query=content, top_k=5)

        assert len(results) > 0
        assert any(content in r.content for r in results)

    @pytest.mark.asyncio
    async def test_retrieve_memory_with_filters(self, ltm_manager):
        """Test retrieving memories with functional filters."""
        memory_function = MemoryFunction.EXPERIENTIAL
        content = "Filter test " + uuid.uuid4().hex

        await ltm_manager.add(
            content=content,
            memory_function=memory_function,
            quality=0.6,
        )

        results = await ltm_manager.retrieve(
            query=content,
            memory_function=memory_function,
            min_quality=0.4,
        )

        assert len(results) > 0
        assert all(r.memory_function == memory_function for r in results)

    @pytest.mark.asyncio
    async def test_update_memory(self, ltm_manager):
        """Test updating an existing memory."""
        content = "Original content"
        updated_content = "Updated content"

        entry = await ltm_manager.add(content=content)
        original_id = entry.id

        updated_entry = await ltm_manager.update(
            entry_id=original_id, content=updated_content
        )

        assert updated_entry.id == original_id
        assert updated_entry.content == updated_content

    @pytest.mark.asyncio
    async def test_delete_memory(self, ltm_manager):
        """Test deleting a memory."""
        content = "To be deleted " + uuid.uuid4().hex
        entry = await ltm_manager.add(content=content)
        memory_id = entry.id

        await ltm_manager.delete(entry_id=memory_id)

        results = await ltm_manager.retrieve(query=content)
        assert not any(r.id == memory_id for r in results)

    @pytest.mark.asyncio
    async def test_update_quality(self, ltm_manager):
        """Test updating memory quality."""
        content = "Quality update test " + uuid.uuid4().hex
        entry = await ltm_manager.add(content=content, quality=0.5)
        memory_id = entry.id

        await ltm_manager.update_quality(entry_id=memory_id, quality=0.9)

        # Retry search as indexing might have lag
        found = False
        for _ in range(3):
            results = await ltm_manager.retrieve(query=content)
            if results and results[0].quality == 0.9:
                found = True
                break
            await asyncio.sleep(0.5)

        assert found

    @pytest.mark.asyncio
    async def test_get_stats(self, ltm_manager):
        """Test getting memory statistics."""
        await ltm_manager.add(content="Stats 1", memory_function=MemoryFunction.FACTUAL)

        stats = await ltm_manager.get_stats()

        assert "total_memories" in stats
        assert "average_quality" in stats
        assert stats["total_memories"] >= 1

    @pytest.mark.asyncio
    async def test_merge_memories(self, ltm_manager):
        """Test merging redundant memories."""
        entry1 = await ltm_manager.add(
            content="Fact: The user likes apples.", keywords=["apples"]
        )
        entry2 = await ltm_manager.add(
            content="The user really enjoys apples.", keywords=["preference"]
        )

        await ltm_manager.merge_memories(
            survivor_id=entry1.id,
            redundant_ids=[entry2.id],
            new_content="Merged apples preference.",
            new_context="Consolidated apple info.",
        )

        results = await ltm_manager.get_by_ids([entry1.id])
        assert len(results) == 1
        assert results[0].content == "Merged apples preference."

        results2 = await ltm_manager.get_by_ids([entry2.id])
        assert len(results2) == 0

    @pytest.mark.asyncio
    async def test_link_traversal_and_decay(self, ltm_manager):
        """Test link weight reinforcement and temporal decay."""
        entry1 = await ltm_manager.add(content="Source node")
        entry2 = await ltm_manager.add(content="Target node")

        await ltm_manager.update_links(entry1.id, [entry2.id])

        # Verify initial link
        entry1_updated = (await ltm_manager.get_by_ids([entry1.id]))[0]
        assert entry2.id in entry1_updated.links

        # Apply decay
        await ltm_manager.decay_links(decay_factor=0.5)

        # Decay to oblivion
        await ltm_manager.decay_links(decay_factor=0.01)
        entry1_decayed = (await ltm_manager.get_by_ids([entry1.id]))[0]
        assert entry2.id not in entry1_decayed.links
