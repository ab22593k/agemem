"""
Long-Term Memory (LTM) Manager using Weaviate.

Manages persistent memory storage with quality scoring, usage tracking,
and semantic search capabilities.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, cast

from weaviate import connect_to_local
from weaviate.collections import Collection
from weaviate.collections.classes.config import (
    Configure,
    DataType,
    Property,
    Tokenization,
)
from weaviate.collections.classes.filters import Filter


class MemoryFunction(str, Enum):
    """Functional taxonomy as per Memory in the Age of AI Agents survey."""

    FACTUAL = "factual"  # Knowledge from interactions with users/environment
    EXPERIENTIAL = "experiential"  # Problem-solving traces and task execution history
    WORKING = "working"  # Workspace information during individual tasks


@dataclass
class MemoryEntry:  # pylint: disable=too-many-instance-attributes
    """Represents a single memory stored in LTM."""

    id: str  # pylint: disable=invalid-name
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    memory_function: MemoryFunction = MemoryFunction.FACTUAL
    quality: float = 0.5
    keywords: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    context_description: Optional[str] = None
    links: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None  # For Hierarchical (3D) memory form
    link_metadata: Dict[str, Any] = field(
        default_factory=dict
    )  # target_id -> {weight, last_traversed}
    usage_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_used_at: Optional[datetime] = None


class LTMManager:
    """Handles Long-Term Memory operations backed by Weaviate."""

    CLASS_NAME = "MemoryEntry"
    MAX_CONTENT_LENGTH = 100000
    DEFAULT_QUALITY = 0.5

    def __init__(self, host: str = "localhost:8080", use_vector_search: bool = False):
        self.host = host
        self.use_vector_search = use_vector_search
        self.client = None
        self.collection = None

    async def initialize(self) -> None:
        """Initialize Weaviate client and ensure schema exists."""
        self.client = connect_to_local(
            host=self.host.split(":")[0],
            port=int(self.host.split(":")[1]) if ":" in self.host else 8080,
        )

        await self._ensure_schema()

    async def __aenter__(self) -> "LTMManager":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def close(self) -> None:
        """Close Weaviate client connection."""
        if self.client:
            self.client.close()

    async def _ensure_schema(self) -> None:
        """Ensure Weaviate schema for MemoryEntry exists."""
        if self.client is None:
            return

        if self.client.collections.exists(self.CLASS_NAME):
            self.collection = self.client.collections.get(self.CLASS_NAME)
            # Add missing properties if any (for migration)
            config = self.collection.config.get()
            existing_props = set()
            for p in cast(List[Any], config.properties):
                existing_props.add(p.name)

            new_props = [
                Property(name="keywords", data_type=DataType.TEXT_ARRAY),
                Property(name="tags", data_type=DataType.TEXT_ARRAY),
                Property(name="context_description", data_type=DataType.TEXT),
                Property(name="links", data_type=DataType.TEXT_ARRAY),
                Property(name="link_metadata", data_type=DataType.TEXT),
                Property(name="parent_id", data_type=DataType.TEXT),
            ]

            for prop in new_props:
                if prop.name not in existing_props:
                    self.collection.config.add_property(prop)
            return

        vectorizer_config: Any
        if self.use_vector_search:
            vectorizer_config = Configure.Vectorizer.text2vec_transformers()
        else:
            vectorizer_config = None

        properties = [
            Property(name="content", data_type=DataType.TEXT),
            Property(name="metadata", data_type=DataType.TEXT),
            Property(
                name="memory_type",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
            ),
            Property(
                name="original_id",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
            ),
            Property(name="quality", data_type=DataType.NUMBER),
            Property(name="keywords", data_type=DataType.TEXT_ARRAY),
            Property(name="tags", data_type=DataType.TEXT_ARRAY),
            Property(name="context_description", data_type=DataType.TEXT),
            Property(name="links", data_type=DataType.TEXT_ARRAY),
            Property(name="link_metadata", data_type=DataType.TEXT),
            Property(name="parent_id", data_type=DataType.TEXT),
            Property(name="usage_count", data_type=DataType.INT),
            Property(name="created_at", data_type=DataType.DATE),
            Property(name="updated_at", data_type=DataType.DATE),
            Property(name="last_used_at", data_type=DataType.DATE),
        ]

        self.collection = self.client.collections.create(
            name=self.CLASS_NAME,
            properties=properties,
            vectorizer_config=vectorizer_config,
        )

    def _get_collection(self) -> Collection:
        """Helper to get collection or raise error if not initialized."""
        if self.collection is not None:
            return cast(Collection, self.collection)
        if self.client is None:
            raise RuntimeError("LTMManager client not initialized")
        self.collection = self.client.collections.get(self.CLASS_NAME)
        return cast(Collection, self.collection)

    async def _find_weaviate_uuid(self, original_id: str) -> Optional[str]:
        """Find Weaviate internal UUID for a given original_id."""
        coll = self._get_collection()

        try:
            result = coll.query.fetch_objects(
                filters=Filter.by_property("original_id").equal(original_id),
                limit=1,
            )

            if result.objects:
                return str(result.objects[0].uuid)
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        return None

    async def add(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        memory_function: MemoryFunction = MemoryFunction.FACTUAL,
        quality: float = 0.5,
        keywords: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        context_description: Optional[str] = None,
        links: Optional[List[str]] = None,
        link_metadata: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
    ) -> MemoryEntry:
        """Create a new memory entry."""
        if len(content) > self.MAX_CONTENT_LENGTH:
            raise ValueError(f"Content exceeds maximum length of {self.MAX_CONTENT_LENGTH} bytes")

        if quality <= 0:
            quality = self.DEFAULT_QUALITY

        coll = self._get_collection()

        entry_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc)
        metadata = metadata or {}

        props = {
            "content": content,
            "metadata": json.dumps(metadata),
            "memory_type": memory_function.value,
            "original_id": entry_id,
            "quality": float(quality),
            "usage_count": 0,
            "created_at": now,
            "updated_at": now,
            "last_used_at": now,
        }
        if keywords:
            props["keywords"] = keywords
        if tags:
            props["tags"] = tags
        if context_description:
            props["context_description"] = context_description
        if links:
            props["links"] = links
        if link_metadata:
            props["link_metadata"] = json.dumps(link_metadata)
        if parent_id:
            props["parent_id"] = parent_id

        coll.data.insert(properties=props)

        # Give Weaviate a moment to index for immediate operations
        await asyncio.sleep(0.5)

        return MemoryEntry(
            id=entry_id,
            content=content,
            metadata=metadata,
            memory_function=memory_function,
            quality=quality,
            keywords=keywords or [],
            tags=tags or [],
            context_description=context_description,
            links=links or [],
            link_metadata=link_metadata or {},
            parent_id=parent_id,
            usage_count=0,
            created_at=now,
            updated_at=now,
            last_used_at=now,
        )

    async def update(
        self,
        entry_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryEntry:
        """Update an existing memory entry by its original ID."""
        if len(content) > self.MAX_CONTENT_LENGTH:
            raise ValueError(f"Content exceeds maximum length of {self.MAX_CONTENT_LENGTH} bytes")

        weaviate_uuid = await self._find_weaviate_uuid(entry_id)
        if not weaviate_uuid:
            # Retry once after a short sleep if not found (might be indexing lag)
            await asyncio.sleep(1)
            weaviate_uuid = await self._find_weaviate_uuid(entry_id)
            if not weaviate_uuid:
                raise ValueError(f"Memory not found: {entry_id}")

        coll = self._get_collection()
        now = datetime.now(timezone.utc)

        update_props: Dict[str, Any] = {
            "content": content,
            "updated_at": now,
        }
        if metadata:
            update_props["metadata"] = json.dumps(metadata)

        coll.data.update(
            uuid=weaviate_uuid,
            properties=update_props,
        )

        return MemoryEntry(id=entry_id, content=content, metadata=metadata or {}, updated_at=now)

    async def delete(self, entry_id: str) -> None:
        """Remove a memory entry by its original ID."""
        weaviate_uuid = await self._find_weaviate_uuid(entry_id)
        if not weaviate_uuid:
            # Retry once
            await asyncio.sleep(1)
            weaviate_uuid = await self._find_weaviate_uuid(entry_id)
            if not weaviate_uuid:
                raise ValueError(f"Memory not found: {entry_id}")

        coll = self._get_collection()
        coll.data.delete_by_id(weaviate_uuid)

    def _build_filters(
        self,
        memory_function: Optional[MemoryFunction] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        min_quality: float = 0,
    ) -> Any:
        """Build Weaviate filters from parameters."""
        filter_parts = []
        if memory_function:
            filter_parts.append(Filter.by_property("memory_type").equal(memory_function.value))

        if metadata_filter:
            # Note: Filtering on JSON-encoded metadata string is limited in Weaviate
            # For now, we only support exact matches if the user knows what they are
            # doing
            for _, _ in metadata_filter.items():
                # This is a placeholder for better implementation in the future
                # e.g. using Filter.by_property("metadata").contains_any([value])
                # but requires specific tokenization.
                pass

        if min_quality > 0:
            filter_parts.append(Filter.by_property("quality").greater_than(min_quality))

        if not filter_parts:
            return None

        filters = filter_parts[0]
        for part in filter_parts[1:]:
            filters = filters & part
        return filters

    async def get_by_ids(self, entry_ids: List[str]) -> List[MemoryEntry]:
        """Fetch multiple memories by their original IDs."""
        if not entry_ids:
            return []

        coll = self._get_collection()
        results = coll.query.fetch_objects(
            filters=Filter.by_property("original_id").contains_any(entry_ids),
            limit=len(entry_ids),
        )
        return self._parse_results(results.objects)

    async def _update_link_traversal(self, source_id: str, target_id: str) -> None:
        """Increment link weight and update last_traversed timestamp."""
        source = await self.get_by_ids([source_id])
        if not source:
            return

        link_meta = source[0].link_metadata or {}
        if target_id not in link_meta:
            # First traversal, start at 1.1 to show reinforcement from baseline 1.0
            link_meta[target_id] = {
                "weight": 1.1,
                "last_traversed": datetime.now(timezone.utc).isoformat(),
            }
        else:
            link_meta[target_id]["weight"] = round(link_meta[target_id].get("weight", 1.0) + 0.1, 2)
            link_meta[target_id]["last_traversed"] = datetime.now(timezone.utc).isoformat()

        weaviate_uuid = await self._find_weaviate_uuid(source_id)
        if weaviate_uuid:
            coll = self._get_collection()
            coll.data.update(
                uuid=weaviate_uuid, properties={"link_metadata": json.dumps(link_meta)}
            )

    async def _handle_linked_memories(
        self, entries: List[MemoryEntry], include_links: bool, link_threshold: float
    ) -> List[MemoryEntry]:
        """Fetch linked memories and update weights."""
        if not include_links or not entries:
            return entries

        linked_ids_to_fetch = []
        traversal_updates = []  # (source_id, target_id)

        for entry in entries:
            if not entry.links:
                continue
            for link_id in entry.links:
                # Check weight threshold
                meta = entry.link_metadata.get(link_id, {"weight": 1.0})
                if meta.get("weight", 1.0) >= link_threshold:
                    linked_ids_to_fetch.append(link_id)
                    traversal_updates.append((entry.id, link_id))

        if not linked_ids_to_fetch:
            return entries

        existing_ids = {e.id for e in entries}
        ids_to_fetch = [lid for lid in set(linked_ids_to_fetch) if lid not in existing_ids]
        if ids_to_fetch:
            linked_memories = await self.get_by_ids(ids_to_fetch)
            entries.extend(linked_memories)

        # Update weights for followed links
        for src_id, tgt_id in traversal_updates:
            await self._update_link_traversal(src_id, tgt_id)

        return entries

    async def retrieve(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        self,
        query: str,
        *,
        top_k: int = 3,
        memory_function: Optional[MemoryFunction] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        min_quality: float = 0,
        update_usage: bool = False,
        search_type: Optional[Literal["vector", "keyword"]] = None,
        min_similarity: Optional[float] = None,
        include_links: bool = False,
        link_threshold: float = 0.5,
    ) -> List[MemoryEntry]:
        """
        Search for memories matching query (Retrieval Operator).
        """
        top_k = min(top_k, 20)
        coll = self._get_collection()

        filters = self._build_filters(memory_function, metadata_filter, min_quality)

        # Determine search type
        effective_search_type = search_type
        if effective_search_type is None:
            effective_search_type = "vector" if self.use_vector_search else "keyword"

        if effective_search_type == "vector" and self.use_vector_search and query:
            results = coll.query.near_text(
                query=query,
                limit=top_k,
                filters=filters,
                distance=1.0 - min_similarity if min_similarity is not None else None,
            )
        elif query:
            results = coll.query.bm25(query=query, limit=top_k, filters=filters)
        else:
            results = coll.query.fetch_objects(limit=top_k, filters=filters)

        entries = self._parse_results(results.objects)

        entries = await self._handle_linked_memories(entries, include_links, link_threshold)

        if update_usage and entries:
            await self._increment_usage(entries)

        return entries

    async def _increment_usage(self, entries: List[MemoryEntry]) -> None:
        """Update usage_count and last_used_at for retrieved memories."""
        now = datetime.now(timezone.utc)
        coll = self._get_collection()

        for entry in entries:
            weaviate_uuid = await self._find_weaviate_uuid(entry.id)
            if weaviate_uuid:
                coll.data.update(
                    uuid=weaviate_uuid,
                    properties={
                        "usage_count": int(entry.usage_count or 0) + 1,
                        "last_used_at": now,
                    },
                )

    async def update_quality(self, entry_id: str, quality: float) -> None:
        """Adjust quality score of a memory."""
        if quality < 0 or quality > 1:
            raise ValueError("Quality must be between 0 and 1")

        weaviate_uuid = await self._find_weaviate_uuid(entry_id)
        if not weaviate_uuid:
            # Retry
            await asyncio.sleep(1)
            weaviate_uuid = await self._find_weaviate_uuid(entry_id)
            if not weaviate_uuid:
                raise ValueError(f"Memory not found: {entry_id}")

        coll = self._get_collection()
        now = datetime.now(timezone.utc)
        coll.data.update(
            uuid=weaviate_uuid,
            properties={
                "quality": quality,
                "updated_at": now,
            },
        )

    async def update_links(self, entry_id: str, links: List[str]) -> None:
        """Update the links for a memory entry."""
        weaviate_uuid = await self._find_weaviate_uuid(entry_id)
        if not weaviate_uuid:
            raise ValueError(f"Memory not found: {entry_id}")

        coll = self._get_collection()
        coll.data.update(
            uuid=weaviate_uuid,
            properties={"links": links},
        )

    async def update_agentic_fields(
        self,
        entry_id: str,
        keywords: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        context_description: Optional[str] = None,
    ) -> None:
        """Update agentic enrichment fields."""
        weaviate_uuid = await self._find_weaviate_uuid(entry_id)
        if not weaviate_uuid:
            raise ValueError(f"Memory not found: {entry_id}")

        coll = self._get_collection()
        props: Dict[str, Any] = {"updated_at": datetime.now(timezone.utc)}
        if keywords is not None:
            props["keywords"] = keywords
        if tags is not None:
            props["tags"] = tags
        if context_description is not None:
            props["context_description"] = context_description

        coll.data.update(
            uuid=weaviate_uuid,
            properties=props,
        )

    async def apply_merge_plan(self, plan: Dict[str, Any]) -> None:
        """Apply a merge plan generated by AgenticMemoryProcessor."""
        await self.merge_memories(
            survivor_id=plan["survivor_id"],
            redundant_ids=plan["redundant_ids"],
            new_content=plan["new_content"],
            new_context=plan["new_context"],
        )

    async def merge_memories(
        self,
        survivor_id: str,
        redundant_ids: List[str],
        new_content: str,
        new_context: str,
    ) -> None:
        """
        Merge multiple redundant memories into a survivor.
        """
        if not redundant_ids:
            return

        survivor = await self.get_by_ids([survivor_id])
        if not survivor:
            raise ValueError(f"Survivor memory not found: {survivor_id}")

        redundants = await self.get_by_ids(redundant_ids)

        # 1. Collect all unique links and metadata
        all_links = set(survivor[0].links)
        all_keywords = set(survivor[0].keywords)
        all_tags = set(survivor[0].tags)

        for r_entry in redundants:
            all_links.update(r_entry.links)
            all_keywords.update(r_entry.keywords)
            all_tags.update(r_entry.tags)

        # Remove self and redundant IDs from links
        all_links.discard(survivor_id)
        for rid in redundant_ids:
            all_links.discard(rid)

        # 2. Update survivor
        await self.update(survivor_id, new_content)
        await self.update_agentic_fields(
            survivor_id,
            keywords=list(all_keywords),
            tags=list(all_tags),
            context_description=new_context,
        )
        await self.update_links(survivor_id, list(all_links))

        # 3. Update other memories that link to redundant IDs
        await self._redirect_links(survivor_id, redundant_ids)

        # 4. Delete redundants
        for rid in redundant_ids:
            await self.delete(rid)

    # ruff: noqa: E741
    async def _redirect_links(self, survivor_id: str, redundant_ids: List[str]) -> None:
        """Redirect links from redundant IDs to survivor ID."""
        coll = self._get_collection()
        # Find objects that link to any of the redundant IDs
        results = coll.query.fetch_objects(
            filters=Filter.by_property("links").contains_any(redundant_ids), limit=1000
        )
        for obj in results.objects:
            links = cast(List[str], obj.properties.get("links") or [])
            new_links = [survivor_id if (l in redundant_ids) else l for l in links]
            # Unique-ify
            new_links = list(set(new_links))
            await self.update_links(str(obj.properties.get("original_id")), new_links)

    def _parse_results(self, objects: List[Any]) -> List[MemoryEntry]:
        """Convert Weaviate response to MemoryEntry list."""
        entries = []

        for obj in objects:
            props = obj.properties
            meta_val = props.get("metadata")
            metadata = {}
            if isinstance(meta_val, str):
                try:
                    metadata = json.loads(meta_val)
                except json.JSONDecodeError:
                    pass
            elif isinstance(meta_val, dict):
                metadata = meta_val

            link_meta_val = props.get("link_metadata")
            link_metadata = {}
            if isinstance(link_meta_val, str):
                try:
                    link_metadata = json.loads(link_meta_val)
                except json.JSONDecodeError:
                    pass

            # Map legacy or unknown memory types to MemoryFunction
            raw_mem_type = str(props.get("memory_type"))
            try:
                mem_func = MemoryFunction(raw_mem_type)
            except ValueError:
                mem_func = MemoryFunction.FACTUAL

            entry = MemoryEntry(
                id=str(props.get("original_id") or ""),
                content=str(props.get("content") or ""),
                metadata=metadata,
                memory_function=mem_func,
                quality=float(props.get("quality") if props.get("quality") is not None else 0.5),
                keywords=cast(List[str], props.get("keywords") or []),
                tags=cast(List[str], props.get("tags") or []),
                context_description=str(props.get("context_description"))
                if props.get("context_description") is not None
                else None,
                links=cast(List[str], props.get("links") or []),
                parent_id=str(props.get("parent_id"))
                if props.get("parent_id") is not None
                else None,
                link_metadata=link_metadata,
                usage_count=int(
                    props.get("usage_count") if props.get("usage_count") is not None else 0
                ),
                created_at=cast(datetime, props.get("created_at")) or datetime.now(timezone.utc),
                updated_at=cast(datetime, props.get("updated_at")) or datetime.now(timezone.utc),
                last_used_at=cast(datetime, props.get("last_used_at")),
            )
            entries.append(entry)

        return entries

    async def decay_links(self, decay_factor: float = 0.9) -> int:
        """
        Apply temporal decay to all links.
        Returns number of links removed.
        """
        coll = self._get_collection()
        all_objects = coll.query.fetch_objects(limit=1000)
        removed_count = 0

        for obj in all_objects.objects:
            links = cast(List[str], obj.properties.get("links") or [])
            if not links:
                continue

            link_meta_val = obj.properties.get("link_metadata")
            link_meta = {}
            if link_meta_val and isinstance(link_meta_val, str):
                try:
                    link_meta = json.loads(link_meta_val)
                except json.JSONDecodeError:
                    pass

            new_links, new_meta, changed = self._apply_decay_to_obj_links(
                links, link_meta, decay_factor
            )

            if changed:
                removed_count += len(links) - len(new_links)
                coll.data.update(
                    uuid=obj.uuid,
                    properties={
                        "links": new_links,
                        "link_metadata": json.dumps(new_meta),
                    },
                )

        return removed_count

    def _apply_decay_to_obj_links(
        self, links: List[str], link_meta: Dict[str, Any], decay_factor: float
    ) -> tuple[List[str], Dict[str, Any], bool]:
        """Apply decay to links of a single object."""
        new_links = []
        new_meta = {}
        changed = False

        for lid in links:
            meta = link_meta.get(lid, {"weight": 1.0})
            old_weight = meta.get("weight", 1.0)
            new_weight = old_weight * decay_factor

            if new_weight < 0.1:
                changed = True
                continue

            new_links.append(lid)
            new_meta[lid] = {
                "weight": round(new_weight, 2),
                "last_traversed": meta.get("last_traversed"),
            }
            if new_weight != old_weight:
                changed = True

        return new_links, new_meta, changed

    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about memory store."""
        coll = self._get_collection()

        results = coll.query.fetch_objects(limit=100)
        entries = self._parse_results(results.objects)

        total_memories = len(entries)
        total_quality = 0.0
        total_usage = 0

        for entry in entries:
            total_quality += float(entry.quality or 0)
            total_usage += int(entry.usage_count or 0)

        avg_quality = total_quality / total_memories if total_memories > 0 else 0.0

        return {
            "total_memories": total_memories,
            "average_quality": avg_quality,
            "total_retrievals": total_usage,
        }
