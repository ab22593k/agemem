"""
Long-Term Memory (LTM) Manager using Weaviate.

Manages persistent memory storage with quality scoring, usage tracking,
and semantic search capabilities.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, cast

import weaviate
from weaviate import connect_to_local
from weaviate.collections import Collection
from weaviate.collections.classes.config import (
    Configure,
    DataType,
    Property,
    Tokenization,
)
from weaviate.collections.classes.filters import Filter


@dataclass
class MemoryEntry:  # pylint: disable=too-many-instance-attributes
    """Represents a single memory stored in LTM."""

    id: str  # pylint: disable=invalid-name
    content: str
    tags: List[str] = field(default_factory=list)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    quality: float = 0.5
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
        self.client: Optional[weaviate.WeaviateClient] = None
        self.collection: Optional[Any] = None

    async def initialize(self) -> None:
        """Initialize Weaviate client and ensure schema exists."""
        self.client = connect_to_local(
            host=self.host.split(":")[0],
            port=int(self.host.split(":")[1]) if ":" in self.host else 8080,
        )

        await self._ensure_schema()

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
            return

        vectorizer_config: Any
        if self.use_vector_search:
            vectorizer_config = Configure.Vectorizer.text2vec_transformers()
        else:
            vectorizer_config = None

        properties = [
            Property(name="content", data_type=DataType.TEXT),
            Property(name="tags", data_type=DataType.TEXT_ARRAY),
            Property(
                name="original_id",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
            ),
            Property(
                name="session_id",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
            ),
            Property(
                name="user_id", data_type=DataType.TEXT, tokenization=Tokenization.FIELD
            ),
            Property(name="quality", data_type=DataType.NUMBER),
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
        tags: List[str],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        quality: float = 0.5,
    ) -> MemoryEntry:
        """Create a new memory entry."""
        if len(content) > self.MAX_CONTENT_LENGTH:
            raise ValueError(
                f"Content exceeds maximum length of {self.MAX_CONTENT_LENGTH} bytes"
            )

        if quality <= 0:
            quality = self.DEFAULT_QUALITY

        coll = self._get_collection()

        entry_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc)

        coll.data.insert(
            properties={
                "content": content,
                "tags": tags,
                "original_id": entry_id,
                "session_id": session_id,
                "user_id": user_id,
                "quality": float(quality),
                "usage_count": 0,
                "created_at": now,
                "updated_at": now,
                "last_used_at": now,
            }
        )

        # Give Weaviate a moment to index for immediate operations
        await asyncio.sleep(0.5)

        return MemoryEntry(
            id=entry_id,
            content=content,
            tags=tags,
            session_id=session_id,
            user_id=user_id,
            quality=quality,
            usage_count=0,
            created_at=now,
            updated_at=now,
            last_used_at=now,
        )

    async def update(self, entry_id: str, content: str) -> MemoryEntry:
        """Update an existing memory entry by its original ID."""
        if len(content) > self.MAX_CONTENT_LENGTH:
            raise ValueError(
                f"Content exceeds maximum length of {self.MAX_CONTENT_LENGTH} bytes"
            )

        weaviate_uuid = await self._find_weaviate_uuid(entry_id)
        if not weaviate_uuid:
            # Retry once after a short sleep if not found (might be indexing lag)
            await asyncio.sleep(1)
            weaviate_uuid = await self._find_weaviate_uuid(entry_id)
            if not weaviate_uuid:
                raise ValueError(f"Memory not found: {entry_id}")

        coll = self._get_collection()
        now = datetime.now(timezone.utc)

        coll.data.update(
            uuid=weaviate_uuid,
            properties={
                "content": content,
                "updated_at": now,
            },
        )

        return MemoryEntry(id=entry_id, content=content, updated_at=now)

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
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        min_quality: float = 0,
    ) -> Any:
        """Build Weaviate filters from parameters."""
        filter_parts = []
        if session_id:
            filter_parts.append(Filter.by_property("session_id").equal(session_id))
        if user_id:
            filter_parts.append(Filter.by_property("user_id").equal(user_id))
        if min_quality > 0:
            filter_parts.append(Filter.by_property("quality").greater_than(min_quality))

        if not filter_parts:
            return None

        filters = filter_parts[0]
        for part in filter_parts[1:]:
            filters = filters & part
        return filters

    async def retrieve(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        query: str,
        *,
        limit: int = 5,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        min_quality: float = 0,
        update_usage: bool = False,
        search_type: Optional[Literal["vector", "keyword"]] = None,
        min_similarity: Optional[float] = None,
    ) -> List[MemoryEntry]:
        """
        Search for memories matching query.

        Args:
            query: Search string.
            limit: Max results.
            session_id: Filter by session.
            user_id: Filter by user.
            min_quality: Filter by quality score.
            update_usage: If true, increments usage count.
            search_type: "vector" (semantic) or "keyword" (BM25).
                        Defaults to "vector" if use_vector_search=True.
            min_similarity: Min similarity score for vector search (0-1).
        """
        limit = min(limit, 20)
        coll = self._get_collection()

        filters = self._build_filters(session_id, user_id, min_quality)

        # Determine search type
        effective_search_type = search_type
        if effective_search_type is None:
            effective_search_type = "vector" if self.use_vector_search else "keyword"

        if effective_search_type == "vector" and self.use_vector_search and query:
            results = coll.query.near_text(
                query=query,
                limit=limit,
                filters=filters,
                distance=1.0 - min_similarity if min_similarity is not None else None,
            )
        elif query:
            results = coll.query.bm25(query=query, limit=limit, filters=filters)
        else:
            results = coll.query.fetch_objects(limit=limit, filters=filters)

        entries = self._parse_results(results.objects)

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

    def _parse_results(self, objects: List[Any]) -> List[MemoryEntry]:
        """Convert Weaviate response to MemoryEntry list."""
        entries = []

        for obj in objects:
            props = obj.properties
            entry = MemoryEntry(
                id=str(props.get("original_id") or ""),
                content=str(props.get("content") or ""),
                tags=list(cast(List, props.get("tags") or [])),
                session_id=str(props.get("session_id"))
                if props.get("session_id") is not None
                else None,
                user_id=str(props.get("user_id"))
                if props.get("user_id") is not None
                else None,
                quality=float(
                    props.get("quality") if props.get("quality") is not None else 0.5
                ),
                usage_count=int(
                    props.get("usage_count")
                    if props.get("usage_count") is not None
                    else 0
                ),
                created_at=cast(datetime, props.get("created_at"))
                or datetime.now(timezone.utc),
                updated_at=cast(datetime, props.get("updated_at"))
                or datetime.now(timezone.utc),
                last_used_at=cast(datetime, props.get("last_used_at")),
            )
            entries.append(entry)

        return entries
