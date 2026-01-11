"""
Long-Term Memory (LTM) Manager using Weaviate.

Manages persistent memory storage with quality scoring, usage tracking,
and semantic search capabilities.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, cast

import weaviate
from weaviate import connect_to_local


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

        # Use the named vectorizer config when vector search is enabled,
        # otherwise signal that no vectorizer should be used by providing
        # an explicit (typed) config as required by the SDK. To satisfy
        # static type checkers we cast the values to Any here.
        vectorizer_config: Any
        if self.use_vector_search:
            vectorizer_config = {"name": "text2vec-transformers"}
        else:
            # Provide an explicit disabled vectorizer configuration object
            # acceptable by the SDK (cast to Any for type-checking).
            vectorizer_config = {"skip": True}

        properties = [
            {"name": "content", "dataType": ["text"]},
            {"name": "tags", "dataType": ["text[]"]},
            {"name": "original_id", "dataType": ["string"]},
            {"name": "session_id", "dataType": ["string"]},
            {"name": "user_id", "dataType": ["string"]},
            {"name": "quality", "dataType": ["number"]},
            {"name": "usage_count", "dataType": ["int"]},
            {"name": "created_at", "dataType": ["date"]},
            {"name": "updated_at", "dataType": ["date"]},
            {"name": "last_used_at", "dataType": ["date"]},
        ]

        # Cast to Any to avoid static type-checker mismatches with the
        # weaviate client types while keeping runtime behavior intact.
        self.collection = self.client.collections.create(
            name=self.CLASS_NAME,
            properties=cast(Any, properties),
            vectorizer_config=cast(Any, vectorizer_config),
        )

    async def _find_weaviate_uuid(self, original_id: str) -> Optional[str]:
        """Find Weaviate internal UUID for a given original_id."""
        if self.collection is None:
            return None
        try:
            result = self.collection.query.fetch_objects(
                filters={
                    "path": "original_id",
                    "operator": "Equal",
                    "valueText": original_id,
                },
                limit=1,
            )

            if result.objects:
                return result.objects[0].uuid
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

        if self.collection is None:
            if self.client is None:
                raise RuntimeError("LTMManager not initialized")
            self.collection = self.client.collections.get(self.CLASS_NAME)

        entry_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc)

        self.collection.data.insert(
            properties={
                "content": content,
                "tags": tags,
                "original_id": entry_id,
                "session_id": session_id,
                "user_id": user_id,
                "quality": quality,
                "usage_count": 0,
                "created_at": now,
                "updated_at": now,
                "last_used_at": now,
            }
        )

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
            raise ValueError(f"Memory not found: {entry_id}")

        if self.collection is None:
            if self.client is None:
                raise RuntimeError("LTMManager not initialized")
            self.collection = self.client.collections.get(self.CLASS_NAME)

        now = datetime.now(timezone.utc)

        self.collection.data.update(
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
            raise ValueError(f"Memory not found: {entry_id}")

        if self.collection is None:
            if self.client is None:
                raise RuntimeError("LTMManager not initialized")
            self.collection = self.client.collections.get(self.CLASS_NAME)

        self.collection.data.delete(uuid=weaviate_uuid)

    async def retrieve(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        query: str,
        *,
        limit: int = 5,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        min_quality: float = 0,
        update_usage: bool = False,
    ) -> List[MemoryEntry]:
        """Search for memories matching query."""
        limit = min(limit, 20)

        if self.collection is None:
            if self.client is None:
                raise RuntimeError("LTMManager not initialized")
            self.collection = self.client.collections.get(self.CLASS_NAME)

        filters = None
        filter_list = []

        if session_id:
            filter_list.append({
                "path": "session_id",
                "operator": "Equal",
                "valueText": session_id,
            })

        if user_id:
            filter_list.append({
                "path": "user_id",
                "operator": "Equal",
                "valueText": user_id,
            })

        if min_quality > 0:
            filter_list.append({
                "path": "quality",
                "operator": "GreaterThan",
                "valueNumber": min_quality,
            })

        if len(filter_list) > 0:
            filters = filter_list[0]
            for f in filter_list[1:]:
                filters = {"operator": "And", "operands": [filters, f]}
        else:
            filters = None

        if self.use_vector_search and query:
            results = self.collection.query.near_text(
                query=query, limit=limit, filters=filters
            )
        elif query:
            results = self.collection.query.bm25(
                query=query, limit=limit, filters=filters
            )
        else:
            results = self.collection.query.fetch_objects(limit=limit, filters=filters)

        entries = self._parse_results(results.objects)

        if update_usage and entries:
            await self._increment_usage(entries)

        return entries

    async def _increment_usage(self, entries: List[MemoryEntry]) -> None:
        """Update usage_count and last_used_at for retrieved memories."""
        now = datetime.now(timezone.utc)

        for entry in entries:
            weaviate_uuid = await self._find_weaviate_uuid(entry.id)
            if weaviate_uuid:
                if self.collection is None:
                    if self.client is None:
                        raise RuntimeError("LTMManager not initialized")
                    self.collection = self.client.collections.get(self.CLASS_NAME)

                self.collection.data.update(
                    uuid=weaviate_uuid,
                    properties={
                        "usage_count": entry.usage_count + 1,
                        "last_used_at": now,
                    },
                )

    async def update_quality(self, entry_id: str, quality: float) -> None:
        """Adjust quality score of a memory."""
        if quality < 0 or quality > 1:
            raise ValueError("Quality must be between 0 and 1")

        weaviate_uuid = await self._find_weaviate_uuid(entry_id)
        if not weaviate_uuid:
            raise ValueError(f"Memory not found: {entry_id}")

        if self.collection is None:
            if self.client is None:
                raise RuntimeError("LTMManager not initialized")
            self.collection = self.client.collections.get(self.CLASS_NAME)

        now = datetime.now(timezone.utc)
        self.collection.data.update(
            uuid=weaviate_uuid,
            properties={
                "quality": quality,
                "updated_at": now,
            },
        )

    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about memory store."""
        if self.collection is None:
            if self.client is None:
                raise RuntimeError("LTMManager not initialized")
            self.collection = self.client.collections.get(self.CLASS_NAME)

        results = self.collection.query.fetch_objects(limit=100)
        entries = self._parse_results(results.objects)

        total_memories = len(entries)
        total_quality = 0
        total_usage = 0

        for entry in entries:
            total_quality += entry.quality
            total_usage += entry.usage_count

        avg_quality = total_quality / total_memories if total_memories > 0 else 0

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
                id=props.get("original_id", ""),
                content=props.get("content", ""),
                tags=props.get("tags", []),
                session_id=props.get("session_id"),
                user_id=props.get("user_id"),
                quality=props.get("quality", 0.5),
                usage_count=props.get("usage_count", 0),
                created_at=props.get("created_at", datetime.now(timezone.utc)),
                updated_at=props.get("updated_at", datetime.now(timezone.utc)),
                last_used_at=props.get("last_used_at"),
            )
            entries.append(entry)

        return entries
