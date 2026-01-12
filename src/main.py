#!/usr/bin/env python3
"""
Agentic Memory (AgeMem) MCP Server - Python Implementation.

This server provides tools for Long-Term Memory (LTM) and Short-Term Memory (STM)
management for LLM agents, implementing Agentic Memory framework.
"""

import json
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, Literal, Optional, Union

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from src.ltm import LTMManager
from src.stm import STMManager

# Configuration
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost:8080")
USE_VECTOR_SEARCH = os.getenv("AGEMEM_VECTOR_SEARCH", "false").lower() == "true"
MAX_TOKENS = int(os.getenv("AGEMEM_MAX_TOKENS", "128000"))


# Lifespan manager for persistent connections
@asynccontextmanager
async def app_lifespan(_server: FastMCP):
    """Manage resources that live for server's lifetime."""
    ltm_mgr = None
    stm_mgr = None

    try:
        # Initialize LTM manager
        ltm_mgr = LTMManager(host=WEAVIATE_HOST, use_vector_search=USE_VECTOR_SEARCH)
        await ltm_mgr.initialize()

        # Initialize STM manager
        stm_mgr = STMManager(max_tokens=MAX_TOKENS)

        yield {"ltm": ltm_mgr, "stm": stm_mgr}

    finally:
        if ltm_mgr:
            await ltm_mgr.close()


# Initialize MCP server
mcp = FastMCP("agemem_mcp", lifespan=app_lifespan)


class AddMemoryInput(BaseModel):
    """Input model for adding a new memory to LTM."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    content: str = Field(
        ...,
        description="The content to store in memory",
        min_length=1,
        max_length=100000,
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata tags to categorize and filter the memory.",
    )
    memory_type: Optional[str] = Field(
        default=None,
        description="The type of memory being stored."
        "Examples: 'fact', 'preference', 'context', 'plan'.",
    )
    quality: Optional[float] = Field(
        default=0.5, description="Initial quality score 0-1 (default 0.5)", ge=0, le=1
    )


class UpdateMemoryInput(BaseModel):
    """Input model for updating an existing memory."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    memory_id: str = Field(
        ...,
        description="The unique identifier of the memory to update."
        "Must be obtained from a previous memory retrieval operation.",
    )
    content: str = Field(
        ...,
        description="The new content to replace the existing memory content.",
        min_length=1,
        max_length=100000,
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Updated metadata for the memory."
    )


class DeleteMemoryInput(BaseModel):
    """Input model for deleting a memory."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    memory_id: str = Field(
        ...,
        description="The unique identifier of the memory to delete."
        "Must be obtained from a previous memory retrieval operation.",
    )
    confirmation: bool = Field(
        ..., description="Confirmation that this memory should be permanently deleted."
    )


class RetrieveMemoryInput(BaseModel):
    """Input model for retrieving memories from LTM."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    query: str = Field(
        ...,
        description="The search query to find relevant memories."
        "Should describe what kind of information or context is needed.",
        min_length=1,
    )
    top_k: Optional[int] = Field(
        default=3,
        description="The maximum number of memories to retrieve. Defaults to 3.",
        ge=1,
        le=20,
    )
    memory_type: Optional[str] = Field(
        default=None,
        description="Filter by memory type (e.g., 'context', 'fact', 'preference').",
    )
    metadata_filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata filters to narrow down memory search.",
    )
    min_quality: Optional[float] = Field(
        default=0, description="Minimum quality score to include (0-1)", ge=0, le=1
    )
    search_type: Optional[Literal["vector", "keyword"]] = Field(
        default=None,
        description="Type of search. 'vector' for semantic similarity"
        ", 'keyword' for BM25. Defaults to vector if enabled on server.",
    )
    min_similarity: Optional[float] = Field(
        default=None,
        description="Minimum similarity score for vector search (0-1)."
        "Higher is more strict.",
        ge=0,
        le=1,
    )


class RateMemoryInput(BaseModel):
    """Input model for rating a memory."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    memory_id: str = Field(
        ..., description="The ID of memory to rate (from retrieve_memory output)"
    )
    quality: float = Field(..., description="New quality score (0-1)", ge=0, le=1)


class SummarizeContextInput(BaseModel):
    """Input model for summarizing context."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    content: str = Field(..., description="The text to summarize", min_length=1)
    aggressive: Optional[bool] = Field(
        default=False, description="If true, use more aggressive compression"
    )
    span: Optional[Union[str, int]] = Field(
        default=None,
        description="The range of conversation rounds to summarize."
        "Can be 'all' for entire context, or a number (e.g., '5')"
        "for the last N rounds.",
    )


class FilterContextInput(BaseModel):
    """Input model for filtering context."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    content: str = Field(..., description="The text to filter", min_length=1)
    criteria: str = Field(
        ...,
        description=(
            "The criteria for content removal. Can be keywords, phrases, or "
            "descriptions of content types to remove"
            "(e.g., 'the birthday of John', 'the age of Mary')."
        ),
    )
    keep_context: Optional[int] = Field(
        default=0,
        description=(
            "Number of surrounding lines to keep (only for keyword filter, default 0)"
        ),
        ge=0,
        le=10,
    )
    semantic: Optional[bool] = Field(
        default=True,
        description=(
            "If True (default), uses LLM for semantic filtering. "
            "If False, uses exact keyword matching."
        ),
    )


class ContextStatsInput(BaseModel):
    """Input model for getting context stats."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    current_context: Optional[str] = Field(
        default=None, description="Optional: provide current context to analyze"
    )


# ============================================================================
# LTM Tools (Long-Term Memory)
# ============================================================================


@mcp.tool(
    name="add_memory",
    annotations=ToolAnnotations(
        title="Add Memory to Long-Term Storage",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
async def add_memory(params: AddMemoryInput, ctx: Context) -> str:
    """
    Adds new information to external memory store for future reference.
    Use this when user provides important information that should be remembered.

    Args:
        params (AddMemoryInput): Validated input parameters.

    Returns:
        str: Confirmation message with memory ID and quality score
    """
    ltm = ctx.request_context.lifespan_context["ltm"]

    try:
        entry = await ltm.add(
            content=params.content,
            metadata=params.metadata,
            memory_type=params.memory_type,
            quality=params.quality or 0.5,
        )
        return (
            f"Memory added successfully. ID: {entry.id} (Quality: {entry.quality:.2f})"
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        return f"Error: Failed to add memory: {type(e).__name__}"


@mcp.tool(
    name="update_memory",
    annotations=ToolAnnotations(
        title="Update Existing Memory",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def update_memory(params: UpdateMemoryInput, ctx: Context) -> str:
    """
    Updates existing memory. Requires memory_id from prior retrieval.
    Use this to refine or correct information in LTM.

    Args:
        params (UpdateMemoryInput): Validated input parameters.

    Returns:
        str: Confirmation message with updated memory ID
    """
    ltm = ctx.request_context.lifespan_context["ltm"]

    try:
        await ltm.update(
            entry_id=params.memory_id, content=params.content, metadata=params.metadata
        )
        return f"Memory {params.memory_id} updated."
    except Exception as e:  # pylint: disable=broad-exception-caught
        if "not found" in str(e).lower():
            return f"Error: Memory not found: {params.memory_id}"
        return f"Error: Failed to update memory: {type(e).__name__}"


@mcp.tool(
    name="delete_memory",
    annotations=ToolAnnotations(
        title="Delete Memory from Long-Term Storage",
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def delete_memory(params: DeleteMemoryInput, ctx: Context) -> str:
    """
    Removes memory from store. Requires confirmation.
    Use this to remove obsolete or incorrect information.

    Args:
        params (DeleteMemoryInput): Validated input parameters.

    Returns:
        str: Confirmation message
    """
    if not params.confirmation:
        return "Error: Deletion not confirmed."

    ltm = ctx.request_context.lifespan_context["ltm"]

    try:
        await ltm.delete(entry_id=params.memory_id)
        return f"Memory {params.memory_id} deleted."
    except Exception as e:  # pylint: disable=broad-exception-caught
        if "not found" in str(e).lower():
            return f"Error: Memory not found: {params.memory_id}"
        return f"Error: Failed to delete memory: {type(e).__name__}"


@mcp.tool(
    name="rate_memory",
    annotations=ToolAnnotations(
        title="Rate Memory Quality",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def rate_memory(params: RateMemoryInput, ctx: Context) -> str:
    """
    Adjust quality rating of a memory.
    Higher quality memories are prioritized in retrieval.

    Args:
        params (RateMemoryInput): Validated input parameters.
    """
    ltm = ctx.request_context.lifespan_context["ltm"]

    try:
        await ltm.update_quality(entry_id=params.memory_id, quality=params.quality)
        return f"Memory {params.memory_id} quality updated to {params.quality:.2f}"
    except Exception as e:  # pylint: disable=broad-exception-caught
        if "not found" in str(e).lower():
            return f"Error: Memory not found: {params.memory_id}"
        return f"Error: Failed to rate memory: {type(e).__name__}"


@mcp.tool(
    name="memory_stats",
    annotations=ToolAnnotations(
        title="Get Memory Statistics",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def memory_stats(ctx: Context) -> str:
    """
    Get statistics about long-term memory store.
    """
    ltm = ctx.request_context.lifespan_context["ltm"]

    try:
        stats = await ltm.get_stats()
        return json.dumps(stats, indent=2)
    except Exception as e:  # pylint: disable=broad-exception-caught
        return f"Error: Failed to get stats: {type(e).__name__}"


# ============================================================================
# STM Tools (Short-Term Memory / Context)
# ============================================================================


@mcp.tool(
    name="retrieve_memory",
    annotations=ToolAnnotations(
        title="Retrieve Relevant Memories",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def retrieve_memory(params: RetrieveMemoryInput, ctx: Context) -> str:
    """
    Retrieves relevant memories and adds them to current context.
    Use this to recall past information from LTM.

    Args:
        params (RetrieveMemoryInput): Validated input parameters.

    Returns:
        str: Formatted list of matching memories
    """
    ltm = ctx.request_context.lifespan_context["ltm"]

    try:
        results = await ltm.retrieve(
            query=params.query,
            top_k=params.top_k or 3,
            memory_type=params.memory_type,
            metadata_filter=params.metadata_filter,
            min_quality=params.min_quality or 0,
            update_usage=True,
            search_type=params.search_type,
            min_similarity=params.min_similarity,
        )

        if not results:
            return "No relevant memories found."

        lines = [f"Found {len(results)} memories:"]
        for r in results:
            line = f"- [{r.id}] (Q:{r.quality:.2f}, Used:{r.usage_count}) {r.content}"
            if r.metadata:
                line += f" [Metadata: {json.dumps(r.metadata)}]"
            lines.append(line)

        return "\n".join(lines)
    except Exception as e:  # pylint: disable=broad-exception-caught
        return f"Error: Retrieval failed: {type(e).__name__}"


@mcp.tool(
    name="summarize_context",
    annotations=ToolAnnotations(
        title="Summarize Context Text",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def summarize_context(params: SummarizeContextInput, ctx: Context) -> str:
    """
    Summarizes conversation rounds to reduce tokens while preserving key information.
    Use this to compress information and save context window space.

    Args:
        params (SummarizeContextInput): Validated input parameters.
    """
    stm = ctx.request_context.lifespan_context["stm"]

    summary = await stm.summary(
        params.content, aggressive=params.aggressive, span=params.span
    )

    original_tokens = stm.estimate_tokens(params.content)
    new_tokens = stm.estimate_tokens(summary)
    savings = original_tokens - new_tokens

    return (
        f"{summary}\n\n[Compression: {original_tokens} -> "
        f"{new_tokens} tokens, saved {savings}]"
    )


@mcp.tool(
    name="filter_context",
    annotations=ToolAnnotations(
        title="Filter Context by Criteria",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def filter_context(params: FilterContextInput, ctx: Context) -> str:
    """
    Filters out irrelevant or outdated content from the conversation context
    to improve task-solving efficiency.

    Args:
        params (FilterContextInput): Validated input parameters.
    """
    stm = ctx.request_context.lifespan_context["stm"]

    filtered = await stm.filter(
        content=params.content,
        criteria=params.criteria,
        keep_context=params.keep_context,
        semantic=params.semantic,
    )

    return filtered


@mcp.tool(
    name="context_stats",
    annotations=ToolAnnotations(
        title="Get Context Usage Statistics",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def context_stats(params: ContextStatsInput, ctx: Context) -> str:
    """
    Get current context window usage statistics. Use this to decide when to
    summarize or filter.
    """
    stm = ctx.request_context.lifespan_context["stm"]

    if params.current_context:
        stm.track_context(params.current_context)

    stats = stm.get_stats()
    result = json.dumps(stats, indent=2)

    if stats["should_summarize"]:
        msg = (
            "\n\n⚠️ RECOMMENDATION: Usage is high. "
            "Use summarize_context or filter_context."
        )
        result += msg

    return result


if __name__ == "__main__":
    mcp.run()
