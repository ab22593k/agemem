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

from src.agentic import AgenticMemoryProcessor
from src.ltm import LTMManager, MemoryFunction
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
    agentic_proc = None

    try:
        # Initialize LTM manager
        ltm_mgr = LTMManager(host=WEAVIATE_HOST, use_vector_search=USE_VECTOR_SEARCH)
        await ltm_mgr.initialize()

        # Initialize STM manager
        stm_mgr = STMManager(max_tokens=MAX_TOKENS)

        # Initialize Agentic Processor
        agentic_proc = AgenticMemoryProcessor()

        yield {"ltm": ltm_mgr, "stm": stm_mgr, "agentic": agentic_proc}

    finally:
        if ltm_mgr:
            await ltm_mgr.close()


# Initialize MCP server
mcp = FastMCP("agemem_mcp", lifespan=app_lifespan)


class AddMemoryInput(BaseModel):
    """Input model for adding a new memory to LTM."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

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
    memory_function: Optional[MemoryFunction] = Field(
        default=MemoryFunction.FACTUAL,
        description="The functional type of memory being stored. "
        "factual: Knowledge from environment/users. "
        "experiential: Problem-solving traces. "
        "working: Transient workspace data.",
    )
    quality: Optional[float] = Field(
        default=0.5, description="Initial quality score 0-1 (default 0.5)", ge=0, le=1
    )
    agentic: bool = Field(
        default=True,
        description="If true, use agentic note construction, linking, and evolution.",
    )


class UpdateMemoryInput(BaseModel):
    """Input model for updating an existing memory."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

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

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

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

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

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
    memory_function: Optional[MemoryFunction] = Field(
        default=None,
        description="Filter by memory function (factual, experiential, working).",
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
        description="Minimum similarity score for vector search (0-1).Higher is more strict.",
        ge=0,
        le=1,
    )
    include_links: bool = Field(
        default=True,
        description="If true, retrieve contextually linked memories (Box retrieval).",
    )


class RateMemoryInput(BaseModel):
    """Input model for rating a memory."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    memory_id: str = Field(
        ..., description="The ID of memory to rate (from retrieve_memory output)"
    )
    quality: float = Field(..., description="New quality score (0-1)", ge=0, le=1)


class SummarizeContextInput(BaseModel):
    """Input model for summarizing context."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

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

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

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
        description=("Number of surrounding lines to keep (only for keyword filter, default 0)"),
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

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

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
    """
    ltm = ctx.request_context.lifespan_context["ltm"]
    agentic_proc = ctx.request_context.lifespan_context["agentic"]

    try:
        enrichment = {}
        if params.agentic and agentic_proc.llm:
            enrichment = await agentic_proc.form_memory(
                params.content,
                function=params.memory_function or MemoryFunction.FACTUAL,
            )

        entry = await ltm.add(
            content=params.content,
            metadata=params.metadata,
            memory_function=params.memory_function or MemoryFunction.FACTUAL,
            quality=params.quality or 0.5,
            keywords=enrichment.get("keywords", []),
            tags=enrichment.get("tags", []),
            context_description=enrichment.get("context_description"),
        )

        if params.agentic and agentic_proc.llm:
            await agentic_proc.orchestrate_lifecycle(ltm, entry, params.content)

        res = f"Memory added successfully. ID: {entry.id} (Quality: {entry.quality:.2f})"
        if entry.keywords:
            res += f"\nKeywords: {', '.join(entry.keywords)}"
        if entry.tags:
            res += f"\nTags: {', '.join(entry.tags)}"
        return res
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
    """
    ltm = ctx.request_context.lifespan_context["ltm"]

    try:
        await ltm.update_quality(entry_id=params.memory_id, quality=params.quality)
        return f"Memory {params.memory_id} quality updated to {params.quality:.2f}"
    except Exception as e:  # pylint: disable=broad-exception-caught
        if "not found" in str(e).lower():
            return f"Error: Memory not found: {params.memory_id}"
        return f"Error: Failed to rate memory: {type(e).__name__}"


class PruneMemoryInput(BaseModel):
    """Input model for pruning memories."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    query: Optional[str] = Field(
        default=None,
        description="Search query to identify a neighborhood to prune.",
    )


@mcp.tool(
    name="prune_memories",
    annotations=ToolAnnotations(
        title="Prune Redundant Memories",
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=False,
        openWorldHint=True,
    ),
)
async def prune_memories(params: PruneMemoryInput, ctx: Context) -> str:
    """
    Identifies and merges redundant memories in a neighborhood.
    """
    ltm = ctx.request_context.lifespan_context["ltm"]
    agentic_proc = ctx.request_context.lifespan_context["agentic"]

    if not agentic_proc.llm:
        return "Error: LLM not configured for agentic operations."

    try:
        if params.query:
            candidates = await ltm.retrieve(params.query, top_k=10)
        else:
            stats = await ltm.get_stats()
            if stats["total_memories"] == 0:
                return "No memories to prune."
            candidates = await ltm.retrieve("", top_k=10)

        if len(candidates) < 2:
            return "Not enough memories in the neighborhood to prune."

        merge_plan = await agentic_proc.plan_merge(candidates)
        if not merge_plan:
            return "No redundant memories found in this neighborhood."

        await ltm.apply_merge_plan(merge_plan)

        num_merged = len(merge_plan["redundant_ids"])

        # ruff: noqa: E501
        return f"Successfully merged {num_merged} memories into {merge_plan['survivor_id']}."

    except Exception as e:  # pylint: disable=broad-exception-caught
        return f"Error: Pruning failed: {type(e).__name__}: {str(e)}"


# ============================================================================
# STM Tools (Short-Term Memory / Context)
# ============================================================================


class DecayLinksInput(BaseModel):
    """Input model for decaying links."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    decay_factor: Optional[float] = Field(
        default=0.9, description="Factor to multiply weights by (0-1).", ge=0, le=1
    )


@mcp.tool(
    name="decay_memory_links",
    annotations=ToolAnnotations(
        title="Decay Memory Link Weights",
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=False,
        openWorldHint=False,
    ),
)
async def decay_memory_links(params: DecayLinksInput, ctx: Context) -> str:
    """
    Applies temporal decay to all memory links.
    """
    ltm = ctx.request_context.lifespan_context["ltm"]

    try:
        removed = await ltm.decay_links(decay_factor=params.decay_factor or 0.9)
        return f"Link decay applied. {removed} weak associations were forgotten."
    except Exception as e:  # pylint: disable=broad-exception-caught
        return f"Error: Decay failed: {type(e).__name__}: {str(e)}"


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
    """
    ltm = ctx.request_context.lifespan_context["ltm"]

    try:
        results = await ltm.retrieve(
            query=params.query,
            top_k=params.top_k or 3,
            memory_function=params.memory_function,
            metadata_filter=params.metadata_filter,
            min_quality=params.min_quality or 0,
            update_usage=True,
            search_type=params.search_type,
            min_similarity=params.min_similarity,
            include_links=params.include_links,
        )

        if not results:
            return "No relevant memories found."

        found_msg = (
            f"Found {len(results)} memories (including links):"
            if params.include_links
            else f"Found {len(results)} memories:"
        )
        lines = [found_msg]
        for r in results:
            line = f"- [{r.id}] (Q:{r.quality:.2f}, Used:{r.usage_count}) {r.content}"
            if r.context_description:
                line += f"\n  Context: {r.context_description}"
            if r.keywords:
                line += f"\n  Keywords: {', '.join(r.keywords)}"
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
    """
    stm = ctx.request_context.lifespan_context["stm"]

    summary = await stm.summary(params.content, aggressive=params.aggressive, span=params.span)

    original_tokens = stm.estimate_tokens(params.content)
    new_tokens = stm.estimate_tokens(summary)
    savings = original_tokens - new_tokens

    return f"{summary}\n\n[Compression: {original_tokens} -> {new_tokens} tokens, saved {savings}]"


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
    Filters out irrelevant or outdated content from the conversation context.
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
    Get current context window usage statistics.
    """
    stm = ctx.request_context.lifespan_context["stm"]

    if params.current_context:
        stm.track_context(params.current_context)

    stats = stm.get_stats()
    result = json.dumps(stats, indent=2)

    if stats["should_summarize"]:
        msg = "\n\n⚠️ RECOMMENDATION: Usage is high. Use summarize_context or filter_context."
        result += msg

    return result


if __name__ == "__main__":
    mcp.run()
