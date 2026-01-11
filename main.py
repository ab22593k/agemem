#!/usr/bin/env python3
"""
Agentic Memory (AgeMem) MCP Server - Python Implementation.

This server provides tools for Long-Term Memory (LTM) and Short-Term Memory (STM)
management for LLM agents, implementing Agentic Memory framework.
"""

import json
import os
from contextlib import asynccontextmanager
from typing import Optional

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ltm import LTMManager
from stm import STMManager

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
    tags: Optional[str] = Field(
        default=None,
        description="Comma-separated tags (e.g., 'user-preferences,project-details')",
    )
    session_id: Optional[str] = Field(
        default=None, description="Optional session identifier for scoping"
    )
    user_id: Optional[str] = Field(
        default=None, description="Optional user identifier for scoping"
    )
    quality: Optional[float] = Field(
        default=0.5, description="Initial quality score 0-1 (default 0.5)", ge=0, le=1
    )

    @field_validator("tags")
    @classmethod
    def parse_tags(cls, v: Optional[str]) -> list:
        """Parse comma-separated tags string into a list."""
        if v is None:
            return []
        tags = [t.strip() for t in v.split(",") if t.strip()]
        return tags


class UpdateMemoryInput(BaseModel):
    """Input model for updating an existing memory."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    id: str = Field(
        ..., description="The ID of memory to update (from retrieve_memory output)"
    )
    content: str = Field(
        ..., description="The new content", min_length=1, max_length=100000
    )


class DeleteMemoryInput(BaseModel):
    """Input model for deleting a memory."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    id: str = Field(
        ..., description="The ID of memory to delete (from retrieve_memory output)"
    )


class RetrieveMemoryInput(BaseModel):
    """Input model for retrieving memories from LTM."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    query: str = Field(..., description="The search query", min_length=1)
    limit: Optional[int] = Field(
        default=5,
        description="Maximum number of results (default 5, max 20)",
        ge=1,
        le=20,
    )
    session_id: Optional[str] = Field(default=None, description="Filter by session ID")
    user_id: Optional[str] = Field(default=None, description="Filter by user ID")
    min_quality: Optional[float] = Field(
        default=0, description="Minimum quality score to include (0-1)", ge=0, le=1
    )


class RateMemoryInput(BaseModel):
    """Input model for rating a memory."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    id: str = Field(
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


class FilterContextInput(BaseModel):
    """Input model for filtering context."""

    model_config = ConfigDict(
        str_strip_whitespace=True, validate_assignment=True, extra="forbid"
    )

    content: str = Field(..., description="The text to filter", min_length=1)
    keywords: str = Field(
        ...,
        description="Keywords to keep (e.g., 'important,urgent,deadline')",
    )
    keep_context: Optional[int] = Field(
        default=0,
        description="Number of surrounding lines to keep (default 0)",
        ge=0,
        le=10,
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
    Add new knowledge to Long-Term Memory (LTM). Use this when user provides
    important information that should be remembered for future sessions.

    Args:
        params (AddMemoryInput): Validated input parameters containing:
            - content (str): The content to store in memory
            - tags (Optional[str]): Comma-separated tags for categorization
            - session_id (Optional[str]): Optional session identifier for scoping
            - user_id (Optional[str]): Optional user identifier for scoping
            - quality (Optional[float]): Initial quality score 0-1 (default 0.5)

    Returns:
        str: Confirmation message with memory ID and quality score

    Error Handling:
        - Returns "Error: Failed to add memory" if storage fails
        - Validates content length (max 100KB)
        - Validates quality range (0-1)
    """
    ltm = ctx.request_context.lifespan_context["ltm"]

    try:
        entry = await ltm.add(
            content=params.content,
            tags=params.tags,
            session_id=params.session_id,
            user_id=params.user_id,
            quality=params.quality,
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
    Update an existing memory entry. Use this to refine or correct information in LTM.

    Args:
        params (UpdateMemoryInput): Validated input parameters containing:
            - id (str): The ID of memory to update (from retrieve_memory output)
            - content (str): The new content

    Returns:
        str: Confirmation message with updated memory ID

    Error Handling:
        - Returns "Error: Memory not found" if ID is invalid
        - Returns "Error: Failed to update memory" if update fails
    """
    ltm = ctx.request_context.lifespan_context["ltm"]

    try:
        await ltm.update(entry_id=params.id, content=params.content)
        return f"Memory {params.id} updated."
    except Exception as e:  # pylint: disable=broad-exception-caught
        if "not found" in str(e).lower():
            return f"Error: Memory not found: {params.id}"
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
    Delete memory entry from LTM. Use this to remove obsolete or incorrect information.

    Args:
        params (DeleteMemoryInput): Validated input parameters containing:
            - id (str): The ID of memory to delete (from retrieve_memory output)

    Returns:
        str: Confirmation message with deleted memory ID

    Error Handling:
        - Returns "Error: Memory not found" if ID is invalid
        - Returns "Error: Failed to delete memory" if deletion fails
    """
    ltm = ctx.request_context.lifespan_context["ltm"]

    try:
        await ltm.delete(entry_id=params.id)
        return f"Memory {params.id} deleted."
    except Exception as e:  # pylint: disable=broad-exception-caught
        if "not found" in str(e).lower():
            return f"Error: Memory not found: {params.id}"
        return f"Error: Failed to delete memory: {type(e).__name__}"


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
    Retrieve relevant memories from LTM based on a query.
    Use this to recall past information.

    Args:
        params (RetrieveMemoryInput): Validated input parameters containing:
            - query (str): The search query
            - limit (Optional[int]): Maximum number of results (default 5, max 20)
            - session_id (Optional[str]): Filter by session ID
            - user_id (Optional[str]): Filter by user ID
            - min_quality (Optional[float]): Minimum quality score to include (0-1)

    Returns:
        str: Formatted list of matching memories

    Error Handling:
        - Returns "No relevant memories found" if no matches
        - Returns "Error: Retrieval failed" if search fails
    """
    ltm = ctx.request_context.lifespan_context["ltm"]

    try:
        results = await ltm.retrieve(
            query=params.query,
            limit=params.limit,
            session_id=params.session_id,
            user_id=params.user_id,
            min_quality=params.min_quality,
            update_usage=True,
        )

        if not results:
            return "No relevant memories found."

        lines = [f"Found {len(results)} memories:"]
        for r in results:
            line = f"- [{r.id}] (Q:{r.quality:.2f}, Used:{r.usage_count}) {r.content}"
            if r.tags:
                line += f" [Tags: {', '.join(r.tags)}]"
            lines.append(line)

        return "\n".join(lines)
    except Exception as e:  # pylint: disable=broad-exception-caught
        return f"Error: Retrieval failed: {type(e).__name__}"


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
        params (RateMemoryInput): Validated input parameters containing:
            - id (str): The ID of memory to rate (from retrieve_memory output)
            - quality (float): New quality score (0-1)

    Returns:
        str: Confirmation message with updated quality score

    Examples:
        - Use when: "This memory is very important" -> id="abc123", quality=0.95
        - Use when: "This memory isn't useful" -> id="def456", quality=0.2
        - Don't use when: You need to change content (use update_memory instead)

    Error Handling:
        - Returns "Error: Memory not found" if ID is invalid
        - Returns "Error: Failed to rate memory" if rating fails
    """
    ltm = ctx.request_context.lifespan_context["ltm"]

    try:
        await ltm.update_quality(entry_id=params.id, quality=params.quality)
        return f"Memory {params.id} quality updated to {params.quality:.2f}"
    except Exception as e:  # pylint: disable=broad-exception-caught
        if "not found" in str(e).lower():
            return f"Error: Memory not found: {params.id}"
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

    Returns:
        str: JSON-formatted statistics including total memories, average quality,
             and total retrieval count

    Examples:
        - Use when: "How many memories do we have stored?"
        - Use when: "What's average quality of our memories?"
        - Use when: Monitoring memory health and usage

    Error Handling:
        - Returns "Error: Failed to get stats" if retrieval fails
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
    Summarize a block of text or context (STM). Use this to compress information
    and save context window space.

    Args:
        params (SummarizeContextInput): Validated input parameters containing:
            - content (str): The text to summarize
            - aggressive (Optional[bool]): If true, use more aggressive compression

    Returns:
        str: Summarized text with compression statistics (original tokens → new tokens)

    Examples:
        - Use when: "This context is getting long, compress it"
        - Use when: "Summarize this conversation" -> content="[long text]"
        - Use when:
            "Aggressive compression needed" -> content="[long text]", aggressive=True

    Error Handling:
        - Returns summarized text with compression statistics
    """
    stm = ctx.request_context.lifespan_context["stm"]

    summary = await stm.summary(params.content, aggressive=params.aggressive)

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
        title="Filter Context by Keywords",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def filter_context(params: FilterContextInput, ctx: Context) -> str:
    """
    Filter text to keep only relevant information based on keywords (STM).
    Use this to remove noise from context.

    Args:
        params (FilterContextInput): Validated input parameters containing:
            - content (str): The text to filter
            - keywords (str): Keywords to keep (comma separated)
            - keep_context (Optional[int]): Number of surrounding lines to keep

    Returns:
        str: Filtered text showing only lines matching keywords, with
             optional surrounding context lines

    Examples:
        - Use when:
            "Keep only important parts" -> content="[text]", keywords="important,urgent"
        - Use when: "Filter with context" -> keywords="error", keep_context=2
        - Don't use when: You need to preserve all information

    Error Handling:
        - Returns "[Content filtered out - no matches for: X]" if no keywords found
    """
    stm = ctx.request_context.lifespan_context["stm"]

    filtered = await stm.filter(
        content=params.content,
        keywords=params.keywords,
        keep_context=params.keep_context,
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

    Args:
        params (ContextStatsInput): Validated input parameters containing:
            - current_context (Optional[str]): Optional: current context to analyze

    Returns:
        str: JSON-formatted statistics including current tokens, max tokens,
             usage percentage, and whether summarization is recommended

    Examples:
        - Use when: "How much context are we using?"
        - Use when: "Should I summarize this?"
        - Use when: Monitoring before adding more content

    Error Handling:
        - Returns statistics with recommendation if usage is high
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
