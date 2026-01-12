"""CLI bridge for LTM operations used by the compaction agent."""

import os
import sys
import asyncio
from src.ltm import LTMManager, MemoryFunction
from src.agentic import AgenticMemoryProcessor

# Configuration (Sync with main.py)
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost:8080")
USE_VECTOR_SEARCH = os.getenv("AGEMEM_VECTOR_SEARCH", "false").lower() == "true"


async def handle_retrieve(ltm, payload, include_links, memory_func=None):
    """Handle the 'retrieve' action."""
    results = await ltm.retrieve(
        query=payload,
        top_k=5,
        include_links=include_links,
        memory_function=memory_func,
    )
    if not results:
        print("")
        return

    # Output as formatted string with agentic fields
    output = []
    for r in results:
        # Include functional metadata in the output for the plugin to see
        func_label = f"[{r.memory_function.value.upper()}]"
        line = f"ID: {r.id} {func_label} | Content: {r.content}"
        if r.context_description:
            line += f"\n   Significance: {r.context_description}"
        if r.keywords:
            line += f"\n   Keywords: {', '.join(r.keywords)}"
        output.append(line)

    print("\n---\n".join(output))


async def handle_memorize(ltm, agentic_proc, payload, memory_func=MemoryFunction.FACTUAL):
    """Handle the 'memorize' action."""
    if not payload:
        print("Error: content required for memorize", file=sys.stderr)
        sys.exit(1)

    # 1. Note Construction
    keywords = []
    tags = []
    context_description = None

    if agentic_proc.llm:
        enrichment = await agentic_proc.form_memory(payload, function=memory_func)
        keywords = enrichment.get("keywords", [])
        tags = enrichment.get("tags", [])
        context_description = enrichment.get("context_description")

    # 2. Add Memory
    entry = await ltm.add(
        content=payload,
        metadata={"source": "opencode_compaction"},
        memory_function=memory_func,
        keywords=keywords,
        tags=tags,
        context_description=context_description,
    )

    # 3. Neighborhood analysis, Linking, Evolution & Pruning
    await agentic_proc.orchestrate_lifecycle(ltm, entry, payload)

    print(f"Success: {entry.id}")


async def handle_prune(ltm, agentic_proc, payload):
    """Handle the 'prune' action."""
    candidates = await ltm.retrieve(payload if payload else "", top_k=10)
    if len(candidates) >= 2 and agentic_proc.llm:
        merge_plan = await agentic_proc.plan_merge(candidates)
        if merge_plan:
            await ltm.apply_merge_plan(merge_plan)
            print(f"Pruned: {len(merge_plan['redundant_ids'])} memories merged.")
        else:
            print("No redundancy found.")
    else:
        print("Not enough memories to prune.")


async def main():
    """Main entry point for the bridge CLI."""
    if len(sys.argv) < 2:
        print("Usage: bridge.py [memorize|retrieve|prune|decay|list|delete] <content> [--links]")
        sys.exit(1)

    action = sys.argv[1]
    include_links = "--links" in sys.argv

    # Parse functional filter
    memory_func = None
    if "--factual" in sys.argv:
        memory_func = MemoryFunction.FACTUAL
    elif "--experiential" in sys.argv:
        memory_func = MemoryFunction.EXPERIENTIAL
    elif "--working" in sys.argv:
        memory_func = MemoryFunction.WORKING

    payload_parts = [a for a in sys.argv[2:] if not a.startswith("--")]
    payload = " ".join(payload_parts) if payload_parts else ""

    try:
        async with LTMManager(host=WEAVIATE_HOST, use_vector_search=USE_VECTOR_SEARCH) as ltm:
            agentic_proc = AgenticMemoryProcessor()

            if action == "memorize":
                await handle_memorize(
                    ltm, agentic_proc, payload, memory_func or MemoryFunction.FACTUAL
                )
            elif action == "delete":
                await ltm.delete(entry_id=payload)
                print(f"Deleted {payload}")
            elif action == "list":
                results = await ltm.retrieve(query="", top_k=50)
                for r in results:
                    print(f"ID: {r.id} | {r.content}")
            elif action == "retrieve":
                await handle_retrieve(ltm, payload, include_links, memory_func)
            elif action == "prune":
                await handle_prune(ltm, agentic_proc, payload)
            elif action == "decay":
                decay_factor = float(payload) if payload else 0.9
                removed = await ltm.decay_links(decay_factor=decay_factor)
                print(f"Decay applied: {removed} links removed.")
            else:
                print(f"Unknown action: {action}", file=sys.stderr)
                sys.exit(1)
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Action failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
