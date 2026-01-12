"""CLI bridge for LTM operations used by the compaction agent."""

import os
import sys
import asyncio
from src.ltm import LTMManager

# Configuration (Sync with main.py)
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost:8080")
USE_VECTOR_SEARCH = os.getenv("AGEMEM_VECTOR_SEARCH", "false").lower() == "true"


async def main():
    """Main entry point for the bridge CLI."""
    if len(sys.argv) < 3:
        print("Usage: bridge.py [memorize|retrieve] <content>")
        sys.exit(1)

    action = sys.argv[1]
    payload = sys.argv[2]

    try:
        async with LTMManager(
            host=WEAVIATE_HOST, use_vector_search=USE_VECTOR_SEARCH
        ) as ltm:
            if action == "memorize":
                await ltm.add(
                    content=payload,
                    metadata={"source": "opencode_compaction"},
                    memory_type="context",
                )
                print("Success")

            elif action == "delete":
                await ltm.delete(entry_id=payload)
                print(f"Deleted {payload}")

            elif action == "list":
                results = await ltm.retrieve(query="", top_k=50)
                for r in results:
                    print(f"ID: {r.id} | {r.content}")

            elif action == "retrieve":
                results = await ltm.retrieve(query=payload, top_k=3)
                if not results:
                    print("")
                    return

                formatted = "\n---\n".join(
                    [f"ID: {r.id} | {r.content}" for r in results]
                )
                print(formatted)
            else:
                print(f"Unknown action: {action}", file=sys.stderr)
                sys.exit(1)
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Action failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
