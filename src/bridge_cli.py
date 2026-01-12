import os
import sys
import asyncio
from src.ltm import LTMManager
from src.stm import STMManager

# Configuration (Sync with main.py)
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost:8080")
USE_VECTOR_SEARCH = os.getenv("AGEMEM_VECTOR_SEARCH", "false").lower() == "true"


async def main():
    if len(sys.argv) < 3:
        print("Usage: bridge.py [memorize|retrieve] <content>")
        sys.exit(1)

    action = sys.argv[1]
    payload = sys.argv[2]

    # Initialize LTM
    try:
        ltm = LTMManager(host=WEAVIATE_HOST, use_vector_search=USE_VECTOR_SEARCH)
        await ltm.initialize()
    except Exception as e:
        # Fallback if connection fails
        print(f"Error initializing LTM: {e}", file=sys.stderr)
        sys.exit(1)

    if action == "memorize":
        await ltm.add(
            content=payload,
            metadata={"source": "opencode_compaction"},
            memory_type="context",
        )
        await ltm.close()
        print("Success")

    elif action == "delete":
        await ltm.delete(entry_id=payload)
        await ltm.close()
        print(f"Deleted {payload}")

    elif action == "list":
        results = await ltm.retrieve(query="", top_k=50)
        await ltm.close()
        for r in results:
            print(f"ID: {r.id} | {r.content}")

    elif action == "retrieve":
        results = await ltm.retrieve(query=payload, top_k=3)
        await ltm.close()
        if not results:
            print("")
            return

        formatted = "\n---\n".join([f"ID: {r.id} | {r.content}" for r in results])
        print(formatted)


if __name__ == "__main__":
    asyncio.run(main())
