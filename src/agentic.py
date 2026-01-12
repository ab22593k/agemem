"""
Agentic Memory Processor implementing A-Mem principles.

Handles note construction, link generation, and memory evolution using LLMs.
"""

import json
import os
from typing import Any, Dict, List, Optional, cast

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from src.ltm import MemoryEntry, MemoryFunction


class AgenticMemoryProcessor:
    """Orchestrates LLM-driven memory dynamics: Formation, Evolution, and Retrieval."""

    def __init__(self, model_name: str = "gemini-2.5-flash-lite"):
        if os.getenv("GOOGLE_API_KEY"):
            self.llm = ChatGoogleGenerativeAI(
                model=model_name, temperature=0.1, convert_system_message_to_human=True
            )
        else:
            self.llm = None

    def _parse_json_response(self, response: Any) -> Dict[str, Any]:
        """Extract and parse JSON from LLM response."""
        content = ""
        if hasattr(response, "content"):
            content = str(response.content)
        else:
            content = str(response)

        # Basic JSON extraction
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        try:
            return cast(Dict[str, Any], json.loads(content))
        except json.JSONDecodeError:
            # Fallback for poorly formatted JSON
            return {}

    async def form_memory(
        self, content: str, function: MemoryFunction = MemoryFunction.FACTUAL
    ) -> Dict[str, Any]:
        """
        Extract features to create a structured memory (Formation Operator).
        Implements Equation (2) from A-Mem paper, aligned with Survey Dynamics.
        """
        if not self.llm:
            return {"keywords": [], "tags": [], "context_description": content[:100]}

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an AI memory formation assistant. "
                    "Your goal is to transform raw interaction artifacts"
                    "into discrete memory units.",
                ),
                (
                    "human",
                    (
                        "Analyze the following content and extract for {function} memory:\n"
                        "1. Keywords: A list of 3-5 key concepts.\n"
                        "2. Tags: A list of 1-3 high-level categories.\n"
                        "3. Contextual Description: A 1-2 sentence explanation of "
                        "WHY this information is useful and what it implies.\n\n"
                        "Output as JSON with keys: "
                        "'keywords', 'tags', 'context_description'.\n\n"
                        "CONTENT:\n{content}"
                    ),
                ),
            ]
        )

        chain = prompt | self.llm
        try:
            response = await chain.ainvoke(
                {
                    "content": content,
                    "function": function.value,
                }
            )
            return self._parse_json_response(response)
        except Exception:  # pylint: disable=broad-exception-caught
            return {"keywords": [], "tags": [], "context_description": content[:100]}

    async def evolve_memory(
        self, new_memory: MemoryEntry, neighbors: List[MemoryEntry]
    ) -> Dict[str, Any]:
        """
        Integrate new memory into existing neighborhood (Evolution Operator).
        Identifies links and updates historical context.
        """
        if not self.llm or not neighbors:
            return {"links": [], "evolutions": []}

        neighbors_text = "\n".join(
            [
                f"ID: {m.id} | Content: {m.content} | Context: {m.context_description}"
                for m in neighbors
            ]
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a memory network orchestrator. "
                    "Your goal is to integrate a new memory into an existing neighborhood "
                    "by establishing links and evolving historical context.",
                ),
                (
                    "human",
                    (
                        "New Memory:\n{new_content}\nContext: {new_context}\n\n"
                        "Historical Neighbors:\n{neighbors_text}\n\n"
                        "Task:\n"
                        "1. Identify IDs of neighbors that are contextually related "
                        "to the new memory (Linking).\n"
                        "2. For any neighbor whose understanding is refined or changed "
                        "by this new memory, provide an updated 'context_description' "
                        "(Evolution).\n\n"
                        "Output as JSON with keys: 'links' (list of IDs) and 'evolutions' "
                        "(list of objects with 'id' and 'context_description')."
                    ),
                ),
            ]
        )

        chain = prompt | self.llm
        try:
            response = await chain.ainvoke(
                {
                    "new_content": new_memory.content,
                    "new_context": new_memory.context_description,
                    "neighbors_text": neighbors_text,
                }
            )
            return self._parse_json_response(response)
        except Exception:  # pylint: disable=broad-exception-caught
            return {"links": [], "evolutions": []}

    async def plan_merge(self, memories: List[MemoryEntry]) -> Optional[Dict[str, Any]]:
        """
        Analyze a group of similar memories and plan a merge if they are redundant.
        Returns a merge plan:
            { 'survivor_id', 'new_content', 'new_context', 'redundant_ids' }
        """
        if not self.llm or len(memories) < 2:
            return None

        memories_text = "\n".join(
            [
                f"ID: {m.id} | Content: {m.content} | Context: {m.context_description} "
                f"| Keywords: {m.keywords}"
                for m in memories
            ]
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a memory pruning assistant. You identify redundant "
                        "information and merge it into a single, comprehensive record."
                    ),
                ),
                (
                    "human",
                    (
                        "Review these memories and determine if any of them are "
                        "redundant (describing the same fact/preference/event with "
                        "high overlap).\n\n"
                        "Memories:\n{memories_text}\n\n"
                        "If merges are needed:\n"
                        "1. Select the best 'survivor' ID (usually most detailed or recent).\n"
                        "2. Create a 'new_content' that combines all unique details.\n"
                        "3. Create a 'new_context' that captures the evolved understanding.\n"
                        "4. List the 'redundant_ids' that should be deleted.\n"
                        "If no redundancy is found, return 'NO_REDUNDANCY'.\n"
                        "Output as JSON with keys: 'survivor_id', 'new_content', "
                        "'new_context', 'redundant_ids'."
                    ),
                ),
            ],
        )

        chain = prompt | self.llm
        try:
            response = await chain.ainvoke({"memories_text": memories_text})
            content = str(response.content) if hasattr(response, "content") else str(response)
            if "NO_REDUNDANCY" in content:
                return None

            return self._parse_json_response(response)
        except Exception:  # pylint: disable=broad-exception-caught
            return None

    async def orchestrate_lifecycle(self, ltm: Any, entry: MemoryEntry, content: str) -> None:
        """Orchestrate the memory lifecycle dynamics: Evolution and Pruning."""
        if not self.llm:
            return

        neighbors = await ltm.retrieve(content, top_k=5)
        neighbors = [n for n in neighbors if n.id != entry.id]

        if not neighbors:
            return

        # 1. Memory Evolution (Linking & Context refinement)
        analysis = await self.evolve_memory(entry, neighbors)

        # Apply links
        if analysis.get("links"):
            await ltm.update_links(entry.id, analysis["links"])

        # Apply evolutions (Updating historical neighbors)
        for update in analysis.get("evolutions", []):
            target_id = update.pop("id")
            await ltm.update_agentic_fields(target_id, **update)

        # 2. Consolidation (Evolution through pruning)
        all_candidates = [entry] + neighbors
        merge_plan = await self.plan_merge(all_candidates)
        if merge_plan:
            await ltm.apply_merge_plan(merge_plan)
