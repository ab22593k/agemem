"""
Short-Term Memory (STM) Manager using LangChain.

Manages context window usage, summarization, and filtering using LLMs.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate  # type: ignore
from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore


@dataclass
class ContextStats:
    """Statistics about current context usage."""

    current_tokens: int
    max_tokens: int
    usage_percent: float
    should_summarize: bool
    compression_threshold: float


class STMManager:
    """Handles Short-Term Memory operations (Context Management) using LangChain."""

    DEFAULT_MAX_TOKENS = 128000
    DEFAULT_COMPRESSION_THRESHOLD = 0.7

    def __init__(
        self,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        compression_threshold: float = DEFAULT_COMPRESSION_THRESHOLD,
        model_name: str = "gemini-1.5-flash",
    ):
        self.max_tokens = max_tokens if max_tokens > 0 else self.DEFAULT_MAX_TOKENS
        self.compression_threshold = compression_threshold
        self.current_tokens = 0

        # Initialize LangChain LLM
        # Requires GOOGLE_API_KEY environment variable
        if os.getenv("GOOGLE_API_KEY"):
            self.llm = ChatGoogleGenerativeAI(
                model=model_name, temperature=0.3, convert_system_message_to_human=True
            )
        else:
            self.llm = None
            print("Warning: GOOGLE_API_KEY not set. STM summarization will be limited.")

    def estimate_tokens(self, content: str) -> int:
        """
        Provides a token count estimate.

        If available, we could use the LLM's tokenizer.
        For now, we use a heuristic (1 token ~= 4 chars) to be fast and free.
        """
        return len(content) // 4

    def track_context(self, content: str) -> None:
        """Update the current token count based on provided context."""
        self.current_tokens = self.estimate_tokens(content)

    def get_stats(self) -> Dict[str, Any]:
        """Return current context statistics."""
        usage_percent = (
            (self.current_tokens / self.max_tokens) * 100 if self.max_tokens else 0
        )
        return {
            "current_tokens": self.current_tokens,
            "max_tokens": self.max_tokens,
            "usage_percent": usage_percent,
            "should_summarize": usage_percent >= (self.compression_threshold * 100),
            "compression_threshold": self.compression_threshold * 100,
        }

    async def summary(self, content: str, aggressive: bool = False) -> str:
        """
        Create a concise version of the input content using LLM.
        """
        if not self.llm:
            return self._heuristic_summary(content, aggressive)

        prompt_text = (
            "Summarize the following text, preserving key facts and actionable items."
        )
        if aggressive:
            prompt_text = (
                "Aggressively compress the following text. Keep only the most "
                "critical facts. Remove all fluff."
            )

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert context compressor for LLM agents."),
            ("human", f"{prompt_text}\n\nTEXT:\n{{text}}"),
        ])

        chain = prompt | self.llm

        try:
            response = await chain.ainvoke({"text": content})
            # Normalize response to a string. The LLM result can be a str, a list,
            # or an object with a .content attribute that is one of those types.
            raw = None
            if hasattr(response, "content"):
                raw = response.content
            elif isinstance(response, dict) and "content" in response:
                raw = response["content"]
            else:
                raw = response

            # If raw is a list, convert each element to string and join.
            if isinstance(raw, list):
                parts = []
                for item in raw:
                    if isinstance(item, str):
                        parts.append(item)
                    else:
                        try:
                            parts.append(json.dumps(item))
                        except (FileNotFoundError, json.JSONDecodeError):
                            parts.append(str(item))
                return "\n".join(parts)
            # For all other cases, ensure we return a string.
            return str(raw)
        except Exception as e:  # pylint: disable=broad-exception-caught
            return (
                f"[Error in summarization: {str(e)}]\n"
                f"{self._heuristic_summary(content, aggressive)}"
            )

    async def filter(self, content: str, keywords: str, keep_context: int = 0) -> str:
        """
        Remove irrelevant segments based on keywords.
        """
        # Simple heuristic filtering is often more reliable for "exact keyword"
        # requirements
        kw_list = [k.strip().lower() for k in keywords.split(",") if k.strip()]
        if not kw_list:
            return content

        lines = content.split("\n")
        kept_indices = set()

        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(k in line_lower for k in kw_list):
                # Keep this line and context
                start = max(0, i - keep_context)
                end = min(len(lines), i + keep_context + 1)
                for j in range(start, end):
                    kept_indices.add(j)

        if not kept_indices:
            return f"[Filtered: No matches found for keywords '{keywords}']"

        result_lines = [lines[i] for i in sorted(kept_indices)]

        return "\n".join(result_lines)

    def _heuristic_summary(self, content: str, aggressive: bool) -> str:
        """Fallback summary if LLM is unavailable."""
        lines = [line for line in content.split("\n") if line.strip()]
        if not lines:
            return ""

        limit = 5 if aggressive else 15
        if len(lines) <= limit:
            return content

        head = lines[: limit // 2]
        tail = lines[-limit // 2 :]
        omitted_count = len(lines) - limit
        return (
            "\n".join(head)
            + f"\n\n... [{omitted_count} lines omitted] ...\n\n"
            + "\n".join(tail)
        )
