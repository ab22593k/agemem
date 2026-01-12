"""
Short-Term Memory (STM) Manager using LangChain.

Manages context window usage, summarization, and filtering using LLMs.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

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
        model_name: str = "gemini-2.5-flash-lite",
    ):
        self.max_tokens = max_tokens if max_tokens > 0 else self.DEFAULT_MAX_TOKENS
        self.compression_threshold = compression_threshold
        self.current_tokens = 0

        # Initialize LangChain LLM
        if os.getenv("GOOGLE_API_KEY"):  # Requires GOOGLE_API_KEY environment variable
            self.llm = ChatGoogleGenerativeAI(
                model=model_name, temperature=0.3, convert_system_message_to_human=True
            )
        else:
            self.llm = None
            print("Warning: GOOGLE_API_KEY not set. STM operations will be limited.")

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

    async def summary(
        self,
        content: Union[str, List[str]],
        aggressive: bool = False,
        span: Optional[Union[str, int]] = None,
    ) -> str:
        """
        Create a concise version of the input content using LLM.

        Args:
            content: Text string or list of messages.
            aggressive: If true, use more aggressive compression.
            span: Optional span to summarize.
                 If int N: last N lines/messages.
                 If "all": everything.
        """
        # Handle span
        processed_content = ""
        if isinstance(content, list):
            if span == "all" or span is None:
                processed_content = "\n".join(content)
            elif isinstance(span, int):
                processed_content = "\n".join(content[-span:])
            else:
                processed_content = "\n".join(content)
        else:
            if isinstance(span, int):
                lines = content.split("\n")
                processed_content = "\n".join(lines[-span:])
            else:
                processed_content = content

        if not self.llm:
            return self._heuristic_summary(processed_content, aggressive)

        prompt_text = (
            "Your goal is to compress the given conversation span into a concise "
            "summary that preserves all important information, intentions, "
            "decisions, and unresolved questions."
        )
        if aggressive:
            prompt_text = (
                "Aggressively compress the following text. Keep only the most "
                "critical facts and unresolved questions. Remove all redundancy."
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a conversation summarization assistant. The summary "
                        "will later be used to replace the original conversation in "
                        "the context, so make sure nothing essential is lost."
                    ),
                ),
                ("human", f"{prompt_text}\n\nTEXT:\n{{text}}"),
            ]
        )

        chain = prompt | self.llm

        try:
            response = await chain.ainvoke({"text": processed_content})
            return self._parse_llm_response(response)
        except Exception as e:  # pylint: disable=broad-exception-caught
            return (
                f"[Error in summarization: {str(e)}]\n"
                f"{self._heuristic_summary(processed_content, aggressive)}"
            )

    async def filter(
        self, content: str, criteria: str, keep_context: int = 0, semantic: bool = True
    ) -> str:
        """
        Remove irrelevant segments based on criteria.

        Args:
            content: The text to filter.
            criteria: Keywords (if semantic=False) or natural language description.
            keep_context: Surrounding lines to keep (only for keyword filter).
            semantic: If True, uses LLM to identify relevant parts.
        """
        if not semantic or not self.llm:
            return self._keyword_filter(content, criteria, keep_context)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are an expert context filter for LLM agents. Your "
                        "goal is to filter out irrelevant or redundant segments "
                        "while keeping everything related to the criteria."
                    ),
                ),
                (
                    "human",
                    "FILTER CRITERIA: {criteria}\n\nCONTENT:\n{content}\n\n"
                    "Output ONLY the filtered content that is relevant. Do "
                    "not add explanations.",
                ),
            ]
        )

        chain = prompt | self.llm

        try:
            response = await chain.ainvoke({"criteria": criteria, "content": content})
            return self._parse_llm_response(response)
        except Exception as e:  # pylint: disable=broad-exception-caught
            return (
                f"[Error in semantic filtering: {str(e)}]\n"
                f"{self._keyword_filter(content, criteria, keep_context)}"
            )

    def _keyword_filter(
        self, content: str, keywords: str, keep_context: int = 0
    ) -> str:
        """Fallback keyword-based filtering."""
        kw_list = [k.strip().lower() for k in keywords.split(",") if k.strip()]
        if not kw_list:
            return content

        lines = content.split("\n")
        kept_indices = set()

        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(k in line_lower for k in kw_list):
                start = max(0, i - keep_context)
                end = min(len(lines), i + keep_context + 1)
                for j in range(start, end):
                    kept_indices.add(j)

        if not kept_indices:
            return f"[Filtered: No matches found for criteria '{keywords}']"

        result_lines = [lines[i] for i in sorted(kept_indices)]
        return "\n".join(result_lines)

    def _parse_llm_response(self, response: Any) -> str:
        """Extract string content from LangChain response."""
        raw = None
        if hasattr(response, "content"):
            raw = response.content
        elif isinstance(response, dict) and "content" in response:
            raw = response["content"]
        else:
            raw = response

        if isinstance(raw, list):
            parts = []
            for item in raw:
                if isinstance(item, str):
                    parts.append(item)
                else:
                    try:
                        parts.append(json.dumps(item))
                    except (TypeError, ValueError):
                        parts.append(str(item))
            return "\n".join(parts)
        return str(raw)

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
