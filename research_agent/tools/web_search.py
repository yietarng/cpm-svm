from __future__ import annotations

import os
from typing import Any

from langchain_core.tools import BaseTool


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "Search the web for recent information. "
        "Input: a natural-language search query. "
        "Output: a list of snippets with source URLs."
    )

    def _run(self, query: str, **kwargs: Any) -> str:
        from tavily import TavilyClient
        api_key = os.environ.get("TAVILY_API_KEY", "")
        client = TavilyClient(api_key=api_key)
        response = client.search(query, max_results=5)
        results = response.get("results", [])
        if not results:
            return "No results found."
        lines = []
        for r in results:
            lines.append(f"- {r.get('title', 'Untitled')}: {r.get('url', '')}\n  {r.get('content', '')[:300]}")
        return "\n".join(lines)

    async def _arun(self, query: str, **kwargs: Any) -> str:
        return self._run(query, **kwargs)


def build_web_search_tool() -> WebSearchTool:
    return WebSearchTool()
