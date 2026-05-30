from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from research_agent.memory.ltm import LTMStore


class RAGTool(BaseTool):
    name: str = "rag_retrieval"
    description: str = (
        "Retrieve relevant documents from the knowledge base using semantic search. "
        "Input: a natural-language query. "
        "Output: relevant document excerpts with source citations."
    )
    ltm_store: Any  # LTMStore — declared as Any to avoid Pydantic issues with custom types

    class Config:
        arbitrary_types_allowed = True

    def _run(self, query: str, **kwargs: Any) -> str:
        docs = self.ltm_store.retrieve(query)
        return self.ltm_store.format_docs_for_prompt(docs)

    async def _arun(self, query: str, **kwargs: Any) -> str:
        return self._run(query, **kwargs)


def build_rag_tool(ltm_store: LTMStore) -> RAGTool:
    return RAGTool(ltm_store=ltm_store)
