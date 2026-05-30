from __future__ import annotations

import functools
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent

from research_agent.memory.ltm import LTMStore
from research_agent.memory.stm import STMStore
from research_agent.prompts import RESEARCH_PROMPT
from research_agent.state import AgentState
from research_agent.tools.rag_tool import build_rag_tool
from research_agent.tools.web_search import build_web_search_tool


def external_search(
    state: AgentState,
    config: RunnableConfig | None = None,
    *,
    llm: Any,
    ltm_store: LTMStore,
) -> dict:
    """Run a ReAct agent loop with web_search and rag_retrieval tools."""
    web_tool = build_web_search_tool()
    rag_tool = build_rag_tool(ltm_store)
    agent = create_react_agent(llm, [web_tool, rag_tool])

    stm_summary = STMStore.format_for_prompt(state)
    ltm_summary = "\n".join(
        entry.get("content", "") for entry in state.get("ltm_context", [])
    )
    system_message = (
        RESEARCH_PROMPT
        + f"\n\nShort-term memory:\n{stm_summary}"
        + f"\n\nLong-term memory:\n{ltm_summary}"
    )

    result = agent.invoke({"messages": [
        {"role": "system", "content": system_message},
        {"role": "user", "content": state["user_query"]},
    ]})

    citations: list[str] = []
    retrieved_docs: list[str] = []
    for msg in result["messages"]:
        content = getattr(msg, "content", "")
        if "http" in content:
            for word in content.split():
                if word.startswith("http"):
                    citations.append(word.strip(".,)\"'"))
        if content and getattr(msg, "type", "") == "tool":
            retrieved_docs.append(content[:500])

    return {
        "retrieved_docs": retrieved_docs,
        "citations": list(dict.fromkeys(citations)),
        "intermediate_notes": ["[Search] Agent completed external search."],
    }


def build_external_search(llm: Any, ltm_store: LTMStore):
    return functools.partial(external_search, llm=llm, ltm_store=ltm_store)
