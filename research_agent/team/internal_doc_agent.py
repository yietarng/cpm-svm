from __future__ import annotations

import functools
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from research_agent.config import ResearchAgentConfig
from research_agent.memory.ltm import LTMStore
from research_agent.prompts import INTERNAL_DOC_AGENT_PROMPT
from research_agent.state import AgentState
from research_agent.tools.ltm_tools import build_ltm_tools
from research_agent.tools.rag_tool import build_rag_tool


def internal_doc_search(state: AgentState, *, llm: Any, ltm_store: LTMStore) -> dict:
    retrieve_ltm, store_ltm = build_ltm_tools(ltm_store)
    # RAG tool pointed at the same store; in production pass a separate
    # LTMStore configured with collection_name="internal_docs"
    rag_tool = build_rag_tool(ltm_store)
    tools = [rag_tool, retrieve_ltm, store_ltm]

    agent = create_react_agent(llm, tools)
    messages = [
        {"role": "system", "content": INTERNAL_DOC_AGENT_PROMPT},
        {"role": "user", "content": state["user_query"]},
    ]
    result = agent.invoke({"messages": messages})
    final = result["messages"][-1].content

    return {
        "retrieved_docs": [final],
        "intermediate_notes": ["[InternalDocAgent] Completed internal document search."],
    }


def build_internal_doc_agent_graph(config: ResearchAgentConfig, llm: Any, ltm_store: LTMStore):
    graph = StateGraph(AgentState)
    graph.add_node("internal_search", functools.partial(internal_doc_search, llm=llm, ltm_store=ltm_store))
    graph.add_edge(START, "internal_search")
    graph.add_edge("internal_search", END)
    return graph.compile()
