from __future__ import annotations

import functools
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from research_agent.config import ResearchAgentConfig
from research_agent.memory.ltm import LTMStore
from research_agent.prompts import WEB_AGENT_PROMPT
from research_agent.state import AgentState
from research_agent.tools.ltm_tools import build_ltm_tools
from research_agent.tools.web_search import build_web_search_tool


def web_search_node(state: AgentState, *, llm: Any, ltm_store: LTMStore) -> dict:
    retrieve_ltm, store_ltm = build_ltm_tools(ltm_store)
    web_tool = build_web_search_tool()
    tools = [web_tool, retrieve_ltm, store_ltm]

    agent = create_react_agent(llm, tools)
    messages = [
        {"role": "system", "content": WEB_AGENT_PROMPT},
        {"role": "user", "content": state["user_query"]},
    ]
    result = agent.invoke({"messages": messages})
    final = result["messages"][-1].content

    citations: list[str] = []
    for word in final.split():
        if word.startswith("http"):
            citations.append(word.strip(".,)\"'"))

    return {
        "retrieved_docs": [final],
        "citations": list(dict.fromkeys(citations)),
        "intermediate_notes": ["[WebAgent] Completed web search."],
    }


def build_web_agent_graph(config: ResearchAgentConfig, llm: Any, ltm_store: LTMStore):
    graph = StateGraph(AgentState)
    graph.add_node("web_search", functools.partial(web_search_node, llm=llm, ltm_store=ltm_store))
    graph.add_edge(START, "web_search")
    graph.add_edge("web_search", END)
    return graph.compile()
