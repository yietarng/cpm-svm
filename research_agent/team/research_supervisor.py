from __future__ import annotations

import functools
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from research_agent.config import ResearchAgentConfig
from research_agent.memory.ltm import LTMStore
from research_agent.nodes.summarize import build_summarize
from research_agent.nodes.update_stm import build_update_stm
from research_agent.nodes.write_ltm import build_write_ltm
from research_agent.state import AgentState
from research_agent.team.internal_doc_agent import build_internal_doc_agent_graph
from research_agent.team.literature_agent import build_literature_agent_graph
from research_agent.team.web_agent import build_web_agent_graph
from research_agent.utils.embeddings import get_embeddings_model


def plan_research(state: AgentState, *, llm: Any) -> dict:
    """LLM decides which sub-agents to activate."""
    prompt = (
        f"User query: {state['user_query']}\n\n"
        "Which research sub-agents are needed? Choose one or more from:\n"
        "- literature (academic papers, surveys)\n"
        "- web (recent news, blogs, benchmarks)\n"
        "- internal (organization internal documents)\n\n"
        "Reply with a comma-separated list, e.g. 'literature, web'."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    plan = response.content.lower()
    return {
        "current_plan": plan,
        "intermediate_notes": [f"[ResearchSupervisor] Plan: {plan}"],
    }


def route_research_tasks(state: AgentState) -> list[Send]:
    """Fan-out to sub-agents in parallel using LangGraph Send API."""
    plan = state.get("current_plan", "")
    sends: list[Send] = []
    if "literature" in plan:
        sends.append(Send("literature_agent", state))
    if "web" in plan:
        sends.append(Send("web_agent", state))
    if "internal" in plan:
        sends.append(Send("internal_doc_agent", state))
    if not sends:
        sends.append(Send("web_agent", state))
    return sends


def merge_results(state: AgentState) -> dict:
    docs = state.get("retrieved_docs", [])
    merged_note = f"[Merge] Collected {len(docs)} document(s) from sub-agents."
    return {"intermediate_notes": [merged_note]}


def build_research_team_graph(
    config: ResearchAgentConfig | None = None,
    checkpointer: Any = None,
):
    config = config or ResearchAgentConfig()
    embeddings = get_embeddings_model(config)
    ltm_store = LTMStore(config, embeddings)

    llm = ChatOpenAI(model=config.llm_model, temperature=config.llm_temperature)

    lit_graph = build_literature_agent_graph(config, llm, ltm_store)
    web_graph = build_web_agent_graph(config, llm, ltm_store)
    int_graph = build_internal_doc_agent_graph(config, llm, ltm_store)

    graph = StateGraph(AgentState)

    graph.add_node("plan_research", functools.partial(plan_research, llm=llm))
    graph.add_node("literature_agent", lit_graph.invoke)
    graph.add_node("web_agent", web_graph.invoke)
    graph.add_node("internal_doc_agent", int_graph.invoke)
    graph.add_node("merge_results", merge_results)
    graph.add_node("summarize", build_summarize(llm))
    graph.add_node("update_shared_stm", build_update_stm(config))
    graph.add_node("write_shared_ltm", build_write_ltm(ltm_store))

    graph.add_edge(START, "plan_research")
    graph.add_conditional_edges("plan_research", route_research_tasks)
    graph.add_edge("literature_agent", "merge_results")
    graph.add_edge("web_agent", "merge_results")
    graph.add_edge("internal_doc_agent", "merge_results")
    graph.add_edge("merge_results", "summarize")
    graph.add_edge("summarize", "update_shared_stm")
    graph.add_edge("update_shared_stm", "write_shared_ltm")
    graph.add_edge("write_shared_ltm", END)

    cp = checkpointer or MemorySaver()
    return graph.compile(checkpointer=cp)
