from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from research_agent.config import ResearchAgentConfig, STMBackend
from research_agent.memory.ltm import LTMStore
from research_agent.nodes import (
    build_external_search,
    build_retrieve_ltm,
    build_summarize,
    build_write_ltm,
    retrieve_stm,
    return_to_supervisor,
    update_stm,
)
from research_agent.state import AgentState, default_state
from research_agent.utils.embeddings import get_embeddings_model


def _build_llm(config: ResearchAgentConfig) -> Any:
    return ChatOpenAI(model=config.llm_model, temperature=config.llm_temperature)


def _build_checkpointer(config: ResearchAgentConfig) -> Any:
    if config.stm_backend == STMBackend.SQLITE:
        from langgraph.checkpoint.sqlite import SqliteSaver  # optional dep
        return SqliteSaver.from_conn_string(config.sqlite_path)
    return MemorySaver()


def build_research_graph(
    config: ResearchAgentConfig | None = None,
    checkpointer: Any = None,
):
    """Assemble and compile the single Research Agent graph.

    Returns a CompiledStateGraph. Pass a thread_id at invoke time:
        graph.invoke({"user_query": "..."}, config={"configurable": {"thread_id": "t1"}})
    """
    config = config or ResearchAgentConfig()

    embeddings = get_embeddings_model(config)
    ltm_store = LTMStore(config, embeddings)
    llm = _build_llm(config)

    graph = StateGraph(AgentState)

    graph.add_node("retrieve_stm", retrieve_stm)
    graph.add_node("retrieve_ltm", build_retrieve_ltm(ltm_store))
    graph.add_node("external_search", build_external_search(llm, ltm_store))
    graph.add_node("summarize", build_summarize(llm))
    graph.add_node("update_stm", update_stm)
    graph.add_node("write_ltm", build_write_ltm(ltm_store))
    graph.add_node("return_to_supervisor", return_to_supervisor)

    graph.add_edge(START, "retrieve_stm")
    graph.add_edge("retrieve_stm", "retrieve_ltm")
    graph.add_edge("retrieve_ltm", "external_search")
    graph.add_edge("external_search", "summarize")
    graph.add_edge("summarize", "update_stm")
    graph.add_edge("update_stm", "write_ltm")
    graph.add_edge("write_ltm", "return_to_supervisor")
    graph.add_edge("return_to_supervisor", END)

    cp = checkpointer or _build_checkpointer(config)
    return graph.compile(checkpointer=cp)


def run_research(
    query: str,
    thread_id: str = "default",
    config: ResearchAgentConfig | None = None,
) -> str:
    """Convenience entry point for CLI and tests."""
    compiled = build_research_graph(config)
    result = compiled.invoke(
        {**default_state(), "user_query": query},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result.get("research_result", "")
