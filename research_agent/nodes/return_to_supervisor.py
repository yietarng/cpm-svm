from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from research_agent.state import AgentState


def return_to_supervisor(state: AgentState, config: RunnableConfig | None = None) -> dict:
    """Format the final result for the supervisor."""
    citations = state.get("citations", [])
    result = state.get("research_result", "(no result)")

    formatted = result
    if citations:
        formatted += "\n\nSources:\n" + "\n".join(f"- {c}" for c in citations)

    return {
        "research_result": formatted,
        "intermediate_notes": ["[Done] Research complete. Result ready for supervisor."],
    }
