from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from research_agent.state import AgentState


def update_stm(state: AgentState, config: RunnableConfig | None = None) -> dict:
    result = state.get("research_result", "")
    if not result:
        return {}

    query = state.get("user_query", "")
    new_note = f"Q: {query[:80]} → {result[:200]}"

    return {
        "stm_notes": [new_note],  # _capped_add reducer enforces the 20-note ceiling
        "intermediate_notes": ["[STM] Updated with new research note."],
    }
