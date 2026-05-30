from __future__ import annotations

import functools
from typing import Any

from langchain_core.runnables import RunnableConfig

from research_agent.config import ResearchAgentConfig
from research_agent.state import AgentState


def update_stm(
    state: AgentState,
    config: RunnableConfig | None = None,
    *,
    agent_config: ResearchAgentConfig,
) -> dict:
    result = state.get("research_result", "")
    if not result:
        return {}

    query = state.get("user_query", "")
    new_note = f"Q: {query[:80]} → {result[:200]}"
    max_notes = agent_config.stm_max_notes

    existing = list(state.get("stm_notes", []))
    updated = (existing + [new_note])[-max_notes:]

    return {
        "stm_notes": [new_note],  # additive reducer appends this
        "intermediate_notes": [f"[STM] Updated with new research note."],
    }


def build_update_stm(agent_config: ResearchAgentConfig):
    return functools.partial(update_stm, agent_config=agent_config)
