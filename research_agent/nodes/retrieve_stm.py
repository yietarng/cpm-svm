from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from research_agent.memory.stm import STMStore
from research_agent.state import AgentState


def retrieve_stm(state: AgentState, config: RunnableConfig | None = None) -> dict:
    """Make the restored STM visible as intermediate_notes for downstream nodes."""
    stm_context = STMStore.format_for_prompt(state)
    note = f"[STM] {stm_context}" if stm_context != "(empty)" else "[STM] (no prior session context)"
    return {"intermediate_notes": [note]}
