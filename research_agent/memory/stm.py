from __future__ import annotations

from typing import Any

from research_agent.state import AgentState


class STMStore:
    """Thin façade over LangGraph's checkpointer-backed state.

    STM is the LangGraph state itself — this class provides convenience
    helpers for reading/writing notes without coupling node logic to the
    raw state dict structure.
    """

    @staticmethod
    def get_notes(state: AgentState) -> list[str]:
        return list(state.get("stm_notes", []))

    @staticmethod
    def get_plan(state: AgentState) -> str:
        return state.get("current_plan", "")

    @staticmethod
    def format_for_prompt(state: AgentState, max_notes: int = 10) -> str:
        notes = STMStore.get_notes(state)[-max_notes:]
        plan = STMStore.get_plan(state)
        parts: list[str] = []
        if plan:
            parts.append(f"Current plan: {plan}")
        if notes:
            parts.append("Recent session notes:\n" + "\n".join(f"- {n}" for n in notes))
        return "\n".join(parts) if parts else "(empty)"

    @staticmethod
    def make_note_update(new_notes: list[str], max_total: int = 20) -> dict[str, Any]:
        """Return a partial state dict that appends new_notes to stm_notes."""
        return {"stm_notes": new_notes[-max_total:]}
