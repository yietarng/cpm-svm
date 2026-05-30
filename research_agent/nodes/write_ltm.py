from __future__ import annotations

import functools
from typing import Any

from langchain_core.runnables import RunnableConfig

from research_agent.memory.ltm import LTMStore
from research_agent.state import AgentState
from research_agent.utils.ltm_policy import should_store_in_ltm


def write_ltm(
    state: AgentState,
    config: RunnableConfig | None = None,
    *,
    ltm_store: LTMStore,
) -> dict:
    result = state.get("research_result", "")
    if not result:
        return {}

    query = state.get("user_query", "")
    should_store, decision = should_store_in_ltm(result, source="research_agent")

    if not should_store:
        return {"intermediate_notes": [f"[LTM] Skipped write ({decision.value})."]}

    doc_id = ltm_store.store(
        result,
        metadata={"query": query[:200], "source": "research_agent", "tags": ["survey"]},
    )
    return {"intermediate_notes": [f"[LTM] Stored research result (id={doc_id})."]}


def build_write_ltm(ltm_store: LTMStore):
    return functools.partial(write_ltm, ltm_store=ltm_store)
