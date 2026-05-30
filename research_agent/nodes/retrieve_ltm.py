from __future__ import annotations

import functools
from typing import Any

from langchain_core.runnables import RunnableConfig

from research_agent.memory.ltm import LTMStore
from research_agent.state import AgentState


def retrieve_ltm(state: AgentState, config: RunnableConfig | None = None, *, ltm_store: LTMStore) -> dict:
    query = state.get("user_query", "")
    if not query:
        return {"ltm_context": [], "intermediate_notes": ["[LTM] No query — skipped LTM retrieval."]}

    docs = ltm_store.retrieve(query)
    formatted = ltm_store.format_docs_for_prompt(docs)
    note = f"[LTM] Retrieved {len(docs)} prior knowledge entries."
    ltm_dicts = [{"content": d.page_content, "metadata": d.metadata} for d in docs]
    return {"ltm_context": ltm_dicts, "intermediate_notes": [note, formatted]}


def build_retrieve_ltm(ltm_store: LTMStore):
    return functools.partial(retrieve_ltm, ltm_store=ltm_store)
