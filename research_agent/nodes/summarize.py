from __future__ import annotations

import functools
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from research_agent.prompts import SUMMARIZER_PROMPT
from research_agent.state import AgentState


def summarize(state: AgentState, config: RunnableConfig | None = None, *, llm: Any) -> dict:
    stm_notes = "\n".join(f"- {n}" for n in state.get("stm_notes", [])[-10:]) or "(none)"
    ltm_context = "\n".join(
        entry.get("content", "") for entry in state.get("ltm_context", [])
    ) or "(none)"
    retrieved = "\n\n".join(state.get("retrieved_docs", [])) or "(none)"

    prompt = SUMMARIZER_PROMPT.format(
        stm_notes=stm_notes,
        ltm_context=ltm_context,
        user_query=state.get("user_query", ""),
        retrieved_docs=retrieved,
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    summary = response.content if hasattr(response, "content") else str(response)

    return {
        "research_result": summary,
        "intermediate_notes": ["[Summarize] Summary generated."],
    }


def build_summarize(llm: Any):
    return functools.partial(summarize, llm=llm)
