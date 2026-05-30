from __future__ import annotations

import operator
from typing import Annotated

from typing_extensions import TypedDict

_STM_MAX = 20


def _capped_add(a: list, b: list) -> list:
    """Additive reducer that keeps only the most recent _STM_MAX notes."""
    combined = a + b
    return combined[-_STM_MAX:]


class AgentState(TypedDict):
    user_query: str
    retrieved_docs: Annotated[list, operator.add]
    citations: Annotated[list, operator.add]
    research_result: str
    stm_notes: Annotated[list, _capped_add]
    ltm_context: Annotated[list, operator.add]
    current_plan: str
    intermediate_notes: Annotated[list, operator.add]


def default_state() -> AgentState:
    return AgentState(
        user_query="",
        retrieved_docs=[],
        citations=[],
        research_result="",
        stm_notes=[],
        ltm_context=[],
        current_plan="",
        intermediate_notes=[],
    )
