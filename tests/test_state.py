import operator

import pytest

from research_agent.state import AgentState, default_state


def test_default_state_keys():
    state = default_state()
    assert set(state.keys()) == {
        "user_query", "retrieved_docs", "citations",
        "research_result", "stm_notes", "ltm_context",
        "current_plan", "intermediate_notes",
    }


def test_default_state_list_fields_empty():
    state = default_state()
    for key in ("retrieved_docs", "citations", "stm_notes", "ltm_context", "intermediate_notes"):
        assert state[key] == []


def test_additive_reducer():
    # Simulate LangGraph state merging for additive fields
    a = [1, 2]
    b = [3, 4]
    assert operator.add(a, b) == [1, 2, 3, 4]
