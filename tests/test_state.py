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
    a = [1, 2]
    b = [3, 4]
    assert operator.add(a, b) == [1, 2, 3, 4]


def test_capped_add_trims_to_max():
    from research_agent.state import _STM_MAX, _capped_add
    big = list(range(_STM_MAX))
    result = _capped_add(big, ["new"])
    assert len(result) == _STM_MAX
    assert result[-1] == "new"


def test_capped_add_under_limit():
    from research_agent.state import _capped_add
    result = _capped_add([1, 2], [3])
    assert result == [1, 2, 3]
