import pytest

from research_agent.nodes.retrieve_stm import retrieve_stm
from research_agent.nodes.return_to_supervisor import return_to_supervisor
from research_agent.nodes.update_stm import update_stm
from research_agent.config import ResearchAgentConfig
from research_agent.state import default_state


def test_retrieve_stm_empty_state():
    state = default_state()
    result = retrieve_stm(state)
    assert "intermediate_notes" in result
    assert len(result["intermediate_notes"]) == 1
    assert "[STM]" in result["intermediate_notes"][0]


def test_retrieve_stm_with_notes():
    state = {**default_state(), "stm_notes": ["Note A", "Note B"], "current_plan": "Survey KV cache"}
    result = retrieve_stm(state)
    note = result["intermediate_notes"][0]
    assert "Note A" in note or "Survey KV cache" in note


def test_return_to_supervisor_formats_citations():
    state = {
        **default_state(),
        "research_result": "KV cache reuse improves throughput.",
        "citations": ["https://arxiv.org/1234", "https://github.com/example"],
    }
    result = return_to_supervisor(state)
    assert "https://arxiv.org/1234" in result["research_result"]
    assert "Sources:" in result["research_result"]


def test_update_stm_appends_note():
    config = ResearchAgentConfig(stm_max_notes=5)
    state = {
        **default_state(),
        "user_query": "Survey KV cache papers",
        "research_result": "CacheBlend uses selective recomputation.",
        "stm_notes": [],
    }
    result = update_stm(state, agent_config=config)
    assert "stm_notes" in result
    assert len(result["stm_notes"]) == 1
    assert "Survey KV cache papers" in result["stm_notes"][0]


def test_update_stm_no_result_no_op():
    config = ResearchAgentConfig()
    state = {**default_state(), "research_result": ""}
    result = update_stm(state, agent_config=config)
    assert result == {}
