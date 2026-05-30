import pytest

import research_agent.utils.ltm_policy as policy_module
from research_agent.utils.ltm_policy import StorageDecision, should_store_in_ltm


@pytest.fixture(autouse=True)
def clear_seen_hashes():
    """Isolate the in-process dedup cache between every test."""
    policy_module._seen_hashes.clear()
    yield
    policy_module._seen_hashes.clear()


def test_too_short_content_rejected():
    ok, decision = should_store_in_ltm("short", source="research_agent")
    assert not ok
    assert decision == StorageDecision.SKIP_TOO_SHORT


def test_temporary_source_rejected():
    content = "This is a survey of recent findings in KV cache reuse techniques across multiple papers."
    ok, decision = should_store_in_ltm(content, source="web_search_raw")
    assert not ok
    assert decision == StorageDecision.SKIP_TEMPORARY


def test_storable_signal_accepted():
    content = (
        "User frequently studies KV cache reuse. "
        "This survey covers CacheBlend, EPIC, KVFlow, and Continuum. "
        "Key finding: position-independent caching improves hit rates by 40%."
    )
    ok, decision = should_store_in_ltm(content, source="research_agent")
    assert ok
    assert decision == StorageDecision.STORE


def test_duplicate_rejected():
    content = (
        "User prefers concise summaries with citations. "
        "This is a stable research interest that should be stored for future sessions."
    )
    should_store_in_ltm(content, source="research_agent")
    ok, decision = should_store_in_ltm(content, source="research_agent")
    assert not ok
    assert decision == StorageDecision.SKIP_DUPLICATE


def test_force_store_overrides_no_signal():
    content = "x " * 50
    ok, decision = should_store_in_ltm(content, source="research_agent", metadata={"force_store": True})
    assert ok
    assert decision == StorageDecision.STORE
