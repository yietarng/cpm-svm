from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any


class StorageDecision(Enum):
    STORE = "store"
    SKIP_TEMPORARY = "skip_temporary"
    SKIP_DUPLICATE = "skip_duplicate"
    SKIP_TOO_SHORT = "skip_too_short"


# Sources that produce temporary outputs — never stored in LTM
_TEMPORARY_SOURCES = {
    "web_search_raw",
    "intermediate_reasoning",
    "tool_output_raw",
    "stm_note",
}

# Keywords that signal storable, reusable content
_STORABLE_SIGNALS = [
    "survey",
    "report",
    "interest",
    "prefer",
    "reference",
    "benchmark",
    "summary",
    "finding",
    "conclusion",
    "project",
    "always",
    "frequently",
]

# Minimum character length for stored content
_MIN_CHARS = 80

# In-process dedup cache (content hash → True)
# For production, this check should run against the vector store metadata.
_seen_hashes: set[str] = set()


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def should_store_in_ltm(
    content: str,
    source: str = "unknown",
    metadata: dict[str, Any] | None = None,
) -> tuple[bool, StorageDecision]:
    """Return (should_store, decision_reason)."""
    metadata = metadata or {}

    if len(content.strip()) < _MIN_CHARS:
        return False, StorageDecision.SKIP_TOO_SHORT

    if source in _TEMPORARY_SOURCES:
        return False, StorageDecision.SKIP_TEMPORARY

    h = _content_hash(content)
    if h in _seen_hashes:
        return False, StorageDecision.SKIP_DUPLICATE

    lower = content.lower()
    has_signal = any(kw in lower for kw in _STORABLE_SIGNALS)
    is_explicit = metadata.get("force_store", False)

    if not has_signal and not is_explicit:
        return False, StorageDecision.SKIP_TEMPORARY

    _seen_hashes.add(h)
    return True, StorageDecision.STORE
