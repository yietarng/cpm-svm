"""
SubContext data model for NOC (New OpenClaw).

A SubContext is a semantically independent unit of context that can be
maintained independently, cached independently, and reused across multiple
LLM inference requests without coupling to other subcontexts.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class SubContextType(Enum):
    """Classification of what a subcontext represents."""
    SYSTEM = auto()       # System prompt / instructions
    USER = auto()         # Direct user message
    PLAN = auto()         # Planner-generated execution plan
    TOOL_RESULT = auto()  # Output from a single tool execution
    DOCUMENT = auto()     # A retrieved document fragment (e.g. RAG)
    WEB_SEARCH = auto()   # Web-search result for a single item
    CONVERSATION = auto() # General conversation history
    SUMMARY = auto()      # A compressed summary of other subcontexts
    PRODUCT_INFO = auto() # Product / entity information block


class SubContextStatus(Enum):
    """Lifecycle state of a subcontext."""
    ACTIVE = auto()     # In use, being sent with requests
    STALE = auto()      # Content has changed; KV cache must be invalidated
    PRUNED = auto()     # Removed from active use (but kept for reference)
    SUMMARIZED = auto() # Replaced by a SUMMARY subcontext


@dataclass
class SubContext:
    """
    A semantically independent unit of session context.

    Design principles
    -----------------
    * The KV cache entries built from this subcontext are *position-independent*
      with respect to other subcontexts, so the same KV cache can be reused
      even when the subcontext appears at different positions in different requests.
    * Each subcontext carries its own embedding vector so it can be routed to the
      right agent or ranked for relevance without an extra LLM call.
    * The ``content_hash`` lets the KV cache manager detect staleness quickly.
    """

    # ---- Identity ----
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: SubContextType = SubContextType.CONVERSATION
    status: SubContextStatus = SubContextStatus.ACTIVE

    # ---- Content ----
    content: str = ""
    token_count: int = 0          # Estimated number of tokens in ``content``
    content_hash: str = ""        # SHA-256 of content (used for cache invalidation)

    # ---- Semantic metadata ----
    embedding: Optional[List[float]] = None   # Dense semantic vector
    tags: List[str] = field(default_factory=list)
    source: str = ""              # E.g. tool name, document URL, agent id
    summary: Optional[str] = None # Optional human-readable one-liner

    # ---- KV cache metadata ----
    kv_cache_key: str = ""        # Key used in KVCacheManager
    position_independent: bool = True  # True ↔ cache is not tied to position in window

    # ---- Provenance ----
    parent_ids: List[str] = field(default_factory=list)  # Subcontexts this was derived from
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---- Timestamps ----
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    last_modified: float = field(default_factory=time.time)

    # ---- Compression ----
    original_token_count: int = 0  # Before any compression
    compression_ratio: float = 1.0  # original / current token count

    def __post_init__(self) -> None:
        if self.content and not self.content_hash:
            self.content_hash = self._hash(self.content)
        if self.content and not self.kv_cache_key:
            self.kv_cache_key = f"{self.type.name}:{self.content_hash[:16]}"
        if self.content and self.original_token_count == 0:
            self.original_token_count = self.token_count

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def update_content(self, new_content: str, new_token_count: int = 0) -> None:
        """Replace content and mark KV cache as stale."""
        self.content = new_content
        self.content_hash = self._hash(new_content)
        self.kv_cache_key = f"{self.type.name}:{self.content_hash[:16]}"
        self.token_count = new_token_count
        self.last_modified = time.time()
        self.status = SubContextStatus.STALE
        if self.original_token_count > 0 and new_token_count > 0:
            self.compression_ratio = self.original_token_count / new_token_count

    def touch(self) -> None:
        """Record that this subcontext was used in a request."""
        self.last_accessed = time.time()

    def is_cache_valid(self, stored_hash: str) -> bool:
        """Return True if the stored cache hash still matches current content."""
        return stored_hash == self.content_hash

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.name,
            "status": self.status.name,
            "content": self.content,
            "token_count": self.token_count,
            "content_hash": self.content_hash,
            "tags": self.tags,
            "source": self.source,
            "summary": self.summary,
            "kv_cache_key": self.kv_cache_key,
            "position_independent": self.position_independent,
            "parent_ids": self.parent_ids,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "last_modified": self.last_modified,
            "original_token_count": self.original_token_count,
            "compression_ratio": self.compression_ratio,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubContext":
        data = dict(data)
        data["type"] = SubContextType[data["type"]]
        data["status"] = SubContextStatus[data["status"]]
        return cls(**data)

    def __repr__(self) -> str:
        return (
            f"SubContext(id={self.id[:8]}, type={self.type.name}, "
            f"tokens={self.token_count}, status={self.status.name})"
        )


@dataclass
class SubContextWindow:
    """
    A composed context window built from multiple SubContexts.

    This is what gets sent to an LLM for a single inference request.
    It records which subcontexts were included and their order so the
    KVCacheManager can figure out which cache entries may be reused.
    """

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    subcontexts: List[SubContext] = field(default_factory=list)
    total_tokens: int = 0
    max_tokens: int = 8192

    # Cache reuse bookkeeping
    cache_hits: int = 0
    cache_misses: int = 0

    def add(self, sc: SubContext) -> bool:
        """
        Append a subcontext if it fits within the token budget.
        Returns True if the subcontext was added, False if it would overflow.
        """
        if self.total_tokens + sc.token_count > self.max_tokens:
            return False
        self.subcontexts.append(sc)
        self.total_tokens += sc.token_count
        sc.touch()
        return True

    def to_prompt(self) -> str:
        """Flatten all subcontext content into a single prompt string."""
        parts: List[str] = []
        for sc in self.subcontexts:
            if sc.content:
                parts.append(sc.content)
        return "\n\n".join(parts)

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"SubContextWindow(request={self.request_id[:8]}, "
            f"subcontexts={len(self.subcontexts)}, tokens={self.total_tokens})"
        )
