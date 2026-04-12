"""
Independent KV-cache manager for NOC (New OpenClaw).

Core idea (Feature III from the spec)
--------------------------------------
Traditional prefix caching ties KV cache reuse to token *position*: if the
first token of a new request mismatches the first token of a cached request,
the entire cache is thrown away.

NOC instead keeps a **per-subcontext KV cache** that is *position-independent*.
When the same subcontext S appears in both request R1 and R2 (possibly at
different positions), S's KV cache entries computed during R1 can be reused
for R2 without rebuilding them.

This module implements:

* ``KVCacheEntry``   – cached data + metadata for one subcontext.
* ``KVCacheManager`` – storage, retrieval, and composition logic.
* ``CacheCompositionPlan`` – describes how to assemble independent cache
  fragments for a new inference request.

Integration note
----------------
Actual KV tensors live on the inference server (e.g. SGLang / vLLM).  This
module stores *handles* (opaque cache keys) that are passed to the server.
The server is responsible for the tensor memory; we are responsible for the
bookkeeping that tells it *which* cache entries to inject into the attention
layers for each new request.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .subcontext import SubContext, SubContextType


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class KVCacheEntry:
    """
    Metadata + opaque handle for the KV cache of one SubContext.

    ``cache_handle`` is whatever the inference backend uses to identify
    the stored KV tensors (e.g. a UUID string for SGLang, a pointer for a
    local llama.cpp instance, etc.).
    """

    subcontext_id: str
    content_hash: str          # SHA-256 of the subcontext content at cache time
    cache_handle: Any          # Opaque; backend-specific

    # Position-independence flag.  When True, the backend can inject these
    # KV tensors regardless of where the subcontext appears in the window.
    position_independent: bool = True

    # Token span in the original subcontext
    token_count: int = 0

    # Usage statistics
    hit_count: int = 0
    miss_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_hit_at: float = field(default_factory=time.time)

    # Eviction metadata
    size_bytes: int = 0        # Approximate memory footprint of KV tensors
    pinned: bool = False       # Pinned entries are never evicted

    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    def record_hit(self) -> None:
        self.hit_count += 1
        self.last_hit_at = time.time()

    def record_miss(self) -> None:
        self.miss_count += 1


@dataclass
class CacheCompositionPlan:
    """
    Instructions for the inference backend on how to assemble the KV cache
    for a new request from independent subcontext caches.
    """

    request_id: str
    # Ordered list of (subcontext_id, cache_handle) pairs to inject
    cache_segments: List[Tuple[str, Any]] = field(default_factory=list)
    # Subcontexts whose KV cache must be computed from scratch
    cold_subcontext_ids: List[str] = field(default_factory=list)
    # Total tokens covered by cache hits
    cached_tokens: int = 0
    # Total tokens that need fresh prefill
    uncached_tokens: int = 0

    @property
    def cache_hit_rate(self) -> float:
        total = self.cached_tokens + self.uncached_tokens
        return self.cached_tokens / total if total > 0 else 0.0

    @property
    def prefill_savings_pct(self) -> float:
        return self.cache_hit_rate * 100.0


# ---------------------------------------------------------------------------
# Cache manager
# ---------------------------------------------------------------------------

class KVCacheManager:
    """
    Manages position-independent KV caches for SubContexts.

    Thread-safe: uses a reentrant lock for all mutations.

    Parameters
    ----------
    max_entries:      Maximum number of KVCacheEntry objects to keep.
    max_bytes:        Optional cap on total ``size_bytes`` across all entries.
    ttl_seconds:      Entries not hit within this period are eligible for eviction.
    eviction_policy:  ``"lru"`` (least-recently used) or ``"lfu"`` (least frequently used).
    """

    def __init__(
        self,
        max_entries: int = 1024,
        max_bytes: Optional[int] = None,
        ttl_seconds: float = 3600.0,
        eviction_policy: str = "lru",
    ) -> None:
        self._entries: Dict[str, KVCacheEntry] = {}   # subcontext_id → entry
        self._hash_index: Dict[str, str] = {}         # content_hash  → subcontext_id
        self._lock = threading.RLock()
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.ttl_seconds = ttl_seconds
        self.eviction_policy = eviction_policy

        # Global statistics
        self.total_hits = 0
        self.total_misses = 0
        self.total_evictions = 0

    # ------------------------------------------------------------------
    # Store / update
    # ------------------------------------------------------------------

    def store(
        self,
        sc: SubContext,
        cache_handle: Any,
        size_bytes: int = 0,
        pinned: bool = False,
    ) -> KVCacheEntry:
        """
        Register a newly computed KV cache for *sc*.

        If an entry for the same ``content_hash`` already exists (identical
        content, different subcontext id), the existing handle is reused to
        avoid duplicate storage on the inference server.
        """
        with self._lock:
            # Check if we already have a cache for this content
            if sc.content_hash in self._hash_index:
                existing_id = self._hash_index[sc.content_hash]
                if existing_id in self._entries:
                    existing = self._entries[existing_id]
                    # Alias this subcontext to the existing cache entry
                    new_entry = KVCacheEntry(
                        subcontext_id=sc.id,
                        content_hash=sc.content_hash,
                        cache_handle=existing.cache_handle,
                        position_independent=sc.position_independent,
                        token_count=sc.token_count,
                        size_bytes=existing.size_bytes,
                        pinned=pinned,
                    )
                    self._entries[sc.id] = new_entry
                    return new_entry

            entry = KVCacheEntry(
                subcontext_id=sc.id,
                content_hash=sc.content_hash,
                cache_handle=cache_handle,
                position_independent=sc.position_independent,
                token_count=sc.token_count,
                size_bytes=size_bytes,
                pinned=pinned,
            )
            self._entries[sc.id] = entry
            self._hash_index[sc.content_hash] = sc.id
            self._maybe_evict()
            return entry

    def update(
        self,
        sc: SubContext,
        new_cache_handle: Any,
        new_size_bytes: int = 0,
    ) -> Optional[KVCacheEntry]:
        """
        Replace the cache handle for *sc* after its content has changed.
        Returns None if the subcontext was not previously cached.
        """
        with self._lock:
            if sc.id not in self._entries:
                return None
            old_entry = self._entries[sc.id]
            # Remove old hash index
            if old_entry.content_hash in self._hash_index:
                del self._hash_index[old_entry.content_hash]

            old_entry.content_hash = sc.content_hash
            old_entry.cache_handle = new_cache_handle
            old_entry.size_bytes = new_size_bytes
            old_entry.created_at = time.time()
            old_entry.hit_count = 0
            self._hash_index[sc.content_hash] = sc.id
            return old_entry

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def get(self, subcontext_id: str) -> Optional[KVCacheEntry]:
        """Return the cache entry for a subcontext, or None on miss."""
        with self._lock:
            entry = self._entries.get(subcontext_id)
            if entry is None:
                self.total_misses += 1
                return None
            # TTL check
            if entry.age_seconds > self.ttl_seconds and not entry.pinned:
                self._evict_entry(subcontext_id)
                self.total_misses += 1
                return None
            entry.record_hit()
            self.total_hits += 1
            return entry

    def get_by_hash(self, content_hash: str) -> Optional[KVCacheEntry]:
        """
        Look up a cache entry by content hash.

        This enables *cross-subcontext* reuse: if two different subcontext
        objects have identical content (same hash), the second one can reuse
        the first one's KV cache without rebuilding it.
        """
        with self._lock:
            sc_id = self._hash_index.get(content_hash)
            if sc_id is None:
                return None
            return self.get(sc_id)

    def has(self, subcontext_id: str) -> bool:
        return subcontext_id in self._entries

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def plan_composition(
        self,
        subcontexts: List[SubContext],
        request_id: str,
    ) -> CacheCompositionPlan:
        """
        Given an ordered list of subcontexts for an inference request, produce
        a ``CacheCompositionPlan`` that tells the backend which KV segments to
        reuse and which must be computed fresh.

        The plan preserves the order of *subcontexts* so the backend can
        assemble them in the correct sequence.
        """
        plan = CacheCompositionPlan(request_id=request_id)

        with self._lock:
            for sc in subcontexts:
                # 1) Direct hit: cache exists for this subcontext id
                entry = self._entries.get(sc.id)
                if entry is not None and sc.is_cache_valid(entry.content_hash):
                    if entry.age_seconds <= self.ttl_seconds or entry.pinned:
                        plan.cache_segments.append((sc.id, entry.cache_handle))
                        plan.cached_tokens += sc.token_count
                        entry.record_hit()
                        self.total_hits += 1
                        continue

                # 2) Hash hit: same content under a different subcontext id
                if sc.content_hash:
                    hash_entry = self._entries.get(
                        self._hash_index.get(sc.content_hash, "")
                    )
                    if hash_entry is not None and hash_entry.position_independent:
                        plan.cache_segments.append((sc.id, hash_entry.cache_handle))
                        plan.cached_tokens += sc.token_count
                        hash_entry.record_hit()
                        self.total_hits += 1
                        continue

                # 3) Cache miss
                plan.cold_subcontext_ids.append(sc.id)
                plan.uncached_tokens += sc.token_count
                self.total_misses += 1

        return plan

    # ------------------------------------------------------------------
    # Invalidation
    # ------------------------------------------------------------------

    def invalidate(self, subcontext_id: str) -> bool:
        """Remove the cache entry for a subcontext.  Returns True if removed."""
        with self._lock:
            if subcontext_id not in self._entries:
                return False
            self._evict_entry(subcontext_id)
            return True

    def invalidate_stale(self, subcontexts: List[SubContext]) -> int:
        """
        For each subcontext whose content has changed (hash mismatch),
        invalidate its cache entry.  Returns the number of entries invalidated.
        """
        count = 0
        with self._lock:
            for sc in subcontexts:
                entry = self._entries.get(sc.id)
                if entry is not None and not sc.is_cache_valid(entry.content_hash):
                    self._evict_entry(sc.id)
                    count += 1
        return count

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._hash_index.clear()

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def total_bytes(self) -> int:
        return sum(e.size_bytes for e in self._entries.values())

    @property
    def global_hit_rate(self) -> float:
        total = self.total_hits + self.total_misses
        return self.total_hits / total if total > 0 else 0.0

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "entries": self.size,
                "total_bytes": self.total_bytes,
                "total_hits": self.total_hits,
                "total_misses": self.total_misses,
                "total_evictions": self.total_evictions,
                "global_hit_rate": self.global_hit_rate,
                "pinned_entries": sum(1 for e in self._entries.values() if e.pinned),
            }

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def _evict_entry(self, subcontext_id: str) -> None:
        """Remove one entry (must be called under self._lock)."""
        entry = self._entries.pop(subcontext_id, None)
        if entry is None:
            return
        self._hash_index.pop(entry.content_hash, None)
        self.total_evictions += 1

    def _maybe_evict(self) -> None:
        """Evict entries if we have exceeded resource limits (under self._lock)."""
        # TTL sweep
        now = time.time()
        stale = [
            sc_id
            for sc_id, e in self._entries.items()
            if not e.pinned and (now - e.last_hit_at) > self.ttl_seconds
        ]
        for sc_id in stale:
            self._evict_entry(sc_id)

        # Count limit
        while len(self._entries) > self.max_entries:
            victim = self._pick_eviction_victim()
            if victim:
                self._evict_entry(victim)
            else:
                break

        # Byte limit
        if self.max_bytes is not None:
            while self.total_bytes > self.max_bytes:
                victim = self._pick_eviction_victim()
                if victim:
                    self._evict_entry(victim)
                else:
                    break

    def _pick_eviction_victim(self) -> Optional[str]:
        """Return the subcontext_id of the entry to evict (under self._lock)."""
        candidates = [
            (sc_id, e)
            for sc_id, e in self._entries.items()
            if not e.pinned
        ]
        if not candidates:
            return None

        if self.eviction_policy == "lru":
            # Evict the entry least recently used
            return min(candidates, key=lambda x: x[1].last_hit_at)[0]
        elif self.eviction_policy == "lfu":
            # Evict the entry with the fewest hits (break ties by age)
            return min(candidates, key=lambda x: (x[1].hit_count, -x[1].age_seconds))[0]
        else:
            # Default: oldest entry
            return min(candidates, key=lambda x: x[1].created_at)[0]
