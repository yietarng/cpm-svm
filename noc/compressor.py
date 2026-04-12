"""
Compression algorithms for NOC (New OpenClaw).

Three compression strategies are provided:

1. TRUNCATE  – Hard-cut the content at a token limit.  Fast, lossy.
2. PRUNE     – Remove sentences / lines below a relevance threshold.
               Can operate without an LLM (TF-IDF / embedding similarity).
3. SUMMARIZE – Call a (local or backend) LLM to produce a shorter version.
               Highest quality but requires an LLM call.

A ``ContextCompressor`` orchestrates these strategies and decides which one
to apply based on the subcontext type and the configured policy.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from .subcontext import SubContext, SubContextStatus, SubContextType


# ---------------------------------------------------------------------------
# Enumerations and configuration
# ---------------------------------------------------------------------------

class CompressionAlgorithm(Enum):
    NONE = auto()       # Pass through unchanged
    TRUNCATE = auto()   # Hard truncation to token budget
    PRUNE = auto()      # Relevance-based sentence removal
    SUMMARIZE = auto()  # LLM-assisted abstractive summary
    HYBRID = auto()     # Prune first, summarize what remains if still too large


@dataclass
class CompressionConfig:
    """Per-subcontext-type compression policy."""

    algorithm: CompressionAlgorithm = CompressionAlgorithm.HYBRID

    # Token budgets
    target_token_ratio: float = 0.5    # Target size as fraction of original
    max_absolute_tokens: int = 2048    # Hard ceiling after compression
    min_absolute_tokens: int = 64      # Do not compress below this

    # PRUNE settings
    prune_min_sentence_score: float = 0.15  # Drop sentences below this TF-IDF score
    prune_query: str = ""                   # Optional query to bias pruning

    # SUMMARIZE settings
    summarize_prompt_template: str = (
        "Summarize the following content concisely, preserving all key facts "
        "and action items. Output only the summary.\n\n{content}"
    )
    summarize_max_output_tokens: int = 512

    # Fallback
    fallback_algorithm: CompressionAlgorithm = CompressionAlgorithm.TRUNCATE


# Default policies per subcontext type
DEFAULT_POLICIES: Dict[SubContextType, CompressionConfig] = {
    SubContextType.SYSTEM:       CompressionConfig(algorithm=CompressionAlgorithm.NONE),
    SubContextType.USER:         CompressionConfig(algorithm=CompressionAlgorithm.NONE),
    SubContextType.PLAN:         CompressionConfig(
        algorithm=CompressionAlgorithm.PRUNE,
        target_token_ratio=0.7,
    ),
    SubContextType.TOOL_RESULT:  CompressionConfig(
        algorithm=CompressionAlgorithm.HYBRID,
        target_token_ratio=0.4,
        prune_min_sentence_score=0.1,
    ),
    SubContextType.DOCUMENT:     CompressionConfig(
        algorithm=CompressionAlgorithm.HYBRID,
        target_token_ratio=0.35,
    ),
    SubContextType.WEB_SEARCH:   CompressionConfig(
        algorithm=CompressionAlgorithm.PRUNE,
        target_token_ratio=0.45,
    ),
    SubContextType.CONVERSATION: CompressionConfig(
        algorithm=CompressionAlgorithm.SUMMARIZE,
        target_token_ratio=0.3,
    ),
    SubContextType.SUMMARY:      CompressionConfig(algorithm=CompressionAlgorithm.NONE),
    SubContextType.PRODUCT_INFO: CompressionConfig(
        algorithm=CompressionAlgorithm.PRUNE,
        target_token_ratio=0.5,
    ),
}


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in ``text`` without a tokenizer.

    Uses the rule-of-thumb: 1 token ≈ 4 characters for English prose.
    Replace this with a real tokenizer (e.g. tiktoken) if available.
    """
    return max(1, len(text) // 4)


def _split_sentences(text: str) -> List[str]:
    """Split text into a list of sentences / logical units."""
    # Split on sentence-ending punctuation or newlines
    raw = re.split(r"(?<=[.!?])\s+|\n{2,}", text.strip())
    return [s.strip() for s in raw if s.strip()]


# ---------------------------------------------------------------------------
# Scoring helpers (lightweight, no LLM required)
# ---------------------------------------------------------------------------

def _build_tf_idf_scores(sentences: List[str]) -> List[float]:
    """
    Return a per-sentence relevance score using a simple TF-IDF variant.

    Words that appear in many sentences are down-weighted; words that appear
    in few sentences but prominently in one are up-weighted.
    """
    import collections
    vocab: Dict[str, int] = collections.Counter()
    tokenised = []
    for s in sentences:
        words = re.findall(r"\w+", s.lower())
        tokenised.append(words)
        vocab.update(set(words))  # document frequency

    n = len(sentences)
    scores: List[float] = []
    for words in tokenised:
        if not words:
            scores.append(0.0)
            continue
        tf_idf = 0.0
        wf: Dict[str, int] = collections.Counter(words)
        for w, cnt in wf.items():
            tf = cnt / len(words)
            idf = math.log((n + 1) / (vocab[w] + 1)) + 1.0
            tf_idf += tf * idf
        scores.append(tf_idf / len(wf))

    # Normalise to [0, 1]
    max_s = max(scores) if scores else 1.0
    if max_s == 0:
        return [0.0] * len(scores)
    return [s / max_s for s in scores]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two dense vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Individual algorithm implementations
# ---------------------------------------------------------------------------

def truncate(
    content: str,
    max_tokens: int,
    from_end: bool = False,
) -> Tuple[str, int]:
    """
    Hard-truncate *content* to at most *max_tokens*.

    Parameters
    ----------
    content:    The text to truncate.
    max_tokens: Maximum allowed tokens in the output.
    from_end:   If True keep the *last* max_tokens worth of content (useful
                for conversation history where the latest turns are most relevant).

    Returns
    -------
    (truncated_text, estimated_token_count)
    """
    char_limit = max_tokens * 4  # 1 token ≈ 4 chars
    if len(content) <= char_limit:
        return content, estimate_tokens(content)

    if from_end:
        truncated = "…" + content[-char_limit:]
    else:
        truncated = content[:char_limit] + "…"

    return truncated, estimate_tokens(truncated)


def prune(
    content: str,
    config: CompressionConfig,
    query_embedding: Optional[List[float]] = None,
) -> Tuple[str, int]:
    """
    Remove low-relevance sentences from *content*.

    If *query_embedding* is provided, sentences are also scored by their
    cosine similarity to the query embedding.

    Returns
    -------
    (pruned_text, estimated_token_count)
    """
    sentences = _split_sentences(content)
    if len(sentences) <= 2:
        # Too short to prune meaningfully
        return content, estimate_tokens(content)

    tf_idf_scores = _build_tf_idf_scores(sentences)

    # Optionally blend with query similarity (requires embeddings per sentence)
    final_scores = tf_idf_scores[:]  # shallow copy

    # Filter out sentences below the minimum score threshold
    threshold = config.prune_min_sentence_score
    kept: List[str] = []
    for sentence, score in zip(sentences, final_scores):
        if score >= threshold:
            kept.append(sentence)

    # Ensure we keep at least 20% of sentences
    min_keep = max(1, len(sentences) // 5)
    if len(kept) < min_keep:
        # Fall back to top-scored sentences
        ranked = sorted(
            zip(final_scores, sentences), key=lambda x: x[0], reverse=True
        )
        kept = [s for _, s in ranked[:max(min_keep, len(sentences) // 2)]]

    pruned = " ".join(kept)

    # If still too large, truncate the remainder
    target_tokens = int(estimate_tokens(content) * config.target_token_ratio)
    if estimate_tokens(pruned) > max(target_tokens, config.min_absolute_tokens):
        pruned, _ = truncate(pruned, target_tokens)

    return pruned, estimate_tokens(pruned)


def summarize(
    content: str,
    config: CompressionConfig,
    llm_callable: Optional[Callable[[str], str]] = None,
) -> Tuple[str, int]:
    """
    Produce an abstractive summary of *content*.

    Parameters
    ----------
    content:      The text to summarise.
    config:       Compression configuration (template, max output tokens, …).
    llm_callable: A function (prompt: str) -> str.  If None, falls back to
                  extractive truncation (best-effort without an LLM).

    Returns
    -------
    (summary_text, estimated_token_count)
    """
    if llm_callable is None:
        # No LLM available – fall back to truncation
        return truncate(content, config.max_absolute_tokens)

    prompt = config.summarize_prompt_template.format(content=content)
    try:
        summary_text = llm_callable(prompt)
        # Enforce hard output limit
        if estimate_tokens(summary_text) > config.summarize_max_output_tokens:
            summary_text, _ = truncate(summary_text, config.summarize_max_output_tokens)
        return summary_text, estimate_tokens(summary_text)
    except Exception:
        # Graceful degradation
        return truncate(content, config.max_absolute_tokens)


# ---------------------------------------------------------------------------
# Main compressor
# ---------------------------------------------------------------------------

@dataclass
class CompressionResult:
    original_tokens: int
    compressed_tokens: int
    algorithm_used: CompressionAlgorithm
    compression_ratio: float
    elapsed_ms: float

    @property
    def savings_pct(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return 100.0 * (1 - self.compressed_tokens / self.original_tokens)


class ContextCompressor:
    """
    Orchestrates compression of SubContext objects.

    Usage
    -----
    ::

        compressor = ContextCompressor(
            policies=DEFAULT_POLICIES,
            llm_callable=my_local_llm.generate,
        )
        compressed_sc, result = compressor.compress(subcontext)
        print(f"Saved {result.savings_pct:.1f}% tokens")
    """

    def __init__(
        self,
        policies: Optional[Dict[SubContextType, CompressionConfig]] = None,
        llm_callable: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.policies: Dict[SubContextType, CompressionConfig] = (
            policies if policies is not None else DEFAULT_POLICIES
        )
        self.llm_callable = llm_callable

    def get_policy(self, sc_type: SubContextType) -> CompressionConfig:
        return self.policies.get(sc_type, CompressionConfig())

    def compress(
        self,
        sc: SubContext,
        algorithm_override: Optional[CompressionAlgorithm] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> Tuple[SubContext, CompressionResult]:
        """
        Compress *sc* and return a new SubContext plus compression metadata.

        The original SubContext is NOT mutated; a new one is returned.
        """
        policy = self.get_policy(sc.type)
        algorithm = algorithm_override or policy.algorithm

        original_tokens = sc.token_count or estimate_tokens(sc.content)
        t0 = time.perf_counter()

        compressed_text, new_tokens = self._apply(
            sc.content, algorithm, policy, query_embedding
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Build new SubContext
        new_sc = SubContext(
            type=sc.type,
            status=SubContextStatus.ACTIVE,
            content=compressed_text,
            token_count=new_tokens,
            tags=sc.tags[:],
            source=sc.source,
            parent_ids=sc.parent_ids + [sc.id],
            metadata={**sc.metadata, "compressed_from": sc.id},
            original_token_count=sc.original_token_count or original_tokens,
        )
        new_sc.compression_ratio = (
            new_sc.original_token_count / new_tokens if new_tokens > 0 else 1.0
        )

        # Mark original as summarized if we replaced it
        if algorithm in (CompressionAlgorithm.SUMMARIZE, CompressionAlgorithm.HYBRID):
            sc.status = SubContextStatus.SUMMARIZED
        else:
            sc.status = SubContextStatus.STALE

        result = CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=new_tokens,
            algorithm_used=algorithm,
            compression_ratio=new_sc.compression_ratio,
            elapsed_ms=elapsed_ms,
        )

        return new_sc, result

    def compress_to_budget(
        self,
        sc: SubContext,
        token_budget: int,
        query_embedding: Optional[List[float]] = None,
    ) -> Tuple[SubContext, CompressionResult]:
        """
        Compress *sc* until its token count fits within *token_budget*.

        Uses an escalating strategy: PRUNE → SUMMARIZE → TRUNCATE.
        """
        if sc.token_count <= token_budget:
            result = CompressionResult(
                original_tokens=sc.token_count,
                compressed_tokens=sc.token_count,
                algorithm_used=CompressionAlgorithm.NONE,
                compression_ratio=1.0,
                elapsed_ms=0.0,
            )
            return sc, result

        policy = self.get_policy(sc.type)

        for algo in [
            CompressionAlgorithm.PRUNE,
            CompressionAlgorithm.SUMMARIZE,
            CompressionAlgorithm.TRUNCATE,
        ]:
            override_policy = CompressionConfig(
                algorithm=algo,
                target_token_ratio=token_budget / max(sc.token_count, 1),
                max_absolute_tokens=token_budget,
                min_absolute_tokens=policy.min_absolute_tokens,
                prune_min_sentence_score=policy.prune_min_sentence_score,
                summarize_prompt_template=policy.summarize_prompt_template,
                summarize_max_output_tokens=min(
                    token_budget, policy.summarize_max_output_tokens
                ),
                fallback_algorithm=CompressionAlgorithm.TRUNCATE,
            )
            new_sc, res = self.compress(sc, algorithm_override=algo)
            if new_sc.token_count <= token_budget:
                return new_sc, res

        # Last resort: hard truncate
        return self.compress(
            sc, algorithm_override=CompressionAlgorithm.TRUNCATE
        )

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _apply(
        self,
        content: str,
        algorithm: CompressionAlgorithm,
        policy: CompressionConfig,
        query_embedding: Optional[List[float]],
    ) -> Tuple[str, int]:
        if algorithm == CompressionAlgorithm.NONE:
            return content, estimate_tokens(content)

        if algorithm == CompressionAlgorithm.TRUNCATE:
            return truncate(content, policy.max_absolute_tokens)

        if algorithm == CompressionAlgorithm.PRUNE:
            return prune(content, policy, query_embedding)

        if algorithm == CompressionAlgorithm.SUMMARIZE:
            return summarize(content, policy, self.llm_callable)

        if algorithm == CompressionAlgorithm.HYBRID:
            # Step 1: prune aggressively
            pruned_text, pruned_tokens = prune(content, policy, query_embedding)
            target = int(estimate_tokens(content) * policy.target_token_ratio)
            if pruned_tokens <= max(target, policy.min_absolute_tokens):
                return pruned_text, pruned_tokens
            # Step 2: summarise what remains
            pruned_policy = CompressionConfig(
                algorithm=CompressionAlgorithm.SUMMARIZE,
                max_absolute_tokens=max(target, policy.min_absolute_tokens),
                summarize_max_output_tokens=min(
                    target, policy.summarize_max_output_tokens
                ),
                summarize_prompt_template=policy.summarize_prompt_template,
                min_absolute_tokens=policy.min_absolute_tokens,
                target_token_ratio=policy.target_token_ratio,
            )
            return summarize(pruned_text, pruned_policy, self.llm_callable)

        # Unknown – fall through to TRUNCATE
        return truncate(content, policy.max_absolute_tokens)
