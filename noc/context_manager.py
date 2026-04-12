"""
Context decomposition and relevance-based composition for NOC.

Feature II from the spec
-------------------------
Instead of maintaining one monolithic context window per session, NOC
decomposes the session context into a set of *semantically independent*
SubContexts and then, for each LLM inference request, assembles a
*minimal context window* containing only the relevant SubContexts.

This module provides:

``ContextDecomposer``
    Splits raw session text (or structured messages) into SubContexts.
    Knows about tool-call boundaries, document separators, plan blocks, etc.

``SemanticIndex``
    Lightweight in-memory index of SubContext embeddings for fast nearest-
    neighbour retrieval without a vector database dependency.

``ContextComposer``
    Given a query and a set of candidate SubContexts, selects and orders the
    most relevant ones within a token budget.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .compressor import estimate_tokens
from .subcontext import SubContext, SubContextType, SubContextWindow


# ---------------------------------------------------------------------------
# Decomposition patterns
# ---------------------------------------------------------------------------

# Markers that signal the beginning of a tool-result block in typical
# OpenClaw / Claude Code output.
_TOOL_RESULT_START = re.compile(
    r"(<result>|<tool_result>|\[Tool Result\]|```tool_output)", re.IGNORECASE
)
_TOOL_RESULT_END = re.compile(
    r"(</result>|</tool_result>|\[/Tool Result\]|```)", re.IGNORECASE
)

# Markers for a planning block
_PLAN_START = re.compile(r"(<plan>|\[Plan\]|## Plan)", re.IGNORECASE)
_PLAN_END = re.compile(r"(</plan>|\[/Plan\]|## (?!Plan))", re.IGNORECASE)

# Markers for a retrieved document fragment
_DOCUMENT_START = re.compile(
    r"(<document>|<doc>|\[Document\]|## Document|---\s*Source:)", re.IGNORECASE
)
_DOCUMENT_END = re.compile(r"(</document>|</doc>|\[/Document\]|---\s*End)", re.IGNORECASE)

# Markers for a web-search result block
_WEB_START = re.compile(
    r"(<web_result>|\[Web Result\]|### Search Result \d+)", re.IGNORECASE
)
_WEB_END = re.compile(r"(</web_result>|\[/Web Result\])", re.IGNORECASE)


def _detect_type(content: str) -> SubContextType:
    """Heuristically classify a text chunk by its content."""
    if _TOOL_RESULT_START.search(content):
        return SubContextType.TOOL_RESULT
    if _PLAN_START.search(content):
        return SubContextType.PLAN
    if _DOCUMENT_START.search(content):
        return SubContextType.DOCUMENT
    if _WEB_START.search(content):
        return SubContextType.WEB_SEARCH
    return SubContextType.CONVERSATION


# ---------------------------------------------------------------------------
# Semantic index (lightweight)
# ---------------------------------------------------------------------------

class SemanticIndex:
    """
    In-memory nearest-neighbour index over SubContext embeddings.

    Uses brute-force cosine similarity — good up to ~10 k subcontexts.
    Swap the ``_score`` method for an HNSW-based ANN library for larger scales.

    If an embedding callable is not provided, falls back to a simple
    TF-IDF bag-of-words representation so the index still works without
    a neural encoder.
    """

    def __init__(
        self,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
    ) -> None:
        self._embed_fn = embed_fn
        # subcontext_id → embedding vector
        self._vectors: Dict[str, List[float]] = {}
        # subcontext_id → SubContext reference
        self._subcontexts: Dict[str, SubContext] = {}
        # Vocabulary for fallback TF-IDF
        self._vocab: Dict[str, int] = {}
        self._doc_freq: Dict[str, int] = {}
        self._n_docs: int = 0

    def add(self, sc: SubContext) -> None:
        """Index a SubContext (compute and store its embedding)."""
        vec = self._vectorize(sc.content)
        self._vectors[sc.id] = vec
        self._subcontexts[sc.id] = sc

    def remove(self, subcontext_id: str) -> None:
        self._vectors.pop(subcontext_id, None)
        self._subcontexts.pop(subcontext_id, None)

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        type_filter: Optional[List[SubContextType]] = None,
        min_score: float = 0.0,
    ) -> List[Tuple[SubContext, float]]:
        """
        Return the top-k SubContexts most relevant to *query_text*.

        Parameters
        ----------
        query_text:  The text of the incoming request / question.
        top_k:       Maximum number of results.
        type_filter: If provided, only return SubContexts of these types.
        min_score:   Discard results with similarity below this threshold.

        Returns
        -------
        List of (SubContext, score) sorted by descending similarity.
        """
        if not self._vectors:
            return []

        q_vec = self._vectorize(query_text)
        scored: List[Tuple[SubContext, float]] = []
        for sc_id, vec in self._vectors.items():
            sc = self._subcontexts[sc_id]
            if type_filter and sc.type not in type_filter:
                continue
            score = self._cosine(q_vec, vec)
            if score >= min_score:
                scored.append((sc, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def update(self, sc: SubContext) -> None:
        """Re-index a SubContext after its content has changed."""
        self.remove(sc.id)
        self.add(sc)

    @property
    def size(self) -> int:
        return len(self._vectors)

    # ------------------------------------------------------------------
    # Internal vectorisation
    # ------------------------------------------------------------------

    def _vectorize(self, text: str) -> List[float]:
        if self._embed_fn is not None:
            try:
                return self._embed_fn(text)
            except Exception:
                pass
        return self._tfidf_vector(text)

    def _tfidf_vector(self, text: str) -> List[float]:
        """Build a sparse TF-IDF vector for *text* using the current vocabulary."""
        words = re.findall(r"\w+", text.lower())
        if not words:
            return [0.0]

        # Update vocabulary and document-frequency counts
        new_words = set(words) - set(self._vocab)
        for w in new_words:
            self._vocab[w] = len(self._vocab)
            self._doc_freq[w] = 0
        for w in set(words):
            self._doc_freq[w] = self._doc_freq.get(w, 0) + 1
        self._n_docs += 1

        n = len(self._vocab)
        if n == 0:
            return [0.0]

        # Compute TF-IDF vector
        from collections import Counter
        tf = Counter(words)
        vec = [0.0] * n
        for w, cnt in tf.items():
            if w in self._vocab:
                idx = self._vocab[w]
                term_tf = cnt / len(words)
                idf = math.log((self._n_docs + 1) / (self._doc_freq.get(w, 0) + 1)) + 1.0
                vec[idx] = term_tf * idf

        return vec

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        if len(a) != len(b):
            # Pad shorter vector with zeros
            n = max(len(a), len(b))
            a = a + [0.0] * (n - len(a))
            b = b + [0.0] * (n - len(b))
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)


# ---------------------------------------------------------------------------
# Decomposer
# ---------------------------------------------------------------------------

@dataclass
class DecompositionConfig:
    """Tuneable parameters for ContextDecomposer."""

    # Minimum token count for a chunk to become its own SubContext
    min_chunk_tokens: int = 20
    # Maximum tokens in a single SubContext before it is split further
    max_chunk_tokens: int = 1500
    # Treat each tool-result block as a separate subcontext
    split_tool_results: bool = True
    # Treat each document / RAG fragment as a separate subcontext
    split_documents: bool = True
    # Treat each web-search result as a separate subcontext
    split_web_results: bool = True
    # Tag applied to all subcontexts produced by this session
    session_tag: str = ""


class ContextDecomposer:
    """
    Converts a raw session context (string or list of messages) into a list
    of semantically independent SubContexts.

    Usage
    -----
    ::

        decomposer = ContextDecomposer()
        subcontexts = decomposer.decompose(session_text)
        # or
        subcontexts = decomposer.decompose_messages(openai_messages)
    """

    def __init__(self, config: Optional[DecompositionConfig] = None) -> None:
        self.config = config or DecompositionConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(self, session_text: str, source: str = "") -> List[SubContext]:
        """
        Split *session_text* into SubContexts.

        The method first attempts structured decomposition (tool results,
        documents, web results, plan blocks).  Any remaining text is split
        into conversation chunks by blank lines.
        """
        chunks: List[Tuple[str, SubContextType]] = []

        if self.config.split_tool_results:
            session_text, extracted = self._extract_blocks(
                session_text,
                _TOOL_RESULT_START,
                _TOOL_RESULT_END,
                SubContextType.TOOL_RESULT,
            )
            chunks.extend(extracted)

        if self.config.split_documents:
            session_text, extracted = self._extract_blocks(
                session_text,
                _DOCUMENT_START,
                _DOCUMENT_END,
                SubContextType.DOCUMENT,
            )
            chunks.extend(extracted)

        if self.config.split_web_results:
            session_text, extracted = self._extract_blocks(
                session_text,
                _WEB_START,
                _WEB_END,
                SubContextType.WEB_SEARCH,
            )
            chunks.extend(extracted)

        # Split plan blocks
        session_text, plan_chunks = self._extract_blocks(
            session_text, _PLAN_START, _PLAN_END, SubContextType.PLAN
        )
        chunks.extend(plan_chunks)

        # Remaining text → conversation chunks
        for para in re.split(r"\n{2,}", session_text.strip()):
            para = para.strip()
            if para and estimate_tokens(para) >= self.config.min_chunk_tokens:
                chunks.append((para, SubContextType.CONVERSATION))

        # Convert to SubContext objects and split oversized chunks
        result: List[SubContext] = []
        for text, sc_type in chunks:
            if not text.strip():
                continue
            sub_chunks = self._split_if_large(text)
            for chunk_text in sub_chunks:
                tokens = estimate_tokens(chunk_text)
                sc = SubContext(
                    type=sc_type,
                    content=chunk_text,
                    token_count=tokens,
                    source=source,
                    tags=(
                        [self.config.session_tag]
                        if self.config.session_tag
                        else []
                    ),
                )
                result.append(sc)

        return result

    def decompose_messages(
        self,
        messages: List[Dict[str, Any]],
        source: str = "",
    ) -> List[SubContext]:
        """
        Decompose a list of OpenAI-style message dicts into SubContexts.

        Each message becomes at least one SubContext.  Tool messages and
        documents embedded in user/assistant turns are extracted separately.
        """
        result: List[SubContext] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "") or ""

            if isinstance(content, list):
                # Multi-part content (text + images, etc.)
                text_parts = [
                    p.get("text", "") for p in content if isinstance(p, dict)
                ]
                content = " ".join(t for t in text_parts if t)

            sc_type = {
                "system": SubContextType.SYSTEM,
                "user": SubContextType.USER,
                "assistant": SubContextType.CONVERSATION,
                "tool": SubContextType.TOOL_RESULT,
            }.get(role, SubContextType.CONVERSATION)

            # For tool messages, treat the whole thing as one SubContext
            if sc_type == SubContextType.TOOL_RESULT:
                tokens = estimate_tokens(content)
                result.append(
                    SubContext(
                        type=sc_type,
                        content=content,
                        token_count=tokens,
                        source=msg.get("name", source),
                        tags=[self.config.session_tag] if self.config.session_tag else [],
                    )
                )
                continue

            # For system/user/assistant messages, run sub-decomposition
            sub_scs = self.decompose(content, source=source)
            if sub_scs:
                # Override type for the first sub-subcontext if it came out as
                # CONVERSATION but the role says otherwise
                if sc_type in (SubContextType.SYSTEM, SubContextType.USER):
                    for sc in sub_scs:
                        if sc.type == SubContextType.CONVERSATION:
                            sc.type = sc_type
                result.extend(sub_scs)
            elif content.strip():
                tokens = estimate_tokens(content)
                result.append(
                    SubContext(
                        type=sc_type,
                        content=content,
                        token_count=tokens,
                        source=source,
                    )
                )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_blocks(
        self,
        text: str,
        start_re: re.Pattern,
        end_re: re.Pattern,
        sc_type: SubContextType,
    ) -> Tuple[str, List[Tuple[str, SubContextType]]]:
        """
        Extract all blocks delimited by *start_re* … *end_re* from *text*.

        Returns (remaining_text, list_of_(block_content, sc_type)).
        """
        extracted: List[Tuple[str, SubContextType]] = []
        remaining_parts: List[str] = []
        pos = 0

        while pos < len(text):
            m_start = start_re.search(text, pos)
            if m_start is None:
                remaining_parts.append(text[pos:])
                break

            # Text before the block is remaining context
            remaining_parts.append(text[pos : m_start.start()])

            m_end = end_re.search(text, m_start.end())
            if m_end is None:
                # No closing tag found – treat the rest as this block type
                block = text[m_start.start() :]
                extracted.append((block.strip(), sc_type))
                pos = len(text)
            else:
                block = text[m_start.start() : m_end.end()]
                extracted.append((block.strip(), sc_type))
                pos = m_end.end()

        return "".join(remaining_parts), extracted

    def _split_if_large(self, text: str) -> List[str]:
        """Split *text* into chunks no larger than max_chunk_tokens."""
        max_chars = self.config.max_chunk_tokens * 4
        if len(text) <= max_chars:
            return [text]
        # Split by paragraph first
        chunks: List[str] = []
        current = ""
        for para in re.split(r"\n{1,2}", text):
            if len(current) + len(para) <= max_chars:
                current = (current + "\n" + para).lstrip()
            else:
                if current:
                    chunks.append(current)
                current = para
        if current:
            chunks.append(current)
        return chunks if chunks else [text]


# ---------------------------------------------------------------------------
# Composer
# ---------------------------------------------------------------------------

@dataclass
class CompositionConfig:
    """Parameters controlling context window assembly."""

    max_tokens: int = 8192
    # Always include these types regardless of relevance score
    always_include_types: List[SubContextType] = field(
        default_factory=lambda: [SubContextType.SYSTEM, SubContextType.USER]
    )
    # Minimum relevance score for a SubContext to be included
    min_relevance_score: float = 0.05
    # Maximum number of SubContexts to include (beyond always-include)
    max_subcontexts: int = 20
    # Reserve this many tokens for the model's response generation
    response_token_reserve: int = 1024


class ContextComposer:
    """
    Builds a minimal SubContextWindow for a single LLM inference request.

    Usage
    -----
    ::

        composer = ContextComposer(index, config)
        window = composer.compose(
            query="What products are available?",
            candidates=all_subcontexts,
        )
        prompt = window.to_prompt()
    """

    def __init__(
        self,
        index: Optional[SemanticIndex] = None,
        config: Optional[CompositionConfig] = None,
    ) -> None:
        self.index = index or SemanticIndex()
        self.config = config or CompositionConfig()

    def compose(
        self,
        query: str,
        candidates: List[SubContext],
        request_id: str = "",
    ) -> SubContextWindow:
        """
        Select and order SubContexts for a single inference request.

        Strategy
        --------
        1. Always include SYSTEM and USER subcontexts first (no relevance filter).
        2. Fill the remaining token budget with the most relevant optional
           subcontexts, ordered by relevance score descending.
        3. Stop when the budget is exhausted or max_subcontexts is reached.
        """
        cfg = self.config
        window = SubContextWindow(
            request_id=request_id or "",
            max_tokens=cfg.max_tokens - cfg.response_token_reserve,
        )

        # Step 1: mandatory subcontexts
        mandatory = [
            sc for sc in candidates
            if sc.type in cfg.always_include_types
        ]
        for sc in mandatory:
            window.add(sc)

        remaining_budget = window.max_tokens - window.total_tokens
        if remaining_budget <= 0:
            return window

        # Step 2: rank optional subcontexts by relevance
        optional = [sc for sc in candidates if sc.type not in cfg.always_include_types]

        if not optional:
            return window

        # Use the semantic index if available, otherwise fall back to
        # including optional subcontexts in reverse-chronological order
        if self.index.size > 0:
            ranked = self.index.query(
                query_text=query,
                top_k=cfg.max_subcontexts,
                min_score=cfg.min_relevance_score,
            )
            ranked_ids = {sc.id for sc, _ in ranked}
            ordered = [sc for sc, _ in ranked if sc in optional]
            # Also append any optional subcontexts that weren't in the index
            ordered += [sc for sc in optional if sc.id not in ranked_ids]
        else:
            # Reverse-chronological fallback: most recent is most relevant
            ordered = list(reversed(optional))

        # Step 3: fill token budget
        included = 0
        for sc in ordered:
            if included >= cfg.max_subcontexts:
                break
            if not window.add(sc):
                # Doesn't fit; try smaller subcontexts
                continue
            included += 1

        return window

    def register_all(self, subcontexts: List[SubContext]) -> None:
        """Add all SubContexts to the semantic index."""
        for sc in subcontexts:
            self.index.add(sc)

    def update_index(self, sc: SubContext) -> None:
        """Re-index a SubContext after its content has changed."""
        self.index.update(sc)
