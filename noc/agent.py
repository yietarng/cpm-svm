"""
Multi-agent architecture for NOC (Feature II from the spec).

Feature II recap
----------------
Instead of one monolithic LLM with one KV cache, NOC sets up **multiple
agents**.  Each agent:

* Works on a well-defined subtask (planning, execution, re-planning, …).
* Maintains its own set of SubContexts (its "memory").
* Has its own independent KV-cache state on the inference backend.
* Can accept requests from the session orchestrator and produce outputs that
  become SubContexts for other agents.

Each agent dispatches inference requests via a ``ContextComposer`` so it only
sends the *relevant* subcontexts for each request rather than its entire history.

Class hierarchy
---------------
``AgentBase``       – abstract base with shared bookkeeping.
``PlannerAgent``    – generates an execution plan from the user request.
``ExecutorAgent``   – executes tool calls, records results as SubContexts.
``ReplannerAgent``  – merges latest plan + new tool results → updated plan.
``SummarizerAgent`` – compresses old SubContexts to free token budget.
"""

from __future__ import annotations

import abc
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from .compressor import ContextCompressor, estimate_tokens
from .context_manager import (
    CompositionConfig,
    ContextComposer,
    ContextDecomposer,
    DecompositionConfig,
    SemanticIndex,
)
from .kv_cache import KVCacheManager
from .subcontext import SubContext, SubContextStatus, SubContextType, SubContextWindow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AgentRole(Enum):
    PLANNER = auto()
    EXECUTOR = auto()
    REPLANNER = auto()
    SUMMARIZER = auto()
    GENERAL = auto()


class AgentStatus(Enum):
    IDLE = auto()
    BUSY = auto()
    FAILED = auto()
    SHUTDOWN = auto()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    role: AgentRole = AgentRole.GENERAL
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    max_context_tokens: int = 8192
    response_token_reserve: int = 1024
    # If True, the agent auto-compresses its context when it grows too large
    auto_compress: bool = True
    compress_threshold: float = 0.85  # Compress when context is >85% full
    # Always-include subcontext types for this agent
    always_include_types: List[SubContextType] = field(
        default_factory=lambda: [SubContextType.SYSTEM]
    )
    # Minimum relevance score to include an optional subcontext
    min_relevance_score: float = 0.05


# ---------------------------------------------------------------------------
# Request / Response objects
# ---------------------------------------------------------------------------

@dataclass
class AgentRequest:
    """A single inference request directed at one agent."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    extra_context: List[SubContext] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class AgentResponse:
    """The result of one agent inference call."""
    request_id: str = ""
    agent_id: str = ""
    content: str = ""
    # New SubContexts produced by this response (e.g. a new plan step)
    produced_subcontexts: List[SubContext] = field(default_factory=list)
    window: Optional[SubContextWindow] = None
    latency_ms: float = 0.0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Abstract base agent
# ---------------------------------------------------------------------------

class AgentBase(abc.ABC):
    """
    Abstract base class for all NOC agents.

    Each agent owns:
    * A list of SubContexts (its persistent memory).
    * A SemanticIndex over those SubContexts.
    * A KVCacheManager tracking cache state for its SubContexts.
    * A ContextCompressor for auto-compression.
    * An LLM callable (local or backend) for inference.
    """

    def __init__(
        self,
        config: AgentConfig,
        llm_callable: Callable[[str], str],
        kv_cache_manager: Optional[KVCacheManager] = None,
    ) -> None:
        self.config = config
        self.agent_id = config.agent_id
        self.role = config.role
        self.status = AgentStatus.IDLE

        self._llm = llm_callable
        self._subcontexts: List[SubContext] = []
        self._index = SemanticIndex()
        self._kv_cache = kv_cache_manager or KVCacheManager()
        self._compressor = ContextCompressor(llm_callable=llm_callable)
        self._composer = ContextComposer(
            index=self._index,
            config=CompositionConfig(
                max_tokens=config.max_context_tokens,
                response_token_reserve=config.response_token_reserve,
                always_include_types=config.always_include_types,
                min_relevance_score=config.min_relevance_score,
            ),
        )

        # Statistics
        self._request_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._created_at = time.time()

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    def add_subcontext(self, sc: SubContext) -> None:
        """Append a SubContext to this agent's memory and index it."""
        self._subcontexts.append(sc)
        self._index.add(sc)
        if self.config.auto_compress:
            self._maybe_compress()

    def add_subcontexts(self, subcontexts: List[SubContext]) -> None:
        for sc in subcontexts:
            self.add_subcontext(sc)

    def get_subcontexts(
        self,
        type_filter: Optional[List[SubContextType]] = None,
        status_filter: Optional[List[SubContextStatus]] = None,
    ) -> List[SubContext]:
        results = self._subcontexts
        if type_filter:
            results = [sc for sc in results if sc.type in type_filter]
        if status_filter:
            results = [sc for sc in results if sc.status in status_filter]
        return results

    @property
    def total_context_tokens(self) -> int:
        return sum(sc.token_count for sc in self._subcontexts if sc.status == SubContextStatus.ACTIVE)

    @property
    def context_utilization(self) -> float:
        budget = self.config.max_context_tokens - self.config.response_token_reserve
        return self.total_context_tokens / budget if budget > 0 else 0.0

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def process(self, request: AgentRequest) -> AgentResponse:
        """
        Handle one inference request.

        1. Compose a minimal context window from relevant SubContexts.
        2. Build the prompt.
        3. Call the LLM.
        4. Parse the response into new SubContexts.
        5. Return an AgentResponse.
        """
        self.status = AgentStatus.BUSY
        t0 = time.perf_counter()
        try:
            # Add any extra context supplied by the caller
            for sc in request.extra_context:
                if sc.id not in {s.id for s in self._subcontexts}:
                    self.add_subcontext(sc)

            # Compose context window
            all_active = [
                sc for sc in self._subcontexts
                if sc.status == SubContextStatus.ACTIVE
            ]
            window = self._composer.compose(
                query=request.query,
                candidates=all_active,
                request_id=request.request_id,
            )

            # Build prompt
            prompt = self._build_prompt(request.query, window)

            # LLM inference
            response_text = self._llm(prompt)
            output_tokens = estimate_tokens(response_text)

            # Parse response
            produced = self._parse_response(response_text, request)

            # Update statistics
            self._request_count += 1
            self._total_input_tokens += window.total_tokens
            self._total_output_tokens += output_tokens

            latency_ms = (time.perf_counter() - t0) * 1000.0
            self.status = AgentStatus.IDLE

            return AgentResponse(
                request_id=request.request_id,
                agent_id=self.agent_id,
                content=response_text,
                produced_subcontexts=produced,
                window=window,
                latency_ms=latency_ms,
            )

        except Exception as exc:
            self.status = AgentStatus.FAILED
            logger.exception("Agent %s failed on request %s", self.agent_id, request.request_id)
            return AgentResponse(
                request_id=request.request_id,
                agent_id=self.agent_id,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Abstract methods for subclasses
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _build_prompt(self, query: str, window: SubContextWindow) -> str:
        """Construct the final prompt string from *query* and *window*."""

    def _parse_response(
        self, response_text: str, request: AgentRequest
    ) -> List[SubContext]:
        """
        Parse the LLM response and return new SubContexts.

        Default implementation: wrap the entire response as a single
        CONVERSATION SubContext.  Subclasses override for structured output.
        """
        tokens = estimate_tokens(response_text)
        sc = SubContext(
            type=SubContextType.CONVERSATION,
            content=response_text,
            token_count=tokens,
            source=f"agent:{self.agent_id}",
            metadata={"request_id": request.request_id},
        )
        self.add_subcontext(sc)
        return [sc]

    # ------------------------------------------------------------------
    # Auto-compression
    # ------------------------------------------------------------------

    def _maybe_compress(self) -> None:
        """Compress old SubContexts if context utilization is above threshold."""
        if self.context_utilization < self.config.compress_threshold:
            return

        # Identify compressible candidates: old TOOL_RESULT and DOCUMENT subcontexts
        candidates = sorted(
            [
                sc for sc in self._subcontexts
                if sc.status == SubContextStatus.ACTIVE
                and sc.type in (
                    SubContextType.TOOL_RESULT,
                    SubContextType.DOCUMENT,
                    SubContextType.WEB_SEARCH,
                    SubContextType.CONVERSATION,
                )
            ],
            key=lambda sc: sc.last_accessed,  # Compress least-recently-used first
        )

        budget = self.config.max_context_tokens - self.config.response_token_reserve
        target = int(budget * 0.6)  # Compress down to 60% of budget
        freed = 0

        for sc in candidates:
            if self.total_context_tokens - freed <= target:
                break
            compressed_sc, result = self._compressor.compress(sc)
            freed += result.original_tokens - result.compressed_tokens
            # Replace in list
            idx = self._subcontexts.index(sc)
            self._subcontexts[idx] = compressed_sc
            self._index.update(compressed_sc)
            logger.debug(
                "Compressed subcontext %s: %d → %d tokens (%.1f%% saved)",
                sc.id[:8],
                result.original_tokens,
                result.compressed_tokens,
                result.savings_pct,
            )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "role": self.role.name,
            "status": self.status.name,
            "request_count": self._request_count,
            "subcontext_count": len(self._subcontexts),
            "total_context_tokens": self.total_context_tokens,
            "context_utilization": f"{self.context_utilization:.1%}",
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "kv_cache": self._kv_cache.stats(),
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(id={self.agent_id[:8]}, "
            f"role={self.role.name}, status={self.status.name})"
        )


# ---------------------------------------------------------------------------
# Concrete agent implementations
# ---------------------------------------------------------------------------

class PlannerAgent(AgentBase):
    """
    Generates an execution plan from a user task description.

    Input:  User task (natural language).
    Output: A PLAN SubContext containing an ordered list of steps,
            each step specifying what tool/action to invoke.
    """

    SYSTEM_PROMPT = (
        "You are a planning agent.  Given a user task, produce a concise "
        "numbered execution plan.  Each step should specify: "
        "(1) what to do, (2) which tool to call (if any), "
        "(3) what output is expected.  Be brief and precise."
    )

    def __init__(self, llm_callable: Callable[[str], str], **kwargs: Any) -> None:
        config = kwargs.pop(
            "config",
            AgentConfig(
                role=AgentRole.PLANNER,
                always_include_types=[SubContextType.SYSTEM, SubContextType.USER],
            ),
        )
        super().__init__(config, llm_callable, **kwargs)
        # Seed with system prompt
        self.add_subcontext(
            SubContext(
                type=SubContextType.SYSTEM,
                content=self.SYSTEM_PROMPT,
                token_count=estimate_tokens(self.SYSTEM_PROMPT),
                source="planner_system",
            )
        )

    def _build_prompt(self, query: str, window: SubContextWindow) -> str:
        context = window.to_prompt()
        return f"{context}\n\nUser task: {query}\n\nPlan:"

    def _parse_response(
        self, response_text: str, request: AgentRequest
    ) -> List[SubContext]:
        tokens = estimate_tokens(response_text)
        sc = SubContext(
            type=SubContextType.PLAN,
            content=response_text,
            token_count=tokens,
            source=f"planner:{self.agent_id}",
            metadata={"request_id": request.request_id},
        )
        self.add_subcontext(sc)
        return [sc]


class ExecutorAgent(AgentBase):
    """
    Executes individual plan steps by calling tools and recording results.

    The executor does NOT call an LLM for most tool executions — it just runs
    the tool and wraps the output as a SubContext.  The LLM is called only when
    the executor needs to *interpret* or *summarise* the tool output.
    """

    SYSTEM_PROMPT = (
        "You are an execution agent.  You receive a plan step and tool output. "
        "Summarise the tool output concisely, highlighting key facts relevant "
        "to the plan step."
    )

    def __init__(
        self,
        llm_callable: Callable[[str], str],
        tool_registry: Optional[Dict[str, Callable[..., Any]]] = None,
        **kwargs: Any,
    ) -> None:
        config = kwargs.pop(
            "config",
            AgentConfig(
                role=AgentRole.EXECUTOR,
                always_include_types=[SubContextType.SYSTEM],
                max_context_tokens=4096,
            ),
        )
        super().__init__(config, llm_callable, **kwargs)
        self.tool_registry: Dict[str, Callable[..., Any]] = tool_registry or {}
        self.add_subcontext(
            SubContext(
                type=SubContextType.SYSTEM,
                content=self.SYSTEM_PROMPT,
                token_count=estimate_tokens(self.SYSTEM_PROMPT),
                source="executor_system",
            )
        )

    def execute_tool(
        self, tool_name: str, tool_args: Dict[str, Any]
    ) -> SubContext:
        """
        Run *tool_name* with *tool_args* and wrap the result as a TOOL_RESULT
        SubContext.  This is the main entry point for tool execution — it does
        NOT involve an LLM call.
        """
        t0 = time.perf_counter()
        tool_fn = self.tool_registry.get(tool_name)
        if tool_fn is None:
            result_text = f"Error: tool '{tool_name}' not found in registry."
            success = False
        else:
            try:
                raw_result = tool_fn(**tool_args)
                result_text = str(raw_result)
                success = True
            except Exception as exc:
                result_text = f"Error executing {tool_name}: {exc}"
                success = False

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        tokens = estimate_tokens(result_text)

        sc = SubContext(
            type=SubContextType.TOOL_RESULT,
            content=result_text,
            token_count=tokens,
            source=tool_name,
            metadata={
                "tool_name": tool_name,
                "tool_args": tool_args,
                "success": success,
                "elapsed_ms": elapsed_ms,
            },
        )
        self.add_subcontext(sc)
        return sc

    def _build_prompt(self, query: str, window: SubContextWindow) -> str:
        context = window.to_prompt()
        return f"{context}\n\nPlan step: {query}\n\nSummary of execution result:"

    def _parse_response(
        self, response_text: str, request: AgentRequest
    ) -> List[SubContext]:
        tokens = estimate_tokens(response_text)
        sc = SubContext(
            type=SubContextType.SUMMARY,
            content=response_text,
            token_count=tokens,
            source=f"executor:{self.agent_id}",
            metadata={"request_id": request.request_id},
        )
        self.add_subcontext(sc)
        return [sc]


class ReplannerAgent(AgentBase):
    """
    Takes the current plan and recent tool execution results and produces
    an updated plan.

    This agent maintains a *sliding window* of the plan history and the most
    recent tool results, keeping its context small.
    """

    SYSTEM_PROMPT = (
        "You are a re-planning agent.  You receive the original plan and the "
        "results of completed steps.  Update the plan to reflect what has been "
        "accomplished and what remains.  Output only the updated plan."
    )

    def __init__(self, llm_callable: Callable[[str], str], **kwargs: Any) -> None:
        config = kwargs.pop(
            "config",
            AgentConfig(
                role=AgentRole.REPLANNER,
                always_include_types=[
                    SubContextType.SYSTEM,
                    SubContextType.PLAN,
                ],
                max_context_tokens=6144,
            ),
        )
        super().__init__(config, llm_callable, **kwargs)
        self.add_subcontext(
            SubContext(
                type=SubContextType.SYSTEM,
                content=self.SYSTEM_PROMPT,
                token_count=estimate_tokens(self.SYSTEM_PROMPT),
                source="replanner_system",
            )
        )

    def _build_prompt(self, query: str, window: SubContextWindow) -> str:
        context = window.to_prompt()
        return (
            f"{context}\n\n"
            f"New information: {query}\n\n"
            "Updated plan:"
        )

    def _parse_response(
        self, response_text: str, request: AgentRequest
    ) -> List[SubContext]:
        tokens = estimate_tokens(response_text)
        sc = SubContext(
            type=SubContextType.PLAN,
            content=response_text,
            token_count=tokens,
            source=f"replanner:{self.agent_id}",
            metadata={"request_id": request.request_id},
        )
        self.add_subcontext(sc)
        return [sc]


class SummarizerAgent(AgentBase):
    """
    Compresses old SubContexts to reclaim token budget across all agents.

    The SummarizerAgent is typically invoked by the session orchestrator when
    any agent's context approaches its capacity.  It produces SUMMARY
    SubContexts that replace groups of older, related SubContexts.
    """

    SYSTEM_PROMPT = (
        "You are a summarisation agent.  Given a set of context items, produce "
        "a concise and information-dense summary preserving all key facts, "
        "decisions, and outcomes."
    )

    def __init__(self, llm_callable: Callable[[str], str], **kwargs: Any) -> None:
        config = kwargs.pop(
            "config",
            AgentConfig(
                role=AgentRole.SUMMARIZER,
                always_include_types=[SubContextType.SYSTEM],
                max_context_tokens=2048,
            ),
        )
        super().__init__(config, llm_callable, **kwargs)
        self.add_subcontext(
            SubContext(
                type=SubContextType.SYSTEM,
                content=self.SYSTEM_PROMPT,
                token_count=estimate_tokens(self.SYSTEM_PROMPT),
                source="summarizer_system",
            )
        )

    def summarize_subcontexts(
        self, subcontexts: List[SubContext]
    ) -> AgentResponse:
        """
        Summarise a list of SubContexts and return the result as an AgentResponse.
        """
        combined = "\n\n".join(sc.content for sc in subcontexts if sc.content)
        parent_ids = [sc.id for sc in subcontexts]
        request = AgentRequest(
            query=combined,
            metadata={"parent_ids": parent_ids},
        )
        response = self.process(request)
        if response.produced_subcontexts:
            response.produced_subcontexts[0].parent_ids = parent_ids
            response.produced_subcontexts[0].type = SubContextType.SUMMARY
        return response

    def _build_prompt(self, query: str, window: SubContextWindow) -> str:
        return (
            "Summarise the following context items concisely:\n\n"
            f"{query}\n\nSummary:"
        )

    def _parse_response(
        self, response_text: str, request: AgentRequest
    ) -> List[SubContext]:
        parent_ids = request.metadata.get("parent_ids", [])
        tokens = estimate_tokens(response_text)
        sc = SubContext(
            type=SubContextType.SUMMARY,
            content=response_text,
            token_count=tokens,
            source=f"summarizer:{self.agent_id}",
            parent_ids=parent_ids,
            metadata={"request_id": request.request_id},
        )
        self.add_subcontext(sc)
        return [sc]
