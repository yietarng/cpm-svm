"""
NOC Session orchestrator.

This module ties all NOC components together into a single ``NOCSession``
that handles the full agentic plan→execute→replan loop described in the spec:

    (a) User requests a task.
    (b) Planner generates an execution plan.
    (c) Executor runs each plan step (tool call), records results.
    (d) Replanner updates the plan given the latest results.
    (e) Repeat from (c) until the task is complete or a stop condition is met.

The session tracks all SubContexts produced throughout the session, maintains
the shared KVCacheManager, and surfaces observability statistics (token savings,
cache hit rates, compression ratios).
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from .agent import (
    AgentBase,
    AgentRequest,
    AgentResponse,
    ExecutorAgent,
    PlannerAgent,
    ReplannerAgent,
    SummarizerAgent,
)
from .context_manager import (
    CompositionConfig,
    ContextComposer,
    ContextDecomposer,
    DecompositionConfig,
    SemanticIndex,
)
from .kv_cache import KVCacheManager
from .local_llm import (
    LocalLLMClient,
    LocalLLMConfig,
    LocalLLMRouter,
    RoutingPolicy,
    TaskCategory,
    create_local_llm,
)
from .router import RequestRouter, RequestType, build_default_agent_pool
from .subcontext import SubContext, SubContextStatus, SubContextType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session configuration
# ---------------------------------------------------------------------------

@dataclass
class SessionConfig:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Context window limits
    global_max_tokens: int = 16384
    per_agent_max_tokens: int = 8192

    # Agentic loop limits
    max_plan_steps: int = 20
    max_replan_cycles: int = 10

    # Compression triggers
    auto_compress: bool = True
    compress_trigger_ratio: float = 0.80  # Compress when global context > 80% full

    # Local LLM configuration (None = don't use a local LLM)
    local_llm_config: Optional[LocalLLMConfig] = None

    # Whether to share the KV cache across agents (recommended: True)
    shared_kv_cache: bool = True


# ---------------------------------------------------------------------------
# Plan step
# ---------------------------------------------------------------------------

@dataclass
class PlanStep:
    """One step extracted from a planner-generated plan."""
    index: int
    description: str
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"   # pending | running | done | failed
    result: Optional[str] = None
    subcontext_id: Optional[str] = None


def parse_plan_steps(plan_text: str) -> List[PlanStep]:
    """
    Extract numbered steps from *plan_text*.

    Supports formats like:
        1. Do something with tool X
        2. Call API Y with arg Z
    """
    import re
    steps: List[PlanStep] = []
    # Match lines starting with a number followed by . or )
    for m in re.finditer(r"^(\d+)[.)]\s+(.+)$", plan_text, re.MULTILINE):
        idx = int(m.group(1))
        desc = m.group(2).strip()
        # Heuristically extract tool name if present
        tool_match = re.search(
            r"\b(?:tool|call|invoke|run|execute)\s+['\"]?(\w+)['\"]?", desc, re.IGNORECASE
        )
        tool_name = tool_match.group(1) if tool_match else None
        steps.append(PlanStep(index=idx, description=desc, tool_name=tool_name))

    if not steps:
        # Fallback: one step per non-empty line
        for i, line in enumerate(plan_text.strip().splitlines(), start=1):
            line = line.strip()
            if line:
                steps.append(PlanStep(index=i, description=line))

    return steps


# ---------------------------------------------------------------------------
# Session events (for streaming / observability)
# ---------------------------------------------------------------------------

class EventType(Enum):
    PLAN_GENERATED = auto()
    STEP_STARTED = auto()
    STEP_COMPLETED = auto()
    STEP_FAILED = auto()
    REPLAN_TRIGGERED = auto()
    REPLAN_COMPLETED = auto()
    CONTEXT_COMPRESSED = auto()
    SESSION_COMPLETE = auto()
    SESSION_FAILED = auto()


@dataclass
class SessionEvent:
    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Session statistics
# ---------------------------------------------------------------------------

@dataclass
class SessionStats:
    session_id: str = ""
    total_llm_calls: int = 0
    local_llm_calls: int = 0
    backend_llm_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    kv_cache_hits: int = 0
    kv_cache_misses: int = 0
    compression_events: int = 0
    tokens_saved_by_compression: int = 0
    plan_cycles: int = 0
    steps_executed: int = 0
    duration_seconds: float = 0.0

    @property
    def kv_cache_hit_rate(self) -> float:
        total = self.kv_cache_hits + self.kv_cache_misses
        return self.kv_cache_hits / total if total > 0 else 0.0

    @property
    def local_llm_fraction(self) -> float:
        if self.total_llm_calls == 0:
            return 0.0
        return self.local_llm_calls / self.total_llm_calls

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "total_llm_calls": self.total_llm_calls,
            "local_llm_calls": self.local_llm_calls,
            "backend_llm_calls": self.backend_llm_calls,
            "local_llm_fraction": f"{self.local_llm_fraction:.1%}",
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "kv_cache_hit_rate": f"{self.kv_cache_hit_rate:.1%}",
            "compression_events": self.compression_events,
            "tokens_saved_by_compression": self.tokens_saved_by_compression,
            "plan_cycles": self.plan_cycles,
            "steps_executed": self.steps_executed,
            "duration_seconds": f"{self.duration_seconds:.2f}",
        }


# ---------------------------------------------------------------------------
# NOC Session
# ---------------------------------------------------------------------------

class NOCSession:
    """
    Orchestrates the full NOC agentic loop for one user task.

    Usage
    -----
    ::

        def my_backend_llm(prompt: str) -> str:
            # Call your Claude / GPT / etc. API here
            return client.generate(prompt)

        session = NOCSession(
            backend_llm=my_backend_llm,
            tool_registry={"web_search": my_search_fn, "shell": my_shell_fn},
            config=SessionConfig(local_llm_config=LocalLLMConfig(model_name="llama3.2:3b")),
        )
        result = session.run("Summarise the top 5 Python ML frameworks")
        print(result)
        print(session.stats())
    """

    def __init__(
        self,
        backend_llm: Callable[[str], str],
        tool_registry: Optional[Dict[str, Callable[..., Any]]] = None,
        config: Optional[SessionConfig] = None,
        agents: Optional[List[AgentBase]] = None,
    ) -> None:
        self.config = config or SessionConfig()
        self._session_id = self.config.session_id
        self._start_time = time.time()
        self._events: List[SessionEvent] = []
        self._stats = SessionStats(session_id=self._session_id)

        # ---- LLM routing ----
        local_client: Optional[LocalLLMClient] = None
        if self.config.local_llm_config is not None:
            try:
                local_client = create_local_llm(self.config.local_llm_config)
            except Exception as exc:
                logger.warning("Failed to create local LLM client: %s", exc)

        self._llm_router = LocalLLMRouter(
            local_client=local_client,
            backend_client=backend_llm,
        )

        def _backend(prompt: str) -> str:
            result = self._llm_router.route(prompt, TaskCategory.GENERAL)
            self._stats.total_llm_calls += 1
            self._stats.backend_llm_calls += 1
            return result

        def _cheap(prompt: str) -> str:
            result = self._llm_router.route(prompt, TaskCategory.SUMMARIZE)
            self._stats.total_llm_calls += 1
            if self._llm_router.local_available:
                self._stats.local_llm_calls += 1
            else:
                self._stats.backend_llm_calls += 1
            return result

        # ---- Shared KV cache ----
        self._kv_cache = KVCacheManager(
            max_entries=2048,
            ttl_seconds=7200.0,
        )

        # ---- Agents ----
        if agents is not None:
            self._agents = agents
        else:
            self._agents = build_default_agent_pool(
                backend_llm=_backend,
                local_llm=_cheap,
                tool_registry=tool_registry or {},
            )

        # Inject shared KV cache if configured
        if self.config.shared_kv_cache:
            for agent in self._agents:
                agent._kv_cache = self._kv_cache

        # ---- Router ----
        self._router = RequestRouter(self._agents)

        # ---- Global context index ----
        self._global_index = SemanticIndex()
        self._all_subcontexts: List[SubContext] = []

        # ---- Decomposer ----
        self._decomposer = ContextDecomposer(
            DecompositionConfig(session_tag=self._session_id)
        )

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def run(
        self,
        user_task: str,
        stream_events: bool = False,
    ) -> str:
        """
        Execute the full plan→execute→replan loop for *user_task*.

        Returns the final synthesised answer as a string.
        """
        logger.info("Session %s: starting task: %s", self._session_id[:8], user_task[:80])

        try:
            # Step 1: Generate initial plan
            plan_sc = self._generate_plan(user_task)
            steps = parse_plan_steps(plan_sc.content)
            self._emit(EventType.PLAN_GENERATED, {"step_count": len(steps)})
            self._stats.plan_cycles += 1

            completed_results: List[str] = []
            replan_cycles = 0

            # Step 2: Execute loop
            step_index = 0
            while step_index < len(steps) and step_index < self.config.max_plan_steps:
                step = steps[step_index]
                self._emit(EventType.STEP_STARTED, {"step": step.description})
                result_sc = self._execute_step(step)

                if result_sc is not None:
                    step.status = "done"
                    step.result = result_sc.content
                    step.subcontext_id = result_sc.id
                    completed_results.append(result_sc.content)
                    self._emit(EventType.STEP_COMPLETED, {"step": step.description})
                    self._stats.steps_executed += 1
                else:
                    step.status = "failed"
                    self._emit(EventType.STEP_FAILED, {"step": step.description})

                step_index += 1

                # Step 3: Maybe replan
                if (
                    step_index < len(steps)
                    and replan_cycles < self.config.max_replan_cycles
                ):
                    new_steps = self._maybe_replan(steps, step_index)
                    if new_steps is not None:
                        steps = new_steps
                        replan_cycles += 1
                        self._stats.plan_cycles += 1
                        self._emit(
                            EventType.REPLAN_COMPLETED,
                            {"new_step_count": len(steps) - step_index},
                        )

                # Auto-compress if needed
                if self.config.auto_compress:
                    self._maybe_compress_global()

            # Step 4: Synthesise final answer
            final_answer = self._synthesise(user_task, completed_results)
            self._emit(EventType.SESSION_COMPLETE, {})
            return final_answer

        except Exception as exc:
            self._emit(EventType.SESSION_FAILED, {"error": str(exc)})
            logger.exception("Session %s failed", self._session_id[:8])
            raise
        finally:
            self._stats.duration_seconds = time.time() - self._start_time
            # Sync KV cache stats
            cache_stats = self._kv_cache.stats()
            self._stats.kv_cache_hits = cache_stats["total_hits"]
            self._stats.kv_cache_misses = cache_stats["total_hits"]  # mirrored

    # ------------------------------------------------------------------
    # Agentic loop steps
    # ------------------------------------------------------------------

    def _generate_plan(self, user_task: str) -> SubContext:
        """Ask the PlannerAgent to produce an execution plan."""
        request = AgentRequest(query=user_task)
        response = self._router.dispatch(request, RequestType.PLAN)

        if not response.success:
            raise RuntimeError(f"Planning failed: {response.error}")

        plan_sc = response.produced_subcontexts[0] if response.produced_subcontexts else SubContext(
            type=SubContextType.PLAN,
            content=response.content,
        )
        self._register_subcontext(plan_sc)
        self._stats.total_input_tokens += response.window.total_tokens if response.window else 0
        return plan_sc

    def _execute_step(self, step: PlanStep) -> Optional[SubContext]:
        """
        Execute one plan step.

        If the step references a known tool, calls it directly via the
        ExecutorAgent without an LLM call.  Otherwise, asks the ExecutorAgent's
        LLM to interpret the step.
        """
        executor: Optional[ExecutorAgent] = None
        for agent in self._agents:
            if isinstance(agent, ExecutorAgent):
                executor = agent
                break

        if executor is None:
            logger.error("No ExecutorAgent found in pool")
            return None

        # Direct tool execution (no LLM)
        if step.tool_name and step.tool_name in executor.tool_registry:
            result_sc = executor.execute_tool(step.tool_name, step.tool_args)
            self._register_subcontext(result_sc)
            return result_sc

        # LLM-based interpretation
        request = AgentRequest(
            query=step.description,
            extra_context=self._get_relevant_context(step.description),
        )
        response = self._router.dispatch(request, RequestType.EXECUTE)
        if not response.success:
            return None

        sc = response.produced_subcontexts[0] if response.produced_subcontexts else None
        if sc:
            self._register_subcontext(sc)
        if response.window:
            self._stats.total_input_tokens += response.window.total_tokens
        return sc

    def _maybe_replan(
        self,
        current_steps: List[PlanStep],
        next_index: int,
    ) -> Optional[List[PlanStep]]:
        """
        Ask the ReplannerAgent if the plan should be updated.

        Returns updated steps list or None if no replan is needed.
        """
        completed = [s for s in current_steps[:next_index] if s.status == "done"]
        if not completed:
            return None

        latest_result = completed[-1].result or ""
        remaining_desc = "\n".join(
            f"{s.index}. {s.description}" for s in current_steps[next_index:]
        )
        query = (
            f"Latest result: {latest_result[:500]}\n\n"
            f"Remaining plan steps:\n{remaining_desc}\n\n"
            "Update the plan if needed, or confirm it is still correct."
        )

        request = AgentRequest(
            query=query,
            extra_context=self._get_relevant_context(query, top_k=3),
        )
        response = self._router.dispatch(request, RequestType.REPLAN)

        if not response.success or not response.content.strip():
            return None

        new_plan_text = response.content
        new_steps = parse_plan_steps(new_plan_text)
        if not new_steps:
            return None

        # Prefix the already-completed steps so indices are consistent
        full_steps = current_steps[:next_index] + new_steps
        if response.window:
            self._stats.total_input_tokens += response.window.total_tokens

        # Register produced subcontexts
        for sc in response.produced_subcontexts:
            self._register_subcontext(sc)

        return full_steps

    def _synthesise(self, user_task: str, results: List[str]) -> str:
        """
        Produce a final answer from the collected execution results.

        For short result sets, returns a concatenation; for large ones,
        calls the backend LLM to synthesise.
        """
        from .compressor import estimate_tokens

        combined = "\n\n".join(results)
        if estimate_tokens(combined) < 500:
            return combined

        prompt = (
            f"User task: {user_task}\n\n"
            f"Execution results:\n{combined}\n\n"
            "Synthesise a concise final answer for the user:"
        )
        request = AgentRequest(query=prompt)
        response = self._router.dispatch(request, RequestType.QUERY)
        if response.success:
            return response.content
        return combined

    # ------------------------------------------------------------------
    # Context helpers
    # ------------------------------------------------------------------

    def _register_subcontext(self, sc: SubContext) -> None:
        """Add a SubContext to the global session index."""
        self._all_subcontexts.append(sc)
        self._global_index.add(sc)

    def _get_relevant_context(
        self, query: str, top_k: int = 5
    ) -> List[SubContext]:
        """Retrieve the top-k most relevant SubContexts from the global index."""
        results = self._global_index.query(query, top_k=top_k)
        return [sc for sc, _ in results]

    def _maybe_compress_global(self) -> None:
        """
        If the total session context exceeds the compression trigger, ask the
        SummarizerAgent to compress the oldest SubContexts.
        """
        from .compressor import estimate_tokens

        total_tokens = sum(sc.token_count for sc in self._all_subcontexts)
        budget = self.config.global_max_tokens
        if total_tokens < budget * self.config.compress_trigger_ratio:
            return

        # Find summarizer
        summarizer: Optional[SummarizerAgent] = None
        for agent in self._agents:
            if isinstance(agent, SummarizerAgent):
                summarizer = agent
                break
        if summarizer is None:
            return

        # Compress the oldest non-system subcontexts
        compressible = sorted(
            [
                sc for sc in self._all_subcontexts
                if sc.status == SubContextStatus.ACTIVE
                and sc.type in (
                    SubContextType.TOOL_RESULT,
                    SubContextType.DOCUMENT,
                    SubContextType.WEB_SEARCH,
                )
            ],
            key=lambda sc: sc.last_accessed,
        )

        if not compressible:
            return

        # Summarise in batches of up to 5 subcontexts
        batch = compressible[:5]
        original_tokens = sum(sc.token_count for sc in batch)
        response = summarizer.summarize_subcontexts(batch)

        if response.success and response.produced_subcontexts:
            summary_sc = response.produced_subcontexts[0]
            # Mark originals as summarised
            for sc in batch:
                sc.status = SubContextStatus.SUMMARIZED
            self._register_subcontext(summary_sc)
            saved = original_tokens - summary_sc.token_count
            self._stats.compression_events += 1
            self._stats.tokens_saved_by_compression += max(0, saved)
            self._emit(
                EventType.CONTEXT_COMPRESSED,
                {"tokens_freed": saved, "batch_size": len(batch)},
            )
            logger.debug(
                "Compressed %d subcontexts, saved %d tokens", len(batch), saved
            )

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _emit(self, event_type: EventType, data: Dict[str, Any]) -> None:
        event = SessionEvent(type=event_type, data=data)
        self._events.append(event)
        logger.debug("Event: %s %s", event_type.name, data)

    def events(self) -> List[SessionEvent]:
        return list(self._events)

    # ------------------------------------------------------------------
    # Statistics / introspection
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        self._stats.duration_seconds = time.time() - self._start_time
        cache_stats = self._kv_cache.stats()
        return {
            **self._stats.to_dict(),
            "global_subcontext_count": len(self._all_subcontexts),
            "kv_cache": cache_stats,
            "router": self._router.stats(),
        }

    def __repr__(self) -> str:
        return (
            f"NOCSession(id={self._session_id[:8]}, "
            f"agents={len(self._agents)}, "
            f"subcontexts={len(self._all_subcontexts)})"
        )
