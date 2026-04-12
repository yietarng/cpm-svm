"""
Request router for NOC (New OpenClaw).

The ``RequestRouter`` sits between the session orchestrator and the agent pool.
Its responsibilities:

1. **Classify** each incoming request into a ``RequestType`` (plan, execute,
   replan, summarise, query).
2. **Select** the best agent to handle it, based on role, availability, and
   current context utilisation.
3. **Load-balance** across multiple agents of the same role if present.
4. **Record** routing decisions and agent utilisation for observability.

Routing is intentionally simple (rule-based + light heuristics) so it
does not itself require an LLM call.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from .agent import (
    AgentBase,
    AgentRequest,
    AgentResponse,
    AgentRole,
    AgentStatus,
    ExecutorAgent,
    PlannerAgent,
    ReplannerAgent,
    SummarizerAgent,
)
from .subcontext import SubContext, SubContextType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request classification
# ---------------------------------------------------------------------------

class RequestType(Enum):
    PLAN = auto()      # Initial plan generation
    EXECUTE = auto()   # Execute one plan step / tool call
    REPLAN = auto()    # Update plan after tool execution
    SUMMARIZE = auto() # Compress old subcontexts
    QUERY = auto()     # General question / free-form request
    UNKNOWN = auto()


# Keyword signals for each request type
_TYPE_SIGNALS: Dict[RequestType, List[str]] = {
    RequestType.PLAN: [
        r"\bplan\b", r"\bgenerate.*plan\b", r"\bcreate.*plan\b",
        r"\bhow.*accomplish\b", r"\bsteps.*to\b", r"\bbreak.*down\b",
    ],
    RequestType.REPLAN: [
        r"\breplan\b", r"\bupdate.*plan\b", r"\brevise.*plan\b",
        r"\bstep.*completed\b", r"\btool.*result\b", r"\bfailed.*step\b",
    ],
    RequestType.EXECUTE: [
        r"\bexecute\b", r"\brun\b", r"\bcall.*tool\b", r"\binvoke\b",
        r"\bperform.*step\b",
    ],
    RequestType.SUMMARIZE: [
        r"\bsummariz", r"\bcompress\b", r"\bcondense\b",
        r"\bshrink.*context\b", r"\bfree.*token\b",
    ],
}


def classify_request(text: str) -> RequestType:
    """
    Classify *text* into a ``RequestType`` using lightweight regex heuristics.
    No LLM call is made.
    """
    text_lower = text.lower()
    scores: Dict[RequestType, int] = {rt: 0 for rt in RequestType}
    for rt, patterns in _TYPE_SIGNALS.items():
        for pat in patterns:
            if re.search(pat, text_lower):
                scores[rt] += 1

    best_rt = max(scores, key=lambda rt: scores[rt])
    if scores[best_rt] == 0:
        return RequestType.QUERY
    return best_rt


# ---------------------------------------------------------------------------
# Routing decision
# ---------------------------------------------------------------------------

@dataclass
class RoutingDecision:
    request_id: str
    request_type: RequestType
    selected_agent_id: str
    fallback_agent_id: Optional[str] = None
    rationale: str = ""
    decided_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class RequestRouter:
    """
    Routes inference requests to the appropriate agent.

    Parameters
    ----------
    agents:           List of all available agent instances.
    default_role:     Role to use when no better match is found.
    """

    # Mapping from request type to preferred agent role
    _ROLE_PREFERENCE: Dict[RequestType, AgentRole] = {
        RequestType.PLAN: AgentRole.PLANNER,
        RequestType.REPLAN: AgentRole.REPLANNER,
        RequestType.EXECUTE: AgentRole.EXECUTOR,
        RequestType.SUMMARIZE: AgentRole.SUMMARIZER,
        RequestType.QUERY: AgentRole.GENERAL,
        RequestType.UNKNOWN: AgentRole.GENERAL,
    }

    def __init__(
        self,
        agents: List[AgentBase],
        default_role: AgentRole = AgentRole.GENERAL,
    ) -> None:
        self.agents = agents
        self.default_role = default_role
        self._history: List[RoutingDecision] = []

        # Build role → agents index
        self._role_index: Dict[AgentRole, List[AgentBase]] = {}
        for agent in agents:
            self._role_index.setdefault(agent.role, []).append(agent)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(
        self,
        request: AgentRequest,
        request_type: Optional[RequestType] = None,
    ) -> Tuple[AgentBase, RoutingDecision]:
        """
        Select the best agent for *request* and return it with a
        ``RoutingDecision`` explaining the choice.
        """
        rt = request_type or classify_request(request.query)
        preferred_role = self._ROLE_PREFERENCE.get(rt, AgentRole.GENERAL)

        agent, rationale = self._select(preferred_role)
        decision = RoutingDecision(
            request_id=request.request_id,
            request_type=rt,
            selected_agent_id=agent.agent_id,
            rationale=rationale,
        )
        self._history.append(decision)
        logger.debug(
            "Routed request %s (type=%s) → agent %s (%s)",
            request.request_id[:8],
            rt.name,
            agent.agent_id[:8],
            agent.role.name,
        )
        return agent, decision

    def dispatch(
        self,
        request: AgentRequest,
        request_type: Optional[RequestType] = None,
    ) -> AgentResponse:
        """Convenience wrapper: route and immediately process the request."""
        agent, _ = self.route(request, request_type)
        return agent.process(request)

    def add_agent(self, agent: AgentBase) -> None:
        """Register a new agent at runtime."""
        self.agents.append(agent)
        self._role_index.setdefault(agent.role, []).append(agent)

    def remove_agent(self, agent_id: str) -> bool:
        """Deregister an agent by id.  Returns True if it was found."""
        found = False
        self.agents = [a for a in self.agents if a.agent_id != agent_id]
        for role, lst in self._role_index.items():
            before = len(lst)
            self._role_index[role] = [a for a in lst if a.agent_id != agent_id]
            if len(self._role_index[role]) < before:
                found = True
        return found

    def stats(self) -> Dict[str, Any]:
        role_counts = {role.name: len(lst) for role, lst in self._role_index.items()}
        return {
            "total_agents": len(self.agents),
            "agents_by_role": role_counts,
            "total_routed_requests": len(self._history),
            "agents": [a.stats() for a in self.agents],
        }

    # ------------------------------------------------------------------
    # Internal selection
    # ------------------------------------------------------------------

    def _select(self, preferred_role: AgentRole) -> Tuple[AgentBase, str]:
        """
        Choose one agent from the pool.

        Strategy (in priority order):
        1. Pick an IDLE agent of the preferred role with the lowest context utilisation.
        2. If none is IDLE, pick the least-busy agent of the preferred role.
        3. Fall back to any IDLE agent of any role.
        4. Fall back to the least-busy agent overall.
        """
        candidates = self._role_index.get(preferred_role, [])

        # 1. Idle with preferred role
        idle = [a for a in candidates if a.status == AgentStatus.IDLE]
        if idle:
            chosen = min(idle, key=lambda a: a.context_utilization)
            return chosen, f"idle {preferred_role.name} with lowest context utilisation"

        # 2. Any preferred role agent
        if candidates:
            chosen = min(candidates, key=lambda a: a.context_utilization)
            return chosen, f"least-utilised {preferred_role.name} agent (all busy)"

        # 3. Any idle agent
        idle_any = [a for a in self.agents if a.status == AgentStatus.IDLE]
        if idle_any:
            chosen = idle_any[0]
            return chosen, f"fallback idle agent (no {preferred_role.name} available)"

        # 4. Least busy overall
        if self.agents:
            chosen = min(self.agents, key=lambda a: a.context_utilization)
            return chosen, "least-utilised agent (all agents busy)"

        raise RuntimeError("No agents registered in RequestRouter")


# ---------------------------------------------------------------------------
# Agent pool factory helpers
# ---------------------------------------------------------------------------

def build_default_agent_pool(
    backend_llm: Any,
    local_llm: Optional[Any] = None,
    tool_registry: Optional[Dict[str, Any]] = None,
) -> List[AgentBase]:
    """
    Create the default set of NOC agents:
    one Planner, one Executor, one Replanner, one Summarizer.

    *backend_llm* and *local_llm* should be callables with signature
    ``(prompt: str) -> str``.

    The Summarizer and Executor use the local LLM if available, falling
    back to the backend LLM.
    """
    cheap_llm = local_llm if local_llm is not None else backend_llm

    planner = PlannerAgent(llm_callable=backend_llm)
    executor = ExecutorAgent(
        llm_callable=cheap_llm,
        tool_registry=tool_registry or {},
    )
    replanner = ReplannerAgent(llm_callable=backend_llm)
    summarizer = SummarizerAgent(llm_callable=cheap_llm)

    return [planner, executor, replanner, summarizer]
