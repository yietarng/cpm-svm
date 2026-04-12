"""
NOC – New OpenClaw
==================

A refined context-management layer for OpenClaw / Claude Code agentic workloads
with three key features:

I.   Local LLM delegation       – off-load cheap tasks (summarisation, pruning,
                                  re-planning) to a locally installed open-source
                                  model, reducing backend token costs.

II.  Context decomposition      – break the session context into semantically
                                  independent SubContexts and send each LLM
                                  request only the *relevant* subset, keeping
                                  context windows small.

III. Position-independent KV    – maintain a separate KV-cache entry per
     caching                      SubContext so the same subcontext's cache can
                                  be reused across multiple requests even when
                                  it appears at different positions in the window,
                                  overcoming the prefix-caching limitation.

Quick start
-----------
::

    from noc import NOCSession, SessionConfig
    from noc.local_llm import LocalLLMConfig, LocalLLMBackend

    def my_llm(prompt: str) -> str:
        # Replace with your actual LLM API call
        return backend_client.generate(prompt)

    session = NOCSession(
        backend_llm=my_llm,
        tool_registry={"web_search": my_search, "shell": my_shell},
        config=SessionConfig(
            local_llm_config=LocalLLMConfig(
                backend=LocalLLMBackend.OLLAMA,
                model_name="llama3.2:3b",
            )
        ),
    )

    answer = session.run("Find the top 5 Python ML libraries and summarise each")
    print(answer)
    print(session.stats())
"""

from .agent import (
    AgentBase,
    AgentConfig,
    AgentRequest,
    AgentResponse,
    AgentRole,
    AgentStatus,
    ExecutorAgent,
    PlannerAgent,
    ReplannerAgent,
    SummarizerAgent,
)
from .compressor import (
    CompressionAlgorithm,
    CompressionConfig,
    CompressionResult,
    ContextCompressor,
    DEFAULT_POLICIES,
    estimate_tokens,
    prune,
    summarize,
    truncate,
)
from .context_manager import (
    CompositionConfig,
    ContextComposer,
    ContextDecomposer,
    DecompositionConfig,
    SemanticIndex,
)
from .kv_cache import (
    CacheCompositionPlan,
    KVCacheEntry,
    KVCacheManager,
)
from .local_llm import (
    LocalLLMBackend,
    LocalLLMClient,
    LocalLLMConfig,
    LocalLLMError,
    LocalLLMRouter,
    OllamaClient,
    RoutingPolicy,
    TaskCategory,
    VLLMClient,
    create_local_llm,
)
from .router import (
    RequestRouter,
    RequestType,
    RoutingDecision,
    build_default_agent_pool,
    classify_request,
)
from .session import (
    EventType,
    NOCSession,
    PlanStep,
    SessionConfig,
    SessionEvent,
    SessionStats,
    parse_plan_steps,
)
from .subcontext import (
    SubContext,
    SubContextStatus,
    SubContextType,
    SubContextWindow,
)

__all__ = [
    # session
    "NOCSession",
    "SessionConfig",
    "SessionStats",
    "SessionEvent",
    "EventType",
    "PlanStep",
    "parse_plan_steps",
    # agents
    "AgentBase",
    "AgentConfig",
    "AgentRequest",
    "AgentResponse",
    "AgentRole",
    "AgentStatus",
    "PlannerAgent",
    "ExecutorAgent",
    "ReplannerAgent",
    "SummarizerAgent",
    # router
    "RequestRouter",
    "RequestType",
    "RoutingDecision",
    "build_default_agent_pool",
    "classify_request",
    # subcontext
    "SubContext",
    "SubContextType",
    "SubContextStatus",
    "SubContextWindow",
    # context manager
    "ContextDecomposer",
    "DecompositionConfig",
    "SemanticIndex",
    "ContextComposer",
    "CompositionConfig",
    # compressor
    "ContextCompressor",
    "CompressionAlgorithm",
    "CompressionConfig",
    "CompressionResult",
    "DEFAULT_POLICIES",
    "estimate_tokens",
    "truncate",
    "prune",
    "summarize",
    # kv cache
    "KVCacheManager",
    "KVCacheEntry",
    "CacheCompositionPlan",
    # local llm
    "LocalLLMClient",
    "LocalLLMConfig",
    "LocalLLMBackend",
    "LocalLLMError",
    "LocalLLMRouter",
    "OllamaClient",
    "VLLMClient",
    "TaskCategory",
    "RoutingPolicy",
    "create_local_llm",
]

__version__ = "0.1.0"
