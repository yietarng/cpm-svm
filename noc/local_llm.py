"""
Local LLM interface for NOC (Feature I from the spec).

Feature I recap
---------------
NOC can delegate parts of the agentic loop to *locally installed* open-source
LLMs instead of always hitting the expensive backend LLM.  Specifically:

* **Summarisation / compression** calls (small, well-defined tasks) are good
  candidates for a small local model.
* **Re-planning** on straightforward task updates can also be done locally.
* The **backend LLM** is reserved for complex reasoning, tool-call generation,
  and final response synthesis.

Supported backends
------------------
* ``OllamaClient``    – HTTP API exposed by Ollama (ollama.ai).  Zero extra
                        dependencies beyond ``requests``.
* ``LlamaCppClient``  – Direct Python binding via ``llama-cpp-python``.
* ``VLLMClient``      – OpenAI-compatible HTTP API served by vLLM.
* ``TransformersClient`` – HuggingFace Transformers pipeline (CPU / GPU).

All clients implement the ``LocalLLMClient`` abstract base class, exposing a
single ``generate(prompt, **kwargs) -> str`` method so the rest of NOC
doesn't need to know which backend is in use.

A ``LocalLLMRouter`` decides, for each incoming task, whether to use a local
model or delegate to the backend LLM based on configurable heuristics.
"""

from __future__ import annotations

import abc
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations and configuration
# ---------------------------------------------------------------------------

class LocalLLMBackend(Enum):
    OLLAMA = auto()
    LLAMACPP = auto()
    VLLM = auto()
    TRANSFORMERS = auto()


class TaskCategory(Enum):
    """Categories of subtask that can be delegated to a local LLM."""
    SUMMARIZE = auto()          # Summarise / compress a subcontext
    PRUNE_SCORE = auto()        # Score sentences for pruning
    CLASSIFY = auto()           # Classify request type or route
    REPLAN_SIMPLE = auto()      # Re-plan when the change is incremental
    EMBED = auto()              # Generate text embeddings
    GENERAL = auto()            # Fallback – route to backend


@dataclass
class LocalLLMConfig:
    """Configuration for one local LLM backend instance."""

    backend: LocalLLMBackend = LocalLLMBackend.OLLAMA
    model_name: str = "llama3.2:3b"
    base_url: str = "http://localhost:11434"   # Ollama / vLLM API base
    model_path: str = ""                       # llama.cpp / Transformers model file path
    context_length: int = 4096
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    timeout_seconds: float = 30.0
    # Optional extra kwargs forwarded to the backend
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class LocalLLMClient(abc.ABC):
    """Abstract interface for all local LLM backends."""

    def __init__(self, config: LocalLLMConfig) -> None:
        self.config = config

    @abc.abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate text given *prompt*.

        Subclasses should respect ``config.max_new_tokens``, ``config.temperature``,
        and ``config.timeout_seconds``.  Additional keyword arguments override
        config defaults for this specific call.

        Raises
        ------
        LocalLLMError  on any backend failure.
        """

    def is_available(self) -> bool:
        """Return True if the backend is reachable / loaded."""
        try:
            self.generate("ping", max_new_tokens=1)
            return True
        except Exception:
            return False

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        """Allow using the client as a plain callable (e.g. for compressor)."""
        return self.generate(prompt, **kwargs)


class LocalLLMError(RuntimeError):
    """Raised when a local LLM backend call fails."""


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

class OllamaClient(LocalLLMClient):
    """
    Client for the Ollama local LLM server (ollama.ai).

    Requires Ollama to be running (``ollama serve``) and the model to be
    pulled (``ollama pull llama3.2:3b``).

    Only depends on the stdlib ``urllib`` so it works without ``requests``
    installed, but will use ``requests`` if available for better error messages.
    """

    def generate(self, prompt: str, **kwargs: Any) -> str:
        max_tokens = kwargs.get("max_new_tokens", self.config.max_new_tokens)
        temperature = kwargs.get("temperature", self.config.temperature)
        url = f"{self.config.base_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": self.config.top_p,
                **self.config.extra_kwargs,
            },
        }

        try:
            import requests  # type: ignore
            resp = requests.post(
                url,
                json=payload,
                timeout=self.config.timeout_seconds,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
        except ImportError:
            # Fall back to urllib
            import urllib.request
            import urllib.error

            data_bytes = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data_bytes,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    req, timeout=self.config.timeout_seconds
                ) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    return body.get("response", "")
            except urllib.error.URLError as exc:
                raise LocalLLMError(f"Ollama request failed: {exc}") from exc
        except Exception as exc:
            raise LocalLLMError(f"Ollama request failed: {exc}") from exc

    def list_models(self) -> List[str]:
        """Return the list of models available on this Ollama instance."""
        url = f"{self.config.base_url.rstrip('/')}/api/tags"
        try:
            import requests
            resp = requests.get(url, timeout=10.0)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []


# ---------------------------------------------------------------------------
# llama.cpp backend
# ---------------------------------------------------------------------------

class LlamaCppClient(LocalLLMClient):
    """
    Client wrapping ``llama-cpp-python``.

    Install: ``pip install llama-cpp-python``
    """

    def __init__(self, config: LocalLLMConfig) -> None:
        super().__init__(config)
        self._llm: Any = None  # Lazy load

    def _load(self) -> None:
        if self._llm is not None:
            return
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as exc:
            raise LocalLLMError(
                "llama-cpp-python is not installed.  Run: pip install llama-cpp-python"
            ) from exc
        if not self.config.model_path:
            raise LocalLLMError("model_path must be set for LlamaCppClient")
        self._llm = Llama(
            model_path=self.config.model_path,
            n_ctx=self.config.context_length,
            verbose=False,
            **self.config.extra_kwargs,
        )

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self._load()
        max_tokens = kwargs.get("max_new_tokens", self.config.max_new_tokens)
        temperature = kwargs.get("temperature", self.config.temperature)
        try:
            output = self._llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=self.config.top_p,
                echo=False,
            )
            return output["choices"][0]["text"]
        except Exception as exc:
            raise LocalLLMError(f"llama.cpp generation failed: {exc}") from exc


# ---------------------------------------------------------------------------
# vLLM backend (OpenAI-compatible API)
# ---------------------------------------------------------------------------

class VLLMClient(LocalLLMClient):
    """
    Client for a vLLM instance exposing the OpenAI-compatible ``/v1/completions``
    or ``/v1/chat/completions`` endpoint.

    Requires ``requests``.
    """

    def generate(self, prompt: str, **kwargs: Any) -> str:
        max_tokens = kwargs.get("max_new_tokens", self.config.max_new_tokens)
        temperature = kwargs.get("temperature", self.config.temperature)
        url = f"{self.config.base_url.rstrip('/')}/v1/completions"
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": self.config.top_p,
        }
        try:
            import requests
            resp = requests.post(
                url,
                json=payload,
                timeout=self.config.timeout_seconds,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["text"]
        except ImportError:
            raise LocalLLMError("requests is required for VLLMClient.  pip install requests")
        except Exception as exc:
            raise LocalLLMError(f"vLLM request failed: {exc}") from exc


# ---------------------------------------------------------------------------
# HuggingFace Transformers backend
# ---------------------------------------------------------------------------

class TransformersClient(LocalLLMClient):
    """
    Client using a HuggingFace Transformers text-generation pipeline.

    Install: ``pip install transformers torch``
    """

    def __init__(self, config: LocalLLMConfig) -> None:
        super().__init__(config)
        self._pipeline: Any = None  # Lazy load

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline  # type: ignore
        except ImportError as exc:
            raise LocalLLMError(
                "transformers is not installed.  Run: pip install transformers torch"
            ) from exc
        model = self.config.model_path or self.config.model_name
        self._pipeline = hf_pipeline(
            "text-generation",
            model=model,
            **self.config.extra_kwargs,
        )

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self._load()
        max_tokens = kwargs.get("max_new_tokens", self.config.max_new_tokens)
        temperature = kwargs.get("temperature", self.config.temperature)
        try:
            outputs = self._pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )
            generated = outputs[0]["generated_text"]
            # Strip the original prompt from the output
            if generated.startswith(prompt):
                generated = generated[len(prompt):]
            return generated.strip()
        except Exception as exc:
            raise LocalLLMError(f"Transformers generation failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_local_llm(config: LocalLLMConfig) -> LocalLLMClient:
    """Instantiate the correct LocalLLMClient based on *config.backend*."""
    backend_map = {
        LocalLLMBackend.OLLAMA: OllamaClient,
        LocalLLMBackend.LLAMACPP: LlamaCppClient,
        LocalLLMBackend.VLLM: VLLMClient,
        LocalLLMBackend.TRANSFORMERS: TransformersClient,
    }
    cls = backend_map.get(config.backend)
    if cls is None:
        raise ValueError(f"Unknown backend: {config.backend}")
    return cls(config)


# ---------------------------------------------------------------------------
# Router: local vs backend LLM
# ---------------------------------------------------------------------------

@dataclass
class RoutingPolicy:
    """Rules for deciding whether to use a local LLM or the backend LLM."""

    # Tasks always handled by the local LLM (if available)
    local_tasks: List[TaskCategory] = field(
        default_factory=lambda: [
            TaskCategory.SUMMARIZE,
            TaskCategory.PRUNE_SCORE,
            TaskCategory.CLASSIFY,
        ]
    )
    # Tasks always escalated to the backend LLM
    backend_tasks: List[TaskCategory] = field(
        default_factory=lambda: [TaskCategory.GENERAL]
    )
    # If local LLM is unavailable, fall back to backend for everything
    fallback_to_backend: bool = True
    # Maximum tokens in the prompt before we prefer the backend (larger context)
    local_max_prompt_tokens: int = 2000


class LocalLLMRouter:
    """
    Decides, per task, whether to use a local model or the backend LLM.

    Parameters
    ----------
    local_client:    A LocalLLMClient instance (or None if none is configured).
    backend_client:  A callable(prompt) -> str for the backend LLM.
    policy:          Routing rules.
    """

    def __init__(
        self,
        local_client: Optional[LocalLLMClient],
        backend_client: Callable[[str], str],
        policy: Optional[RoutingPolicy] = None,
    ) -> None:
        self.local_client = local_client
        self.backend_client = backend_client
        self.policy = policy or RoutingPolicy()

        # Check availability once at startup
        self._local_available: Optional[bool] = None

    @property
    def local_available(self) -> bool:
        if self._local_available is None:
            if self.local_client is None:
                self._local_available = False
            else:
                self._local_available = self.local_client.is_available()
        return self._local_available

    def route(
        self,
        prompt: str,
        task: TaskCategory = TaskCategory.GENERAL,
        prompt_tokens: int = 0,
    ) -> str:
        """
        Generate text for *prompt*, routing to the appropriate LLM.

        Returns the generated string.
        """
        use_local = self._should_use_local(task, prompt_tokens)

        if use_local and self.local_client is not None:
            try:
                logger.debug("Routing task=%s to local LLM", task.name)
                return self.local_client.generate(prompt)
            except LocalLLMError as exc:
                logger.warning(
                    "Local LLM failed for task=%s (%s); falling back to backend",
                    task.name,
                    exc,
                )
                self._local_available = False
                if not self.policy.fallback_to_backend:
                    raise

        logger.debug("Routing task=%s to backend LLM", task.name)
        return self.backend_client(prompt)

    def _should_use_local(self, task: TaskCategory, prompt_tokens: int) -> bool:
        if not self.local_available:
            return False
        if task in self.policy.backend_tasks:
            return False
        if task in self.policy.local_tasks:
            # Still check prompt size
            if (
                prompt_tokens > 0
                and prompt_tokens > self.policy.local_max_prompt_tokens
            ):
                return False
            return True
        # Default: use backend
        return False

    def reset_availability_cache(self) -> None:
        """Force re-check of local LLM availability on the next call."""
        self._local_available = None
