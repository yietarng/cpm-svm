from __future__ import annotations

from langchain_core.tools import tool


@tool
def read_stm(state_key: str) -> str:
    """Read a value from short-term memory by key name.
    Returns a string representation of the value, or empty string if not found.
    """
    # STM is the LangGraph state. At runtime the node injects the
    # current state dict via a closure; this stub documents the interface.
    return f"(STM key '{state_key}' is accessible via the graph state)"


@tool
def write_stm(key: str, value: str) -> str:
    """Write a value to short-term memory.
    Returns confirmation. The actual write is applied by the node that calls this tool.
    """
    return f"STM updated: {key} = {value[:100]}..."
