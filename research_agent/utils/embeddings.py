from __future__ import annotations

from research_agent.config import LLMProvider, ResearchAgentConfig


def get_embeddings_model(config: ResearchAgentConfig):
    """Return a LangChain Embeddings instance based on config."""
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=config.embedding_model)
