from __future__ import annotations

import datetime
import uuid

_UTC = datetime.timezone.utc
from typing import Any

from langchain_core.documents import Document

from research_agent.config import ResearchAgentConfig


class LTMStore:
    """ChromaDB-backed long-term memory store.

    Wraps langchain_chroma.Chroma with a clean interface so the
    underlying vector store can be swapped (FAISS, Qdrant, pgvector)
    by subclassing and replacing retrieve/store/delete.
    """

    def __init__(self, config: ResearchAgentConfig, embeddings: Any) -> None:
        from langchain_chroma import Chroma
        self._config = config
        self._store = Chroma(
            collection_name=config.ltm_collection_name,
            embedding_function=embeddings,
            persist_directory=config.ltm_persist_directory,
        )

    @classmethod
    def from_config(cls, config: ResearchAgentConfig) -> "LTMStore":
        from research_agent.utils.embeddings import get_embeddings_model
        return cls(config, get_embeddings_model(config))

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        k = k or self._config.ltm_top_k
        results = self._store.similarity_search_with_relevance_scores(query, k=k)
        threshold = self._config.ltm_relevance_threshold
        return [doc for doc, score in results if score >= threshold]

    def store(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        doc_id = str(uuid.uuid4())
        meta = {
            "stored_at": datetime.datetime.now(_UTC).isoformat(),
            "source": "research_agent",
            **(metadata or {}),
        }
        self._store.add_texts(texts=[content], metadatas=[meta], ids=[doc_id])
        return doc_id

    def delete(self, doc_id: str) -> None:
        self._store.delete(ids=[doc_id])

    def collection_name(self) -> str:
        return self._config.ltm_collection_name

    def format_docs_for_prompt(self, docs: list[Document]) -> str:
        if not docs:
            return "(no relevant prior knowledge found)"
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[LTM {i}] ({source}): {doc.page_content}")
        return "\n".join(parts)
