"""Retrieval and reranking tools for RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.vectorstores import VectorStoreRetriever
from sentence_transformers import CrossEncoder

from src.config import RetrievalConfig


@dataclass
class RetrievalDeps:
    """Dependencies for retrieval operations.
    
    Attributes:
        retriever: Vector store retriever for similarity search.
        reranker: Cross-encoder model for reranking results.
        config: Retrieval configuration.
    """
    retriever: VectorStoreRetriever
    reranker: CrossEncoder
    config: RetrievalConfig


async def retrieve_and_rerank(
    deps: RetrievalDeps, search_query: str
) -> list[str]:
    """Retrieve and rerank documents for a search query.
    
    Args:
        deps: Retrieval dependencies.
        search_query: The search query string.
        
    Returns:
        List of reranked document texts sorted by relevance score.
    """
    # Retrieve initial results
    retrieved_results = await deps.retriever.ainvoke(search_query)

    # Prepare input for reranker
    reranker_input = [
        (search_query, doc.page_content) for doc in retrieved_results
    ]

    # Rerank documents
    reranker_scores = deps.reranker.predict(reranker_input)

    # Filter by threshold and sort
    ranked_docs = [
        (doc.page_content, score)
        for doc, score in zip(retrieved_results, reranker_scores)
        if score > deps.config.reranker_threshold
    ]
    ranked_docs.sort(key=lambda x: x[1], reverse=True)

    # Return only the document texts
    return [doc_text for doc_text, _ in ranked_docs]


def create_retrieval_deps(
    retriever: VectorStoreRetriever, config: RetrievalConfig
) -> RetrievalDeps:
    """Create retrieval dependencies.
    
    Args:
        retriever: Vector store retriever.
        config: Retrieval configuration.
        
    Returns:
        Configured RetrievalDeps instance.
    """
    reranker = CrossEncoder(config.reranker_model)
    return RetrievalDeps(retriever=retriever, reranker=reranker, config=config)