## Saint Louis University
## Team 404FoundUs
## @file src/adaptive_routing/modules/legal_retrieval/retriever.py
## @project_ LLM Legal Adaptive Routing Framework
## @desc_ RAG retriever that queries the FAISS vector search index.
## @deps src.adaptive_routing.modules.legal_retrieval.embedding, src.adaptive_routing.config, logging

import logging
from src.adaptive_routing.modules.legal_retrieval.embedding import EmbeddingManager
from src.adaptive_routing.config import FrameworkConfig

logger = logging.getLogger(__name__)

class LegalRetriever:
    """
    @class LegalRetriever
    @desc_ Retrieves relevant legal text chunks from the vector index with quality filtering.
    @attr_ _embedding_manager : (EmbeddingManager) Handles vector search over indexed documents.
    """
    def __init__(self, embedding_manager: EmbeddingManager):
        self._embedding_manager = embedding_manager

    def _retrieve_context_(self, query: str, top_k: int = None, score_threshold: float = None, jurisdiction: str = None) -> list:
        """
        @func_ _retrieve_context_
        @params query : (str) The user's legal question.
        @params top_k : (int, optional) Number of chunks to retrieve.
        @params score_threshold : (float, optional) Minimum similarity score.
        @params jurisdiction : (str, optional) Jurisdiction filter.
        @returns (list) Filtered list of context matches.
        @desc_ Searches the FAISS index and applies relevance filtering.
        """
        search_results = self._embedding_manager._search_(query, top_k=top_k)
        
        ## @logic_ Apply relevance threshold filtering
        threshold = score_threshold if score_threshold is not None else FrameworkConfig._RETRIEVAL_SCORE_THRESHOLD
        
        if threshold > 0.0:
            if search_results and max(r["score"] for r in search_results) < 0.1 and threshold >= 0.1:
                logger.info("RRF scoring detected. Bypassing cosine threshold.")
            else:
                before_count = len(search_results)
                search_results = [r for r in search_results if r["score"] >= threshold]
                filtered_count = before_count - len(search_results)
                if filtered_count > 0:
                    logger.info(f"Filtered {filtered_count} results below threshold.")

        ## @logic_ Apply jurisdiction metadata filter
        if jurisdiction:
            search_results = [
                r for r in search_results 
                if r.get("metadata", {}).get("jurisdiction", "").upper() == jurisdiction.upper()
            ]

        ## @logic_ Deduplicate and inject parent context
        unique_parents = set()
        final_results = []
        ## @iter_ search_results : Processing matches for parent-child deduplication
        for r in search_results:
            parent = r.get("metadata", {}).get("parent_context")
            if parent:
                if parent not in unique_parents:
                    unique_parents.add(parent)
                    r["chunk"] = parent
                    final_results.append(r)
            else:
                final_results.append(r)

        return final_results
