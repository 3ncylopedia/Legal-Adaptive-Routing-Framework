## Saint Louis University
## Team 404FoundUs
## @file src/adaptive_routing/modules/retrieval.py
## @project_ LLM Legal Adaptive Routing Framework
## @desc_ Orchestrator module that coordinates Legal RAG retrieval: embed documents and search.
## @deps src.adaptive_routing.modules.legal_retrieval.embedding, src.adaptive_routing.modules.legal_retrieval.retriever, src.adaptive_routing.config, logging

from src.adaptive_routing.modules.legal_retrieval.embedding import EmbeddingManager
from src.adaptive_routing.modules.legal_retrieval.retriever import LegalRetriever
from src.adaptive_routing.config import FrameworkConfig
from src.adaptive_routing.modules.legal_retrieval.utils import legal_indexing
import os
import json
import logging

logger = logging.getLogger(__name__)

class LegalRetrievalModule:
    """
    @class LegalRetrievalModule
    @desc_ Facade/Orchestrator that manages the RAG pipeline: Ingest -> Search.
    @attr_ _embedding_manager : (EmbeddingManager) Component for document indexing and vector search.
    @attr_ _retriever : (LegalRetriever) Component that queries the index for relevant context.
    """
    def __init__(self, api_key=None, embedding_manager=None, retriever=None, index_path=None, chunks_path=None):
        ## @logic_ Initialize embedding manager with Retrieval-specific configuration if not provided
        self._embedding_manager = embedding_manager or EmbeddingManager(
            api_key=api_key,
            model=FrameworkConfig._RETRIEVAL_MODEL,
            chunk_size=FrameworkConfig._RETRIEVAL_CHUNK_SIZE,
            chunk_overlap=FrameworkConfig._RETRIEVAL_CHUNK_OVERLAP
        )

        ## @logic_ Initialize retriever with filtering capabilities
        self._retriever = retriever or LegalRetriever(self._embedding_manager)
        
        ## @logic_ Auto-load FAISS index if specified in settings
        target_index = index_path or FrameworkConfig._RETRIEVAL_INDEX_PATH
        target_chunks = chunks_path or FrameworkConfig._RETRIEVAL_CHUNKS_PATH
        
        if target_index and target_chunks:
            if os.path.exists(target_index) and os.path.exists(target_chunks):
                self._load_index_(target_index, target_chunks)
            else:
                logger.warning(f"Index or chunk file not found at {target_index} / {target_chunks}.")

    def _ingest_documents_(self, documents: list):
        """
        @func_ _ingest_documents_
        @params documents : (list[str]) Raw legal document texts to add.
        @returns None
        @desc_ Embeds and indexes the provided documents into the FAISS vector store.
        """
        self._embedding_manager._add_documents_(documents, bypass_chunking=True)

    def _process_retrieval_(self, query: str, signals: list = None, top_k: int = None) -> dict:
        """
        @func_ _process_retrieval_
        @params query : (str) The user's legal question.
        @params signals : (list, optional) A list of keyword phrases from the Semantic Router.
        @params top_k : (int, optional) Number of context chunks to retrieve.
        @returns (dict) Contains 'query', 'retrieved_chunks', and 'combined_query'.
        @desc_ Main entry point — retrieves relevant context chunks from the index.
        """
        ## @logic_ Combine original query with search signals for enhanced retrieval
        search_query = query
        if signals and isinstance(signals, list):
            valid_signals = [str(s).strip() for s in signals if s]
            if valid_signals:
                search_query = f"{query} {' '.join(valid_signals)}"
        
        retrieved_chunks = self._retriever._retrieve_context_(search_query, top_k=top_k)
        
        return {
            "query": query,
            "combined_query": search_query,
            "retrieved_chunks": retrieved_chunks
        }

    def _save_index_(self, index_path: str, chunks_path: str):
        """
        @func_ _save_index_
        @params index_path : (str) File path for the FAISS index binary.
        @params chunks_path : (str) File path for the chunk metadata JSON.
        @desc_ Persists the current FAISS index and text chunks to disk.
        """
        self._embedding_manager._save_index_(index_path, chunks_path)

    def _load_index_(self, index_path: str, chunks_path: str):
        """
        @func_ _load_index_
        @params index_path : (str) File path of the saved FAISS index.
        @params chunks_path : (str) File path of the saved chunk metadata JSON.
        @desc_ Loads a previously persisted index and chunks from disk.
        """
        self._embedding_manager._load_index_(index_path, chunks_path)

    def build_and_save_index(self, corpus_dir: str, output_dir: str, index_prefix: str) -> str:
        """
        @func_ build_and_save_index
        @params corpus_dir : (str) Path to the directory containing JSON corpus files.
        @params output_dir : (str) Directory where index will be saved.
        @params index_prefix : (str) Prefix for the output files.
        @returns (str) Path to the created FAISS index file.
        @desc_ Utility function that delegates to rebuild_index to crawl and persist a FAISS store.
        """
        return legal_indexing.rebuild_index(
            corpus_dir=corpus_dir,
            output_dir=output_dir,
            index_prefix=index_prefix
        )
