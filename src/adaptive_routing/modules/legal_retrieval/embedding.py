## Saint Louis University
## Team 404FoundUs
## @file src/adaptive_routing/modules/legal_retrieval/embedding.py
## @project_ LLM Legal Adaptive Routing Framework
## @desc_ Manages document embeddings via OpenRouter and FAISS vector index for legal RAG.
## @deps requests, json, numpy, faiss, re, logging, rank_bm25, src.adaptive_routing.config, src.adaptive_routing.core.exceptions

import json
import re
import numpy as np
import faiss
import logging
from rank_bm25 import BM25Okapi
from src.adaptive_routing.core.engine import LLMRequestEngine
from src.adaptive_routing.config import FrameworkConfig
from src.adaptive_routing.core.exceptions import (
    AuthenticationError,
    APIConnectionError,
    APIResponseError,
    InvalidInputError
)

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """
    @class EmbeddingManager
    @desc_ Handles document chunking, embedding generation, and FAISS index management.
    @attr_ _api_key : (str) OpenRouter API credential.
    @attr_ _model : (str) Embedding model identifier.
    @attr_ _chunk_size : (int) Max characters per chunk.
    @attr_ _chunk_overlap : (int) Overlap between chunks.
    @attr_ _index : (faiss.IndexFlatL2) The FAISS vector index.
    @attr_ _chunks : (list) Stored text chunks and metadata.
    """
    def __init__(self, api_key=None, model=None, chunk_size=None, chunk_overlap=None):
        ## @logic_ Resolve API key and configuration
        self._api_key = api_key or FrameworkConfig._API_KEY
        if not self._api_key:
            raise AuthenticationError("API Key is missing.")

        self._model = model or FrameworkConfig._RETRIEVAL_MODEL
        self._chunk_size = chunk_size if chunk_size is not None else FrameworkConfig._RETRIEVAL_CHUNK_SIZE
        self._chunk_overlap = chunk_overlap if chunk_overlap is not None else FrameworkConfig._RETRIEVAL_CHUNK_OVERLAP
        
        self._engine = LLMRequestEngine(api_key=self._api_key, model=self._model)
        self._engine._url = "https://openrouter.ai/api/v1/embeddings"

        self._index = None
        self._chunks = []
        self._dimension = None
        self._bm25 = None

    def _chunk_text_(self, text: str) -> list:
        """
        @func_ _chunk_text_
        @params text : (str) Raw document text.
        @returns (list) List of text chunks split at sentence boundaries.
        @desc_ Splits a document into overlapping chunks at sentence boundaries.
        """
        if not text or not text.strip():
            raise InvalidInputError("Cannot chunk empty text.")

        ## @logic_ Split text into sentences
        sentences = re.split(r'(?<=[.!?;])\s+', text)
        
        if len(sentences) <= 1 and len(text) > self._chunk_size:
            sentences = re.split(r'\n\s*\n', text)
        
        if len(sentences) <= 1 and len(text) > self._chunk_size:
            sentences = [text[i:i+500] for i in range(0, len(text), 500)]

        chunks = []
        current_chunk = ""

        ## @iter_ sentences : Building chunks from tokenized sentences
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self._chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                if self._chunk_overlap > 0:
                    current_chunk = current_chunk[-self._chunk_overlap:] + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk = (current_chunk + " " + sentence) if current_chunk else sentence

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _get_embeddings_(self, texts: list) -> np.ndarray:
        """
        @func_ _get_embeddings_
        @params texts : (list) Strings to embed.
        @returns (np.ndarray) Matrix of embeddings.
        @desc_ Calls OpenRouter /embeddings endpoint.
        """
        payload = {"model": self._model, "input": texts}
        response_json = self._engine._call_api_(payload=payload, timeout=FrameworkConfig._EMBEDDING_TIMEOUT)

        if "data" not in response_json or len(response_json["data"]) == 0:
            raise APIResponseError("Invalid embedding response.")

        sorted_data = sorted(response_json["data"], key=lambda x: x["index"])
        return np.array([item["embedding"] for item in sorted_data], dtype=np.float32)

    def _add_documents_(self, documents: list, bypass_chunking: bool = False):
        """
        @func_ _add_documents_
        @params documents : (list) Raw document texts or dicts.
        @params bypass_chunking : (bool) Whether to skip splitting.
        @desc_ Embeds and indexes documents into FAISS.
        """
        all_chunks = []
        chunk_metadatas = []
        ## @iter_ documents : Processing each document for indexing
        for doc in documents:
            if isinstance(doc, dict):
                text = doc.get("content", "")
                meta = doc.get("metadata", {})
            else:
                text = doc
                meta = {}
                
            meta_copy = meta.copy()
            meta_copy["parent_context"] = text
                
            if bypass_chunking:
                chunks = [text]
            else:
                original_size = self._chunk_size
                original_overlap = self._chunk_overlap
                self._chunk_size = min(self._chunk_size, 1500)
                self._chunk_overlap = min(self._chunk_overlap, 150)
                chunks = self._chunk_text_(text)
                self._chunk_size = original_size
                self._chunk_overlap = original_overlap
                
            all_chunks.extend(chunks)
            chunk_metadatas.extend([meta_copy] * len(chunks))

        if not all_chunks:
            raise InvalidInputError("No chunks generated.")

        ## @logic_ Batch requests to avoid API limits
        batch_size = 100
        all_embeddings = []
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            batch_emb = self._get_embeddings_(batch)
            all_embeddings.append(batch_emb)
            
        embeddings = np.vstack(all_embeddings)

        if self._index is None:
            self._dimension = embeddings.shape[1]
            self._index = faiss.IndexFlatL2(self._dimension)

        self._index.add(embeddings)
        self._chunks.extend([{"text": c, "metadata": m} for c, m in zip(all_chunks, chunk_metadatas)])
        self._init_bm25_()

    def _init_bm25_(self):
        """
        @func_ _init_bm25_
        @desc_ Initializes the BM25 model for hybrid search.
        """
        if not self._chunks:
            self._bm25 = None
            return
            
        tokenized_corpus = []
        for chunk_data in self._chunks:
            text = chunk_data.get("text", "") if isinstance(chunk_data, dict) else str(chunk_data)
            tokenized_corpus.append(text.lower().split(" "))
            
        self._bm25 = BM25Okapi(tokenized_corpus)

    def _search_(self, query: str, top_k: int = None) -> list:
        """
        @func_ _search_
        @params query : (str) The search query.
        @params top_k : (int, optional) Number of results.
        @returns (list) Ranked results using RRF.
        @desc_ Hybrid vector + BM25 search.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        top_k = top_k if top_k is not None else FrameworkConfig._RETRIEVAL_TOP_K
        top_k = min(top_k, self._index.ntotal)

        ## @logic_ Vector Search
        query_embedding = self._get_embeddings_([query])
        distances, indices = self._index.search(query_embedding, top_k * 2)
        
        vector_results = {}
        for i, idx in enumerate(indices[0]):
            if idx < len(self._chunks):
                vector_results[idx] = 1.0 / (1.0 + float(distances[0][i]))

        ## @logic_ BM25 Search
        bm25_results = {}
        if self._bm25:
            tokenized_query = query.lower().split(" ")
            bm25_scores = self._bm25.get_scores(tokenized_query)
            top_bm25_idx = np.argsort(bm25_scores)[::-1][:top_k * 2]
            for idx in top_bm25_idx:
                if bm25_scores[idx] > 0:
                    bm25_results[idx] = float(bm25_scores[idx])

        ## @logic_ Reciprocal Rank Fusion
        ranked_vector = {idx: r for r, (idx, _) in enumerate(sorted(vector_results.items(), key=lambda x: x[1], reverse=True), 1)}
        ranked_bm25 = {idx: r for r, (idx, _) in enumerate(sorted(bm25_results.items(), key=lambda x: x[1], reverse=True), 1)}
        
        combined_scores = {}
        all_indices = set(ranked_vector.keys()).union(set(ranked_bm25.keys()))
        k_rrf = 60
        for idx in all_indices:
            score = 0.0
            if idx in ranked_vector: score += 1.0 / (k_rrf + ranked_vector[idx])
            if idx in ranked_bm25: score += 1.0 / (k_rrf + ranked_bm25[idx])
            combined_scores[idx] = score

        top_final_idx = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)[:top_k]
        results = []
        for idx in top_final_idx:
            chunk_data = self._chunks[idx]
            results.append({
                "chunk": chunk_data["text"] if isinstance(chunk_data, dict) else chunk_data,
                "metadata": chunk_data.get("metadata", {}) if isinstance(chunk_data, dict) else {},
                "score": combined_scores[idx]
            })
        return results

    def _save_index_(self, index_path: str, chunks_path: str):
        """
        @func_ _save_index_
        @params index_path : (str) Path to FAISS file.
        @params chunks_path : (str) Path to JSON metadata.
        @desc_ Persists index and chunks to disk.
        """
        if self._index is None:
            raise InvalidInputError("No index to save.")
        faiss.write_index(self._index, index_path)
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(self._chunks, f, ensure_ascii=False)

    def _load_index_(self, index_path: str, chunks_path: str):
        """
        @func_ _load_index_
        @params index_path : (str) Path of FAISS index.
        @params chunks_path : (str) Path of metadata.
        @desc_ Loads index and chunks from disk.
        """
        self._index = faiss.read_index(index_path)
        self._dimension = self._index.d
        with open(chunks_path, "r", encoding="utf-8") as f:
            self._chunks = json.load(f)
        self._init_bm25_()
