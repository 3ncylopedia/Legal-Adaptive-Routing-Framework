## Saint Louis University
## Team 404FoundUs
## @file src/adaptive_routing/modules/legal_retrieval/utils/legal_indexing.py
## @project_ LLM Legal Adaptive Routing Framework
## @desc_ Developer utilities for managing legal corpus ingestion and indexing.
## @deps os, json, glob, logging, src.adaptive_routing.modules.retrieval

import os
import json
import glob
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

def crawl_corpus(corpus_dir: str) -> List[str]:
    """
    @func_ crawl_corpus
    @params corpus_dir : (str) Directory to search for legal documents.
    @returns (list) Absolute paths to all .json files found.
    @desc_ Recursively discovers all JSON files within the specified directory.
    """
    if not os.path.exists(corpus_dir):
        logger.error(f"Corpus directory not found: {corpus_dir}")
        return []
        
    pattern = os.path.join(corpus_dir, "**", "*.json")
    files = glob.glob(pattern, recursive=True)
    logger.info(f"Discovered {len(files)} JSON files.")
    return files

def validate_legal_doc(data: Dict[str, Any]) -> bool:
    """
    @func_ validate_legal_doc
    @params data : (dict) Loaded JSON content.
    @returns (bool) True if valid and not repealed.
    @desc_ Ensures the document meets minimum criteria for indexing.
    """
    if not isinstance(data, dict):
        return False
    if data.get("is_repealed") is True:
        return False
    return True

def format_doc_for_indexing(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    @func_ format_doc_for_indexing
    @params data : (dict) Validated JSON content.
    @returns (dict) Formatted dictionary ready for ingestion.
    @desc_ Prepares metadata and content with smart fallbacks.
    """
    content = data.get("content")
    if not content or not str(content).strip():
        content = json.dumps(data, ensure_ascii=False, indent=2)
    
    metadata = data.get("metadata", {})
    
    return {
        "content": content,
        "metadata": {
            "jurisdiction": data.get("jurisdiction", "Information/General"),
            "title": data.get("title", metadata.get("source_file", "Untitled Dataset")),
            "category": metadata.get("corpus_category", "Developer Resource"),
            "source_file": metadata.get("source_file", "Direct Ingestion")
        }
    }

def verify_index_integrity(corpus_dir: str, chunks_path: str) -> Dict[str, Any]:
    """
    @func_ verify_index_integrity
    @params corpus_dir : (str) Path to raw JSON files.
    @params chunks_path : (str) Path to indexed metadata.
    @returns (dict) Statistics about sync status.
    @desc_ Compares files on disk with vectors in the index.
    """
    corpus_files = crawl_corpus(corpus_dir)
    valid_corpus_count = 0
    
    ## @iter_ corpus_files : Validation check for each file in corpus
    for f_path in corpus_files:
        try:
            with open(f_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if validate_legal_doc(data):
                    valid_corpus_count += 1
        except:
            continue
            
    indexed_count = 0
    if os.path.exists(chunks_path):
        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
                indexed_count = len(chunks)
        except:
            pass
            
    is_synced = valid_corpus_count == indexed_count
    
    return {
        "corpus_count": valid_corpus_count,
        "indexed_count": indexed_count,
        "is_synced": is_synced,
        "missing_count": max(0, valid_corpus_count - indexed_count)
    }

def ingest_custom_dataset(retrieval_module, raw_data_list: List[Dict[str, Any]]):
    """
    @func_ ingest_custom_dataset
    @params retrieval_module : (LegalRetrievalModule) Active module instance.
    @params raw_data_list : (list) List of raw JSON-like dicts.
    @desc_ Batch-indexes arbitrary data into the active session.
    """
    valid_docs = []
    ## @iter_ raw_data_list : Formatting each custom doc for ingestion
    for data in raw_data_list:
        if validate_legal_doc(data):
            valid_docs.append(format_doc_for_indexing(data))
            
    if valid_docs:
        retrieval_module._ingest_documents_(valid_docs)
        logger.info(f"Ingested {len(valid_docs)} custom documents.")
    else:
        logger.warning("No valid documents found.")

def rebuild_index(corpus_dir: str, output_dir: str, index_prefix: str = "combined_index"):
    """
    @func_ rebuild_index
    @params corpus_dir : (str) Root of legal corpus.
    @params output_dir : (str) Target directory for FAISS save.
    @params index_prefix : (str) Filename prefix.
    @desc_ Forces a full re-index of all datasets from scratch.
    """
    from src.adaptive_routing.modules.retrieval import LegalRetrievalModule
    
    logger.info(f"Rebuilding index from {corpus_dir}...")
    rm = LegalRetrievalModule(index_path="", chunks_path="")
    
    files = crawl_corpus(corpus_dir)
    docs_to_index = []
    
    ## @iter_ files : Loading docs for indexing
    for f_path in files:
        try:
            with open(f_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if validate_legal_doc(data):
                    docs_to_index.append(format_doc_for_indexing(data))
        except Exception as e:
            logger.error(f"Error processing {f_path}: {e}")
            
    if not docs_to_index:
        logger.error("No valid documents found.")
        return None
        
    rm._ingest_documents_(docs_to_index)
    
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, f"{index_prefix}.faiss")
    chunks_path = os.path.join(output_dir, f"{index_prefix}.json")
    
    rm._save_index_(index_path, chunks_path)
    logger.info(f"Rebuild complete: {index_path}")
    
    return index_path
