## Saint Louis University
## Team 404FoundUs
## @file tests/test_retrieval.py
## @project_ LLM Legal Adaptive Routing Framework
## @desc_ Demonstrates and tests the lifecycle of the LegalRetrievalModule.

import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.adaptive_routing.modules.retrieval import LegalRetrievalModule

def main():
    """
    @func_ main
    @desc_ Demonstrates building and saving the FAISS index.
    """
    print("==================================================")
    print(" STEP 1: Building and Saving the FAISS Index")
    print("==================================================")
    
    builder_module = LegalRetrievalModule()
    output_directory = "./Faiss"
    
    try:
        saved_faiss_path = builder_module.build_and_save_index(
            corpus_dir="legal-corpus/HK",
            output_dir=output_directory,
            index_prefix="hk_test_index"
        )
        print(f"Success: {saved_faiss_path}\n")
    except ValueError as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()
