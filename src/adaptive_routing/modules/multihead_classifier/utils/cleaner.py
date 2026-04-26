## Saint Louis University
## Team 404FoundUs
## @file src/adaptive_routing/modules/multihead_classifier/utils/cleaner.py
## @desc_ Utility functions for cleaning LLM outputs in the Triage module.

import re

def strip_llm_artifacts(text: str) -> str:
    """
    @func strip_llm_artifacts
    @params text : (str) Raw LLM output.
    @returns (str) Cleaned text with reasoning blocks and markdown artifacts removed.
    @desc_ Removes <think>...</think> blocks and other common LLM artifacts.
    """
    if not text:
        return ""
        
    ## @logic_ Remove <think>...</think> blocks (non-greedy, handles multi-line)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    ## @logic_ Remove any leading/trailing whitespace
    return text.strip()
