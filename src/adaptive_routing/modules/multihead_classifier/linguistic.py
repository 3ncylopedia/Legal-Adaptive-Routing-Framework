"""
Saint Louis University : Team 404FoundUs
@file_ linguistic.py
@project_ LLM Legal Adaptive Routing Framework
@desc_ Hardened module for transforming Tagalog/Taglish into standardized English.
@deps_ src.adaptive_routing.core.engine
"""

from src.adaptive_routing.core.engine import LLMRequestEngine
from src.adaptive_routing.config import FrameworkConfig

class LinguisticNormalizer:
    """
    @class_ LinguisticNormalizer
    @desc_ Hardened module for transforming Tagalog/Taglish into standardized English.
    @attr_ _handler : (LLMRequestEngine) Interaction engine for AI requests.
    @attr_ _instruction : (str) System prompt for normalization.
    """
    def __init__(self, handler: LLMRequestEngine):
        self._handler = handler
        self._instruction = FrameworkConfig._TRIAGE_INSTRUCTIONS

    def _normalize_text_(self, raw_input: str, image_path: str = None) -> str:
        """
        @func_ _normalize_text_ (@params raw_input, image_path)
        @params raw_input : (str) The raw user string.
        @params image_path : (str) Optional path/URL to an image.
        @return_ str : The sanitized, English-only output with appended language tag.
        """
        ## @logic_ Using a delimiter to separate the data from the prompt to prevent injection.
        formatted_input = f"TEXT_TO_TRANSLATE: ###\n{raw_input}\n###"
        
        ## @logic_ If image, we pass it as a list to the engine
        images = [image_path] if image_path else None

        result = self._handler._get_completion_(formatted_input, self._instruction, images=images)
        
        ## @logic_ Post-processing: ensure no lingering whitespace or artifacts
        return result.strip()