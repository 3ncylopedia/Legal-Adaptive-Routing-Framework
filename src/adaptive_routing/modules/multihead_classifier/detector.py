## Saint Louis University
## Team 404FoundUs
## @file src/adaptive_routing/modules/multihead_classifier/detector.py
## @project_ LLM Legal Adaptive Routing Framework
## @desc_ Module for storing state: original prompt, detected language, and normalized text.

class LanguageStateDetector:
    """
    @class LanguageStateDetector
    @desc_ Stores the state of linguistic processing and persists RAG context for turns.
    @attr_ _original_prompt : (str) The raw input from the user.
    @attr_ _detected_language : (str) The identified language tag.
    @attr_ _normalized_text : (str) The sanitized English text.
    @attr_ _intent : (str) The classified intent/route of the query.
    @attr_ _last_rag_context : (list) The raw chunks retrieved from the last RAG call.
    """
    def __init__(self):
        self._original_prompt = None
        self._detected_language = None
        self._normalized_text = None
        self._intent = None
        self._last_rag_context = []

    def _update_state_(self, original: str, normalized: str, language: str, intent: str = None, context: list = None):
        """
        @func_ _update_state_
        @params original : (str) The raw input.
        @params normalized : (str) The processed English text.
        @params language : (str) The detected language tag.
        @params intent : (str, optional) The classified route.
        @params context : (list, optional) The list of retrieved RAG chunks.
        @returns None
        @desc_ Updates the internal state with results from a processing cycle.
        """
        self._original_prompt = original
        self._normalized_text = normalized
        self._detected_language = language
        
        if intent is not None:
            self._intent = intent
        if context is not None:
            self._last_rag_context = context

    def _get_state_(self):
        """
        @func_ _get_state_
        @returns (dict) A dictionary containing the current state.
        @desc_ Returns the full captured state.
        """
        return {
            "original_prompt": self._original_prompt,
            "detected_language": self._detected_language,
            "normalized_text": self._normalized_text,
            "intent": self._intent,
            "last_rag_context": self._last_rag_context
        }
