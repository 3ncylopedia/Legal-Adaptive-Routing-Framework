## Saint Louis University
## Team 404FoundUs
## @file src/adaptive_routing/modules/semantic_router/legal_generation.py
## @project_ LLM Legal Adaptive Routing Framework
## @desc_ Generates responses using specific LLMs based on classification.
## @deps src.adaptive_routing.core.engine, src.adaptive_routing.config, logging

import logging
from src.adaptive_routing.core.engine import LLMRequestEngine
from src.adaptive_routing.config import FrameworkConfig

logger = logging.getLogger(__name__)

class LegalGenerator:
    """
    @class LegalGenerator
    @desc_ Handles response generation by dispatching to the appropriate LLM engine.
    @attr_ _general_engine : (LLMRequestEngine) Engine for General pathway.
    @attr_ _reasoning_engine : (LLMRequestEngine) Engine for Reasoning pathway.
    @attr_ _casual_engine : (LLMRequestEngine) Engine for Casual pathway.
    """
    def __init__(self, api_key=None, general_engine=None, reasoning_engine=None, casual_engine=None):
        self._api_key = api_key
        
        self._general_engine = general_engine or LLMRequestEngine(
            api_key=api_key,
            model=FrameworkConfig._GENERAL_MODEL,
            temperature=FrameworkConfig._GENERAL_TEMP,
            max_tokens=FrameworkConfig._GENERAL_MAX_TOKENS,
            use_system_role=FrameworkConfig._GENERAL_USE_SYSTEM,
            include_reasoning=FrameworkConfig._GENERAL_REASONING
        )

        self._reasoning_engine = reasoning_engine or LLMRequestEngine(
            api_key=api_key,
            model=FrameworkConfig._REASONING_MODEL,
            temperature=FrameworkConfig._REASONING_TEMP,
            max_tokens=FrameworkConfig._REASONING_MAX_TOKENS,
            use_system_role=FrameworkConfig._REASONING_USE_SYSTEM,
            include_reasoning=FrameworkConfig._REASONING_REASONING
        )

        self._casual_engine = casual_engine or LLMRequestEngine(
            api_key=api_key,
            model=FrameworkConfig._CASUAL_MODEL,
            temperature=FrameworkConfig._CASUAL_TEMP,
            max_tokens=FrameworkConfig._CASUAL_MAX_TOKENS,
            use_system_role=FrameworkConfig._CASUAL_USE_SYSTEM,
            include_reasoning=FrameworkConfig._CASUAL_REASONING
        )

    def _build_messages_with_system_(self, messages: list, system_prompt: str) -> list:
        """
        @func_ _build_messages_with_system_
        @params messages : (list) Conversation history.
        @params system_prompt : (str) The system instruction to inject.
        @returns (list) Messages list with system prompt at index 0.
        @desc_ Prepend a system message to conversation history.
        """
        if messages and messages[0].get("role") == "system":
            return [{"role": "system", "content": system_prompt}] + messages[1:]
        return [{"role": "system", "content": system_prompt}] + messages

    def _dispatch_(self, query: str, route: str) -> str:
        """
        @func_ _dispatch_
        @params query : (str) The user query.
        @params route : (str) Target route ("Casual-LLM", "General-LLM", "Reasoning-LLM").
        @returns (str) The LLM response.
        @desc_ Single-turn generation dispatch.
        """
        if route == "Casual-LLM":
            system_prompt = FrameworkConfig._CASUAL_INSTRUCTIONS
            return self._casual_engine._get_completion_(query, system_prompt)
        elif route == "Reasoning-LLM":
            system_prompt = FrameworkConfig._REASONING_INSTRUCTIONS
            return self._reasoning_engine._get_completion_(query, system_prompt)
        else:
            system_prompt = FrameworkConfig._GENERAL_INSTRUCTIONS
            return self._general_engine._get_completion_(query, system_prompt)

    def _dispatch_conversation_(self, messages: list, route: str) -> str:
        """
        @func_ _dispatch_conversation_
        @params messages : (list) Conversation history.
        @params route : (str) Target route.
        @returns (str) The LLM response.
        @desc_ Multi-turn generation dispatch with system prompt injection.
        """
        if not messages:
            return None

        if route == "Casual-LLM":
            system_prompt = FrameworkConfig._CASUAL_INSTRUCTIONS
            full_messages = self._build_messages_with_system_(messages, system_prompt)
            return self._casual_engine._get_chat_completion_(full_messages)
        elif route == "Reasoning-LLM":
            system_prompt = FrameworkConfig._REASONING_INSTRUCTIONS
            full_messages = self._build_messages_with_system_(messages, system_prompt)
            return self._reasoning_engine._get_chat_completion_(full_messages)
        else:
            system_prompt = FrameworkConfig._GENERAL_INSTRUCTIONS
            full_messages = self._build_messages_with_system_(messages, system_prompt)
            return self._general_engine._get_chat_completion_(full_messages)
