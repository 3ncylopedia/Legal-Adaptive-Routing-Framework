"""
Saint Louis University : Team 404FoundUs
@file_ legal_generation.py
@project_ LLM Legal Adaptive Routing Framework
@desc_ Generates responses using the specific LLM (Casual vs. General vs. Reasoning) based on classification.
@deps_ src.adaptive_routing.core.engine, src.adaptive_routing.config
"""

import logging
from src.adaptive_routing.core.engine import LLMRequestEngine
from src.adaptive_routing.config import FrameworkConfig

logger = logging.getLogger(__name__)

class LegalGenerator:
    """
    @class_ LegalGenerator
    @desc_ Handles response generation by dispatching to the appropriate LLM engine based on route.
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

    # ------------------------------------------------------------------
    # Internal Helper
    # ------------------------------------------------------------------

    def _build_messages_with_system_(self, messages: list, system_prompt: str) -> list:
        """
        @func_ _build_messages_with_system_
        @desc_ Prepend a system message to conversation history if one is not already present.
               This ensures every API call has a system instruction anchoring the persona/role.
               If a system message already exists at index 0, it is replaced with the correct one.
        @return_ list : Messages list with system prompt guaranteed at index 0.
        """
        if messages and messages[0].get("role") == "system":
            # Replace stale/incorrect system message
            return [{"role": "system", "content": system_prompt}] + messages[1:]
        # Prepend fresh system message
        return [{"role": "system", "content": system_prompt}] + messages

    # ------------------------------------------------------------------
    # Public Dispatch Methods
    # ------------------------------------------------------------------

    def _dispatch_(self, query: str, route: str) -> str:
        """
        @func_ _dispatch_ (@params query, route)
        @desc_ Single-turn generation. Routes to the appropriate engine.
        @params query : (str) The user query.
        @params route : (str) "Casual-LLM", "General-LLM", or "Reasoning-LLM".
        @return_ str : The LLM response.
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
        @func_ _dispatch_conversation_ (@params messages, route)
        @desc_ Multi-turn generation. Always injects the correct system prompt before dispatching.
               Without this, the API receives raw history with no role/persona context, causing
               empty, None, or off-topic responses.
        @params messages : (list[dict]) Full conversation history [{role, content}, ...].
        @params route : (str) "Casual-LLM", "General-LLM", or "Reasoning-LLM".
        @return_ str : The LLM response, or None if messages list is empty.
        """
        if not messages:
            logger.error("_dispatch_conversation_ called with an empty messages list. Skipping API call.")
            return None

        if route == "Casual-LLM":
            system_prompt = FrameworkConfig._CASUAL_INSTRUCTIONS
            full_messages = self._build_messages_with_system_(messages, system_prompt)
            logger.debug(f"[Casual] Dispatching conversation with {len(full_messages)} messages (incl. system).")
            return self._casual_engine._get_chat_completion_(full_messages)
        elif route == "Reasoning-LLM":
            system_prompt = FrameworkConfig._REASONING_INSTRUCTIONS
            full_messages = self._build_messages_with_system_(messages, system_prompt)
            logger.debug(f"[Reasoning] Dispatching conversation with {len(full_messages)} messages (incl. system).")
            return self._reasoning_engine._get_chat_completion_(full_messages)
        else:
            system_prompt = FrameworkConfig._GENERAL_INSTRUCTIONS
            full_messages = self._build_messages_with_system_(messages, system_prompt)
            logger.debug(f"[General] Dispatching conversation with {len(full_messages)} messages (incl. system).")
            return self._general_engine._get_chat_completion_(full_messages)
