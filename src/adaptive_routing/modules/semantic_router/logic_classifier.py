## Saint Louis University
## Team 404FoundUs
## @file src/adaptive_routing/modules/semantic_router/logic_classifier.py
## @project_ LLM Legal Adaptive Routing Framework
## @desc_ Semantic router for classifying legal queries into Information or Advice pathways.
## @deps src.adaptive_routing.core.engine, src.adaptive_routing.config, json, re, logging

import json
import re
import logging
from src.adaptive_routing.core.engine import LLMRequestEngine
from src.adaptive_routing.config import FrameworkConfig

logger = logging.getLogger(__name__)

class RoutingClassifier:
    """
    @class RoutingClassifier
    @desc_ Analyzes user queries to route them to Casual, General/Info, or Advice/Scenario.
    @attr_ _handler : (LLMRequestEngine) Interaction engine for AI requests.
    @attr_ _system_prompt : (str) The instruction set for the routing reasoning.
    """
    def __init__(self, api_key=None, handler=None, system_prompt=None):
        self._handler = handler or LLMRequestEngine(
            api_key=api_key,
            model=FrameworkConfig._ROUTER_MODEL,
            temperature=FrameworkConfig._ROUTER_TEMP,
            max_tokens=FrameworkConfig._ROUTER_MAX_TOKENS, 
            use_system_role=FrameworkConfig._ROUTER_USE_SYSTEM,
            include_reasoning=FrameworkConfig._ROUTER_REASONING
        )

        self._system_prompt = system_prompt or (
            "ROLE: Legal Query Router\n"
            "TASK: Analyze the USER QUERY and decide which LLM should handle it.\n"
            "\n"
            "Casual-LLM:\n"
            "- Greetings (hi, hello, good morning, kumusta)\n"
            "- Gratitude (thank you, thanks, salamat po)\n"
            "- Farewells (bye, goodbye, take care, ingat)\n"
            "- Small talk unrelated to law or legal matters\n"
            "- Single-word affirmations (ok, yes, sure, noted, sige)\n"
            "- Emotional check-ins without legal context\n"
            "- Unrelated inquiries towards migrant worker rights and legal assistance\n"
            "\n"
            "General-LLM:\n"
            "- General legal information and Government DMW/OWWA informations\n"
            "- Definitions, explanations, rights overview, Contact Details\n"
            "- Simple Q&A about law\n"
            "- Summarize Legal Findings\n"
            "- Perform Simplifications\n"
            "- Clarify complex scenarios\n"
            "- No personalized advice\n"
            "- No complex scenario or dispute\n"
            "\n"
            "Reasoning-LLM:\n"
            "- Describes a real or hypothetical situation\n"
            "- Asks what action to take\n"
            "- Involves disputes, violations, contracts, termination, abuse, or legal risk\n"
            "- Requires legal interpretation and structured reasoning\n"
            "\n"
            "Constraints:\n"
            "- Strictly adhere to the ROLE and TASK above\n"
            "- The router must return structured JSON only\n"
            "- No markdown allowed in output\n"
            "- Do NOT answer the question\n"
            "- When in doubt between Casual and Legal, choose the legal route\n"
            "\n"
            "JSON Schema:\n"
            "{\n"
            '  "route": "Casual-LLM" | "General-LLM" | "Reasoning-LLM",\n'
            '  "confidence": float,\n'
            '  "search_signals": [list of short phrases] | null\n'
            "}\n"
            "\n"
            "Signal Generation Rules (for search_signals):\n"
            "- Always include Contact Details in keyword phrases\n"
            "- If the query is a new legal inquiry: Provide 4-6 concise keyword phrases (noun phrases, ≤ 5 words each) optimized for retrieval.\n"
            "- If the query is a follow-up, clarification, summarization, or lacks new legal information: Return null.\n"
            "- Avoid verbs, questions, or full sentences.\n"
            "- Use legal/domain-relevant keywords."
        )

    def _route_query_(self, query: str) -> dict:
        """
        @func_ _route_query_
        @params query : (str) The user's input query.
        @returns (dict) The structured routing decision.
        @desc_ Calls the LLM and parses the JSON response.
        """
        ## @logic_ Snapshot current config for diagnostics
        _cfg = (
            f"[Router Config] model={FrameworkConfig._ROUTER_MODEL}, "
            f"USE_SYSTEM={FrameworkConfig._ROUTER_USE_SYSTEM}, "
            f"REASONING={FrameworkConfig._ROUTER_REASONING}"
        )

        try:
            raw_response = self._handler._get_completion_(query, self._system_prompt)

            ## @logic_ Guard: Detect empty/null responses
            if not raw_response or not str(raw_response).strip():
                error_msg = f"Router LLM returned an empty response. {_cfg}"
                logger.error(error_msg)
                return {
                    "route": None,
                    "confidence": 0.0,
                    "search_signals": ["Empty LLM Response"],
                    "error": error_msg
                }

            return self._parse_json_(raw_response)
        except Exception as e:
            error_msg = f"Routing classification failed: {e}. {_cfg}"
            logger.error(error_msg)
            return {
                "route": None,
                "confidence": 0.0,
                "search_signals": ["Routing Error", str(e)],
                "error": error_msg
            }

    def _parse_json_(self, text: str) -> dict:
        """
        @func_ _parse_json_
        @params text : (str) Raw LLM output.
        @returns (dict) Parsed JSON object.
        @desc_ Removes markdown code blocks and whitespace before parsing.
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
            
        ## @logic_ Strip markdown code blocks (```json ... ```)
        cleaned_text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r"```", "", cleaned_text)
        cleaned_text = cleaned_text.strip()
        
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            preview = repr(text[:200]) if text else "(empty)"
            error_msg = f"Failed to parse router output as JSON. Raw output: {preview}"
            logger.warning(error_msg)
            return {
                "route": None,
                "confidence": 0.0,
                "search_signals": ["JSON Parsing Failed"],
                "error": error_msg
            }
