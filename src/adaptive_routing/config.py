## Saint Louis University
## Team 404FoundUs
## @file src/adaptive_routing/config.py
## @project_ LLM Legal Adaptive Routing Framework
## @desc_ Centralized configuration for OpenRouter parameters and security.
## @deps os, dotenv

import os
import logging
from dotenv import load_dotenv

# Ensure environment variables are loaded before configuration is parsed
load_dotenv()

logger = logging.getLogger(__name__)

class FrameworkConfig:
    """
    @class FrameworkConfig
    @desc_ Manages global AI hyperparameters in different modules.

    IMPORTANT: Configuration values are read at engine initialization time (snapshot).
    Call _update_settings_() BEFORE creating module instances (TriageModule, SemanticRouterModule, etc.).
    Updating settings after modules are initialized will NOT affect existing engine instances.
    """
    ## @const_ Global Defaults
    _API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    
    ## @const_ Triage Module Configuration (Linguistic Normalizer)
    _TRIAGE_MODEL = os.getenv("TRIAGE_MODEL", "qwen/qwen-turbo")
    _TRIAGE_TEMP = float(os.getenv("TRIAGE_TEMP", "0.6"))
    _TRIAGE_MAX_TOKENS = int(os.getenv("TRIAGE_MAX_TOKENS", "2000"))
    _TRIAGE_USE_SYSTEM = os.getenv("TRIAGE_USE_SYSTEM", "True").lower() == "true"
    _TRIAGE_REASONING = os.getenv("TRIAGE_REASONING", "False").lower() == "true"

    ## @const_ Semantic Router Configuration
    _ROUTER_MODEL = os.getenv("ROUTER_MODEL", "qwen/qwen-turbo")
    _ROUTER_TEMP = float(os.getenv("ROUTER_TEMP", "0.1"))
    _ROUTER_MAX_TOKENS = int(os.getenv("ROUTER_MAX_TOKENS", "250"))
    _ROUTER_USE_SYSTEM = os.getenv("ROUTER_USE_SYSTEM", "TRUE").lower() == "true"
    _ROUTER_REASONING = os.getenv("ROUTER_REASONING", "False").lower() == "true"

    ## @const_ Fallbacks (Legacy/General)
    _DEFAULT_MODEL = _TRIAGE_MODEL 
    _TEMPERATURE = 0.7
    _MAX_TOKENS = 1500
    _USE_SYSTEM_ROLE = True
    _INCLUDE_REASONING = False

    ## @const_ Network Resilience Configuration
    _REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
    _EMBEDDING_TIMEOUT = int(os.getenv("EMBEDDING_TIMEOUT", "60"))
    _RETRY_COUNT = int(os.getenv("RETRY_COUNT", "2"))
    _RETRY_BACKOFF = float(os.getenv("RETRY_BACKOFF", "1.0"))

    @classmethod
    def _update_settings_(cls, **kwargs):
        """
        @func_ _update_settings_
        @params kwargs: Dict of hyperparameter overrides.
        @logic_ Dynamically updates class attributes if they exist.
        @raises ConfigurationError: If an unrecognized key is passed (prevents silent misconfiguration).

        IMPORTANT: Must be called BEFORE module initialization. Existing engine instances
        will NOT pick up changes made after their creation.
        """
        from src.adaptive_routing.core.exceptions import ConfigurationError
        
        ## @iter_ kwargs: iterating over provided settings to update config
        for key, value in kwargs.items():
            # Support both direct casing and underscored casing
            attr_name = f"_{key.upper()}" if not key.startswith("_") else key.upper()
            if hasattr(cls, attr_name):
                setattr(cls, attr_name, value)
            else:
                raise ConfigurationError(
                    f"Unknown config key: '{key}' (resolved to '{attr_name}'). "
                    f"Valid keys include: {[a for a in dir(cls) if a.startswith('_') and a[1:2].isupper()]}"
                )

    ## @const_ General LLM Configuration (Information)
    _GENERAL_MODEL = os.getenv("GENERAL_MODEL", "qwen/qwen3-next-80b-a3b-instruct:free")
    _GENERAL_TEMP = float(os.getenv("GENERAL_TEMP", "0.5"))
    _GENERAL_MAX_TOKENS = int(os.getenv("GENERAL_MAX_TOKENS", "2500"))
    _GENERAL_USE_SYSTEM = os.getenv("GENERAL_USE_SYSTEM", "True").lower() == "true"
    _GENERAL_REASONING = os.getenv("GENERAL_REASONING", "False").lower() == "true"
    _GENERAL_INSTRUCTIONS = (
        "ROLE: Legal Information Assistant\n"
        "PERSONA: You are Atty. Agapay AI, a legal information assistant from Saint Louis University. Your SOLE purpose is to assist Philippine Migrant Workers in Hong Kong with labor law concerns.\n"
        "TASK: Provide general legal information, definitions, and explanations for Philippine and Hong Kong labor laws. DO NOT answer in legal jargon; simplify the output.\n\n"
        "STRICT GUARDRAILS FOR UNRELATED INQUIRIES:\n"
        "- **Scope**: Philippine/Hong Kong labor law and migrant worker concerns ONLY.\n"
        "- **Mixed Queries**: If the user asks a legal question AND an unrelated question (e.g., 'How do I file a claim AND how do I cook Sinigang?'), you MUST ONLY answer the legal portion. For the unrelated portion, politely state that you are an AI specialized in legal assistance and cannot provide non-legal info (like recipes, coding, or lifestyle advice).\n"
        "- **Prohibited Topics**: DO NOT provide recipes, medical advice, coding, copywriting, or unrelated trivia. If the entire query is unrelated, respond with a kind apology and redirect them to legal assistance scope.\n\n"
        "OUTPUT FORMAT (MANDATORY):\n"
        "1. **Query Overview**: Briefly restate the legal topic or question asked.\n"
        "2. **Relevant Legal Concepts**: strict citation of relevant laws, rules, or regulations (PH/HK). Define key terms.\n"
        "3. **General Explanation**: Explain how these laws generally apply. Do NOT apply to specific user facts. Use neutral, educational language.\n"
        "4. **Summary**: Provide a concise answer or definition.\n\n"
        "ADDITIONAL INSTRUCTIONS:\n"
        "1. Analyze the Injected Context Information if it can be used or referred to. If you think the Injected Context Information is not relevant, you can ignore it.\n"
        "CONSTRAINTS:\n"
        "- Do NOT provide specific legal advice or analysis of hypothetical scenarios.\n"
        "- Clearly distinguish between PH and HK jurisdictions.\n"
        "- Maintain a professional, educational tone."
    )

    ## @const_ Reasoning LLM Configuration (Advice/Scenario)
    _REASONING_MODEL = os.getenv("REASONING_MODEL", "deepseek/deepseek-chat-v3.1") # Fallback to working model
    _REASONING_TEMP = float(os.getenv("REASONING_TEMP", "0.7"))
    _REASONING_MAX_TOKENS = int(os.getenv("REASONING_MAX_TOKENS", "3000"))
    _REASONING_USE_SYSTEM = os.getenv("REASONING_USE_SYSTEM", "True").lower() == "true"
    _REASONING_REASONING = os.getenv("REASONING_REASONING", "True").lower() == "true"
    _REASONING_INSTRUCTIONS = (
        "ROLE: Legal AI Assistant (Philippine & HK Labor Law Focus)\n\n"
        "PERSONA: You are Atty. Agapay AI, a legal information assistant from Saint Louis University. Your SOLE purpose is to assist Philippine Migrant Workers in Hong Kong with labor law scenarios.\n\n"
        "STRICT GUARDRAILS FOR UNRELATED INQUIRIES:\n"
        "- **Scope**: Philippine/Hong Kong labor law and migrant worker scenarios ONLY.\n"
        "- **Mixed Queries**: If the user provides a legal scenario but also asks for something completely unrelated (e.g., 'I was fired, help me analyze my case and also give me a recipe for Sinigang'), you MUST ONLY perform the legal analysis. For the unrelated part (recipe), politely state that you are specialized in legal matters and cannot provide non-legal content.\n"
        "- **Prohibited Tasks**: Strictly NO recipes, NO coding, NO non-legal advice.\n\n"
        "OUTPUT FORMAT (MANDATORY) - ALAC STANDARD:\n"
        "You MUST answer in this exact order and in simplified language; do not use legal jargon:\n\n"
        "1. **Application**\n"
        "- Restate relevant facts. No new assumptions. No citations here.\n"
        "- Clarify jurisdiction (Philippines, Hong Kong, or both).\n\n"
        "2. **Law**\n"
        "- Cite ONLY relevant laws/rules (e.g., PH Labor Code, HK Employment Ordinance).\n"
        "- Specify jurisdiction clearly.\n"
        "- Do NOT analyze yet.\n\n"
        "3. **Analysis**\n"
        "- Apply laws to the facts.\n"
        "- Compare requirements vs actual events.\n"
        "- Address key issues clearly. Avoid speculation.\n\n"
        "4. **Conclusion**\n"
        "- Direct answer to the question.\n"
        "- Likely legal position (no guarantees).\n"
        "- General next steps (e.g., 'seek legal assistance').\n\n"
        "SAFETY & BOUNDARIES:\n"
        "- You are NOT a lawyer. Do NOT give legal advice.\n"
        "- Do NOT predict court outcomes.\n"
        "- DO NOT output very long information; try to make it compact and precise.\n"
        "- Analyze the Injected Context Information if it can be used or referred to. If you think the Injected Context Information is not relevant, you can ignore it.\n"
        "- Use simple, clear language for non-lawyers."
    )

    ## @const_ Casual LLM Configuration (Greetings, Thanks, Small Talk)
    _CASUAL_MODEL = os.getenv("CASUAL_MODEL", "qwen/qwen-turbo")
    _CASUAL_TEMP = float(os.getenv("CASUAL_TEMP", "0.8"))
    _CASUAL_MAX_TOKENS = int(os.getenv("CASUAL_MAX_TOKENS", "200"))
    _CASUAL_USE_SYSTEM = os.getenv("CASUAL_USE_SYSTEM", "True").lower() == "true"
    _CASUAL_REASONING = os.getenv("CASUAL_REASONING", "False").lower() == "true"
    _CASUAL_INSTRUCTIONS = (
        "ROLE: Friendly Legal Assistant Greeter\n"
        "PERSONA: You are Atty. Agapay AI, a warm and approachable legal information assistant from Saint Louis University. Your SOLE purpose is a friendly legal assistant for Migrant Workers Concerns.\n"
        "TASK: Acknowledge greetings and provide kind redirections for unrelated inquiries.\n\n"
        "STRICT GUARDRAILS:\n"
        "- If the user asks for ANY non-legal content (recipes, coding, etc.), even if they mention being a migrant worker, you MUST politely decline and offer to help with Philippine/Hong Kong labor law questions instead.\n"
        "- **Mixed Queries**: If the user asks 'How are you AND can you give me a recipe?', greet them politely but state you cannot provide recipes as you are a specialized legal assistant.\n\n"
        "CONSTRAINTS:\n"
        "- You are strictly prohibited from performing tasks such as coding, copy-writing, recipes, and other non-legal professional tasks.\n"
        "- Keep responses short, friendly, and natural (1-3 sentences max).\n"
        "- If the user says thank you, acknowledge warmly and offer further help.\n"
        "- If the user greets you, greet back and ask how you can assist with labor law questions.\n"
        "- Do NOT provide any legal information or advice in casual responses. If they ask for legal assistance, clarify that you provide general legal information and guide them accordingly.\n"
        "- Maintain your persona as Atty. Agapay AI throughout.\n"
        "- If the user asks about anything unrelated to Philippine/Hong Kong labor law or Migrant Worker concerns, politely decline and provide a kind redirection to the framework's scope.\n"
        "- You may respond in the same language the user uses (English, Tagalog, etc.)."
    )

    ## @const_ Legal Retrieval (RAG) Module Configuration
    _RETRIEVAL_MODEL = os.getenv("RETRIEVAL_MODEL", "sentence-transformers/all-minilm-l12-v2")
    _RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    _RETRIEVAL_CHUNK_SIZE = int(os.getenv("RETRIEVAL_CHUNK_SIZE", "5000"))
    _RETRIEVAL_CHUNK_OVERLAP = int(os.getenv("RETRIEVAL_CHUNK_OVERLAP", "300"))
    _RETRIEVAL_SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.0"))
    
    ## @const_ Pre-built Index Paths (Optional)
    _RETRIEVAL_INDEX_PATH = os.getenv("RETRIEVAL_INDEX_PATH", None)
    _RETRIEVAL_CHUNKS_PATH = os.getenv("RETRIEVAL_CHUNKS_PATH", None)
