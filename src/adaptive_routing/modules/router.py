"""
Saint Louis University : Team 404FoundUs
@file src/adaptive_routing/modules/router.py
@project_ LLM Legal Adaptive Routing Framework
@desc_ Orchestrator module that coordinates Logic Classification and Legal Generation.
@deps_ src.adaptive_routing.modules.semantic_router.logic_classifier, src.adaptive_routing.modules.semantic_router.legal_generation
"""

from src.adaptive_routing.modules.semantic_router.logic_classifier import RoutingClassifier
from src.adaptive_routing.modules.semantic_router.legal_generation import LegalGenerator

class SemanticRouterModule:
    """
    @class_ SemanticRouterModule
    @desc_ Facade/Orchestrator that manages the pipeline: Classify -> Route -> Generate.
    @attr_ _classifier : (RoutingClassifier) Component to decide the route.
    @attr_ _generator : (LegalGenerator) Component to execute the LLM call.
    """

    def __init__(self, api_key=None, classifier=None, generator=None):
        self._classifier = classifier or RoutingClassifier(api_key)
        self._generator = generator or LegalGenerator(api_key)

    def _process_routing_(self, normalized_text: str, context: str = None) -> dict:
        """
        @func_ _process_routing_ (@params normalized_text, context)
        @params normalized_text : (str) Standardized user query.
        @params context : (str, optional) RAG retrieved context to enhance generation.
        @return_ dict : Contains routing metadata and the final LLM response.
        @logic_ 
            1. Calls classifier to get route (General vs Reasoning) using ONLY the safe normalized query.
            2. If route is valid, appends context to the query and calls generator.
            3. Returns combined result.
        """
        ## @logic_ Classify using clean normalized text
        classification = self._classifier._route_query_(normalized_text)
        route = classification.get('route')
        
        ## @logic_ Generate if valid, passing augmented context
        if route and route != "PATHWAY_2":
             augmented_query = normalized_text
             if context:
                 augmented_query += f"\n\nContext Information:\n{context}\nPlease use the provided context to answer the user query if applicable."
             try:
                 response_text = self._generator._dispatch_(augmented_query, route)
             except Exception as e:
                 print(f"Generator Error: {e}")
                 response_text = "I apologize, but I encountered an error while formulating my legal response. Please try again."
        else:
             response_text = "Hi There, can you please clarify your inquiry and provide specific details."

        ## @logic_ Combine results
        return {
            "classification": classification,
            "response_text": response_text
        }
