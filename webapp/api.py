"""
API endpoint handlers for Web of Microbes.
"""

import sys
import time
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "substrate_preference"))
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

from models import (
    RecommendRequest, RecommendResponse, Recommendation,
    PredictResponseRequest, PredictResponseResponse, CommunityPrediction,
    ChatRequest, ChatResponse, ChatChoice, ChatMessage, ChatUsage,
    HealthResponse, ModelInfo, FunctionalGroup
)
from config import Config, FUNCTIONAL_GROUPS

logger = logging.getLogger(__name__)


class WebOfMicrobesAPI:
    """Main API handler class."""

    def __init__(self, config: Config):
        self.config = config
        self.recommender = None
        self.synthesizer = None
        self._load_models()

    def _load_models(self):
        """Load BSPR model and optionally LLM synthesizer."""
        logger.info("Loading Web of Microbes models...")

        # Import here to avoid circular imports
        from soilfm_recommender import SoilFMRecommender

        self.recommender = SoilFMRecommender(
            model_dir=str(self.config.model_dir),
            kg_dir=str(self.config.kg_dir)
        )

        logger.info(f"BSPR loaded: {len(self.recommender.gnn.organisms)} organisms, "
                    f"{len(self.recommender.gnn.metabolites)} metabolites")

        # Load LLM synthesizer if configured
        if self.config.llm_enabled:
            try:
                from llm_synthesis import LLMSynthesizer, LLMConfig

                # Ensure endpoint has the full path for chat completions
                llm_endpoint = self.config.llm_endpoint
                if not llm_endpoint.endswith('/v1/chat/completions'):
                    llm_endpoint = llm_endpoint.rstrip('/') + '/v1/chat/completions'

                llm_config = LLMConfig(
                    endpoint=llm_endpoint,
                    model=self.config.llm_model
                )
                self.synthesizer = LLMSynthesizer(config=llm_config)
                logger.info(f"LLM synthesizer configured: {self.config.llm_model}")
            except Exception as e:
                logger.warning(f"LLM initialization failed: {e}")
                self.synthesizer = None
        else:
            logger.info("LLM not configured (no API key)")

    def _resolve_target(self, target) -> List[str]:
        """Resolve target to list of genera."""
        if isinstance(target, FunctionalGroup) or (isinstance(target, str) and target in FUNCTIONAL_GROUPS):
            group_name = target.value if isinstance(target, FunctionalGroup) else target
            return FUNCTIONAL_GROUPS[group_name]["genera"]
        elif isinstance(target, list):
            return target
        else:
            return [target]

    # =========================================================================
    # /recommend endpoint
    # =========================================================================

    async def recommend(self, request: RecommendRequest) -> RecommendResponse:
        """Generate prebiotic recommendations."""
        # Resolve target genera
        target_genera = self._resolve_target(request.target)

        # Get BSPR recommendations
        results = self.recommender.recommend(
            target=target_genera,
            community=request.community,
            n_recommendations=request.n_recommendations
        )

        # Format recommendations
        recommendations = [
            Recommendation(
                metabolite=r.metabolite,
                selectivity=float(r.gnn_score),
                target_affinity=float(r.target_affinity),
                competitors=r.competitors,
                warning=r.selectivity_warning
            )
            for r in results
        ]

        # Get LLM explanation if requested and available
        llm_explanation = None
        if request.include_llm_explanation and self.synthesizer:
            try:
                spm_recs = [
                    {
                        "metabolite": r.metabolite,
                        "target_affinity": r.target_affinity,
                        "selectivity": r.selectivity,
                        "competitors": r.competitors,
                        "warning": r.warning
                    }
                    for r in recommendations[:5]
                ]

                llm_explanation = self.synthesizer.recommend(
                    user_goal=f"Promote {', '.join(target_genera)} in the soil microbiome",
                    target_taxa=target_genera,
                    community=request.community,
                    spm_recommendations=spm_recs,
                    kg_context={"note": "Web of Microbes API query"},
                    user_context=""
                )
            except Exception as e:
                logger.error(f"LLM synthesis failed: {e}")
                llm_explanation = f"(LLM explanation unavailable: {e})"

        return RecommendResponse(
            recommendations=recommendations,
            target_genera=target_genera,
            community=request.community,
            llm_explanation=llm_explanation
        )

    # =========================================================================
    # /predict_response endpoint
    # =========================================================================

    async def predict_response(self, request: PredictResponseRequest) -> PredictResponseResponse:
        """Predict community response to metabolite profile."""
        # Use recommender.predict_response which handles genus-to-organism matching
        benefited_results, harmed_results = self.recommender.predict_response(
            metabolite_profile=request.metabolite_profile,
            community=request.community,
            n_top=len(request.community)
        )

        # Convert to CommunityPrediction objects
        benefited = [
            CommunityPrediction(
                genus=r.genus,
                affinity=float(r.gnn_affinity),
                direction="benefited",
                in_model=True
            )
            for r in benefited_results
        ]

        harmed = [
            CommunityPrediction(
                genus=r.genus,
                affinity=float(r.gnn_affinity),
                direction="harmed",
                in_model=True
            )
            for r in harmed_results
        ]

        # Find genera not in model
        found_genera = {r.genus for r in benefited_results + harmed_results}
        unknown = [g for g in request.community if g not in found_genera]

        # Get LLM interpretation if requested
        llm_interpretation = None
        if request.include_llm_interpretation and self.synthesizer:
            try:
                community_predictions = [
                    {"genus": p.genus, "affinity": p.affinity, "direction": p.direction}
                    for p in benefited + harmed
                ]

                llm_interpretation = self.synthesizer.predict_response(
                    metabolite_profile=request.metabolite_profile,
                    community_predictions=community_predictions,
                    kg_context={"note": "Web of Microbes API query"},
                    user_context=""
                )
            except Exception as e:
                logger.error(f"LLM interpretation failed: {e}")
                llm_interpretation = f"(LLM interpretation unavailable: {e})"

        return PredictResponseResponse(
            benefited=benefited,
            harmed=harmed,
            unknown=unknown,
            llm_interpretation=llm_interpretation
        )

    # =========================================================================
    # /v1/chat/completions endpoint (OpenAI-compatible)
    # =========================================================================

    async def chat_completions(self, request: ChatRequest) -> ChatResponse:
        """OpenAI-compatible chat completion endpoint."""
        # Extract the last user message
        user_message = None
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_message = msg.content
                break

        if not user_message:
            return self._create_chat_response(
                "Please provide a question about soil microbiome engineering."
            )

        # Parse intent from message
        response_text = await self._process_chat_message(user_message)

        return self._create_chat_response(response_text)

    async def _process_chat_message(self, message: str) -> str:
        """Process a chat message and generate response."""
        message_lower = message.lower()

        # Try to extract intent
        # Pattern 1: Recommend prebiotics for [target] in community with [community]
        recommend_pattern = r"(?:recommend|suggest|what).+(?:prebiotic|metabolite|substrate).+(?:for|to promote|to help)\s+([^,]+?)(?:\s+in\s+(?:a\s+)?community\s+(?:with|containing|of)\s+(.+))?$"

        # Pattern 2: What would happen if [metabolites] were added to [community]
        predict_pattern = r"(?:what|how).+(?:happen|respond|react).+(?:if|when)\s+(.+?)(?:\s+(?:were|was|is)\s+added)"

        # Pattern 3: Direct target/community specification
        target_pattern = r"(?:promote|help|support|increase)\s+([^,]+?)(?:\s+(?:in|with|against)\s+(.+))?"

        # Try recommendation pattern
        match = re.search(recommend_pattern, message_lower, re.IGNORECASE)
        if match:
            target_text = match.group(1).strip()
            community_text = match.group(2) if match.group(2) else ""

            target = self._parse_genera(target_text)
            community = self._parse_genera(community_text) if community_text else ["Pseudomonas", "Bacillus", "Arthrobacter"]

            request = RecommendRequest(
                target=target,
                community=community,
                n_recommendations=5,
                include_llm_explanation=True
            )

            response = await self.recommend(request)
            return self._format_recommendation_response(response)

        # Try target pattern
        match = re.search(target_pattern, message_lower, re.IGNORECASE)
        if match:
            target_text = match.group(1).strip()
            community_text = match.group(2) if match.group(2) else ""

            # Check for functional groups
            target = self._resolve_functional_group(target_text)
            if not target:
                target = self._parse_genera(target_text)

            community = self._parse_genera(community_text) if community_text else ["Pseudomonas", "Bacillus", "Arthrobacter"]

            request = RecommendRequest(
                target=target,
                community=community,
                n_recommendations=5,
                include_llm_explanation=True
            )

            response = await self.recommend(request)
            return self._format_recommendation_response(response)

        # Default: provide general guidance
        return self._get_help_message()

    def _parse_genera(self, text: str) -> List[str]:
        """Parse genera names from text."""
        # Split by common delimiters
        genera = re.split(r'[,\s]+(?:and|or)?\s*|[,\s]+', text)
        # Clean and capitalize
        genera = [g.strip().capitalize() for g in genera if g.strip() and len(g.strip()) > 2]
        return genera if genera else ["Rhizobium"]

    def _resolve_functional_group(self, text: str) -> Optional[List[str]]:
        """Try to resolve text to a functional group."""
        text_lower = text.lower()

        keywords = {
            "nitrogen fix": "nitrogen_fixation",
            "n-fix": "nitrogen_fixation",
            "nitrif": "nitrification",
            "denitrif": "denitrification",
            "phosphate": "phosphate_solubilization",
            "p-solubil": "phosphate_solubilization",
            "pgpr": "plant_growth_promotion",
            "plant growth": "plant_growth_promotion",
            "aromatic": "aromatic_degradation",
            "bioremediation": "aromatic_degradation",
        }

        for keyword, group in keywords.items():
            if keyword in text_lower:
                return FUNCTIONAL_GROUPS[group]["genera"]

        return None

    def _format_recommendation_response(self, response: RecommendResponse) -> str:
        """Format recommendation response as text."""
        lines = [
            f"## Prebiotic Recommendations for {', '.join(response.target_genera)}",
            f"\n**Community context**: {', '.join(response.community)}",
            "\n### Top Recommendations:\n"
        ]

        for i, rec in enumerate(response.recommendations[:5], 1):
            warning_text = f" [Warning: {rec.warning}]" if rec.warning else ""
            lines.append(f"{i}. **{rec.metabolite}**")
            lines.append(f"   - Selectivity: {rec.selectivity:.2f}")
            lines.append(f"   - Target affinity: {rec.target_affinity:.2f}")
            if rec.competitors:
                lines.append(f"   - Competitors: {', '.join(rec.competitors[:3])}")
            if warning_text:
                lines.append(f"   - {warning_text}")
            lines.append("")

        if response.llm_explanation:
            lines.append("\n### Expert Analysis:\n")
            lines.append(response.llm_explanation)

        return "\n".join(lines)

    def _get_help_message(self) -> str:
        """Return help message."""
        return """# Web of Microbes - Prebiotic Recommender

I can help you find metabolites to promote specific bacteria in your soil microbiome.

## How to ask:

**Recommendation queries:**
- "What prebiotics would promote Rhizobium in a community with Pseudomonas and Bacillus?"
- "Recommend metabolites for nitrogen fixers"
- "Suggest substrates to help Arthrobacter against Pseudomonas"

**Functional groups I understand:**
- nitrogen_fixation (Rhizobium, Bradyrhizobium, etc.)
- phosphate_solubilization (Pseudomonas, Bacillus, etc.)
- plant_growth_promotion (PGPR)
- aromatic_degradation (bioremediation)
- nitrification, denitrification

**Example:**
"Recommend prebiotics for nitrogen fixers in a community with Pseudomonas, Bacillus, and Arthrobacter"

---
*Model: Web of Microbes v1.1 | Validated: AUC 0.94 (genus-level)*
"""

    def _create_chat_response(self, content: str) -> ChatResponse:
        """Create OpenAI-compatible chat response."""
        return ChatResponse(
            id=f"chatcmpl-wom-{int(time.time())}",
            created=int(time.time()),
            choices=[
                ChatChoice(
                    message=ChatMessage(role="assistant", content=content)
                )
            ]
        )

    # =========================================================================
    # Health & Info endpoints
    # =========================================================================

    def health(self) -> HealthResponse:
        """Health check."""
        # Check if KG data is loaded (lazy loading check)
        kg_loaded = False
        if self.recommender:
            # Trigger KG loading if not already done
            self.recommender._load_kg_data()
            kg_loaded = self.recommender._kg_consumption is not None
        return HealthResponse(
            model_loaded=self.recommender is not None,
            kg_loaded=kg_loaded,
            llm_configured=self.synthesizer is not None
        )

    def model_info(self) -> ModelInfo:
        """Get model information."""
        return ModelInfo(
            n_organisms=len(self.recommender.gnn.organisms) if self.recommender else 0,
            n_metabolites=len(self.recommender.gnn.metabolites) if self.recommender else 0,
            functional_groups=list(FUNCTIONAL_GROUPS.keys())
        )
