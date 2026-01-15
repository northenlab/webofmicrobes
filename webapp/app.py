"""
Web of Microbes - FastAPI + Gradio Application

Main application combining:
- FastAPI backend with REST API
- Gradio frontend for web UI
- OpenAI-compatible chat endpoint
"""

import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import gradio as gr

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "substrate_preference"))
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

from config import get_config, FUNCTIONAL_GROUPS
from models import (
    RecommendRequest, RecommendResponse,
    PredictResponseRequest, PredictResponseResponse,
    ChatRequest, ChatResponse,
    HealthResponse, ModelInfo, ErrorResponse
)
from api import WebOfMicrobesAPI

# Global API instance
api: Optional[WebOfMicrobesAPI] = None


# =============================================================================
# Lifespan management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    global api
    logger.info("Starting Web of Microbes API...")

    try:
        config = get_config()
        api = WebOfMicrobesAPI(config)
        logger.info("Models loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

    yield

    logger.info("Shutting down Web of Microbes API...")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Web of Microbes API",
    description="""
    A neural network-based prebiotic recommender for soil microbiome engineering.

    ## Features
    - **Prebiotic Recommendations**: Find metabolites to promote target bacteria
    - **Community Response Prediction**: Predict how communities respond to metabolites
    - **OpenAI-Compatible API**: Use with standard OpenAI clients

    ## Model Performance
    - BSPR Model: AUC 0.94 on held-out validation
    - Resolution: Genus-level predictions (species within same genus get identical predictions)

    ## Links
    - [Documentation](https://webofmicrobes.org/docs)
    - [HuggingFace Model](https://huggingface.co/northenlab/webofmicrobes-v1.1)
    """,
    version="1.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Authentication (simple for alpha)
# =============================================================================

async def verify_auth(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None)
):
    """Simple authentication for alpha testing."""
    config = get_config()

    # Check API key
    if x_api_key and config.api_key:
        if x_api_key == config.api_key:
            return True

    # Check Bearer token / password
    if authorization:
        token = authorization.replace("Bearer ", "")
        if token == config.alpha_password or token == config.api_key:
            return True

    # For alpha: allow if no auth configured
    if not config.api_key and not config.alpha_password:
        return True

    # Default allow for now (can be made strict later)
    return True


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    if api is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return api.health()


@app.get("/info", response_model=ModelInfo, tags=["System"])
async def model_info():
    """Get model information."""
    if api is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return api.model_info()


@app.post("/recommend", response_model=RecommendResponse, tags=["Recommendations"])
async def recommend(
    request: RecommendRequest,
    authenticated: bool = Depends(verify_auth)
):
    """
    Get prebiotic recommendations to promote target taxa.

    **Parameters:**
    - `target`: List of genera to promote OR a functional group name
    - `community`: List of genera in the current community (competitors)
    - `n_recommendations`: Number of recommendations (1-20)
    - `include_llm_explanation`: Whether to include AI explanation

    **Functional groups:**
    - nitrogen_fixation
    - phosphate_solubilization
    - plant_growth_promotion
    - aromatic_degradation
    - nitrification
    - denitrification
    """
    if api is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        return await api.recommend(request)
    except Exception as e:
        logger.error(f"Recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_response", response_model=PredictResponseResponse, tags=["Predictions"])
async def predict_response(
    request: PredictResponseRequest,
    authenticated: bool = Depends(verify_auth)
):
    """
    Predict community response to a metabolite profile.

    **Parameters:**
    - `metabolite_profile`: Dict of metabolite names to relative concentrations
    - `community`: List of genera to predict response for
    - `include_llm_interpretation`: Whether to include AI interpretation
    """
    if api is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        return await api.predict_response(request)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions", response_model=ChatResponse, tags=["OpenAI Compatible"])
async def chat_completions(
    request: ChatRequest,
    authenticated: bool = Depends(verify_auth)
):
    """
    OpenAI-compatible chat completion endpoint.

    Use with standard OpenAI clients:
    ```python
    import openai
    client = openai.OpenAI(base_url="https://webofmicrobes.org/v1", api_key="your-key")
    response = client.chat.completions.create(
        model="webofmicrobes",
        messages=[{"role": "user", "content": "Recommend prebiotics for nitrogen fixers"}]
    )
    ```
    """
    if api is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        return await api.chat_completions(request)
    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models", tags=["OpenAI Compatible"])
async def list_models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "webofmicrobes",
                "object": "model",
                "created": 1704067200,
                "owned_by": "northenlab"
            }
        ]
    }


@app.post("/substrate_preferences", tags=["Predictions"])
async def substrate_preferences(
    request: dict,
    authenticated: bool = Depends(verify_auth)
):
    """
    Get top substrate preferences for a single organism.

    **Parameters:**
    - `genus`: Genus name (e.g., "Pseudomonas")
    - `n_substrates`: Number of substrates to return (default 10)
    """
    if api is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    genus = request.get("genus", "").strip()
    n_substrates = request.get("n_substrates", 10)

    if not genus:
        return {"error": "Please provide a genus name"}

    try:
        # Find matching organism in BSPR
        matched_org = None
        for org in api.recommender.gnn.organisms:
            if org.lower().startswith(genus.lower()):
                matched_org = org
                break

        if not matched_org:
            available = sorted(set(org.split()[0] for org in api.recommender.gnn.organisms))
            return {
                "error": f"{genus} not found in model",
                "available_genera": available[:50]
            }

        # Get predictions
        preds = api.recommender.gnn.predict(matched_org)

        # Sort by score (negative = consumed)
        sorted_preds = sorted(preds.items(), key=lambda x: x[1])

        # Top consumed (most negative scores)
        consumed = [
            {"metabolite": m, "affinity": abs(s)}
            for m, s in sorted_preds
            if s < -0.5 and 'Unk_' not in m
        ][:n_substrates]

        return {
            "matched_organism": matched_org,
            "consumed": consumed
        }

    except Exception as e:
        logger.error(f"Substrate preferences failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Gradio Interface
# =============================================================================

def create_gradio_interface():
    """Create Gradio web interface."""

    def substrate_preferences_ui(genus_input, n_substrates):
        """UI handler for single organism substrate preferences."""
        if api is None:
            return "Error: Models not loaded"

        genus = genus_input.strip()
        if not genus:
            return "Please enter a genus name (e.g., Pseudomonas)"

        try:
            # Find matching organism in BSPR
            matched_org = None
            for org in api.recommender.gnn.organisms:
                if org.lower().startswith(genus.lower()):
                    matched_org = org
                    break

            if not matched_org:
                available = sorted(set(org.split()[0] for org in api.recommender.gnn.organisms))
                return f"**{genus}** not found in model.\n\nAvailable genera ({len(available)}):\n" + ", ".join(available[:50]) + "..."

            # Get predictions
            preds = api.recommender.gnn.predict(matched_org)

            # Sort by score (negative = consumed, positive = produced)
            sorted_preds = sorted(preds.items(), key=lambda x: x[1])

            # Top consumed (most negative scores)
            consumed = [(m, s) for m, s in sorted_preds if s < -0.5 and 'Unk_' not in m][:int(n_substrates)]

            # Top produced (most positive scores)
            produced = [(m, s) for m, s in reversed(sorted_preds) if s > 0.5 and 'Unk_' not in m][:int(n_substrates)]

            # Format results
            results = f"## Substrate Preferences for {matched_org}\n\n"

            results += "### Top Consumed Substrates\n"
            results += "| Rank | Metabolite | Affinity |\n"
            results += "|------|------------|----------|\n"
            for i, (met, score) in enumerate(consumed, 1):
                results += f"| {i} | {met} | {abs(score):.2f} |\n"

            results += "\n### Top Produced Metabolites\n"
            results += "| Rank | Metabolite | Affinity |\n"
            results += "|------|------------|----------|\n"
            for i, (met, score) in enumerate(produced, 1):
                results += f"| {i} | {met} | {score:.2f} |\n"

            if not produced:
                results += "| - | No significant production predicted | - |\n"

            return results

        except Exception as e:
            return f"Error: {e}"

    def recommend_ui(target_input, community_input, functional_group, n_recs, use_llm):
        """UI handler for recommendations."""
        if api is None:
            return "Error: Models not loaded", ""

        # Determine target
        if functional_group and functional_group != "Custom (specify below)":
            target = functional_group  # Pass as string for _resolve_target to handle
        else:
            target = [g.strip() for g in target_input.split(",") if g.strip()]

        # Parse community
        community = [g.strip() for g in community_input.split(",") if g.strip()]

        if not target:
            return "Please specify target genera or select a functional group", ""
        if not community:
            community = ["Pseudomonas", "Bacillus", "Arthrobacter"]

        try:
            import asyncio
            request = RecommendRequest(
                target=target,
                community=community,
                n_recommendations=int(n_recs),
                include_llm_explanation=use_llm
            )
            response = asyncio.run(api.recommend(request))

            # Format results table
            table = "| Metabolite | Selectivity | Target Affinity | Competitors |\n"
            table += "|------------|-------------|-----------------|-------------|\n"
            for rec in response.recommendations:
                competitors = ", ".join(rec.competitors[:3]) if rec.competitors else "-"
                warning = " [!]" if rec.warning else ""
                table += f"| {rec.metabolite}{warning} | {rec.selectivity:.2f} | {rec.target_affinity:.2f} | {competitors} |\n"

            explanation = response.llm_explanation or "(LLM explanation disabled or unavailable)"

            return table, explanation

        except Exception as e:
            return f"Error: {e}", ""

    def predict_ui(metabolites_input, community_input, use_llm):
        """UI handler for community response prediction."""
        if api is None:
            return "Error: Models not loaded", ""

        # Parse metabolite profile
        metabolite_profile = {}
        for line in metabolites_input.strip().split("\n"):
            if ":" in line:
                name, value = line.split(":", 1)
                try:
                    metabolite_profile[name.strip()] = float(value.strip())
                except ValueError:
                    pass

        # Parse community
        community = [g.strip() for g in community_input.split(",") if g.strip()]

        if not metabolite_profile:
            return "Please specify metabolites (format: name: value)", ""
        if not community:
            return "Please specify community genera", ""

        try:
            import asyncio
            request = PredictResponseRequest(
                metabolite_profile=metabolite_profile,
                community=community,
                include_llm_interpretation=use_llm
            )
            response = asyncio.run(api.predict_response(request))

            # Format results
            results = "### Benefited (will increase)\n"
            for p in response.benefited[:5]:
                results += f"- **{p.genus}**: affinity = {p.affinity:.2f}\n"

            results += "\n### Harmed (will decrease)\n"
            for p in response.harmed[:5]:
                results += f"- **{p.genus}**: affinity = {p.affinity:.2f}\n"

            if response.unknown:
                results += f"\n### Unknown (not in model)\n- {', '.join(response.unknown)}\n"

            interpretation = response.llm_interpretation or "(LLM interpretation disabled or unavailable)"

            return results, interpretation

        except Exception as e:
            return f"Error: {e}", ""

    # Build interface
    with gr.Blocks(title="Web of Microbes", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # Web of Microbes
        ### Prebiotic Recommender for Soil Microbiome Engineering

        Find metabolites to promote beneficial bacteria in your soil community.

        **Model**: BSPR v1.0 | **Resolution**: Genus-level | **Validation AUC**: 0.94
        """)

        with gr.Tabs():
            # Tab 1: Substrate Preferences
            with gr.TabItem("Substrate Preferences"):
                gr.Markdown("**Predict which metabolites a microbe consumes or produces**")
                with gr.Row():
                    with gr.Column():
                        genus_input = gr.Textbox(
                            label="Genus Name",
                            placeholder="Pseudomonas",
                            value="Pseudomonas"
                        )
                        n_substrates = gr.Slider(5, 20, value=10, step=1, label="Number of Substrates")
                        predict_substrates_btn = gr.Button("Predict Substrates", variant="primary")

                    with gr.Column():
                        substrates_output = gr.Markdown(label="Substrate Preferences")

                predict_substrates_btn.click(
                    substrate_preferences_ui,
                    inputs=[genus_input, n_substrates],
                    outputs=[substrates_output]
                )

            # Tab 2: Prebiotic Recommender
            with gr.TabItem("Prebiotic Recommender"):
                with gr.Row():
                    with gr.Column():
                        functional_group = gr.Dropdown(
                            choices=["Custom (specify below)"] + list(FUNCTIONAL_GROUPS.keys()),
                            value="Custom (specify below)",
                            label="Functional Group"
                        )
                        target_input = gr.Textbox(
                            label="Target Genera (comma-separated)",
                            placeholder="Rhizobium, Bradyrhizobium",
                            value="Rhizobium, Bradyrhizobium"
                        )
                        community_input = gr.Textbox(
                            label="Community Genera (comma-separated)",
                            placeholder="Pseudomonas, Bacillus, Arthrobacter",
                            value="Pseudomonas, Bacillus, Arthrobacter"
                        )
                        n_recs = gr.Slider(1, 10, value=5, step=1, label="Number of Recommendations")
                        use_llm = gr.Checkbox(value=True, label="Include AI Explanation")
                        recommend_btn = gr.Button("Get Recommendations", variant="primary")

                    with gr.Column():
                        results_table = gr.Markdown(label="Recommendations")
                        explanation_output = gr.Markdown(label="AI Explanation")

                recommend_btn.click(
                    recommend_ui,
                    inputs=[target_input, community_input, functional_group, n_recs, use_llm],
                    outputs=[results_table, explanation_output]
                )

            # Tab 2: Predict Response
            with gr.TabItem("Predict Response"):
                with gr.Row():
                    with gr.Column():
                        metabolites_input = gr.Textbox(
                            label="Metabolite Profile (name: concentration)",
                            placeholder="glucose: 100\nglutamic acid: 50\nbetaine: 74",
                            value="glucose: 100\nglutamic acid: 50\nbetaine: 74",
                            lines=5
                        )
                        community_input2 = gr.Textbox(
                            label="Community Genera (comma-separated)",
                            placeholder="Pseudomonas, Bacillus, Arthrobacter, Rhizobium",
                            value="Pseudomonas, Bacillus, Arthrobacter, Rhizobium"
                        )
                        use_llm2 = gr.Checkbox(value=True, label="Include AI Interpretation")
                        predict_btn = gr.Button("Predict Response", variant="primary")

                    with gr.Column():
                        response_output = gr.Markdown(label="Predictions")
                        interpretation_output = gr.Markdown(label="AI Interpretation")

                predict_btn.click(
                    predict_ui,
                    inputs=[metabolites_input, community_input2, use_llm2],
                    outputs=[response_output, interpretation_output]
                )

            # Tab 3: About
            with gr.TabItem("About"):
                gr.Markdown("""
                ## About Web of Microbes

                Web of Microbes is a neural network trained on exometabolomics data to predict
                microbial substrate preferences. It can recommend metabolites (prebiotics) to
                promote specific bacteria in soil communities.

                ### Model Details
                - **Model**: BSPR (Bayesian Substrate Preference Ranking) v1.0
                - **Training data**: Exometabolomics data from LBNL
                - **Resolution**: Genus-level (species within same genus get identical predictions)
                - **Genera**: 20 trained genera, 403 metabolites
                - **Validation AUC**: 0.94 on held-out test set

                ### Functional Groups
                | Group | Description | Example Genera |
                |-------|-------------|----------------|
                | nitrogen_fixation | N-fixing bacteria | Rhizobium, Bradyrhizobium |
                | phosphate_solubilization | P-solubilizers | Pseudomonas, Bacillus |
                | plant_growth_promotion | PGPR | Azospirillum, Enterobacter |
                | aromatic_degradation | Bioremediation | Arthrobacter, Rhodococcus |

                ### API Access
                ```python
                import openai
                client = openai.OpenAI(
                    base_url="https://webofmicrobes.org/v1",
                    api_key="your-api-key"
                )
                response = client.chat.completions.create(
                    model="webofmicrobes",
                    messages=[{"role": "user", "content": "Recommend prebiotics for nitrogen fixers"}]
                )
                ```

                ### Links
                - [HuggingFace Model](https://huggingface.co/northenlab/webofmicrobes-v1.1)
                - [GitHub](https://github.com/northenlab/webofmicrobes)
                - [Northen Lab @ LBNL](https://biosciences.lbl.gov/profiles/trent-northen/)

                ### Citation
                ```
                Northen Lab (2026). Web of Microbes: A Neural Network for Predicting
                Microbial Substrate Preferences. Lawrence Berkeley National Laboratory.
                ```
                """)

    return interface


# Mount Gradio app at /app
gradio_app = create_gradio_interface()
app = gr.mount_gradio_app(app, gradio_app, path="/app")

# Serve static landing page at /
static_dir = Path(__file__).parent / "static"

@app.get("/", response_class=FileResponse)
async def landing_page():
    """Serve the landing page."""
    return FileResponse(static_dir / "index.html")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    config = get_config()
    uvicorn.run(
        "app:app",
        host=config.host,
        port=config.port,
        reload=config.debug
    )
