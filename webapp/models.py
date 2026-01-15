"""
Pydantic models for Web of Microbes API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from enum import Enum


# =============================================================================
# Enums
# =============================================================================

class FunctionalGroup(str, Enum):
    """Predefined functional groups for target selection."""
    nitrogen_fixation = "nitrogen_fixation"
    nitrification = "nitrification"
    denitrification = "denitrification"
    phosphate_solubilization = "phosphate_solubilization"
    plant_growth_promotion = "plant_growth_promotion"
    aromatic_degradation = "aromatic_degradation"


# =============================================================================
# Recommendation Endpoint
# =============================================================================

class RecommendRequest(BaseModel):
    """Request model for /recommend endpoint."""
    target: Union[List[str], FunctionalGroup] = Field(
        ...,
        description="Target genera to promote OR a functional group name"
    )
    community: List[str] = Field(
        ...,
        description="List of genera in the current community (competitors)"
    )
    n_recommendations: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of recommendations to return"
    )
    include_llm_explanation: bool = Field(
        default=True,
        description="Whether to include LLM-generated explanation"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "target": ["Rhizobium", "Bradyrhizobium"],
                "community": ["Pseudomonas", "Bacillus", "Arthrobacter"],
                "n_recommendations": 5,
                "include_llm_explanation": True
            }
        }


class Recommendation(BaseModel):
    """Single metabolite recommendation."""
    metabolite: str
    selectivity: float = Field(description="Selectivity score (higher = more selective for target)")
    target_affinity: float = Field(description="Predicted affinity for target taxa")
    competitors: List[str] = Field(description="Genera that also consume this metabolite")
    warning: Optional[str] = Field(default=None, description="Selectivity warning if applicable")


class RecommendResponse(BaseModel):
    """Response model for /recommend endpoint."""
    recommendations: List[Recommendation]
    target_genera: List[str]
    community: List[str]
    llm_explanation: Optional[str] = None
    model_version: str = "webofmicrobes-v1.1"


# =============================================================================
# Predict Response Endpoint
# =============================================================================

class PredictResponseRequest(BaseModel):
    """Request model for /predict_response endpoint."""
    metabolite_profile: Dict[str, float] = Field(
        ...,
        description="Metabolite concentrations (name: relative abundance)"
    )
    community: List[str] = Field(
        ...,
        description="List of genera to predict response for"
    )
    include_llm_interpretation: bool = Field(
        default=True,
        description="Whether to include LLM interpretation"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "metabolite_profile": {
                    "glutamic acid": 1.0,
                    "aspartic acid": 0.8,
                    "glucose": 2.0
                },
                "community": ["Pseudomonas", "Bacillus", "Arthrobacter", "Rhizobium"]
            }
        }


class CommunityPrediction(BaseModel):
    """Prediction for a single genus."""
    genus: str
    affinity: float = Field(description="Predicted affinity (weighted sum)")
    direction: str = Field(description="'benefited' or 'harmed'")
    in_model: bool = Field(default=True, description="Whether genus is in GNN vocabulary")


class PredictResponseResponse(BaseModel):
    """Response model for /predict_response endpoint."""
    benefited: List[CommunityPrediction]
    harmed: List[CommunityPrediction]
    unknown: List[str] = Field(default_factory=list, description="Genera not in model")
    llm_interpretation: Optional[str] = None
    model_version: str = "webofmicrobes-v1.1"


# =============================================================================
# OpenAI-Compatible Chat Endpoint
# =============================================================================

class ChatMessage(BaseModel):
    """OpenAI-compatible message format."""
    role: str = Field(..., description="Role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str = Field(default="webofmicrobes", description="Model name")
    messages: List[ChatMessage] = Field(..., description="Conversation messages")
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=2000, ge=1, le=4096)
    stream: bool = Field(default=False, description="Streaming not yet supported")

    class Config:
        json_schema_extra = {
            "example": {
                "model": "webofmicrobes",
                "messages": [
                    {"role": "user", "content": "What prebiotics would promote nitrogen fixers in a community with Pseudomonas and Bacillus?"}
                ]
            }
        }


class ChatChoice(BaseModel):
    """Single choice in chat response."""
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatUsage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str = Field(default="chatcmpl-wom")
    object: str = "chat.completion"
    created: int
    model: str = "webofmicrobes-v1.1"
    choices: List[ChatChoice]
    usage: ChatUsage = Field(default_factory=ChatUsage)


# =============================================================================
# Health & Info Endpoints
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    model_loaded: bool
    kg_loaded: bool
    llm_configured: bool
    version: str = "1.1.0"


class ModelInfo(BaseModel):
    """Model information response."""
    name: str = "Web of Microbes"
    version: str = "v1.1"
    n_organisms: int
    n_metabolites: int
    validation_rmse: float = 2.18
    validation_correlation: str = "Spearman rho=0.78, p=0.039"
    functional_groups: List[str]


# =============================================================================
# Error Models
# =============================================================================

class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    status_code: int

# =============================================================================
# Name Unification Endpoint
# =============================================================================

class UnifyNamesRequest(BaseModel):
    """Request model for /unify_names endpoint."""
    microbe_names: List[str] = Field(
        default_factory=list,
        description="Raw microbe names to resolve (user input)"
    )
    metabolite_names: List[str] = Field(
        default_factory=list,
        description="Raw metabolite names to resolve (user input)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "microbe_names": ["e coli", "pseudomonas aeruginosa", "rhizobia"],
                "metabolite_names": ["glucose", "glutamate", "organic acid"]
            }
        }


class ResolvedName(BaseModel):
    """Single resolved name with confidence."""
    input_name: str = Field(description="Original user input")
    resolved_name: Optional[str] = Field(description="Resolved name in model vocabulary")
    confidence: str = Field(description="high/medium/low confidence")
    alternatives: List[str] = Field(default_factory=list, description="Alternative matches if ambiguous")
    note: Optional[str] = Field(default=None, description="Explanation or warning")


class UnifyNamesResponse(BaseModel):
    """Response model for /unify_names endpoint."""
    microbes: List[ResolvedName]
    metabolites: List[ResolvedName]
    model_version: str = "webofmicrobes-v1.1"
