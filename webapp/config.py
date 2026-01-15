"""
Configuration management for Web of Microbes API.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """Application configuration."""

    # Paths
    model_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "10_svd_tower" / "trained_gnn_model")
    kg_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "01_knowledge_graph")

    # LLM Configuration
    llm_endpoint: str = field(default_factory=lambda: os.getenv(
        "LLM_ENDPOINT",
        "https://api.together.xyz/v1/chat/completions"
    ))
    llm_model: str = field(default_factory=lambda: os.getenv(
        "LLM_MODEL",
        "Qwen/Qwen2.5-14B-Instruct"
    ))
    llm_api_key: Optional[str] = field(default_factory=lambda: os.getenv("LLM_API_KEY"))

    # API Configuration
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("WOM_API_KEY"))
    alpha_password: Optional[str] = field(default_factory=lambda: os.getenv("WOM_ALPHA_PASSWORD"))

    # Server
    host: str = "0.0.0.0"
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8000")))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")

    # Model settings
    default_n_recommendations: int = 5
    max_n_recommendations: int = 20
    request_timeout: int = 60

    def __post_init__(self):
        """Validate paths exist."""
        if not self.model_dir.exists():
            raise ValueError(f"Model directory not found: {self.model_dir}")
        if not self.kg_dir.exists():
            raise ValueError(f"Knowledge graph directory not found: {self.kg_dir}")

    @property
    def llm_enabled(self) -> bool:
        """Check if LLM is configured."""
        # LLM is enabled if we have an API key OR if using local vLLM (localhost endpoint)
        if self.llm_api_key:
            return True
        if 'localhost' in self.llm_endpoint or '127.0.0.1' in self.llm_endpoint:
            return True
        return False


def get_config() -> Config:
    """Get application configuration."""
    return Config()


# Functional groups for target selection
FUNCTIONAL_GROUPS = {
    "nitrogen_fixation": {
        "description": "Nitrogen-fixing bacteria",
        "genera": ["Rhizobium", "Bradyrhizobium", "Mesorhizobium", "Sinorhizobium",
                   "Azorhizobium", "Azospirillum", "Azotobacter", "Frankia",
                   "Paenibacillus", "Klebsiella", "Beijerinckia", "Derxia"]
    },
    "nitrification": {
        "description": "Nitrifying bacteria",
        "genera": ["Nitrosomonas", "Nitrosospira", "Nitrobacter", "Nitrospira",
                   "Nitrosococcus", "Nitrososphaera"]
    },
    "denitrification": {
        "description": "Denitrifying bacteria",
        "genera": ["Pseudomonas", "Paracoccus", "Alcaligenes", "Bacillus",
                   "Thiobacillus", "Achromobacter"]
    },
    "phosphate_solubilization": {
        "description": "Phosphate-solubilizing bacteria",
        "genera": ["Pseudomonas", "Bacillus", "Rhizobium", "Burkholderia",
                   "Enterobacter", "Serratia", "Azotobacter", "Azospirillum", "Micrococcus"]
    },
    "plant_growth_promotion": {
        "description": "Plant growth-promoting rhizobacteria (PGPR)",
        "genera": ["Pseudomonas", "Bacillus", "Azospirillum", "Rhizobium",
                   "Enterobacter", "Burkholderia", "Acinetobacter", "Serratia"]
    },
    "aromatic_degradation": {
        "description": "Aromatic compound degraders (bioremediation)",
        "genera": ["Arthrobacter", "Rhodococcus", "Pseudomonas", "Sphingomonas",
                   "Nocardioides", "Mycobacterium", "Streptomyces"]
    }
}
