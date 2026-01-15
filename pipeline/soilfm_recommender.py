#!/usr/bin/env python3
"""
SoilFM Unified Recommender Pipeline

Integrates:
1. BSPR (Bayesian Substrate Preference Ranking) Model (AUC 0.94, genus-level)
2. KG Competitive Landscape Analysis
3. LLM Expert Synthesis (optional)

Two main use cases:
- recommend(): Predict prebiotics to promote target taxa/function
- predict_response(): Predict community response to metabolite profile

IMPORTANT: All BSPR predictions are at GENUS level - species within the
same genus will receive identical predictions.

Architecture:
  Input -> [KG pathway lookup if function goal] -> BSPR predictions
        -> KG competitive analysis -> LLM synthesis -> Output
"""

import sys
import pickle
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict
from dataclasses import dataclass

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "substrate_preference"))

DATA_ROOT = Path(__file__).parent.parent
KG_DIR = DATA_ROOT / "knowledge_graph"
MODEL_DIR = DATA_ROOT / "substrate_preference" / "trained_gnn_model"
PIPELINE_DIR = DATA_ROOT / "pipeline"


@dataclass
class RecommendationResult:
    """Result from the recommendation pipeline."""
    metabolite: str
    gnn_score: float  # Higher = more selective for target
    target_affinity: float  # How much target consumes this
    competitor_count: int  # How many competitors also consume
    competitors: List[str]  # Names of competing genera
    selectivity_warning: Optional[str] = None
    kg_context: Optional[Dict] = None


@dataclass
class CommunityResponse:
    """Predicted response of community to a metabolite."""
    genus: str
    gnn_affinity: float
    direction: str  # "benefited", "harmed", "neutral"
    kg_consumers: int  # How many KG species in this genus consume it


class SoilFMRecommender:
    """
    Unified pipeline for prebiotic recommendation and community response prediction.
    """

    def __init__(self, model_dir: str = None, kg_dir: str = None):
        """Initialize with GNN model and KG."""
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self.kg_dir = Path(kg_dir) if kg_dir else KG_DIR

        # Lazy loading
        self._gnn = None
        self._kg_consumption = None
        self._chebi_to_genus = None
        self._metabolite_names = None
        self._function_to_genera = None
        self._metabolite_unifier = None

    @property
    def gnn(self):
        """Lazy load BSPR model (backwards-compatible alias for 'bspr')."""
        return self.bspr

    @property
    def bspr(self):
        """Lazy load BSPR (Bayesian Substrate Preference Ranking) model v1.0.

        NOTE: Predictions are at GENUS level - species within the same
        genus will receive identical predictions.
        """
        if self._gnn is None:
            from substrate_preference_model_v1 import SubstratePreferenceModelV1
            profiles_path = self.model_dir.parent / 'genus_profiles.json'
            self._gnn = SubstratePreferenceModelV1(profiles_path=str(profiles_path))
        return self._gnn

    def _load_kg_data(self):
        """Load KG consumption edges."""
        if self._kg_consumption is not None:
            return

        kg_path = self.kg_dir / "soilfm_filtered_kg.pt"
        mappings_path = self.kg_dir / "filtered_node_mappings.pkl"

        if not kg_path.exists():
            print(f"Warning: KG not found at {kg_path}")
            self._kg_consumption = {}
            self._chebi_to_genus = {}
            self._metabolite_names = {}
            return

        kg = torch.load(kg_path, map_location='cpu', weights_only=False)
        with open(mappings_path, 'rb') as f:
            mappings = pickle.load(f)

        idx_to_microbe_id = mappings.get('idx_to_microbe', {})
        idx_to_metabolite_id = mappings.get('idx_to_metabolite', {})
        microbe_id_to_name = mappings.get('microbe_names', {})
        self._metabolite_names = mappings.get('metabolite_names', {})

        # Build genus -> set(chebi_ids) and chebi -> set(genera)
        self._kg_consumption = defaultdict(set)
        self._chebi_to_genus = defaultdict(set)

        for edge_type in kg.edge_types:
            src_type, rel, dst_type = edge_type
            if 'consumes' in rel.lower():
                edge_index = kg[edge_type].edge_index
                for i in range(edge_index.shape[1]):
                    src_idx = edge_index[0, i].item()
                    dst_idx = edge_index[1, i].item()

                    microbe_id = idx_to_microbe_id.get(src_idx, "")
                    metabolite_id = idx_to_metabolite_id.get(dst_idx, "")
                    microbe_name = microbe_id_to_name.get(microbe_id, "")

                    if microbe_name and metabolite_id:
                        genus = self._extract_genus(microbe_name)
                        if genus:
                            self._kg_consumption[genus.lower()].add(metabolite_id)
                            self._chebi_to_genus[metabolite_id].add(genus.lower())

    def _extract_genus(self, name: str) -> Optional[str]:
        """Extract genus from species name."""
        for prefix in ['s__', 'g__', 'f__', 'o__', 'c__', 'p__', 'd__']:
            name = name.replace(prefix, '')
        name = name.replace('_', ' ')
        parts = name.split()
        return parts[0] if parts else None

    def _load_function_lookup(self):
        """Load function -> genera mapping (FAPROTAX-style)."""
        if self._function_to_genera is not None:
            return

        # Built-in lookup for common functions
        self._function_to_genera = {
            'nitrogen_fixation': [
                'Rhizobium', 'Bradyrhizobium', 'Mesorhizobium', 'Sinorhizobium',
                'Azorhizobium', 'Azospirillum', 'Azotobacter', 'Frankia',
                'Paenibacillus', 'Klebsiella', 'Beijerinckia', 'Derxia'
            ],
            'nitrification': [
                'Nitrosomonas', 'Nitrosospira', 'Nitrobacter', 'Nitrospira',
                'Nitrosococcus', 'Nitrososphaera'
            ],
            'denitrification': [
                'Pseudomonas', 'Paracoccus', 'Alcaligenes', 'Bacillus',
                'Thiobacillus', 'Achromobacter'
            ],
            'phosphate_solubilization': [
                'Pseudomonas', 'Bacillus', 'Rhizobium', 'Burkholderia',
                'Enterobacter', 'Serratia', 'Azotobacter', 'Azospirillum',
                'Micrococcus'
            ],
            'plant_growth_promotion': [
                'Pseudomonas', 'Bacillus', 'Azospirillum', 'Rhizobium',
                'Enterobacter', 'Burkholderia', 'Acinetobacter', 'Serratia'
            ],
            'aromatic_degradation': [
                'Arthrobacter', 'Rhodococcus', 'Pseudomonas', 'Sphingomonas',
                'Nocardioides', 'Mycobacterium', 'Streptomyces'
            ],
        }

    def get_target_genera(self, function: str) -> List[str]:
        """Get genera associated with a function."""
        self._load_function_lookup()
        function_key = function.lower().replace(' ', '_')
        return self._function_to_genera.get(function_key, [])

    def recommend(
        self,
        target: Union[str, List[str]],  # Function name OR list of genera
        community: Optional[List[str]] = None,
        metabolite_profile: Optional[Dict[str, float]] = None,
        n_recommendations: int = 10,
        context: Optional[str] = None
    ) -> List[RecommendationResult]:
        """
        Recommend prebiotics to promote target taxa or function.

        Args:
            target: Either a function name ("nitrogen_fixation") or list of genera
            community: List of genera in the community (for competition analysis)
            metabolite_profile: Available metabolites {name: abundance}
            n_recommendations: Number of metabolites to recommend
            context: Optional text context for LLM

        Returns:
            List of RecommendationResult sorted by selectivity
        """
        # Step 1: Resolve target to genera
        if isinstance(target, str):
            target_genera = self.get_target_genera(target)
            if not target_genera:
                # Treat as single genus name
                target_genera = [target]
        else:
            target_genera = target

        # Step 2: Get GNN predictions for target genera
        target_predictions = {}
        for genus in target_genera:
            # Find matching organism in GNN
            for org in self.gnn.organisms:
                if org.lower().startswith(genus.lower()):
                    preds = self.gnn.predict(org)
                    # Merge predictions (average if multiple orgs per genus)
                    for met, score in preds.items():
                        if met not in target_predictions:
                            target_predictions[met] = []
                        target_predictions[met].append(score)
                    break

        if not target_predictions:
            return []

        # Average scores per metabolite
        target_scores = {
            met: sum(scores) / len(scores)
            for met, scores in target_predictions.items()
        }

        # Step 3: Get competitor predictions (if community provided)
        competitor_scores = defaultdict(list)
        if community:
            for genus in community:
                if genus.lower() not in [g.lower() for g in target_genera]:
                    for org in self.gnn.organisms:
                        if org.lower().startswith(genus.lower()):
                            preds = self.gnn.predict(org)
                            for met, score in preds.items():
                                competitor_scores[met].append((genus, score))
                            break

        # Step 4: Compute selectivity scores
        # Negative score = consumption, we want metabolites the target consumes strongly
        results = []
        for met, target_score in target_scores.items():
            # Skip unknown/weird metabolites and those not consumed
            if 'Unk_' in met or target_score > -1.0:
                continue

            # Selectivity = how much target consumes vs competitors
            # target_score is negative (e.g., -5 = strong consumption)
            # Higher absolute value = stronger consumption
            target_consumption = abs(target_score)

            competitors_consuming = []
            competitor_total = 0
            for genus, score in competitor_scores.get(met, []):
                if score < -1.0:  # Competitor also consumes
                    competitors_consuming.append(genus)
                    competitor_total += abs(score)

            # Selectivity = target consumption - weighted competitor consumption
            # Higher = better (target consumes more, competitors consume less)
            selectivity = target_consumption - 0.3 * competitor_total

            # Step 5: KG competitive analysis
            self._load_kg_data()
            kg_competitors = []
            if community and self._chebi_to_genus:
                # Find matching ChEBI ID
                for chebi_id, kg_name in self._metabolite_names.items():
                    if self._normalize_name(met) == self._normalize_name(kg_name):
                        kg_consumers = self._chebi_to_genus.get(chebi_id, set())
                        kg_competitors = [
                            g for g in community
                            if g.lower() in kg_consumers
                        ]
                        break

            warning = None
            if len(competitors_consuming) > 3:
                warning = f"Low selectivity: {len(competitors_consuming)} competitors also consume this"
            elif kg_competitors and len(kg_competitors) > len(competitors_consuming):
                warning = f"KG suggests additional competitors: {kg_competitors[:3]}"

            results.append(RecommendationResult(
                metabolite=met,
                gnn_score=selectivity,
                target_affinity=target_consumption,
                competitor_count=len(competitors_consuming),
                competitors=competitors_consuming[:5],
                selectivity_warning=warning,
                kg_context={'kg_competitors': kg_competitors}
            ))

        # Sort by selectivity score (higher = better)
        results.sort(key=lambda x: x.gnn_score, reverse=True)
        return results[:n_recommendations]

    def predict_response(
        self,
        metabolite_profile: Dict[str, float],
        community: List[str],
        n_top: int = 10
    ) -> Tuple[List[CommunityResponse], List[CommunityResponse]]:
        """
        Predict which community members will benefit/suffer from metabolite profile.

        Args:
            metabolite_profile: {metabolite_name: abundance}
            community: List of genera in the community
            n_top: Number of top benefited/harmed to return

        Returns:
            (benefited, harmed) - Lists of CommunityResponse
        """
        self._load_kg_data()

        responses = []
        for genus in community:
            # Get GNN affinity
            affinity = 0.0
            found = False
            for org in self.gnn.organisms:
                if org.lower().startswith(genus.lower()):
                    found = True
                    result = self.gnn.predict_community_response(
                        metabolite_profile,
                        community=[org],
                        n_top=1
                    )
                    if result['benefited']:
                        affinity = result['benefited'][0][1]
                    elif result['harmed']:
                        affinity = -result['harmed'][0][1]
                    break

            # Get KG consumer count
            kg_count = 0
            for met in metabolite_profile.keys():
                for chebi_id, kg_name in self._metabolite_names.items():
                    if self._normalize_name(met) == self._normalize_name(kg_name):
                        if genus.lower() in self._chebi_to_genus.get(chebi_id, set()):
                            kg_count += 1
                        break

            direction = "neutral"
            if affinity > 0:
                direction = "benefited"
            elif affinity < 0:
                direction = "harmed"

            responses.append(CommunityResponse(
                genus=genus,
                gnn_affinity=affinity,
                direction=direction,
                kg_consumers=kg_count
            ))

        # Sort and split
        responses.sort(key=lambda x: x.gnn_affinity, reverse=True)
        benefited = [r for r in responses if r.direction == "benefited"][:n_top]
        harmed = [r for r in responses if r.direction == "harmed"][:n_top]

        return benefited, harmed

    @property
    def metabolite_unifier(self):
        """Lazy load metabolite unifier."""
        if self._metabolite_unifier is None:
            from simple_organism_unifier import SimpleMetaboliteUnifier
            self._metabolite_unifier = SimpleMetaboliteUnifier()
        return self._metabolite_unifier

    def _normalize_name(self, name: str) -> str:
        """Normalize metabolite name for matching using full synonym database."""
        return self.metabolite_unifier.normalize(name)


def main():
    """Demo the unified recommender."""
    print("=" * 70)
    print("SOILFM UNIFIED RECOMMENDER DEMO")
    print("=" * 70)

    recommender = SoilFMRecommender()

    # Check what genera we have in GNN
    print("\n[0] GNN ORGANISM COVERAGE")
    print("-" * 70)
    gnn_genera = set()
    for org in recommender.gnn.organisms:
        genus = org.split()[0]
        gnn_genera.add(genus)
    print(f"GNN has {len(gnn_genera)} unique genera")
    print(f"Sample: {sorted(list(gnn_genera))[:15]}")

    # Demo 1: Recommend prebiotics for specific genus in GNN (Arthrobacter)
    print("\n[1] RECOMMEND PREBIOTICS TO PROMOTE ARTHROBACTER")
    print("-" * 70)

    community = ['Pseudomonas', 'Bacillus', 'Chitinophaga', 'Massilia',
                 'Pedobacter', 'Arthrobacter', 'Lysobacter', 'Cupriavidus']

    results = recommender.recommend(
        target=['Arthrobacter'],  # Single target genus
        community=community,
        n_recommendations=5
    )

    print(f"Target: Arthrobacter")
    print(f"Community: {community}")
    print(f"\nTop recommendations (sorted by selectivity):")
    print("  (Selectivity = target consumption - 0.3 * competitor consumption)")
    print("  Higher = more selective for target")
    print()
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r.metabolite}")
        print(f"     Selectivity: {r.gnn_score:.2f} | Target consumes: {r.target_affinity:.2f}")
        print(f"     Competitors consuming: {r.competitor_count} ({', '.join(r.competitors[:3])})")
        if r.selectivity_warning:
            print(f"     WARNING: {r.selectivity_warning}")

    # Demo 2: Predict community response to exudate profile
    print("\n\n[2] PREDICT COMMUNITY RESPONSE TO ROOT EXUDATE")
    print("-" * 70)

    exudate = {
        'glutamic acid': 1.0,
        'aspartic acid': 0.8,
        'tyrosine': 0.5,
        'phenylalanine': 0.7,
        'adenine': 0.3,
    }

    benefited, harmed = recommender.predict_response(
        metabolite_profile=exudate,
        community=community,
        n_top=5
    )

    print(f"Exudate profile: {list(exudate.keys())}")
    print(f"\nPredicted benefited:")
    for r in benefited:
        print(f"  {r.genus}: affinity={r.gnn_affinity:.2e}, KG matches={r.kg_consumers}")

    print(f"\nPredicted harmed:")
    for r in harmed:
        print(f"  {r.genus}: affinity={r.gnn_affinity:.2e}")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
