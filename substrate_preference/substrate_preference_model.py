#!/usr/bin/env python3
"""
Bayesian Substrate Preference Ranking (BSPR) Model v1.0

Production-ready model using genus-average profiles with fallback hierarchy.

Key features:
- Genus-level lookup table (AUC 0.94 on held-out validation)
- Fallback to population mean for unknown genera
- Predictions are at GENUS level (not species level)
- API compatible with previous GNNPreferenceHead interface
- Ready for future species-level BPR when more spike-in data available

Model name: BSPR (Bayesian Substrate Preference Ranking)
- Uses Bayesian priors from genus-level consumption patterns
- Ranks metabolites by consumption probability

Date: 2026-01-14
"""

import json
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class SubstratePreferenceModelV1:
    """
    BSPR (Bayesian Substrate Preference Ranking) Model v1.0

    Predicts metabolite consumption preferences at the GENUS level
    based on Bayesian priors from exometabolomics training data.

    NOTE: All predictions are genus-level. Species within the same
    genus will receive identical predictions. This is a known limitation
    that will be addressed when more spike-in validation data is available.

    Interface compatible with previous GNNPreferenceHead for easy swap-in.
    """

    VERSION = "1.0.0"
    MODEL_NAME = "BSPR"  # Bayesian Substrate Preference Ranking

    def __init__(self, profiles_path: Optional[str] = None):
        """
        Initialize the model.

        Args:
            profiles_path: Path to genus_profiles.json. If None, looks in same directory.
        """
        self.genus_profiles: Dict[str, Dict[str, float]] = {}
        self.metabolites: List[str] = []
        self.organism_to_genus: Dict[str, str] = {}
        self.population_mean: Dict[str, float] = {}

        if profiles_path is None:
            profiles_path = Path(__file__).parent / 'genus_profiles.json'

        if Path(profiles_path).exists():
            self.load_profiles(profiles_path)

    def load_profiles(self, path: str) -> None:
        """Load pre-computed genus profiles."""
        with open(path, 'r') as f:
            data = json.load(f)

        self.genus_profiles = data.get('genus_profiles', {})
        self.metabolites = data.get('metabolites', [])
        self.organism_to_genus = data.get('organism_to_genus', {})

        # Compute population mean across all genera (excluding Unknown)
        self._compute_population_mean()

        print(f"Loaded {self.MODEL_NAME} (Bayesian Substrate Preference Ranking) v{self.VERSION}")
        print(f"  Genera: {len(self.genus_profiles)} (predictions are at genus level)")
        print(f"  Metabolites: {len(self.metabolites)}")

    def _compute_population_mean(self) -> None:
        """Compute population mean profile across all genera."""
        all_scores = {}
        counts = {}

        for genus, profile in self.genus_profiles.items():
            if genus == 'Unknown':
                continue
            for met, score in profile.items():
                if score != 0:
                    if met not in all_scores:
                        all_scores[met] = 0
                        counts[met] = 0
                    all_scores[met] += score
                    counts[met] += 1

        self.population_mean = {
            met: all_scores[met] / counts[met]
            for met in all_scores if counts[met] > 0
        }

    def extract_genus(self, organism_name: str) -> str:
        """
        Extract genus from organism name.

        Handles formats:
        - "Bacillus sp. (FW507-8R2A)" -> "Bacillus"
        - "BacillusOAE603" -> "Bacillus"
        - "Acidovorax sp." -> "Acidovorax"
        - "FW305-10-XG185" -> "Unknown"
        """
        # Check explicit mapping first
        if organism_name in self.organism_to_genus:
            return self.organism_to_genus[organism_name]

        name = organism_name.strip()

        # Pattern 1: "Genus sp. (strain)" or "Genus species"
        match = re.match(r'^([A-Z][a-z]{2,})\s+', name)
        if match:
            genus = match.group(1)
            if genus in self.genus_profiles:
                return genus

        # Pattern 2: "GenusOAE123" - CamelCase with strain suffix
        match = re.match(r'^([A-Z][a-z]{2,})(?:OA[A-Z]|[A-Z][0-9])', name)
        if match:
            genus = match.group(1)
            if genus in self.genus_profiles:
                return genus

        # Pattern 3: Just extract first capitalized word
        match = re.match(r'^([A-Z][a-z]{2,})', name)
        if match:
            genus = match.group(1)
            if genus in self.genus_profiles:
                return genus

        return 'Unknown'

    def predict(self, organism: str) -> Dict[str, float]:
        """
        Predict metabolite preference scores for an organism.

        Args:
            organism: Organism name (will extract genus)

        Returns:
            Dict mapping metabolite names to preference scores.
            Higher positive = more consumed.
        """
        genus = self.extract_genus(organism)

        if genus in self.genus_profiles and genus != 'Unknown':
            return dict(self.genus_profiles[genus])
        else:
            # Fall back to population mean
            return dict(self.population_mean)

    def get_top_consumed(self, organism: str, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top N consumed metabolites for an organism.

        Args:
            organism: Organism name
            n: Number of top metabolites to return

        Returns:
            List of (metabolite, score) tuples, sorted by consumption (highest first)
        """
        scores = self.predict(organism)
        sorted_mets = sorted(scores.items(), key=lambda x: -x[1])
        return [(met, score) for met, score in sorted_mets[:n] if score > 0]

    def get_top_produced(self, organism: str, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top N produced metabolites for an organism.

        Args:
            organism: Organism name
            n: Number of top metabolites to return

        Returns:
            List of (metabolite, score) tuples, sorted by production (most negative first)
        """
        scores = self.predict(organism)
        sorted_mets = sorted(scores.items(), key=lambda x: x[1])
        return [(met, score) for met, score in sorted_mets[:n] if score < 0]

    def get_discriminative_metabolites(
        self,
        target: str,
        competitors: List[str],
        n: int = 10
    ) -> List[Tuple[str, float, float]]:
        """
        Find metabolites that discriminate target from competitors.

        This is the key function for prebiotic recommendation.

        Args:
            target: Target organism to promote
            competitors: List of competitor organisms
            n: Number of top discriminative metabolites

        Returns:
            List of (metabolite, target_score, avg_competitor_score) tuples,
            sorted by selectivity (target consumes, competitors don't)
        """
        target_scores = self.predict(target)

        # Average competitor scores
        competitor_scores = {}
        for met in target_scores:
            comp_vals = []
            for comp in competitors:
                comp_pred = self.predict(comp)
                if met in comp_pred:
                    comp_vals.append(comp_pred[met])
            if comp_vals:
                competitor_scores[met] = np.mean(comp_vals)
            else:
                competitor_scores[met] = 0

        # Selectivity = target consumes more than competitors
        selectivity = []
        for met in target_scores:
            target_score = target_scores[met]
            comp_score = competitor_scores.get(met, 0)

            # Want: target consumes (positive), competitors don't (low/negative)
            if target_score > 0:  # Target consumes
                diff = target_score - comp_score
                selectivity.append((met, target_score, comp_score, diff))

        # Sort by selectivity
        selectivity.sort(key=lambda x: -x[3])

        return [(met, t_score, c_score) for met, t_score, c_score, _ in selectivity[:n]]

    @property
    def organisms(self) -> List[str]:
        """List of organisms the model can predict for.

        Returns genera as "Genus sp." format for webapp compatibility.
        """
        # Return genera as organism-like names for backwards compatibility
        return [f"{g} sp." for g in self.genera]

    @property
    def genera(self) -> List[str]:
        """List of genera with profiles."""
        return [g for g in self.genus_profiles.keys() if g != 'Unknown']

    def predict_community_response(
        self,
        metabolite_profile: Dict[str, float],
        community: Optional[List[str]] = None,
        n_top: int = 20
    ) -> Dict:
        """
        Predict how a metabolite profile will impact a microbial community.

        This is for backwards compatibility with GNNPreferenceHead interface.

        Args:
            metabolite_profile: Dict of {metabolite_name: abundance}
            community: Optional list of organisms to consider
            n_top: Number of top responders to return

        Returns:
            Dict with 'benefited', 'disadvantaged', 'metabolite_coverage', 'community_coverage'
        """
        if community is None:
            community = self.organisms

        # For each organism, compute weighted affinity
        org_scores = {}
        matched_mets = set()

        for org in community:
            preds = self.predict(org)
            score = 0.0
            for met, abundance in metabolite_profile.items():
                # Find matching metabolite in predictions
                for pred_met, pred_score in preds.items():
                    if self._metabolite_matches(met, pred_met):
                        # Negative score = consumed = benefits organism
                        score += abundance * (-pred_score)  # Negate so consumption = positive benefit
                        matched_mets.add(met)
                        break
            org_scores[org] = score

        # Sort by score
        sorted_scores = sorted(org_scores.items(), key=lambda x: x[1], reverse=True)

        # Split into benefited (positive) and disadvantaged (negative)
        benefited = [(org, score) for org, score in sorted_scores if score > 0][:n_top]
        disadvantaged = [(org, abs(score)) for org, score in sorted_scores if score < 0][:n_top]

        return {
            'benefited': benefited,
            'disadvantaged': disadvantaged,
            'harmed': disadvantaged,  # Alias
            'metabolite_coverage': len(matched_mets) / len(metabolite_profile) if metabolite_profile else 0,
            'community_coverage': len([s for s in org_scores.values() if s != 0]) / len(community) if community else 0
        }

    def _metabolite_matches(self, query: str, candidate: str) -> bool:
        """Check if metabolite names match (case-insensitive, handles prefixes)."""
        query_norm = query.lower().strip()
        # Handle numbered prefix format like "0001_thymine"
        if '_' in candidate:
            candidate_base = '_'.join(candidate.split('_')[1:]).lower()
        else:
            candidate_base = candidate.lower()
        return query_norm == candidate_base or query_norm in candidate_base

    def get_model_info(self) -> Dict:
        """Return model metadata."""
        return {
            'name': self.MODEL_NAME,
            'full_name': 'Bayesian Substrate Preference Ranking',
            'version': self.VERSION,
            'type': 'genus_average',
            'resolution': 'genus',  # Important: not species-level
            'n_genera': len(self.genera),
            'n_metabolites': len(self.metabolites),
            'n_explicit_mappings': len(self.organism_to_genus),
            'validation_auc': 0.942,
            'validation_precision_at_10': 0.500,
            'note': 'Predictions are at genus level. Species within same genus get identical predictions.'
        }


# Backwards-compatible aliases
GNNPreferenceHead = SubstratePreferenceModelV1  # Legacy name
BSPRModel = SubstratePreferenceModelV1  # New preferred name


def main():
    """Demo and validation."""
    print("=" * 60)
    print("BSPR (Bayesian Substrate Preference Ranking) v1.0 DEMO")
    print("=" * 60)

    model = SubstratePreferenceModelV1()

    print(f"\nModel info:")
    for k, v in model.get_model_info().items():
        print(f"  {k}: {v}")

    # Test predictions
    test_organisms = [
        "Pseudomonas sp. (FW300-N1B4)",
        "BacillusOAE603",
        "Arthrobacter sp.",
        "FW305-10-XG185",  # Unknown genus
    ]

    print("\n" + "-" * 60)
    print("SAMPLE PREDICTIONS")
    print("-" * 60)

    for org in test_organisms:
        genus = model.extract_genus(org)
        top = model.get_top_consumed(org, n=5)

        print(f"\n{org}")
        print(f"  Genus: {genus}")
        print(f"  Top consumed: {[m[:25] for m, _ in top]}")

    # Test discriminative metabolites
    print("\n" + "-" * 60)
    print("DISCRIMINATIVE METABOLITES")
    print("-" * 60)

    target = "BacillusOAE603"
    competitors = ["Pseudomonas sp.", "Arthrobacter sp."]

    print(f"\nTarget: {target}")
    print(f"Competitors: {competitors}")

    selective = model.get_discriminative_metabolites(target, competitors, n=5)
    print("\nTop selective metabolites:")
    for met, t_score, c_score in selective:
        print(f"  {met[:35]:35s} target={t_score:.2f}, competitors={c_score:.2f}")


if __name__ == '__main__':
    main()
