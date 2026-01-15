#!/usr/bin/env python3
"""
LLM Expert Synthesis Module for SoilFM Pipeline.

This module integrates a local LLM (Qwen 14B or similar) to:
1. Interpret BSPR (Bayesian Substrate Preference Ranking) predictions
2. Add KG pathway context
3. Synthesize reasoned recommendations
4. Flag concerns and suggest validation
5. Detect red flags (e.g., all microbes predicted to increase)
"""

import json
import requests
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class LLMConfig:
    """Configuration for LLM backend."""
    endpoint: str = "http://localhost:8000/v1/chat/completions"
    model: str = "Qwen/Qwen2.5-14B-Instruct"
    max_tokens: int = 1024
    temperature: float = 0.3  # Low for factual responses


SYSTEM_PROMPT = """You are an expert in soil microbiology and metabolomics helping a researcher
design targeted prebiotics for rhizosphere engineering.

You have access to two data sources:
1. **BSPR (Bayesian Substrate Preference Ranking)**: A model trained on exometabolomics data
   that predicts substrate preferences at the GENUS level (AUC 0.94 on held-out validation).
   Higher affinity = more consumption. NOTE: All predictions are at genus level - species
   within the same genus receive identical predictions.
2. **Knowledge Graph (KG)**: Metabolic pathway information from BacDive/KEGG.
   Less predictive but captures pathway presence/absence.

Your role is to INTERPRET and SYNTHESIZE these signals - NOT to override the BSPR predictions.

IMPORTANT RED FLAG: If the BSPR predicts that ALL microbes in a community are "benefited"
(increasing) and NONE are "harmed" (decreasing), this contradicts exometabolite niche
partitioning theory. In this case, include this warning:
"Warning: all microbes are predicted to increase, which contradicts exometabolite niche
partitioning theory. This metabolite profile may not be selective."

When responding:
- Trust the BSPR predictions but note they are genus-level
- Use KG for pathway context and competitive analysis
- Flag concerns when BSPR and KG disagree
- Be concise but thorough
- Cite specific scores when relevant"""


def build_recommendation_prompt(
    user_goal: str,
    target_taxa: List[str],
    community: List[str],
    spm_recommendations: List[Dict],
    kg_context: Dict,
    user_context: Optional[str] = None
) -> str:
    """Build prompt for prebiotic recommendation task."""

    # Format SPM recommendations
    spm_text = "Top recommended metabolites from BSPR:\n"
    for i, rec in enumerate(spm_recommendations[:5], 1):
        spm_text += f"  {i}. {rec['metabolite']}\n"
        spm_text += f"     - Target affinity: {rec['target_affinity']:.2f}\n"
        spm_text += f"     - Selectivity score: {rec['selectivity']:.2f}\n"
        spm_text += f"     - Competitors: {rec['competitors']}\n"
        if rec.get('warning'):
            spm_text += f"     - Warning: {rec['warning']}\n"

    # Format KG context
    kg_text = "Knowledge Graph context:\n"
    if kg_context.get('target_pathways'):
        kg_text += f"  Target pathways: {kg_context['target_pathways']}\n"
    if kg_context.get('competitor_overlap'):
        kg_text += f"  Competitor overlap: {kg_context['competitor_overlap']}\n"
    if kg_context.get('warnings'):
        kg_text += f"  KG warnings: {kg_context['warnings']}\n"

    prompt = f"""## User's Goal
{user_goal}

## Target Taxa
{', '.join(target_taxa)}

## Community Context
{', '.join(community)}

{f"## Additional Context" + chr(10) + user_context if user_context else ""}

## Model Predictions

### BSPR Output (Genus-level, AUC 0.94)
{spm_text}

### KG Context
{kg_text}

## Your Task

Based on the BSPR predictions (genus-level) and KG context:

1. **Recommendation**: Which metabolite(s) best achieve the user's goal? Select from SPM top picks.

2. **Reasoning**: Explain why, referencing:
   - SPM affinity scores (quantitative support)
   - Relevant metabolic pathways from KG
   - Selectivity concerns

3. **Caveats**: Note any concerns:
   - Potential off-target effects
   - Pathway conflicts
   - Missing information

4. **Validation**: Suggest how to experimentally test this recommendation.

Be concise. Trust the SPM predictions."""

    return prompt


def build_community_response_prompt(
    metabolite_profile: Dict[str, float],
    community_predictions: List[Dict],
    kg_context: Dict,
    user_context: Optional[str] = None
) -> str:
    """Build prompt for community response prediction task."""

    # Format metabolite profile
    profile_text = "Metabolite profile:\n"
    for met, intensity in sorted(metabolite_profile.items(), key=lambda x: -x[1])[:10]:
        profile_text += f"  - {met}: {intensity:.2f}\n"

    # Format community predictions
    pred_text = "SPM predictions (who benefits):\n"
    benefited = [p for p in community_predictions if p['direction'] == 'benefited']
    harmed = [p for p in community_predictions if p['direction'] == 'harmed']

    pred_text += "  Benefited:\n"
    for p in benefited[:5]:
        pred_text += f"    - {p['genus']}: affinity={p['affinity']:.2e}\n"

    if harmed:
        pred_text += "  Harmed:\n"
        for p in harmed[:5]:
            pred_text += f"    - {p['genus']}: affinity={p['affinity']:.2e}\n"

    prompt = f"""## Metabolite Profile
{profile_text}

## Community
{', '.join([p['genus'] for p in community_predictions])}

{f"## Context" + chr(10) + user_context if user_context else ""}

## Model Predictions

### BSPR Community Response (Genus-level)
{pred_text}

### KG Pathway Context
{json.dumps(kg_context, indent=2) if kg_context else "No pathway data available."}

## Your Task

Interpret these predictions:

1. **Winners**: Which taxa will benefit most and why?

2. **Losers**: Which taxa may be disadvantaged?

3. **Ecological implications**: How might this shift the community?

4. **Confidence**: How confident are these predictions? Note any caveats.

Be concise."""

    return prompt


def build_blind_spot_prompt(
    observation: str,
    bspr_prediction: Dict,
    kg_context: Dict
) -> str:
    """Build prompt for explaining BSPR blind spots (like Arthrobacter/dopamine)."""

    prompt = f"""## Observed Result
{observation}

## BSPR Prediction
{json.dumps(bspr_prediction, indent=2)}

## KG Pathway Context
{json.dumps(kg_context, indent=2) if kg_context else "Limited pathway data available."}

## Your Task

The BSPR prediction differs from the observed result. Explain this discrepancy:

1. **Why might the BSPR be wrong?**
   - Training data limitations
   - Metabolic mechanisms not captured
   - Ecological factors

2. **What does the KG or literature suggest?**
   - Relevant pathways
   - Known metabolic capabilities
   - Ecological interactions

3. **How could the model be improved?**
   - Additional training data
   - New features
   - Pathway integration

Be specific and cite mechanisms where possible."""

    return prompt


class LLMSynthesizer:
    """LLM-based synthesis layer for SoilFM pipeline."""

    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig()
        self._check_connection()

    def _check_connection(self):
        """Check if LLM endpoint is available."""
        try:
            response = requests.get(
                self.config.endpoint.replace('/chat/completions', '/models'),
                timeout=5
            )
            if response.status_code == 200:
                print(f"Connected to LLM at {self.config.endpoint}")
                return True
        except requests.exceptions.RequestException:
            pass
        print(f"Warning: LLM endpoint not available at {self.config.endpoint}")
        return False

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API."""
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }

        try:
            response = requests.post(
                self.config.endpoint,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            return f"Error calling LLM: {e}"

    def recommend(
        self,
        user_goal: str,
        target_taxa: List[str],
        community: List[str],
        spm_recommendations: List[Dict],
        kg_context: Dict = None,
        user_context: str = None
    ) -> str:
        """Get LLM recommendation synthesis."""
        prompt = build_recommendation_prompt(
            user_goal=user_goal,
            target_taxa=target_taxa,
            community=community,
            spm_recommendations=spm_recommendations,
            kg_context=kg_context or {},
            user_context=user_context
        )
        return self._call_llm(prompt)

    def predict_response(
        self,
        metabolite_profile: Dict[str, float],
        community_predictions: List[Dict],
        kg_context: Dict = None,
        user_context: str = None
    ) -> str:
        """Get LLM interpretation of community response."""
        prompt = build_community_response_prompt(
            metabolite_profile=metabolite_profile,
            community_predictions=community_predictions,
            kg_context=kg_context or {},
            user_context=user_context
        )
        return self._call_llm(prompt)

    def explain_blind_spot(
        self,
        observation: str,
        bspr_prediction: Dict,
        kg_context: Dict = None
    ) -> str:
        """Get LLM explanation for BSPR prediction failures."""
        prompt = build_blind_spot_prompt(
            observation=observation,
            bspr_prediction=bspr_prediction,
            kg_context=kg_context or {}
        )
        return self._call_llm(prompt)


def main():
    """Test the LLM synthesis module."""
    print("=" * 70)
    print("LLM SYNTHESIS MODULE TEST")
    print("=" * 70)

    # Test without actual LLM (just prompt generation)
    print("\n[1] Testing prompt generation...")

    # Example recommendation task
    spm_recs = [
        {
            'metabolite': 'glutamic acid',
            'target_affinity': 6.16,
            'selectivity': -5.20,
            'competitors': ['Pseudomonas', 'Bacillus', 'Chitinophaga'],
            'warning': 'Low selectivity - all competitors consume'
        },
        {
            'metabolite': 'pantothenic acid',
            'target_affinity': 1.09,
            'selectivity': -0.25,
            'competitors': ['Pseudomonas', 'Chitinophaga'],
            'warning': None
        }
    ]

    kg_context = {
        'target_pathways': ['Aromatic degradation', 'Amino acid metabolism'],
        'competitor_overlap': 'High - common amino acid consumers',
        'warnings': None
    }

    prompt = build_recommendation_prompt(
        user_goal="Promote Arthrobacter growth in soil",
        target_taxa=['Arthrobacter'],
        community=['Pseudomonas', 'Bacillus', 'Chitinophaga', 'Arthrobacter'],
        spm_recommendations=spm_recs,
        kg_context=kg_context
    )

    print("\nGenerated recommendation prompt:")
    print("-" * 50)
    print(prompt[:1500] + "..." if len(prompt) > 1500 else prompt)

    # Example blind spot explanation
    print("\n\n[2] Testing blind spot prompt...")

    observation = """Dopamine addition to soil caused Arthrobacter to increase 6.5x,
making it the dominant responder. Other Actinobacteria (Nocardioides, Agromyces)
also increased 5-6x. Proteobacteria showed no response."""

    bspr_pred = {
        'metabolite': 'dopamine',
        'Arthrobacter_score': 1.52,
        'prediction': 'PRODUCES (not consumes)',
        'related_metabolites': {
            'tyrosine': -5.39,
            'phenylalanine': -5.55
        }
    }

    kg_ctx = {
        'Arthrobacter_pathways': ['Aromatic compound degradation', 'Catechol metabolism'],
        'dopamine_pathway': 'Tyrosine metabolism / Catecholamine biosynthesis',
        'note': 'Actinobacteria are known aromatic degraders'
    }

    blind_prompt = build_blind_spot_prompt(
        observation=observation,
        bspr_prediction=bspr_pred,
        kg_context=kg_ctx
    )

    print("\nGenerated blind spot prompt:")
    print("-" * 50)
    print(blind_prompt)

    # Try to connect to LLM
    print("\n\n[3] Testing LLM connection...")
    synthesizer = LLMSynthesizer()

    print("\n" + "=" * 70)
    print("PROMPT GENERATION COMPLETE")
    print("=" * 70)
    print("""
To test with actual LLM:
1. Start vLLM server with Qwen2.5-14B-Instruct
2. Set endpoint in LLMConfig
3. Call synthesizer.recommend() or synthesizer.explain_blind_spot()
""")


if __name__ == '__main__':
    main()

    def validate_predictions_with_kg(
        self,
        predictions: List[Dict],
        genus: str,
        kg_pathways: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Validate predictions against KG pathway data.

        Args:
            predictions: List of metabolite predictions with affinities
            genus: Target genus name
            kg_pathways: Knowledge graph pathway data for the genus

        Returns:
            Predictions enriched with validation flags
        """
        validated = []

        for pred in predictions:
            validated_pred = pred.copy()
            metabolite = pred.get('metabolite', '')
            affinity = pred.get('affinity', 0)

            # Default: no validation issues
            validated_pred['kg_validated'] = True
            validated_pred['kg_warning'] = None

            # If KG data available, check for conflicts
            if kg_pathways:
                # Check if metabolite makes biological sense
                # For example: amino acid degradation pathways
                if 'amino acid' in metabolite.lower() or metabolite in [
                    'glutamic acid', 'aspartic acid', 'alanine', 'leucine',
                    'valine', 'isoleucine', 'phenylalanine', 'tyrosine'
                ]:
                    # Check for amino acid metabolism pathways
                    has_aa_pathways = any('amino acid' in str(p).lower()
                                         for p in kg_pathways.values())
                    if not has_aa_pathways and abs(affinity) > 3.0:
                        validated_pred['kg_validated'] = False
                        validated_pred['kg_warning'] = f"High affinity for {metabolite} but no amino acid metabolism pathways found in KG"

                # Check for aromatic compounds
                if any(arom in metabolite.lower() for arom in ['benzoate', 'phenol', 'catechol', 'dopamine', 'tyrosine']):
                    has_aromatic_pathways = any('aromatic' in str(p).lower() or 'catechol' in str(p).lower()
                                               for p in kg_pathways.values())
                    if not has_aromatic_pathways and abs(affinity) > 3.0:
                        validated_pred['kg_validated'] = False
                        validated_pred['kg_warning'] = f"High affinity for aromatic compound {metabolite} but no aromatic degradation pathways in KG"

                # Check for sugars
                if any(sugar in metabolite.lower() for sugar in ['glucose', 'fructose', 'sucrose', 'maltose']):
                    has_carb_pathways = any('carbohydrate' in str(p).lower() or 'glycolysis' in str(p).lower()
                                           for p in kg_pathways.values())
                    if not has_carb_pathways and abs(affinity) > 3.0:
                        validated_pred['kg_validated'] = False
                        validated_pred['kg_warning'] = f"High affinity for sugar {metabolite} but limited carbohydrate metabolism pathways in KG"

            validated.append(validated_pred)

        return validated

    def flag_unlikely_predictions(
        self,
        community_predictions: List[Dict]
    ) -> Dict[str, Any]:
        """
        Flag predictions that seem unlikely based on ecological theory.

        Args:
            community_predictions: List of genus predictions with affinities and directions

        Returns:
            Dictionary with flags and warnings
        """
        benefited = [p for p in community_predictions if p.get('direction') == 'benefited']
        harmed = [p for p in community_predictions if p.get('direction') == 'harmed']

        flags = {
            'all_benefit': False,
            'none_harmed': False,
            'warnings': []
        }

        # Flag 1: All microbes benefit (violates niche partitioning)
        if len(benefited) == len(community_predictions) and len(benefited) > 2:
            flags['all_benefit'] = True
            flags['warnings'].append(
                "Warning: All microbes predicted to increase. This contradicts exometabolite "
                "niche partitioning theory. The metabolite profile may not be selective."
            )

        # Flag 2: No microbes harmed (suspicious if community > 3)
        if len(harmed) == 0 and len(community_predictions) > 3:
            flags['none_harmed'] = True
            flags['warnings'].append(
                "Note: No microbes predicted to decrease. This is unusual for larger communities "
                "and may indicate limited metabolite diversity or modeling limitations."
            )

        # Flag 3: Extreme dominance (one microbe >>  others)
        if benefited:
            max_affinity = max(abs(p.get('affinity', 0)) for p in benefited)
            avg_affinity = sum(abs(p.get('affinity', 0)) for p in benefited) / len(benefited)

            if max_affinity > 3 * avg_affinity and max_affinity > 5.0:
                dominant = [p for p in benefited if abs(p.get('affinity', 0)) == max_affinity][0]
                flags['warnings'].append(
                    f"Note: {dominant.get('genus', 'Unknown')} shows dominant response "
                    f"(affinity={dominant.get('affinity', 0):.1f}). Consider if this aligns with your goal."
                )

        return flags
