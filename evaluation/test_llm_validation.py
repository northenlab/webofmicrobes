#!/usr/bin/env python3
"""
GNN + LLM Validation Test Suite

Tests the complete SoilFM pipeline (GNN + KG + LLM) on three validation datasets:
1. Spike-in study (Arthrobacter/dopamine blind spot)
2. Zheng 2023 (16S recruitment)
3. Zhalnina 2018 (colonization)

Usage:
    # Test without LLM (just check GNN predictions)
    python test_gnn_llm_validation.py --no-llm

    # Full test with LLM
    python test_gnn_llm_validation.py --endpoint http://localhost:8000/v1/chat/completions

    # Test specific study
    python test_gnn_llm_validation.py --test spike_in
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "substrate_preference"))
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).parent))

from soilfm_recommender import SoilFMRecommender
from llm_synthesis import LLMSynthesizer, LLMConfig


def test_spike_in(recommender, synthesizer=None, verbose=True):
    """
    Test 1: Spike-in Study - Arthrobacter/Dopamine Blind Spot

    GNN predicts Arthrobacter PRODUCES dopamine (+1.52)
    Reality: Arthrobacter bloomed 6.5x when dopamine was added

    Can LLM explain this discrepancy using pathway knowledge?
    """
    print("\n" + "=" * 70)
    print("TEST 1: SPIKE-IN STUDY - ARTHROBACTER/DOPAMINE BLIND SPOT")
    print("=" * 70)

    # Get GNN predictions for Arthrobacter
    gnn = recommender.gnn

    # Find Arthrobacter
    arthro_org = None
    for org in gnn.organisms:
        if org.lower().startswith('arthrobacter'):
            arthro_org = org
            break

    if not arthro_org:
        print("ERROR: Arthrobacter not found in GNN")
        return {'test': 'spike_in', 'success': False, 'error': 'Arthrobacter not found'}

    preds = gnn.predict(arthro_org)

    # Get dopamine-related predictions
    dopamine_score = preds.get('dopamine', 0)
    tyramine_score = preds.get('tyramine', 0)
    tyrosine_scores = [(m, s) for m, s in preds.items() if 'tyrosine' in m.lower()]

    gnn_prediction = {
        'organism': arthro_org,
        'dopamine_score': float(dopamine_score),
        'dopamine_prediction': 'PRODUCES' if dopamine_score > 0 else 'CONSUMES',
        'tyramine_score': float(tyramine_score),
        'tyrosine_avg': float(sum(s for _, s in tyrosine_scores) / max(len(tyrosine_scores), 1)),
        'phenylalanine': float(preds.get('phenylalanine', 0))
    }

    observation = """In a soil spike-in experiment, dopamine addition caused:
- Arthrobacter: 6.5x - 136x increase (dominant responder)
- Nocardioides: 6.3x increase
- Agromyces: 5.6x increase
- Proteobacteria (Pseudomonas, Bacillus): NO significant response

All top responders are Actinobacteria, suggesting a phylum-level capability."""

    kg_context = {
        'Actinobacteria_capabilities': [
            'Aromatic compound degradation',
            'Catechol ring cleavage',
            'Biphenyl degradation',
            'PAH metabolism'
        ],
        'dopamine_metabolism': 'Catechol ring structure - can be degraded via catechol pathway',
        'literature': 'Arthrobacter species are well-known aromatic degraders (Karigar & Rao 2011)',
        'pathway_note': 'Dopamine has catechol structure - Actinobacteria have catechol dioxygenases'
    }

    print(f"\n[GNN PREDICTION]")
    print(f"  Organism: {arthro_org}")
    print(f"  Dopamine: {gnn_prediction['dopamine_score']:.2f} ({gnn_prediction['dopamine_prediction']})")
    print(f"  Tyramine: {gnn_prediction['tyramine_score']:.2f}")
    print(f"  Tyrosine (avg): {gnn_prediction['tyrosine_avg']:.2f}")
    print(f"  Phenylalanine: {gnn_prediction['phenylalanine']:.2f}")

    print(f"\n[OBSERVED RESULT]")
    print(observation)

    result = {
        'test': 'spike_in',
        'gnn_prediction': gnn_prediction,
        'observation': observation,
        'kg_context': kg_context,
        'llm_response': None
    }

    # Get LLM explanation if available
    if synthesizer:
        print("\n" + "-" * 50)
        print("[LLM EXPLANATION]")
        print("-" * 50)

        response = synthesizer.explain_blind_spot(
            observation=observation,
            gnn_prediction=gnn_prediction,
            kg_context=kg_context
        )
        print(response)
        result['llm_response'] = response

    result['success'] = True
    return result


def test_zheng(recommender, synthesizer=None, verbose=True):
    """
    Test 2: Zheng 2023 - 16S Recruitment

    GNN validated with rho=0.78, p=0.039
    Can LLM explain WHY certain genera were recruited?
    """
    print("\n" + "=" * 70)
    print("TEST 2: ZHENG 2023 - 16S RECRUITMENT INTERPRETATION")
    print("=" * 70)

    # Load Zheng data
    DATA_ROOT = Path(__file__).parent.parent
    validation_file = DATA_ROOT / "evaluation" / "gnn_recruitment_validation.json"

    if not validation_file.exists():
        print(f"ERROR: Validation file not found: {validation_file}")
        return {'test': 'zheng', 'success': False, 'error': 'Validation file not found'}

    with open(validation_file, 'r') as f:
        validation = json.load(f)

    # Top exudate metabolites (from brachy)
    metabolite_profile = {
        'betaine': 74.3,
        'guanine': 24.2,
        'cytosine': 6.0,
        'phenylalanine': 6.0,
        'adenosine': 5.3,
        'glutamic acid': 0.52,
        'aspartic acid': 0.50
    }

    # Format for LLM
    community_predictions = [
        {
            'genus': d['genus'],
            'affinity': d['gnn_affinity'],
            'direction': 'benefited' if d['gnn_affinity'] > 0 else 'harmed',
            'observed_log2fc': d['log2fc_recruit']
        }
        for d in validation['data']
    ]

    print(f"\n[VALIDATION RESULTS]")
    print(f"  Spearman rho: {validation['spearman_rho']:.3f}")
    print(f"  p-value: {validation['spearman_p']:.4f}")
    print(f"  n genera: {validation['n_overlap']}")

    print(f"\n[COMMUNITY PREDICTIONS]")
    for p in sorted(community_predictions, key=lambda x: -x['affinity']):
        print(f"  {p['genus']:20s}: GNN={p['affinity']:.2e}, observed={p['observed_log2fc']:+.2f}")

    kg_context = {
        'correlation': f"Spearman rho={validation['spearman_rho']:.3f}, p={validation['spearman_p']:.4f}",
        'interpretation': 'GNN predictions significantly correlate with observed recruitment',
        'top_recruited': ['Chitinophaga', 'Cupriavidus', 'Lysobacter'],
        'exudate_note': 'Brachypodium exudate rich in purines, amino acids, betaines'
    }

    result = {
        'test': 'zheng',
        'correlation': validation['spearman_rho'],
        'p_value': validation['spearman_p'],
        'community_predictions': community_predictions,
        'metabolite_profile': metabolite_profile,
        'kg_context': kg_context,
        'llm_response': None
    }

    # Get LLM interpretation if available
    if synthesizer:
        print("\n" + "-" * 50)
        print("[LLM INTERPRETATION]")
        print("-" * 50)

        response = synthesizer.predict_response(
            metabolite_profile=metabolite_profile,
            community_predictions=community_predictions,
            kg_context=kg_context,
            user_context="This is the Zheng 2023 Brachypodium rhizosphere study. All genera shown were detected in the rhizosphere."
        )
        print(response)
        result['llm_response'] = response

    result['success'] = True
    return result


def test_zhalnina(recommender, synthesizer=None, verbose=True):
    """
    Test 3: Zhalnina 2018 - Colonization Prediction

    Original study showed metabolite preferences predict colonization.
    Can our pipeline recommend prebiotics for specific targets?
    """
    print("\n" + "=" * 70)
    print("TEST 3: ZHALNINA-STYLE PREBIOTIC RECOMMENDATION")
    print("=" * 70)

    # Simulate Avena rhizosphere scenario
    community = [
        'Pseudomonas', 'Bacillus', 'Arthrobacter', 'Burkholderia',
        'Chitinophaga', 'Streptomyces', 'Sphingomonas', 'Cupriavidus'
    ]

    # Get recommendations for promoting nitrogen fixers
    print(f"\n[SCENARIO]")
    print(f"  Goal: Promote nitrogen-fixing bacteria")
    print(f"  Target: Rhizobium, Bradyrhizobium")
    print(f"  Community: {community}")

    results = recommender.recommend(
        target=['Rhizobium', 'Bradyrhizobium'],
        community=community,
        n_recommendations=5
    )

    # Format for LLM
    spm_recs = [
        {
            'metabolite': r.metabolite,
            'target_affinity': float(r.target_affinity),
            'selectivity': float(r.gnn_score),
            'competitors': r.competitors,
            'warning': r.selectivity_warning
        }
        for r in results
    ]

    print(f"\n[SPM RECOMMENDATIONS]")
    for i, r in enumerate(spm_recs[:5], 1):
        print(f"  {i}. {r['metabolite']}")
        print(f"     Selectivity: {r['selectivity']:.2f}, Affinity: {r['target_affinity']:.2f}")
        print(f"     Competitors: {r['competitors']}")
        if r['warning']:
            print(f"     Warning: {r['warning']}")

    kg_context = {
        'target_function': 'Nitrogen fixation',
        'target_taxa': ['Rhizobium', 'Bradyrhizobium'],
        'known_preferences': 'Rhizobia typically prefer organic acids, amino acids',
        'competition_note': 'High competition from Pseudomonas for most substrates'
    }

    result = {
        'test': 'zhalnina',
        'target': ['Rhizobium', 'Bradyrhizobium'],
        'community': community,
        'spm_recommendations': spm_recs,
        'kg_context': kg_context,
        'llm_response': None
    }

    # Get LLM recommendation if available
    if synthesizer:
        print("\n" + "-" * 50)
        print("[LLM RECOMMENDATION]")
        print("-" * 50)

        response = synthesizer.recommend(
            user_goal="Promote nitrogen-fixing bacteria (Rhizobium, Bradyrhizobium) in the rhizosphere to improve plant nitrogen availability.",
            target_taxa=['Rhizobium', 'Bradyrhizobium'],
            community=community,
            spm_recommendations=spm_recs,
            kg_context=kg_context,
            user_context="Based on the Zhalnina 2018 methodology for predicting colonization from metabolite preferences."
        )
        print(response)
        result['llm_response'] = response

    result['success'] = True
    return result


def main():
    parser = argparse.ArgumentParser(description='Test GNN + LLM validation pipeline')
    parser.add_argument('--endpoint', type=str,
                        default='http://localhost:8000/v1/chat/completions',
                        help='LLM API endpoint')
    parser.add_argument('--model', type=str,
                        default='Qwen/Qwen2.5-14B-Instruct',
                        help='Model name')
    parser.add_argument('--test', type=str, default='all',
                        choices=['all', 'spike_in', 'zheng', 'zhalnina'],
                        help='Which test to run')
    parser.add_argument('--no-llm', action='store_true',
                        help='Skip LLM (only test GNN predictions)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')

    args = parser.parse_args()

    print("=" * 70)
    print("SOILFM GNN + LLM VALIDATION TEST SUITE")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"LLM: {'Disabled' if args.no_llm else args.endpoint}")

    # Initialize components
    print("\n[1] Initializing pipeline...")
    recommender = SoilFMRecommender()
    print(f"    GNN loaded: {len(recommender.gnn.organisms)} organisms, {len(recommender.gnn.metabolites)} metabolites")

    synthesizer = None
    if not args.no_llm:
        config = LLMConfig(endpoint=args.endpoint, model=args.model)
        synthesizer = LLMSynthesizer(config=config)

    # Run tests
    results = {}

    if args.test in ['all', 'spike_in']:
        results['spike_in'] = test_spike_in(recommender, synthesizer)

    if args.test in ['all', 'zheng']:
        results['zheng'] = test_zheng(recommender, synthesizer)

    if args.test in ['all', 'zhalnina']:
        results['zhalnina'] = test_zhalnina(recommender, synthesizer)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for test_name, result in results.items():
        status = "[PASS]" if result.get('success') else "[FAIL]"
        llm_status = "with LLM" if result.get('llm_response') else "GNN only"
        print(f"  {status} {test_name}: {llm_status}")

    # Save results
    output_file = args.output or Path(__file__).parent / 'gnn_llm_validation_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'llm_enabled': not args.no_llm,
            'results': results
        }, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
