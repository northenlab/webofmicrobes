#!/usr/bin/env python3
"""
Test: Use KG for competitive pathway analysis.

Workflow:
1. GNN recommends a metabolite for target taxon
2. KG checks: How many OTHER taxa in the community can also use this metabolite?
3. Output: Selectivity score / competitive warning

This uses KG as a "second opinion" rather than a filter.
"""

import sys
import pickle
import torch
from pathlib import Path
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent / "substrate_preference"))

DATA_ROOT = Path(__file__).parent.parent
KG_DIR = DATA_ROOT / "knowledge_graph"


def load_kg_consumption():
    """Load KG consumption edges by genus."""
    kg_path = KG_DIR / "soilfm_filtered_kg.pt"
    mappings_path = KG_DIR / "filtered_node_mappings.pkl"

    kg = torch.load(kg_path, map_location='cpu', weights_only=False)
    with open(mappings_path, 'rb') as f:
        mappings = pickle.load(f)

    idx_to_microbe_id = mappings.get('idx_to_microbe', {})
    idx_to_metabolite_id = mappings.get('idx_to_metabolite', {})
    microbe_id_to_name = mappings.get('microbe_names', {})
    metabolite_id_to_name = mappings.get('metabolite_names', {})

    # Build genus -> set of ChEBI IDs they consume
    genus_consumes_chebi = defaultdict(set)

    # Also build reverse: ChEBI -> set of genera that consume it
    chebi_consumed_by = defaultdict(set)

    for edge_type in kg.edge_types:
        src_type, rel, dst_type = edge_type

        if 'consumes' in rel.lower():
            edge_index = kg[edge_type].edge_index
            n_edges = edge_index.shape[1]

            for i in range(n_edges):
                src_idx = edge_index[0, i].item()
                dst_idx = edge_index[1, i].item()

                microbe_id = idx_to_microbe_id.get(src_idx, "")
                metabolite_id = idx_to_metabolite_id.get(dst_idx, "")

                microbe_name = microbe_id_to_name.get(microbe_id, "")

                if microbe_name and metabolite_id:
                    # Extract genus
                    genus = extract_genus(microbe_name)
                    if genus:
                        genus_consumes_chebi[genus.lower()].add(metabolite_id)
                        chebi_consumed_by[metabolite_id].add(genus.lower())

    return genus_consumes_chebi, chebi_consumed_by, metabolite_id_to_name


def extract_genus(name):
    """Extract genus from species name."""
    for prefix in ['s__', 'g__', 'f__', 'o__', 'c__', 'p__', 'd__']:
        name = name.replace(prefix, '')
    name = name.replace('_', ' ')
    parts = name.split()
    if parts:
        return parts[0]
    return None


def competitive_analysis(target_genus, metabolite_chebi, community_genera,
                         genus_consumes, chebi_consumed_by, metabolite_names):
    """
    Analyze competitive landscape for a metabolite recommendation.

    Args:
        target_genus: The genus we want to promote
        metabolite_chebi: ChEBI ID of recommended metabolite
        community_genera: List of other genera in the community
        genus_consumes: Dict[genus] -> set of ChEBI IDs
        chebi_consumed_by: Dict[ChEBI] -> set of genera
        metabolite_names: Dict[ChEBI] -> name

    Returns:
        Dict with selectivity analysis
    """
    met_name = metabolite_names.get(metabolite_chebi, metabolite_chebi)

    # Who else in the KG can consume this?
    all_consumers = chebi_consumed_by.get(metabolite_chebi, set())

    # Intersection with community
    community_lower = [g.lower() for g in community_genera]
    community_consumers = [g for g in community_lower if g in all_consumers]

    # Does target consume it?
    target_consumes = target_genus.lower() in all_consumers

    result = {
        'metabolite': met_name,
        'chebi_id': metabolite_chebi,
        'target_genus': target_genus,
        'target_in_kg': target_genus.lower() in genus_consumes,
        'target_consumes_metabolite': target_consumes,
        'total_kg_consumers': len(all_consumers),
        'community_competitors': community_consumers,
        'n_competitors': len(community_consumers),
        'selectivity_warning': None
    }

    # Generate warning if needed
    if len(community_consumers) > 0:
        if target_consumes:
            result['selectivity_warning'] = (
                f"WARNING: {len(community_consumers)} other community members "
                f"({', '.join(community_consumers[:3])}) also consume {met_name} according to KG"
            )
        else:
            result['selectivity_warning'] = (
                f"CAUTION: Target {target_genus} not known to consume {met_name} in KG, "
                f"but {len(community_consumers)} competitors do"
            )
    elif not target_consumes and target_genus.lower() in genus_consumes:
        result['selectivity_warning'] = (
            f"INFO: KG has no consumption data for {met_name} for {target_genus}, "
            f"but does have other consumption data for this genus"
        )

    return result


def main():
    print("=" * 70)
    print("KG COMPETITIVE ANALYSIS TEST")
    print("=" * 70)

    print("\n[1] Loading KG consumption data...")
    genus_consumes, chebi_consumed_by, metabolite_names = load_kg_consumption()

    print(f"  Loaded consumption data for {len(genus_consumes)} genera")
    print(f"  {len(chebi_consumed_by)} metabolites with consumer info")

    # Test scenario: Promoting Bradyrhizobium in a diverse community
    print("\n" + "-" * 70)
    print("TEST SCENARIO: Promoting nitrogen fixers")
    print("-" * 70)

    target = "Bradyrhizobium"
    community = ["Pseudomonas", "Bacillus", "Streptomyces", "Rhizobium",
                 "Chitinophaga", "Massilia", "Pedobacter"]

    # Check a few metabolites
    test_metabolites = [
        ("CHEBI:17992", "sucrose"),
        ("CHEBI:16899", "D-mannitol"),
        ("CHEBI:16467", "L-arginine"),
        ("CHEBI:18019", "L-lysine"),
    ]

    for chebi_id, expected_name in test_metabolites:
        result = competitive_analysis(
            target_genus=target,
            metabolite_chebi=chebi_id,
            community_genera=community,
            genus_consumes=genus_consumes,
            chebi_consumed_by=chebi_consumed_by,
            metabolite_names=metabolite_names
        )

        print(f"\n{result['metabolite']} ({chebi_id}):")
        print(f"  Target ({target}) in KG: {result['target_in_kg']}")
        print(f"  Target consumes this: {result['target_consumes_metabolite']}")
        print(f"  Total KG consumers: {result['total_kg_consumers']}")
        print(f"  Community competitors: {result['community_competitors']}")
        if result['selectivity_warning']:
            print(f"  >> {result['selectivity_warning']}")

    # Show what metabolites Bradyrhizobium uniquely consumes
    print("\n" + "-" * 70)
    print(f"WHAT DOES {target.upper()} UNIQUELY CONSUME?")
    print("-" * 70)

    if target.lower() in genus_consumes:
        target_metabolites = genus_consumes[target.lower()]
        print(f"\n{target} consumes {len(target_metabolites)} metabolites in KG")

        # Find selective ones (few other consumers in typical community)
        selective = []
        for chebi_id in target_metabolites:
            n_total = len(chebi_consumed_by.get(chebi_id, set()))
            # Check community overlap
            consumers = chebi_consumed_by.get(chebi_id, set())
            community_overlap = len([g for g in community if g.lower() in consumers])

            if community_overlap == 0 and n_total < 50:
                met_name = metabolite_names.get(chebi_id, chebi_id)
                selective.append((met_name, n_total))

        if selective:
            print(f"\nSelective metabolites (no community competitors):")
            for met, n in sorted(selective, key=lambda x: x[1])[:10]:
                print(f"  {met}: {n} total consumers in KG")
        else:
            print("\nNo uniquely selective metabolites found")
    else:
        print(f"\n{target} not in KG consumption data")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
The KG can serve as a "competitive landscape analyzer":
1. GNN picks metabolites that target consumes
2. KG checks: Who else can consume this?
3. Output: Selectivity score or warning

This doesn't require a second GNN pass - it's post-hoc context.
The LLM can then synthesize:
  "GNN recommends glutamate for Bradyrhizobium, but KG shows
   3 competitors also consume it. Consider more selective options."
""")


if __name__ == '__main__':
    main()
