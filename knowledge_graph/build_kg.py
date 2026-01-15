#!/usr/bin/env python3
"""
Build a filtered Knowledge Graph containing only metabolites from our standards.

This script:
1. Loads HILIC standards (positive and negative) to get target metabolites
2. Parses KG-Microbe source files
3. Filters to keep only edges involving our target metabolites
4. Optionally filters microbes to only those with metabolite edges
5. Builds a PyTorch Geometric HeteroData object
6. Saves with clear provenance

Input:
- HILIC standards TSV files (for metabolite filtering)
- KG-Microbe: merged-kg_nodes.tsv, merged-kg_edges.tsv

Output:
- soilfm_filtered_kg.pt: Filtered PyG HeteroData
- filtered_node_mappings.pkl: ID mappings
- filtered_metadata.json: Statistics and provenance

Note: KG-Microbe only has consumption/production data for ~655 metabolites
(from BacDive phenotypic assays). The overlap with our ~782 ChEBI standards
is only ~96 metabolites. This is a data coverage limitation, not a bug.
"""

import os
import re
import json
import pickle
import pandas as pd
import torch
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, Set, List, Tuple

try:
    from torch_geometric.data import HeteroData
except ImportError:
    print("Installing PyTorch Geometric...")
    os.system("pip install torch-geometric")
    from torch_geometric.data import HeteroData


def load_standards(standards_dir: Path) -> Tuple[Set[str], Set[str], Dict[str, str]]:
    """
    Load metabolite standards and extract identifiers.

    Returns:
        inchi_keys: Set of InChI keys from standards
        chebi_ids: Set of ChEBI IDs extracted from synonyms
        name_to_inchi: Mapping of compound names to InChI keys
    """
    inchi_keys = set()
    chebi_ids = set()
    name_to_inchi = {}

    for polarity in ['negative', 'positive']:
        filepath = standards_dir / f'HILIC_EMA-standards_{polarity}.tsv'
        if not filepath.exists():
            print(f"Warning: {filepath} not found")
            continue

        print(f"Loading {polarity} standards...")
        df = pd.read_csv(filepath, sep='\t')

        for _, row in df.iterrows():
            # Get InChI key
            inchi_key = str(row.get('inchi_key', '')).strip()
            if inchi_key and inchi_key != 'nan':
                inchi_keys.add(inchi_key)

                # Map compound name to InChI key
                name = str(row.get('compound_name', '')).strip()
                if name:
                    name_to_inchi[name.lower()] = inchi_key

            # Extract ChEBI IDs from synonyms
            synonyms = str(row.get('synonyms', ''))
            chebi_matches = re.findall(r'CHEBI:(\d+)', synonyms)
            for chebi in chebi_matches:
                chebi_ids.add(f"CHEBI:{chebi}")

    print(f"  Unique InChI keys: {len(inchi_keys)}")
    print(f"  Unique ChEBI IDs: {len(chebi_ids)}")
    print(f"  Name mappings: {len(name_to_inchi)}")

    return inchi_keys, chebi_ids, name_to_inchi


def parse_kg_microbe(kg_dir: Path, target_chebi: Set[str], target_inchi: Set[str],
                     filter_microbes: bool = True) -> Tuple[Dict, Dict, List]:
    """
    Parse KG-Microbe TSV files and filter to target metabolites.

    Args:
        kg_dir: Path to KG-Microbe data directory
        target_chebi: Set of target ChEBI IDs from standards
        target_inchi: Set of target InChI keys from standards
        filter_microbes: If True, only keep microbes that have metabolite edges

    Returns:
        microbes: Dict of microbe nodes
        metabolites: Dict of metabolite nodes (filtered)
        edges: List of (src_type, rel, dst_type, src_id, dst_id) tuples
    """
    print("\n=== Parsing KG-Microbe ===")

    nodes_file = kg_dir / "merged-kg_nodes.tsv"
    edges_file = kg_dir / "merged-kg_edges.tsv"

    # Load nodes
    print(f"Loading nodes from {nodes_file}...")
    nodes_df = pd.read_csv(nodes_file, sep='\t', low_memory=False)
    print(f"  Total nodes: {len(nodes_df)}")

    all_microbes = {}  # All microbes before filtering
    metabolites = {}
    metabolite_ids = set()  # Track which metabolite IDs we're keeping

    # Categorize nodes
    for _, row in tqdm(nodes_df.iterrows(), total=len(nodes_df), desc="Filtering nodes"):
        node_id = str(row['id'])
        category = str(row.get('category', ''))
        name = str(row.get('name', ''))

        # Check if it's a microbe
        if 'Organism' in category or 'Taxon' in category:
            all_microbes[node_id] = {
                'id': node_id,
                'name': name,
                'category': category
            }
        # Check if it's a metabolite we want
        elif 'Chemical' in category or 'Metabolite' in category:
            # Check if this metabolite matches our standards
            keep = False

            # Check by ChEBI ID
            if node_id.startswith('CHEBI:') and node_id in target_chebi:
                keep = True

            if keep:
                metabolites[node_id] = {
                    'id': node_id,
                    'name': name,
                    'category': category
                }
                metabolite_ids.add(node_id)

    print(f"  All microbes: {len(all_microbes)}")
    print(f"  Filtered metabolites: {len(metabolites)} (from {len(target_chebi)} target ChEBI IDs)")

    # Load edges - only keep those involving our filtered metabolites
    print(f"\nLoading edges from {edges_file}...")
    edges_df = pd.read_csv(edges_file, sep='\t', low_memory=False)
    print(f"  Total edges: {len(edges_df)}")

    edges = []
    edge_counts = defaultdict(int)
    microbes_with_metabolite_edges = set()  # Track microbes that have metabolite edges

    for _, row in tqdm(edges_df.iterrows(), total=len(edges_df), desc="Filtering edges"):
        subject = str(row['subject'])
        predicate = str(row['predicate'])
        obj = str(row['object'])

        # Determine if this edge involves our filtered metabolites
        src_is_microbe = subject in all_microbes
        dst_is_microbe = obj in all_microbes
        src_is_metabolite = subject in metabolite_ids
        dst_is_metabolite = obj in metabolite_ids

        # Keep edges that:
        # 1. Connect microbe to our filtered metabolites
        # 2. Connect microbes to microbes (taxonomy) - only if microbes have metabolite edges
        if src_is_microbe and dst_is_metabolite:
            edges.append(('microbe', predicate, 'metabolite', subject, obj))
            edge_counts[predicate] += 1
            microbes_with_metabolite_edges.add(subject)
        elif src_is_metabolite and dst_is_microbe:
            edges.append(('metabolite', predicate, 'microbe', subject, obj))
            edge_counts[predicate] += 1
            microbes_with_metabolite_edges.add(obj)
        elif src_is_microbe and dst_is_microbe:
            # Keep microbe-microbe edges (taxonomy hierarchy)
            edges.append(('microbe', predicate, 'microbe', subject, obj))
            edge_counts[predicate] += 1

    print(f"  Filtered edges: {len(edges)}")
    print(f"  Microbes with metabolite edges: {len(microbes_with_metabolite_edges)}")
    print("  Edge type counts:")
    for pred, count in sorted(edge_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {pred}: {count}")

    # Filter microbes if requested
    if filter_microbes:
        print(f"\n  Filtering microbes to only those with metabolite edges...")
        microbes = {k: v for k, v in all_microbes.items() if k in microbes_with_metabolite_edges}
        print(f"  Microbes after filtering: {len(microbes)} (reduced from {len(all_microbes)})")

        # Also filter edges to only include filtered microbes
        filtered_edges = []
        for edge in edges:
            src_type, rel, dst_type, src_id, dst_id = edge
            if src_type == 'microbe' and src_id not in microbes:
                continue
            if dst_type == 'microbe' and dst_id not in microbes:
                continue
            filtered_edges.append(edge)

        print(f"  Edges after microbe filtering: {len(filtered_edges)} (from {len(edges)})")
        edges = filtered_edges
    else:
        microbes = all_microbes

    return microbes, metabolites, edges


def build_heterodata(microbes: Dict, metabolites: Dict, edges: List) -> Tuple[HeteroData, Dict]:
    """Build PyTorch Geometric HeteroData from parsed data."""
    print("\n=== Building HeteroData ===")

    data = HeteroData()

    # Create node mappings
    microbe_to_idx = {node_id: idx for idx, node_id in enumerate(microbes.keys())}
    metabolite_to_idx = {node_id: idx for idx, node_id in enumerate(metabolites.keys())}

    # Set node counts
    data['microbe'].num_nodes = len(microbe_to_idx)
    data['metabolite'].num_nodes = len(metabolite_to_idx)

    print(f"  Microbe nodes: {len(microbe_to_idx)}")
    print(f"  Metabolite nodes: {len(metabolite_to_idx)}")

    # Group edges by type
    edge_groups = defaultdict(list)
    for src_type, rel, dst_type, src_id, dst_id in edges:
        edge_groups[(src_type, rel, dst_type)].append((src_id, dst_id))

    # Add edges to HeteroData
    for (src_type, rel, dst_type), edge_list in edge_groups.items():
        # Get appropriate mapping
        if src_type == 'microbe':
            src_mapping = microbe_to_idx
        else:
            src_mapping = metabolite_to_idx

        if dst_type == 'microbe':
            dst_mapping = microbe_to_idx
        else:
            dst_mapping = metabolite_to_idx

        # Convert to indices
        edge_index = []
        for src_id, dst_id in edge_list:
            if src_id in src_mapping and dst_id in dst_mapping:
                edge_index.append([src_mapping[src_id], dst_mapping[dst_id]])

        if edge_index:
            edge_tensor = torch.tensor(edge_index, dtype=torch.long).t()
            data[src_type, rel, dst_type].edge_index = edge_tensor
            print(f"  {src_type} --[{rel}]--> {dst_type}: {len(edge_index)} edges")

    # Build mappings dict
    mappings = {
        'microbe_to_idx': microbe_to_idx,
        'metabolite_to_idx': metabolite_to_idx,
        'idx_to_microbe': {v: k for k, v in microbe_to_idx.items()},
        'idx_to_metabolite': {v: k for k, v in metabolite_to_idx.items()},
        'microbe_names': {node_id: info['name'] for node_id, info in microbes.items()},
        'metabolite_names': {node_id: info['name'] for node_id, info in metabolites.items()}
    }

    return data, mappings


def main():
    """
    Build filtered knowledge graph from KG-Microbe data.

    Configure paths before running:
    - kg_source_dir: Path to KG-Microbe data directory
    """
    # Paths
    base_dir = Path(__file__).parent.parent
    standards_dir = base_dir / '02_metabolite_standards'
    # Configure paths - adjust kg_source_dir to your KG-Microbe data location
    kg_source_dir = Path('path/to/kg_microbe')  # TODO: Update this path
    output_dir = base_dir / '01_knowledge_graph'

    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Building Filtered Knowledge Graph")
    print("="*70)

    # Step 1: Load standards
    print("\n[Step 1] Loading metabolite standards...")
    inchi_keys, chebi_ids, name_to_inchi = load_standards(standards_dir)

    # Step 2: Parse and filter KG-Microbe
    print("\n[Step 2] Parsing KG-Microbe...")
    microbes, metabolites, edges = parse_kg_microbe(
        kg_source_dir, chebi_ids, inchi_keys,
        filter_microbes=True  # Only keep microbes with metabolite edges
    )

    # Step 3: Build HeteroData
    print("\n[Step 3] Building PyTorch Geometric graph...")
    data, mappings = build_heterodata(microbes, metabolites, edges)

    # Step 4: Save outputs
    print("\n[Step 4] Saving outputs...")

    # Save graph
    kg_path = output_dir / 'soilfm_filtered_kg.pt'
    torch.save(data, kg_path)
    print(f"  Saved graph to {kg_path}")

    # Save mappings
    mappings_path = output_dir / 'filtered_node_mappings.pkl'
    with open(mappings_path, 'wb') as f:
        pickle.dump(mappings, f)
    print(f"  Saved mappings to {mappings_path}")

    # Save metadata
    metadata = {
        'num_microbes': len(mappings['microbe_to_idx']),
        'num_metabolites': len(mappings['metabolite_to_idx']),
        'num_edges': sum(data[et].edge_index.shape[1] for et in data.edge_types),
        'edge_types': [str(et) for et in data.edge_types],
        'edge_counts': {str(et): data[et].edge_index.shape[1] for et in data.edge_types},
        'source': 'KG-Microbe (filtered to HILIC standards)',
        'standards_inchi_keys': len(inchi_keys),
        'standards_chebi_ids': len(chebi_ids),
        'filtering_note': 'Only metabolites with ChEBI IDs matching standards are included'
    }

    metadata_path = output_dir / 'filtered_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata to {metadata_path}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Standards: {len(inchi_keys)} InChI keys, {len(chebi_ids)} ChEBI IDs")
    print(f"Filtered KG: {metadata['num_microbes']} microbes, {metadata['num_metabolites']} metabolites")
    print(f"Total edges: {metadata['num_edges']}")
    print("="*70)


if __name__ == '__main__':
    main()
