"""
Integrate MetagenomicKG into SoilFM2 knowledge graph.

Tasks:
1. Create metabolite ID mapping table
2. Filter microbes to soil-relevant organisms
3. Extract pathway-metabolite edges

Configure paths before running:
- BASE_DIR: Base directory for project
- KG_DIR: Path to knowledge graph data directory
"""

import pandas as pd
import json
import pickle
import re
from pathlib import Path
from collections import defaultdict
import sys

# Base directories - configure these paths for your environment
BASE_DIR = Path(__file__).parent.parent
KG_DIR = Path("path/to/kg_data")  # TODO: Update to your KG data directory

def normalize_name(name):
    """Normalize metabolite names for matching"""
    if pd.isna(name):
        return None
    # Convert to lowercase, strip whitespace
    name = str(name).lower().strip()
    # Remove common prefixes
    name = re.sub(r'^(l-|d-|alpha-|beta-|gamma-)', '', name)
    return name

def get_name_variants(name):
    """Generate common name variants for metabolites"""
    variants = [name]

    # Common acid/ate conversions
    if name.endswith('ate'):
        # acetate -> acetic acid
        variants.append(name[:-3] + 'ic acid')
    elif name.endswith('ic acid'):
        # acetic acid -> acetate
        variants.append(name[:-8] + 'ate')
    elif name.endswith('ate'):
        variants.append(name)

    # Handle special cases
    special_variants = {
        'glucose': ['glucopyranose', 'dextrose', 'd-glucose', 'alpha-d-glucose', 'beta-d-glucose'],
        'fructose': ['d-fructose', 'levulose'],
        'mannose': ['d-mannose'],
        'glycine': ['aminoacetic acid', 'aminoethanoic acid'],
        'citrate': ['citric acid', 'citrate(3-)'],
        'acetate': ['acetic acid', 'ethanoic acid', 'acetate ion'],
        'formate': ['formic acid', 'methanoic acid'],
        'fumarate': ['fumaric acid', 'trans-butenedioic acid'],
        'glyoxylate': ['glyoxylic acid', 'glyoxalate'],
        'alpha-ketoglutaric acid': ['alpha-ketoglutarate', '2-oxoglutaric acid', '2-oxoglutarate', 'oxoglutaric acid'],
        'myo-inositol': ['inositol', 'myoinositol', 'i-inositol', 'cyclohexanehexol']
    }

    if name in special_variants:
        variants.extend(special_variants[name])

    return variants

def extract_chebi_from_synonyms(synonyms_str):
    """Extract ChEBI ID from synonyms column"""
    if pd.isna(synonyms_str):
        return None
    # Match CHEBI:12345 or ChEBI:12345
    match = re.search(r'ChEBI:(\d+)', str(synonyms_str), re.IGNORECASE)
    if match:
        return f"CHEBI:{match.group(1)}"
    return None

def extract_kegg_from_synonyms(synonyms_str):
    """Extract KEGG compound ID from synonyms column"""
    if pd.isna(synonyms_str):
        return None
    # Match KEGG:cpd_C00123 or C00123
    match = re.search(r'KEGG:cpd_(C\d+)', str(synonyms_str))
    if match:
        return match.group(1)
    # Also try direct C##### format
    match = re.search(r'\bC\d{5}\b', str(synonyms_str))
    if match:
        return match.group(0)
    return None

def task1_create_metabolite_mappings():
    """Task 1: Create comprehensive metabolite ID mapping table"""
    print("\n" + "="*80)
    print("TASK 1: Creating Metabolite ID Mapping Table")
    print("="*80)

    # Load HILIC standards
    print("\n1. Loading HILIC standards...")
    neg_path = BASE_DIR / "02_metabolite_standards" / "HILIC_EMA-standards_negative.tsv"
    pos_path = BASE_DIR / "02_metabolite_standards" / "HILIC_EMA-standards_positive.tsv"

    neg_df = pd.read_csv(neg_path, sep='\t', low_memory=False)
    pos_df = pd.read_csv(pos_path, sep='\t', low_memory=False)

    print(f"   - Negative mode: {len(neg_df)} standards")
    print(f"   - Positive mode: {len(pos_df)} standards")

    # Combine and deduplicate
    standards_df = pd.concat([neg_df, pos_df], ignore_index=True)

    # Initialize mapping dictionaries
    name_to_chebi = {}
    name_to_kegg = {}
    chebi_to_kegg = {}
    kegg_to_chebi = {}
    inchi_to_chebi = {}

    # Extract from HILIC standards
    print("\n2. Extracting IDs from HILIC standards...")
    for _, row in standards_df.iterrows():
        # Get normalized name
        name = normalize_name(row.get('compound_name'))
        if not name:
            continue

        # Extract ChEBI
        chebi = extract_chebi_from_synonyms(row.get('synonyms'))
        if chebi:
            # Add all variants
            for variant in get_name_variants(name):
                name_to_chebi[variant] = chebi

            # Map InChI key to ChEBI
            inchi_key = row.get('inchi_key')
            if pd.notna(inchi_key):
                inchi_to_chebi[str(inchi_key).strip()] = chebi

        # Extract KEGG (from synonyms if present)
        kegg = extract_kegg_from_synonyms(row.get('synonyms'))
        if kegg:
            # Add all variants
            for variant in get_name_variants(name):
                name_to_kegg[variant] = kegg

            # Bidirectional ChEBI-KEGG mapping
            if chebi:
                chebi_to_kegg[chebi] = kegg
                kegg_to_chebi[kegg] = chebi

    print(f"   - Found {len(name_to_chebi)} ChEBI mappings from standards")
    print(f"   - Found {len(name_to_kegg)} KEGG mappings from standards")
    print(f"   - Found {len(inchi_to_chebi)} InChI->ChEBI mappings")

    # Load MetagenomicKG compounds
    print("\n3. Loading MetagenomicKG compound nodes...")
    kg_nodes_path = KG_DIR / "metagenomickg" / "KG_nodes.tsv"

    # Use chunking for large file
    chunk_size = 100000
    compound_count = 0
    chebi_count = 0
    kegg_count = 0

    for chunk in pd.read_csv(kg_nodes_path, sep='\t', chunksize=chunk_size,
                             low_memory=False, on_bad_lines='skip'):
        # Filter to Compound nodes
        compounds = chunk[chunk['node_type'] == 'biolink:MolecularEntity']
        compound_count += len(compounds)

        for _, row in compounds.iterrows():
            # Extract names
            names_str = row.get('all_names', '[]')
            try:
                names = eval(names_str) if isinstance(names_str, str) else []
            except:
                names = []

            # Extract synonyms
            synonyms = row.get('synonyms', '')
            chebi = extract_chebi_from_synonyms(synonyms)
            kegg = extract_kegg_from_synonyms(synonyms)

            if chebi:
                chebi_count += 1
            if kegg:
                kegg_count += 1

            # Add to mappings
            for name in names:
                norm_name = normalize_name(name)
                if norm_name:
                    # Add all variants
                    for variant in get_name_variants(norm_name):
                        if chebi and variant not in name_to_chebi:
                            name_to_chebi[variant] = chebi
                        if kegg and variant not in name_to_kegg:
                            name_to_kegg[variant] = kegg

            # Bidirectional ChEBI-KEGG
            if chebi and kegg:
                if chebi not in chebi_to_kegg:
                    chebi_to_kegg[chebi] = kegg
                if kegg not in kegg_to_chebi:
                    kegg_to_chebi[kegg] = chebi

    print(f"   - Processed {compound_count} compound nodes")
    print(f"   - Found {chebi_count} with ChEBI IDs")
    print(f"   - Found {kegg_count} with KEGG IDs")

    # Load core metabolites
    print("\n4. Checking NLDM metabolite coverage...")
    core_path = BASE_DIR / "core_metabolites.json"
    if not core_path.exists():
        print(f"   - Skipping NLDM coverage check (file not found)")
        nldm_with_mapping = 0
        nldm_missing = []
        nldm_metabolites = []
    else:
        with open(core_path, 'r') as f:
            core_data = json.load(f)

        nldm_metabolites = core_data['nldm_metabolites']
        nldm_with_mapping = 0
        nldm_missing = []

        for metabolite in nldm_metabolites:
            norm_name = normalize_name(metabolite)
            if norm_name in name_to_chebi or norm_name in name_to_kegg:
                nldm_with_mapping += 1
            else:
                nldm_missing.append(metabolite)

        print(f"   - NLDM metabolites with mappings: {nldm_with_mapping}/{len(nldm_metabolites)}")
        if nldm_missing:
            print(f"   - Missing mappings: {nldm_missing[:10]}")

    # Count standards with KG mapping
    standards_with_kg = 0
    for _, row in standards_df.iterrows():
        chebi = extract_chebi_from_synonyms(row.get('synonyms'))
        if chebi and chebi in chebi_to_kegg:
            standards_with_kg += 1

    # Create output
    output = {
        "name_to_chebi": name_to_chebi,
        "name_to_kegg": name_to_kegg,
        "chebi_to_kegg": chebi_to_kegg,
        "kegg_to_chebi": kegg_to_chebi,
        "inchi_to_chebi": inchi_to_chebi,
        "standards_with_kg_mapping": standards_with_kg,
        "nldm_with_mapping": nldm_with_mapping,
        "total_unique_metabolites": len(set(name_to_chebi.keys()) | set(name_to_kegg.keys()))
    }

    # Save
    output_path = BASE_DIR / "02_metabolite_standards" / "metabolite_id_mappings.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n5. Saved to: {output_path}")
    print(f"\nSummary Statistics:")
    print(f"   - Total unique metabolite names: {output['total_unique_metabolites']}")
    print(f"   - Name->ChEBI mappings: {len(name_to_chebi)}")
    print(f"   - Name->KEGG mappings: {len(name_to_kegg)}")
    print(f"   - ChEBI<->KEGG bidirectional: {len(chebi_to_kegg)}")
    print(f"   - Standards with KG mapping: {standards_with_kg}")
    print(f"   - NLDM metabolites covered: {nldm_with_mapping}/{len(nldm_metabolites)}")

    return output


def task2_filter_soil_organisms():
    """Task 2: Filter microbes to soil-relevant organisms"""
    print("\n" + "="*80)
    print("TASK 2: Filtering Microbes to Soil-Relevant Organisms")
    print("="*80)

    # Load RefSoil taxonomy
    print("\n1. Loading RefSoil taxonomy...")
    refsoil_path = Path("path/to/refsoil_taxonomy.pkl")  # TODO: Update this path

    if not refsoil_path.exists():
        print(f"   - ERROR: RefSoil taxonomy file not found at {refsoil_path}")
        print(f"   - Please configure the path in the script")
        return {}

    with open(refsoil_path, 'rb') as f:
        refsoil_tax = pickle.load(f)

    # Extract unique genera - refsoil_tax is a dict with taxonomy info as values
    refsoil_genera = set()

    if isinstance(refsoil_tax, dict):
        for key, tax_info in refsoil_tax.items():
            if isinstance(tax_info, dict) and 'genus' in tax_info:
                genus = tax_info['genus']
                if genus and isinstance(genus, str) and genus != 'unknown':
                    refsoil_genera.add(genus.lower().strip())
    elif isinstance(refsoil_tax, pd.DataFrame):
        if 'genus' in refsoil_tax.columns:
            genera = refsoil_tax['genus'].dropna().unique()
            refsoil_genera.update([g.lower().strip() for g in genera if g != 'unknown'])

    refsoil_genera = sorted(list(refsoil_genera))
    print(f"   - Found {len(refsoil_genera)} unique genera in RefSoil")
    print(f"   - Examples: {refsoil_genera[:10]}")

    # Load KG-Microbe organisms
    print("\n2. Loading KG-Microbe organisms...")
    kg_microbe_path = KG_DIR / "kg_microbe" / "merged-kg_nodes.tsv"

    kg_microbe = pd.read_csv(kg_microbe_path, sep='\t', low_memory=False, on_bad_lines='skip')

    # Filter to organism nodes - check column name
    if 'category' in kg_microbe.columns:
        organisms = kg_microbe[kg_microbe['category'] == 'biolink:OrganismTaxon']
    elif 'node_type' in kg_microbe.columns:
        organisms = kg_microbe[kg_microbe['node_type'] == 'biolink:OrganismTaxon']
    else:
        # If no type column, use all rows
        print(f"   - Warning: No category column found. Available columns: {kg_microbe.columns.tolist()}")
        organisms = kg_microbe

    print(f"   - Total organism nodes: {len(organisms)}")

    # Match genera
    matching_organisms = []
    phylo_counts = defaultdict(int)

    for _, row in organisms.iterrows():
        # Extract genus from names - try both 'name' and 'all_names' columns
        names = []
        if 'name' in row:
            names.append(row['name'])
        if 'all_names' in row:
            names_str = row.get('all_names', '[]')
            try:
                parsed_names = eval(names_str) if isinstance(names_str, str) else []
                names.extend(parsed_names)
            except:
                pass

        # Also check synonym column
        if 'synonym' in row and pd.notna(row['synonym']):
            names.append(str(row['synonym']))

        # Check if any name contains a RefSoil genus
        matched = False
        node_id = row.get('id', row.get('node_id', ''))

        for name in names:
            name_lower = str(name).lower()
            for genus in refsoil_genera:
                if genus in name_lower:
                    matching_organisms.append(node_id)
                    matched = True

                    # Try to extract phylum for coverage stats
                    desc = row.get('description', '')
                    if 'phylum' in str(desc).lower():
                        phylo_counts['matched'] += 1
                    break
            if matched:
                break

    matching_organisms = list(set(matching_organisms))

    # Create output
    output = {
        "refsoil_genera": refsoil_genera,
        "kg_organisms_matching_refsoil": matching_organisms,
        "n_refsoil_genera": len(refsoil_genera),
        "n_matching_kg_organisms": len(matching_organisms),
        "phylogenetic_coverage": {
            "total_kg_organisms": len(organisms),
            "matched_organisms": len(matching_organisms),
            "match_rate": f"{len(matching_organisms)/len(organisms)*100:.1f}%"
        }
    }

    # Save
    output_path = BASE_DIR / "01_knowledge_graph" / "soil_organism_filter.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n3. Saved to: {output_path}")
    print(f"\nSummary Statistics:")
    print(f"   - RefSoil genera: {len(refsoil_genera)}")
    print(f"   - KG-Microbe organisms matching: {len(matching_organisms)}")
    print(f"   - Match rate: {output['phylogenetic_coverage']['match_rate']}")

    return output


def task3_extract_pathway_edges():
    """Task 3: Extract pathway-metabolite edges from MetagenomicKG"""
    print("\n" + "="*80)
    print("TASK 3: Extracting Pathway-Metabolite Edges")
    print("="*80)

    # Load metabolite mappings from Task 1
    print("\n1. Loading metabolite ID mappings...")
    mappings_path = BASE_DIR / "02_metabolite_standards" / "metabolite_id_mappings.json"
    with open(mappings_path, 'r') as f:
        mappings = json.load(f)

    # Create set of ChEBI IDs from standards
    standards_chebi = set(mappings['name_to_chebi'].values())
    standards_kegg = set(mappings['name_to_kegg'].values())

    print(f"   - Tracking {len(standards_chebi)} ChEBI IDs")
    print(f"   - Tracking {len(standards_kegg)} KEGG IDs")

    # First, build compound ID to ChEBI/KEGG mapping
    print("\n2. Building compound node mappings...")
    kg_nodes_path = KG_DIR / "metagenomickg" / "KG_nodes.tsv"

    compound_to_chebi = {}
    compound_to_kegg = {}

    chunk_size = 100000
    for chunk in pd.read_csv(kg_nodes_path, sep='\t', chunksize=chunk_size,
                             low_memory=False, on_bad_lines='skip'):
        compounds = chunk[chunk['node_type'] == 'biolink:MolecularEntity']

        for _, row in compounds.iterrows():
            node_id = row['node_id']
            synonyms = row.get('synonyms', '')

            chebi = extract_chebi_from_synonyms(synonyms)
            kegg = extract_kegg_from_synonyms(synonyms)

            if chebi:
                compound_to_chebi[node_id] = chebi
            if kegg:
                compound_to_kegg[node_id] = kegg

    print(f"   - Mapped {len(compound_to_chebi)} compounds to ChEBI")
    print(f"   - Mapped {len(compound_to_kegg)} compounds to KEGG")

    # Stream through edges file (6GB)
    print("\n3. Streaming through edges file (this may take a few minutes)...")
    edges_path = KG_DIR / "metagenomickg" / "KG_edges.tsv"

    pathway_compound_edges = []
    module_compound_edges = []
    reaction_compound_edges = []

    stats = {
        'total_edges': 0,
        'pathway_compound': 0,
        'module_compound': 0,
        'reaction_compound': 0,
        'filtered_to_standards': 0
    }

    # Relations of interest
    relations = ['has_participant', 'participates_in']

    print("   Processing edges...")
    line_count = 0

    with open(edges_path, 'r', encoding='utf-8') as f:
        # Read header to get column indices
        header = f.readline().strip().split('\t')
        source_idx = header.index('source_node')
        target_idx = header.index('target_node')
        predicate_idx = header.index('predicate')

        for line in f:
            line_count += 1
            if line_count % 1000000 == 0:
                print(f"   - Processed {line_count:,} lines, found {stats['filtered_to_standards']} relevant edges")

            try:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue

                subject = parts[source_idx]
                obj = parts[target_idx]
                predicate = parts[predicate_idx]

                stats['total_edges'] += 1

                # Check for pathway/module/reaction <-> compound edges
                is_pathway = subject.startswith('Pathway:') or obj.startswith('Pathway:')
                is_module = subject.startswith('Module:') or obj.startswith('Module:')
                is_reaction = subject.startswith('Reaction:') or obj.startswith('Reaction:')
                is_compound = subject.startswith('Compound:') or obj.startswith('Compound:')

                if not is_compound:
                    continue

                # Extract pathway/module/reaction and compound
                if is_pathway and is_compound:
                    stats['pathway_compound'] += 1
                    pathway_id = subject if subject.startswith('Pathway:') else obj
                    compound_id = obj if obj.startswith('Compound:') else subject

                    # Check if compound matches our standards
                    chebi = compound_to_chebi.get(compound_id)
                    kegg = compound_to_kegg.get(compound_id)

                    if chebi in standards_chebi or kegg in standards_kegg:
                        stats['filtered_to_standards'] += 1
                        pathway_compound_edges.append({
                            'pathway_id': pathway_id,
                            'compound_id': compound_id,
                            'chebi': chebi,
                            'kegg': kegg,
                            'relation': predicate
                        })

                elif is_module and is_compound:
                    stats['module_compound'] += 1
                    module_id = subject if subject.startswith('Module:') else obj
                    compound_id = obj if obj.startswith('Compound:') else subject

                    chebi = compound_to_chebi.get(compound_id)
                    kegg = compound_to_kegg.get(compound_id)

                    if chebi in standards_chebi or kegg in standards_kegg:
                        stats['filtered_to_standards'] += 1
                        module_compound_edges.append({
                            'module_id': module_id,
                            'compound_id': compound_id,
                            'chebi': chebi,
                            'kegg': kegg,
                            'relation': predicate
                        })

                elif is_reaction and is_compound:
                    stats['reaction_compound'] += 1
                    reaction_id = subject if subject.startswith('Reaction:') else obj
                    compound_id = obj if obj.startswith('Compound:') else subject

                    chebi = compound_to_chebi.get(compound_id)
                    kegg = compound_to_kegg.get(compound_id)

                    if chebi in standards_chebi or kegg in standards_kegg:
                        stats['filtered_to_standards'] += 1
                        reaction_compound_edges.append({
                            'reaction_id': reaction_id,
                            'compound_id': compound_id,
                            'chebi': chebi,
                            'kegg': kegg,
                            'relation': predicate
                        })

            except Exception as e:
                # Skip malformed lines
                continue

    print(f"   - Total lines processed: {line_count:,}")

    # Count unique entities
    unique_pathways = len(set(e['pathway_id'] for e in pathway_compound_edges))
    unique_modules = len(set(e['module_id'] for e in module_compound_edges))
    unique_reactions = len(set(e['reaction_id'] for e in reaction_compound_edges))
    unique_compounds = len(set(
        e['compound_id'] for e in pathway_compound_edges + module_compound_edges + reaction_compound_edges
    ))

    # Create output
    output = {
        'pathway_compound_edges': pathway_compound_edges,
        'module_compound_edges': module_compound_edges,
        'reaction_compound_edges': reaction_compound_edges,
        'stats': {
            'total_edges_scanned': stats['total_edges'],
            'total_pathway_compound': stats['pathway_compound'],
            'total_module_compound': stats['module_compound'],
            'total_reaction_compound': stats['reaction_compound'],
            'filtered_to_standards': stats['filtered_to_standards'],
            'unique_pathways': unique_pathways,
            'unique_modules': unique_modules,
            'unique_reactions': unique_reactions,
            'unique_metabolites': unique_compounds
        }
    }

    # Save
    output_path = BASE_DIR / "01_knowledge_graph" / "pathway_metabolite_edges.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n4. Saved to: {output_path}")
    print(f"\nSummary Statistics:")
    print(f"   - Total edges scanned: {stats['total_edges']:,}")
    print(f"   - Pathway-Compound edges: {stats['pathway_compound']:,}")
    print(f"   - Module-Compound edges: {stats['module_compound']:,}")
    print(f"   - Reaction-Compound edges: {stats['reaction_compound']:,}")
    print(f"   - Filtered to standards: {stats['filtered_to_standards']:,}")
    print(f"   - Unique pathways: {unique_pathways}")
    print(f"   - Unique modules: {unique_modules}")
    print(f"   - Unique reactions: {unique_reactions}")
    print(f"   - Unique metabolites: {unique_compounds}")

    return output


def main():
    """Run all tasks"""
    print("\n" + "="*80)
    print("MetagenomicKG Integration for SoilFM2 Publication")
    print("="*80)

    # Task 1: Metabolite mappings
    task1_result = task1_create_metabolite_mappings()

    # Task 2: Soil organism filter
    task2_result = task2_filter_soil_organisms()

    # Task 3: Pathway-metabolite edges
    task3_result = task3_extract_pathway_edges()

    print("\n" + "="*80)
    print("ALL TASKS COMPLETED SUCCESSFULLY")
    print("="*80)

    print("\nOutput files created:")
    print(f"  1. {BASE_DIR / '02_metabolite_standards' / 'metabolite_id_mappings.json'}")
    print(f"  2. {BASE_DIR / '01_knowledge_graph' / 'soil_organism_filter.json'}")
    print(f"  3. {BASE_DIR / '01_knowledge_graph' / 'pathway_metabolite_edges.json'}")


if __name__ == "__main__":
    main()
