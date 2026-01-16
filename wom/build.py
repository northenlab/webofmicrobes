"""Web of Microbes - main entry point."""

import logging
import sys
from pathlib import Path

from model import build_bpr_models, plot_all_embeddings
from config import CACHE_DIR

# Add parent to path so we can import sibling modules when run directly
sys.path.insert(0, str(Path(__file__).parent))

# Set up logging before imports so we see download progress
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("wom")


def main():
    logger.info("Initializing Web of Microbes")
    
    # Absolute imports now work whether run as module or directly
    import kg
    import pref
    
    # Step 1: Build knowledge graph
    logger.info("=" * 60)
    logger.info("Step 1: Building knowledge graph")
    logger.info("=" * 60)
    try:
        nodes, edges = kg.build_kg(force=False)
        logger.info(f"KG loaded: {len(nodes):,} nodes, {len(edges):,} edges")
        
        # Show node type breakdown
        type_counts = nodes["node_type"].value_counts()
        logger.info("Node types:")
        for ntype, count in type_counts.items():
            logger.info(f"  {ntype}: {count:,}")
    except Exception as e:
        logger.error(f"Failed to build KG: {e}")
        raise
    

    # Step 2: Build ChEBI lookup
    logger.info("=" * 60)
    logger.info("Step 2: Building ChEBI lookup")
    logger.info("=" * 60)
    try:
        chebi = pref.build_chebi(force=False)
        logger.info(f"ChEBI loaded: {len(chebi):,} name entries, {chebi['compound_id'].nunique():,} unique compounds")
        with_smiles = chebi["smiles"].notna().sum()
        logger.info(f"  Entries with SMILES: {with_smiles:,}")
    except Exception as e:
        logger.error(f"Failed to build ChEBI: {e}")
        raise
    
    # Step 3: Build exometabolite data
    logger.info("=" * 60)
    logger.info("Step 3: Building exometabolite data")
    logger.info("=" * 60)
    try:
        exo = pref.build_exometabolite(force=False)
        logger.info(f"Exometabolite data loaded: {len(exo):,} observations")
        
        # Compound matching stats
        match_counts = exo["compound_match_type"].value_counts()
        logger.info("Compound matching:")
        for mtype, count in match_counts.items():
            logger.info(f"  {mtype}: {count:,}")
        
        # Show unmatched
        unmatched = pref.get_unmatched_compounds()
        if unmatched:
            logger.warning(f"Unmatched compounds ({len(unmatched)}): {unmatched[:10]}...")
    except Exception as e:
        logger.error(f"Failed to build exometabolite data: {e}")
        raise

    
    logger.info("=" * 60)
    logger.info("Step 4: Building BPR recommendation models")
    logger.info("=" * 60)

    bpr_models = build_bpr_models(exo)
    for pref_type, wrapper in bpr_models.items():
        logger.info(f"  {pref_type}: {wrapper.model.n_microbes} microbes, {wrapper.model.n_metabolites} metabolites")
    # Generate plots (optional)
    plot_all_embeddings(bpr_models, output_dir=CACHE_DIR)
    logger.info("=" * 60)
    logger.info("Initialization complete")
    logger.info("=" * 60)








    return nodes, edges, exo


if __name__ == "__main__":
    main()