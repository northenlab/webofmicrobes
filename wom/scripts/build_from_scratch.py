#!/usr/bin/env python
"""Build all data from scratch - no cached files.

This demonstrates the clean refactored architecture:
1. Download ChEBI database (to chebi_source/)
2. Process and cache ChEBI (to processed_data/chebi.parquet)
3. Load and process exometabolite data (to processed_data/exometabolite.parquet)
4. Prepare training data with TrainingData dataclass
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loaders.compound_mapper import build_chebi_database
from loaders.loaders import build_exometabolite_dataframe, prepare_training_data
from loaders.config import CHEBI_SOURCE_DIR, PROCESSED_DATA_DIR, EXOMETABOLITE_DATA_DIR
from core.types import AssertionType

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    """Build everything from scratch."""
    logger.info("=" * 70)
    logger.info("BUILDING WEB OF MICROBES FROM SCRATCH")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Directory structure:")
    logger.info(f"  Source data:      {EXOMETABOLITE_DATA_DIR}")
    logger.info(f"  ChEBI downloads:  {CHEBI_SOURCE_DIR}")
    logger.info(f"  Processed output: {PROCESSED_DATA_DIR}")
    logger.info("")
    
    # Step 1: Build ChEBI database
    logger.info("=" * 70)
    logger.info("STEP 1: Building ChEBI database")
    logger.info("=" * 70)
    logger.info("This will download ChEBI files if not present and build the database")
    logger.info("")
    
    try:
        chebi = build_chebi_database(force=True)  # Force rebuild
        logger.info("")
        logger.info(f"✅ ChEBI built successfully:")
        logger.info(f"   - {len(chebi):,} name entries")
        logger.info(f"   - {chebi['compound_id'].nunique():,} unique compounds")
        with_smiles = chebi['smiles'].notna().sum()
        with_inchi = chebi['inchi_key'].notna().sum()
        logger.info(f"   - {with_smiles:,} entries with SMILES")
        logger.info(f"   - {with_inchi:,} entries with InChIKey")
    except Exception as e:
        logger.error(f"❌ Failed to build ChEBI: {e}", exc_info=True)
        return 1
    
    # Step 2: Build exometabolite dataframe
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 2: Building exometabolite dataframe")
    logger.info("=" * 70)
    logger.info("Loading TSV/CSV files and mapping compound names to ChEBI")
    logger.info("")
    
    try:
        df = build_exometabolite_dataframe(force=True)  # Force rebuild
        logger.info("")
        logger.info(f"✅ Exometabolite data built successfully:")
        logger.info(f"   - {len(df):,} total observations")
        logger.info(f"   - {df['organism_id'].nunique()} unique organisms")
        logger.info(f"   - {df['representative_compound'].nunique()} unique metabolites")
        
        # Show assertion distribution
        assertion_counts = df['assertion'].value_counts()
        logger.info(f"   - Assertion distribution:")
        for assertion, count in assertion_counts.items():
            logger.info(f"     • {assertion}: {count:,}")
    except Exception as e:
        logger.error(f"❌ Failed to build exometabolite data: {e}", exc_info=True)
        return 1
    
    # Step 3: Prepare training data
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 3: Preparing training data")
    logger.info("=" * 70)
    logger.info("Creating TrainingData dataclass with all mappings")
    logger.info("")
    
    try:
        training_data = prepare_training_data(df, AssertionType.CONSUMED)
        logger.info("")
        logger.info(f"✅ Training data prepared successfully:")
        logger.info(f"   - {training_data.n_microbes} microbes")
        logger.info(f"   - {training_data.n_metabolites} metabolites")
        logger.info(f"   - {len(training_data.positives)} organisms with interactions")
        total_interactions = sum(len(s) for s in training_data.positives.values())
        logger.info(f"   - {total_interactions:,} total positive interactions")
        logger.info(f"   - Preference type: {training_data.preference_type.value}")
        
        # Test dataclass methods
        logger.info("")
        logger.info("Testing TrainingData methods:")
        organism_ids = list(training_data.microbe_to_idx.keys())[:3]
        for org_id in organism_ids:
            positives = training_data.get_microbe_positives(org_id)
            taxonomy = training_data.microbe_taxonomy[org_id]
            logger.info(
                f"   • {org_id}: {len(positives)} metabolites, "
                f"genus={taxonomy.genus}, family={taxonomy.family}"
            )
    except Exception as e:
        logger.error(f"❌ Failed to prepare training data: {e}", exc_info=True)
        return 1
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("✅ BUILD COMPLETE")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Generated files:")
    logger.info(f"  {CHEBI_SOURCE_DIR}/chebi_compounds.tsv.gz (downloaded)")
    logger.info(f"  {CHEBI_SOURCE_DIR}/chebi_names.tsv.gz (downloaded)")
    logger.info(f"  {CHEBI_SOURCE_DIR}/chebi_structures.tsv.gz (downloaded)")
    logger.info(f"  {PROCESSED_DATA_DIR}/chebi.parquet (processed)")
    logger.info(f"  {PROCESSED_DATA_DIR}/exometabolite.parquet (processed)")
    logger.info("")
    logger.info("Ready for model training!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
