#!/usr/bin/env python
"""Test script to validate data loading in wom_refactor.

This script tests the new dataclass-based data loading pipeline.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loaders.loaders import (
    load_exometabolite,
    prepare_training_data,
    get_organism_preferences,
)
from loaders.compound_mapper import (
    load_chebi,
    map_compound_name,
)
from core.types import AssertionType

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def test_chebi_loading():
    """Test ChEBI database loading."""
    logger.info("=" * 60)
    logger.info("Testing ChEBI loading...")
    logger.info("=" * 60)
    
    chebi = load_chebi()
    logger.info(f"✓ Loaded {len(chebi):,} ChEBI name entries")
    logger.info(f"✓ {chebi['compound_id'].nunique():,} unique compounds")
    
    # Test a few compound mappings
    test_compounds = ["glutamate", "L-glutamate", "serine", "glucose"]
    logger.info("\nTesting compound name mapping:")
    
    for name in test_compounds:
        mapping = map_compound_name(name, chebi)
        logger.info(
            f"  '{name}' -> '{mapping.representative_name}' "
            f"({mapping.match_type}, score={mapping.match_score:.2f})"
        )
    
    return chebi


def test_exometabolite_loading():
    """Test exometabolite data loading."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing exometabolite loading...")
    logger.info("=" * 60)
    
    df = load_exometabolite()
    logger.info(f"✓ Loaded {len(df):,} observations")
    logger.info(f"✓ {df['organism_id'].nunique()} unique organisms")
    logger.info(f"✓ {df['representative_compound'].nunique()} unique metabolites")
    
    # Check assertions
    assertion_counts = df['assertion'].value_counts()
    logger.info("\nAssertion distribution:")
    for assertion, count in assertion_counts.items():
        logger.info(f"  {assertion}: {count:,}")
    
    return df


def test_training_data_preparation(df):
    """Test training data preparation with new dataclass."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing training data preparation...")
    logger.info("=" * 60)
    
    # Test consumed
    training_data = prepare_training_data(df, AssertionType.CONSUMED)
    logger.info(f"✓ Created TrainingData object")
    logger.info(f"  n_microbes: {training_data.n_microbes}")
    logger.info(f"  n_metabolites: {training_data.n_metabolites}")
    logger.info(f"  preference_type: {training_data.preference_type.value}")
    
    # Test some dataclass methods
    organism_ids = list(training_data.microbe_to_idx.keys())[:3]
    logger.info(f"\nTesting TrainingData methods on {organism_ids[0]}:")
    
    positives = training_data.get_microbe_positives(organism_ids[0])
    logger.info(f"  ✓ get_microbe_positives(): {len(positives)} metabolites")
    
    # Test reverse lookup
    if positives:
        metabolite = list(positives)[0]
        consumers = training_data.get_metabolite_positives(metabolite)
        logger.info(f"  ✓ get_metabolite_positives('{metabolite}'): {len(consumers)} consumers")
    
    # Check taxonomy
    taxonomy = training_data.microbe_taxonomy[organism_ids[0]]
    logger.info(f"  ✓ Taxonomy: genus={taxonomy.genus}, family={taxonomy.family}")
    
    return training_data


def test_data_queries(df):
    """Test convenience query functions."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing data queries...")
    logger.info("=" * 60)
    
    # Get a random organism
    organism_id = df['organism_id'].iloc[0]
    logger.info(f"\nQuerying preferences for organism: {organism_id}")
    
    org_prefs = get_organism_preferences(organism_id, df)
    logger.info(f"✓ Found {len(org_prefs)} preference records")
    
    consumed = org_prefs[org_prefs['assertion'] == 'consumed']
    logger.info(f"  - {len(consumed)} consumed")
    
    if len(consumed) > 0:
        logger.info("\n  Top consumed metabolites:")
        for _, row in consumed.head(5).iterrows():
            logger.info(f"    • {row['representative_compound']} (value={row['assertion_value']:.1f})")


def main():
    """Run all tests."""
    try:
        # Test 1: ChEBI
        chebi = test_chebi_loading()
        
        # Test 2: Exometabolite
        df = test_exometabolite_loading()
        
        # Test 3: Training data
        training_data = test_training_data_preparation(df)
        
        # Test 4: Queries
        test_data_queries(df)
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ ALL TESTS PASSED")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ TEST FAILED: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
