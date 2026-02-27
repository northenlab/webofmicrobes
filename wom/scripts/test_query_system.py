"""Test query system with a small trained model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loaders.loaders import prepare_training_data
from core.training import train_model
from core.io import save_model_artifacts
from api import QueryEngine
from api.query_types import *
import tempfile
import json


def setup_test_model():
    """Train a small model for testing."""
    print("Training small test model...")
    training_data = prepare_training_data()
    artifacts = train_model(
        training_data=training_data,
        model_type="bpr",
        embedding_dim=8,
        n_epochs=10,
        verbose=False,
    )
    
    # Save to temp directory
    tmpdir = tempfile.mkdtemp()
    save_model_artifacts(artifacts, tmpdir)
    print(f"Model saved to {tmpdir}\n")
    
    return tmpdir


def test_query_system(model_dir):
    """Test all query types."""
    print("=" * 70)
    print("QUERY SYSTEM TESTS")
    print("=" * 70)
    
    engine = QueryEngine(model_dir)
    
    # Test 1: Compounds for microbes
    print("\n1. COMPOUNDS FOR MICROBES")
    print("-" * 70)
    query = CompoundsForMicrobesQuery.from_string(
        "genus:Arthrobacter",
        top_k=5,
    )
    response = engine.query_compounds_for_microbes(query)
    print(f"Success: {response.success}")
    if response.results:
        print(f"Top 5 compounds for Arthrobacter:")
        for compound in response.results[0]["compounds"][:5]:
            print(f"  - {compound['name']}: {compound['raw_score']:.3f}")
    
    # Test 2: Microbes for compounds
    print("\n2. MICROBES FOR COMPOUNDS")
    print("-" * 70)
    query = MicrobesForCompoundsQuery.from_string(
        "glucose,alanine",
        level="genus",
        top_k=5,
    )
    response = engine.query_microbes_for_compounds(query)
    print(f"Success: {response.success}")
    if response.results:
        print(f"Top 5 genera consuming glucose and alanine:")
        for result in response.results[:5]:
            print(f"  - {result['taxon']}: {result['aggregate_score']:.3f}")
    
    # Test 3: Selective metabolites
    print("\n3. SELECTIVE METABOLITES")
    print("-" * 70)
    genera = engine.artifacts.get_taxa_at_level("genus")
    if len(genera) >= 3:
        target = genera[0]
        competitors = genera[1:3]
        
        query = SelectiveMetabolitesQuery.from_strings(
            f"genus:{target}",
            f"genus:{competitors[0]},genus:{competitors[1]}",
            top_k=5,
        )
        response = engine.query_selective_metabolites(query)
        print(f"Success: {response.success}")
        if response.results:
            print(f"Top 5 metabolites favoring {target}:")
            for result in response.results[:5]:
                print(f"  - {result['metabolite']}: selectivity={result['selectivity']:.3f}")
    
    # Test 4: Metabolite environment fitness
    print("\n4. METABOLITE ENVIRONMENT FITNESS")
    print("-" * 70)
    query = MetaboliteEnvironmentFitnessQuery.from_strings(
        "glucose,alanine,glycine",
        level="genus",
        top_k=5,
    )
    response = engine.query_metabolite_environment_fitness(query)
    print(f"Success: {response.success}")
    if response.results:
        print(f"Top 5 genera in glucose+alanine+glycine environment:")
        for result in response.results[:5]:
            print(f"  - {result['microbe']}: fitness={result['fitness']:.3f}")
    
    # Test 5: Niche metabolites
    print("\n5. NICHE METABOLITES")
    print("-" * 70)
    isolates = list(engine.artifacts.training_data.microbe_to_idx.keys())[:1]
    if isolates:
        query = NicheMetabolitesQuery.from_string(
            f"isolate:{isolates[0]}",
            top_k=5,
        )
        response = engine.query_niche_metabolites(query)
        print(f"Success: {response.success}")
        if response.results:
            print(f"Top 5 distinctive metabolites for {isolates[0]}:")
            for result in response.results[:5]:
                print(f"  - {result['metabolite']}: distinctive={result['distinctive_score']:.3f}")
    
    # Test 6: Search compound
    print("\n6. SEARCH COMPOUND")
    print("-" * 70)
    query = SearchCompoundQuery(
        query="glucose",
        fuzzy=True,
    )
    response = engine.query_search_compound(query)
    print(f"Success: {response.success}")
    if response.results:
        for result in response.results:
            print(f"  Found: {result['matched_name']}")
            print(f"  In training data: {result['in_training_data']}")
    
    # Test 7: List taxa
    print("\n7. LIST TAXA")
    print("-" * 70)
    query = ListTaxaQuery(
        level="genus",
        include_counts=True,
    )
    response = engine.query_list_taxa(query)
    print(f"Success: {response.success}")
    print(f"Found {len(response.results)} genera")
    if response.results:
        print(f"Sample (first 5):")
        for result in response.results[:5]:
            print(f"  - {result['name']}: {result.get('n_isolates', 'N/A')} isolates")
    
    # Test 8: Model info
    print("\n8. MODEL INFO")
    print("-" * 70)
    query = ModelInfoQuery(include_stats=True)
    response = engine.query_model_info(query)
    print(f"Success: {response.success}")
    if response.results:
        info = response.results[0]
        print(f"Model type: {info['model_type']}")
        print(f"Embedding dim: {info['embedding_dim']}")
        print(f"Microbes: {info['n_microbes']}")
        print(f"Metabolites: {info['n_metabolites']}")
        if 'statistics' in info:
            print(f"Interactions: {info['statistics']['n_interactions']}")
            print(f"Sparsity: {info['statistics']['sparsity']:.4f}")
    
    print("\n" + "=" * 70)
    print("✓ All query tests completed!")
    print("=" * 70)


def test_json_serialization(model_dir):
    """Test JSON serialization of responses."""
    print("\n" + "=" * 70)
    print("JSON SERIALIZATION TEST")
    print("=" * 70)
    
    engine = QueryEngine(model_dir)
    
    query = CompoundsForMicrobesQuery.from_string(
        "genus:Arthrobacter",
        top_k=3,
    )
    response = engine.query_compounds_for_microbes(query)
    
    # Convert to JSON
    json_str = json.dumps(response.to_dict(), indent=2)
    print("\nJSON output (truncated):")
    print(json_str[:500] + "...")
    
    # Verify it can be parsed back
    parsed = json.loads(json_str)
    print(f"\n✓ JSON serialization working")
    print(f"  Success: {parsed['success']}")
    print(f"  Query type: {parsed['query_type']}")
    print(f"  Results: {len(parsed['results'])} items")


def main():
    # Setup
    model_dir = setup_test_model()
    
    # Run tests
    try:
        test_query_system(model_dir)
        test_json_serialization(model_dir)
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(model_dir, ignore_errors=True)
        print(f"\nCleaned up test model directory")


if __name__ == "__main__":
    main()
