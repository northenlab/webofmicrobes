#!/usr/bin/env python
"""Test the explain flag to show how queries are computed."""

import sys
from pathlib import Path
import tempfile
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from loaders.loaders import prepare_training_data
from core.training import train_model, compute_taxonomy_embeddings
from core.io import save_model_artifacts
from api.query_engine import QueryEngine
from api.query_types import CompoundsForMicrobesQuery, QueryConfig


def test_explain_mode():
    """Test explain mode showing computation details."""
    print("=" * 80)
    print("TESTING EXPLAIN MODE - How is genus:Bosea scored?")
    print("=" * 80)
    
    # Setup test model
    with tempfile.TemporaryDirectory() as tmpdir:
        print("\n1. Training test model...")
        
        # Train small model
        training_data = prepare_training_data()
        print(f"   Loaded: {training_data.n_microbes} microbes, {training_data.n_metabolites} metabolites")
        
        artifacts = train_model(
            training_data=training_data,
            model_type="bpr",
            embedding_dim=8,
            n_epochs=5,
            batch_size=32,
            verbose=False,
        )
        
        # Compute taxonomy embeddings
        for level in ["genus", "family"]:
            artifacts.taxonomy_embeddings[level] = compute_taxonomy_embeddings(artifacts, level)
        
        # Save
        save_model_artifacts(artifacts, tmpdir)
        print(f"   Saved to {tmpdir}")
        
        # Load engine
        engine = QueryEngine(tmpdir)
        
        # Get a genus with only 1 isolate (if possible)
        artifacts = engine.artifacts
        genus_counts = {}
        for org_id, tax in artifacts.training_data.microbe_taxonomy.items():
            genus = tax.genus
            if genus:
                genus_counts[genus] = genus_counts.get(genus, 0) + 1
        
        # Find genus with 1 isolate and genus with multiple isolates
        single_isolate_genus = None
        multi_isolate_genus = None
        for genus, count in genus_counts.items():
            if count == 1 and not single_isolate_genus:
                single_isolate_genus = genus
            elif count > 1 and not multi_isolate_genus:
                multi_isolate_genus = genus
            if single_isolate_genus and multi_isolate_genus:
                break
        
        print(f"\n2. Testing with:")
        print(f"   - Single-isolate genus: {single_isolate_genus} ({genus_counts.get(single_isolate_genus, 0)} isolate)")
        print(f"   - Multi-isolate genus: {multi_isolate_genus} ({genus_counts.get(multi_isolate_genus, 0)} isolates)")
        
        # Query WITH explain
        print("\n" + "=" * 80)
        print(f"QUERY 1: Single-isolate genus WITH --explain")
        print("=" * 80)
        
        query = CompoundsForMicrobesQuery(
            microbes=[("genus", single_isolate_genus)],
            top_k=5,
            compute_distinctive=True,
            explain=True,
        )
        
        response = engine.query_compounds_for_microbes(query)
        
        if response.success and response.results:
            result = response.results[0]
            print(f"\nMicrobe: {result['microbe']['level']}:{result['microbe']['name']}")
            
            if "explanation" in result:
                print("\n--- EXPLANATION (How it works) ---")
                exp = result["explanation"]
                
                print(f"\nEmbedding Source: {exp.get('embedding_source', 'N/A')}")
                print(f"Isolates Averaged: {exp.get('n_isolates_averaged', 'N/A')}")
                if exp.get('isolates_in_taxon'):
                    print(f"Isolates in Taxon: {', '.join(exp['isolates_in_taxon'][:3])}")
                
                if exp.get('embedding'):
                    print(f"Embedding (first 5 dims): {[f'{x:.4f}' for x in exp['embedding'][:5]]}")
                
                print(f"\nPrediction Method: {exp.get('prediction_method', 'N/A')}")
                print(f"Metabolites Scored: {exp.get('n_metabolites_scored', 'N/A')}")
                
                stats = exp.get('score_statistics', {})
                print(f"\nScore Distribution:")
                print(f"  Mean: {stats.get('mean', 0):.4f}")
                print(f"  Median: {stats.get('median', 0):.4f}")
                print(f"  Std Dev: {stats.get('std', 0):.4f}")
                
                if exp.get('distinctive_computation'):
                    print(f"\nDistinctive Score Computation: {exp['distinctive_computation']}")
                    print(f"Background Taxa: {exp.get('n_background_taxa', 'N/A')} genera")
            
            print("\n--- TOP RESULTS ---")
            for i, compound in enumerate(result['compounds'][:5], 1):
                print(f"{i}. {compound['name']}")
                print(f"   Raw Score: {compound['raw_score']:.4f}")
                if compound.get('distinctive_score') is not None:
                    print(f"   Distinctive: {compound['distinctive_score']:.4f} (rank {compound.get('rank', 'N/A')})")
        
        # Now multi-isolate genus
        print("\n" + "=" * 80)
        print(f"QUERY 2: Multi-isolate genus WITH --explain")
        print("=" * 80)
        
        query = CompoundsForMicrobesQuery(
            microbes=[("genus", multi_isolate_genus)],
            top_k=5,
            compute_distinctive=True,
            explain=True,
        )
        
        response = engine.query_compounds_for_microbes(query)
        
        if response.success and response.results:
            result = response.results[0]
            print(f"\nMicrobe: {result['microbe']['level']}:{result['microbe']['name']}")
            
            if "explanation" in result:
                print("\n--- EXPLANATION (How it works) ---")
                exp = result["explanation"]
                
                print(f"\nEmbedding Source: {exp.get('embedding_source', 'N/A')}")
                print(f"Isolates Averaged: {exp.get('n_isolates_averaged', 'N/A')}")
                if exp.get('isolates_in_taxon'):
                    print(f"Isolates in Taxon (showing first 3): {', '.join(exp['isolates_in_taxon'][:3])}")
                
                if exp.get('embedding'):
                    print(f"Embedding (first 5 dims): {[f'{x:.4f}' for x in exp['embedding'][:5]]}")
                
                print(f"\nKey Insight: Genus embedding is AVERAGE of {exp.get('n_isolates_averaged', 'N/A')} isolate embeddings")
                print("             Each isolate's embedding is weighted equally in the mean")
        
        # Compare: isolate vs genus for single-isolate genus
        if single_isolate_genus:
            print("\n" + "=" * 80)
            print(f"COMPARISON: Isolate vs Genus (single-isolate genus)")
            print("=" * 80)
            
            # Get the isolate ID
            isolate_id = None
            for org_id, tax in artifacts.training_data.microbe_taxonomy.items():
                if tax.genus == single_isolate_genus:
                    isolate_id = org_id
                    break
            
            if isolate_id:
                print(f"\nComparing:")
                print(f"  isolate:{isolate_id}")
                print(f"  genus:{single_isolate_genus}")
                
                # Query isolate
                query_iso = CompoundsForMicrobesQuery(
                    microbes=[("isolate", isolate_id)],
                    top_k=5,
                    compute_distinctive=False,
                    explain=True,
                )
                response_iso = engine.query_compounds_for_microbes(query_iso)
                
                # Query genus
                query_gen = CompoundsForMicrobesQuery(
                    microbes=[("genus", single_isolate_genus)],
                    top_k=5,
                    compute_distinctive=False,
                    explain=True,
                )
                response_gen = engine.query_compounds_for_microbes(query_gen)
                
                if response_iso.success and response_gen.success:
                    iso_exp = response_iso.results[0].get('explanation', {})
                    gen_exp = response_gen.results[0].get('explanation', {})
                    
                    iso_emb = iso_exp.get('embedding', [])[:5]
                    gen_emb = gen_exp.get('embedding', [])[:5]
                    
                    print(f"\nIsolate embedding:  {[f'{x:.6f}' for x in iso_emb]}")
                    print(f"Genus embedding:    {[f'{x:.6f}' for x in gen_emb]}")
                    
                    if iso_emb and gen_emb:
                        diffs = [abs(a - b) for a, b in zip(iso_emb, gen_emb)]
                        max_diff = max(diffs) if diffs else 0
                        print(f"\nMax difference: {max_diff:.10f}")
                        
                        if max_diff < 1e-6:
                            print("✓ IDENTICAL (within floating point precision)")
                        else:
                            print("✗ DIFFERENT")
                    
                    print("\nConclusion:")
                    print("For single-isolate genera, genus embedding = isolate embedding")
                    print("But they use DIFFERENT LOOKUP PATHS:")
                    print("  - isolate: model.microbe_embeddings[idx]")
                    print("  - genus: taxonomy_embeddings['genus'][name]")


if __name__ == "__main__":
    test_explain_mode()
