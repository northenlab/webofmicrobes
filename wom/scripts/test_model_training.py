"""Test model training and inference."""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loaders.loaders import prepare_training_data
from core.training import train_model, compute_taxonomy_embeddings, compute_distinctive_scores
from core.io import save_model_artifacts, load_model_artifacts
import tempfile
import numpy as np


def test_bpr_training():
    """Test BPR model training."""
    print("\n=== Testing BPR Model ===")
    
    training_data = prepare_training_data()
    print(f"Loaded training data: {training_data.n_microbes} microbes, {training_data.n_metabolites} metabolites")
    
    artifacts = train_model(
        training_data=training_data,
        model_type="bpr",
        embedding_dim=8,
        n_epochs=10,  # Small for testing
        batch_size=32,
        verbose=False,
    )
    
    print(f"✓ Trained BPR model")
    print(f"  Microbe embeddings: {artifacts.model.microbe_embeddings.shape}")
    print(f"  Metabolite embeddings: {artifacts.model.metabolite_embeddings.shape}")
    
    # Test prediction using unified interface
    microbe_id = list(artifacts.training_data.microbe_to_idx.keys())[0]
    scores = artifacts.predict_for_taxon("isolate", microbe_id)
    print(f"✓ Prediction for isolate {microbe_id}: {len(scores)} scores, range [{scores.min():.3f}, {scores.max():.3f}]")
    
    return artifacts


def test_svd_training():
    """Test SVD model training."""
    print("\n=== Testing SVD Model ===")
    
    training_data = prepare_training_data()
    
    artifacts = train_model(
        training_data=training_data,
        model_type="svd",
        embedding_dim=8,
        verbose=False,
    )
    
    print(f"✓ Trained SVD model")
    print(f"  Microbe embeddings: {artifacts.model.microbe_embeddings.shape}")
    print(f"  Metabolite embeddings: {artifacts.model.metabolite_embeddings.shape}")
    
    return artifacts


def test_taxonomy_embeddings(artifacts):
    """Test taxonomy embedding computation."""
    print("\n=== Testing Taxonomy Embeddings ===")
    
    genus_emb = compute_taxonomy_embeddings(artifacts, taxon_level="genus")
    print(f"✓ Computed {len(genus_emb)} genus embeddings")
    
    family_emb = compute_taxonomy_embeddings(artifacts, taxon_level="family")
    print(f"✓ Computed {len(family_emb)} family embeddings")
    
    # Store them in artifacts for unified access
    artifacts.taxonomy_embeddings["genus"] = genus_emb
    artifacts.taxonomy_embeddings["family"] = family_emb
    
    # Show a sample
    if genus_emb:
        genus = list(genus_emb.keys())[0]
        embedding = genus_emb[genus]
        print(f"  Example: {genus} -> embedding shape {embedding.shape}")
    
    # Test unified interface at multiple levels
    print("\n  Testing unified predict_for_taxon interface:")
    isolate_id = list(artifacts.training_data.microbe_to_idx.keys())[0]
    taxonomy = artifacts.training_data.microbe_taxonomy[isolate_id]
    
    # Isolate level
    isolate_scores = artifacts.predict_for_taxon("isolate", isolate_id)
    print(f"    ✓ Isolate '{isolate_id}': {len(isolate_scores)} scores")
    
    # Genus level (if available)
    if taxonomy.genus and taxonomy.genus in genus_emb:
        genus_scores = artifacts.predict_for_taxon("genus", taxonomy.genus)
        print(f"    ✓ Genus '{taxonomy.genus}': {len(genus_scores)} scores")
    
    # Family level (if available)
    if taxonomy.family and taxonomy.family in family_emb:
        family_scores = artifacts.predict_for_taxon("family", taxonomy.family)
        print(f"    ✓ Family '{taxonomy.family}': {len(family_scores)} scores")


def test_distinctive_scores(artifacts):
    """Test distinctive score computation."""
    print("\n=== Testing Distinctive Scores ===")
    
    # Get a microbe
    microbe_id = list(artifacts.training_data.microbe_to_idx.keys())[0]
    taxonomy = artifacts.training_data.microbe_taxonomy[microbe_id]
    
    print(f"Testing with microbe: {microbe_id}")
    print(f"  Taxonomy: {taxonomy}")
    
    distinctive = compute_distinctive_scores(
        artifacts,
        microbe_id,
        taxon_level="genus",
        top_k=10,
    )
    
    print(f"✓ Computed distinctive scores")
    print(f"  Top 10 metabolites:")
    for metabolite_id, score in list(distinctive.items())[:10]:
        print(f"    {metabolite_id}: {score:.3f}")


def test_comprehensive_predictions(artifacts):
    """Test comprehensive prediction with statistics."""
    print("\n=== Testing Comprehensive Predictions ===")
    
    # Get an isolate
    isolate_id = list(artifacts.training_data.microbe_to_idx.keys())[0]
    
    # Get full prediction result
    result = artifacts.predict_with_stats(
        level="isolate",
        name=isolate_id,
        compute_distinctive=True,
        background_level="genus",
    )
    
    print(f"Isolate: {result.name}")
    print(f"  Statistics:")
    print(f"    Mean score: {result.mean_score:.4f}")
    print(f"    Median score: {result.median_score:.4f}")
    print(f"    Std dev: {result.std_score:.4f}")
    print(f"    Std error: {result.stderr_score:.4f}" if result.stderr_score else "    Std error: N/A")
    print(f"  Counts:")
    print(f"    Total metabolites: {result.n_metabolites}")
    print(f"    Training positives: {result.n_positive_training}")
    print(f"    Background isolates: {result.n_isolates_aggregated}")
    print(f"  Background:")
    print(f"    Mean: {result.background_mean:.4f}" if result.background_mean else "    Mean: N/A")
    print(f"    Std: {result.background_std:.4f}" if result.background_std else "    Std: N/A")
    
    # Test top_k
    top10 = result.top_k(10, by="distinctive")
    print(f"\n  Top 10 by distinctive score:")
    for i, name in enumerate(top10.metabolite_names):
        print(f"    {i+1}. {name}: raw={top10.raw_scores[i]:.3f}, distinctive={top10.distinctive_scores[i]:.3f}, rank={top10.ranks[i]}, %ile={top10.percentiles[i]:.1f}")
    
    # Test to_dict
    result_dict = top10.to_dict()
    print(f"\n✓ to_dict() successful with {len(result_dict['metabolites'])} metabolites")
    print(f"✓ All comprehensive prediction features working")


def test_save_load(artifacts):
    """Test saving and loading artifacts."""
    print("\n=== Testing Save/Load ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        save_model_artifacts(artifacts, tmpdir)
        print(f"✓ Saved artifacts to {tmpdir}")
        
        # Load
        loaded_artifacts = load_model_artifacts(tmpdir)
        print(f"✓ Loaded artifacts")
        
        # Verify
        assert loaded_artifacts.model.n_microbes == artifacts.model.n_microbes
        assert loaded_artifacts.model.n_metabolites == artifacts.model.n_metabolites
        assert loaded_artifacts.model.embedding_dim == artifacts.model.embedding_dim
        
        # Check embeddings are the same
        diff = np.abs(
            loaded_artifacts.model.microbe_embeddings - artifacts.model.microbe_embeddings
        ).max()
        print(f"✓ Max embedding difference: {diff:.6f}")
        assert diff < 1e-5, "Embeddings differ after save/load"


def test_training_data_methods():
    """Test TrainingData convenience methods."""
    print("\n=== Testing TrainingData Methods ===")
    
    training_data = prepare_training_data()
    
    # Test get_microbe_positives
    microbe_id = list(training_data.microbe_to_idx.keys())[0]
    positives = training_data.get_microbe_positives(microbe_id)
    print(f"✓ Microbe {microbe_id} has {len(positives)} positive metabolites")
    
    # Test get_metabolite_positives
    metabolite_id = list(training_data.metabolite_to_idx.keys())[0]
    consumers = training_data.get_metabolite_positives(metabolite_id)
    print(f"✓ Metabolite {metabolite_id} consumed by {len(consumers)} microbes")


def test_isolate_embeddings(artifacts):
    """Test isolate embedding access."""
    print("\n=== Testing Isolate Embeddings ===")
    
    # Get a few isolates
    isolate_ids = list(artifacts.training_data.microbe_to_idx.keys())[:3]
    
    print(f"Testing with {len(isolate_ids)} isolates:")
    for isolate_id in isolate_ids:
        # Get embedding using unified interface
        embedding = artifacts.get_embedding("isolate", isolate_id)
        print(f"  {isolate_id}: shape={embedding.shape}, norm={np.linalg.norm(embedding):.3f}")
        
        # Verify it matches direct access
        idx = artifacts.training_data.microbe_to_idx[isolate_id]
        direct_embedding = artifacts.model.microbe_embeddings[idx]
        assert np.allclose(embedding, direct_embedding), "Embedding mismatch!"
    
    # Test metabolite embeddings
    metabolite_ids = list(artifacts.training_data.metabolite_to_idx.keys())[:2]
    print(f"\nTesting metabolite embeddings:")
    for metabolite_id in metabolite_ids:
        embedding = artifacts.get_metabolite_embedding(metabolite_id)
        print(f"  {metabolite_id}: shape={embedding.shape}")
    
    print("✓ All isolate and metabolite embedding accessors working")


def main():
    print("Starting model training tests...")
    
    # Test training data methods
    test_training_data_methods()
    
    # Train and test BPR
    bpr_artifacts = test_bpr_training()
    
    # Test isolate embeddings
    test_isolate_embeddings(bpr_artifacts)
    
    # Train and test SVD
    svd_artifacts = test_svd_training()
    
    # Test taxonomy embeddings
    test_taxonomy_embeddings(bpr_artifacts)
    
    # Test distinctive scores
    test_distinctive_scores(bpr_artifacts)
    
    # Test comprehensive predictions
    test_comprehensive_predictions(bpr_artifacts)
    
    # Test save/load
    test_save_load(bpr_artifacts)
    
    print("\n" + "=" * 50)
    print("✓ All model tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
