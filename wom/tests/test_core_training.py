"""Test core.training module - model training and inference.

This module tests the core/training.py, which handles:
- Training recommender models (BPR, SVD, etc.)
- Computing embeddings at different taxonomy levels
- Computing distinctive scores for microbe-specific metabolites
- Model evaluation and statistics
"""

import pytest
import numpy as np
from pathlib import Path

from core.training import (
    train_model,
    compute_taxonomy_embeddings,
    compute_distinctive_scores,
)
from core.types import AssertionType


class TestModelTraining:
    """Test model training functionality."""

    def test_bpr_model_trains(self, training_data):
        """BPR model should train successfully."""
        artifacts = train_model(
            training_data=training_data,
            model_type="bpr",
            embedding_dim=8,
            n_epochs=3,
            batch_size=32,
            verbose=False,
        )
        
        assert artifacts is not None
        assert artifacts.model is not None

    def test_svd_model_trains(self, training_data):
        """SVD model should train successfully."""
        artifacts = train_model(
            training_data=training_data,
            model_type="svd",
            embedding_dim=8,
            verbose=False,
        )
        
        assert artifacts is not None
        assert artifacts.model is not None

    def test_model_embeddings_have_correct_shape(self, trained_bpr_artifacts):
        """Model embeddings should have expected dimensions."""
        n_microbes = trained_bpr_artifacts.model.n_microbes
        n_metabolites = trained_bpr_artifacts.model.n_metabolites
        embedding_dim = trained_bpr_artifacts.model.embedding_dim
        
        assert trained_bpr_artifacts.model.microbe_embeddings.shape == (
            n_microbes,
            embedding_dim,
        )
        assert trained_bpr_artifacts.model.metabolite_embeddings.shape == (
            n_metabolites,
            embedding_dim,
        )

    def test_model_embeddings_are_reasonable(self, trained_bpr_artifacts):
        """Model embeddings should have reasonable values (not NaN, not all zeros)."""
        microbe_emb = trained_bpr_artifacts.model.microbe_embeddings
        metabolite_emb = trained_bpr_artifacts.model.metabolite_embeddings
        
        # Check for NaN
        assert not np.isnan(microbe_emb).any(), "Microbe embeddings contain NaN"
        assert not np.isnan(metabolite_emb).any(), "Metabolite embeddings contain NaN"
        
        # Check not all zeros (at least some variance)
        assert microbe_emb.std() > 0, "Microbe embeddings have no variance"
        assert metabolite_emb.std() > 0, "Metabolite embeddings have no variance"

    def test_model_artifact_stores_training_data(self, trained_bpr_artifacts, training_data):
        """Artifacts should preserve training data reference."""
        assert trained_bpr_artifacts.training_data == training_data


class TestTaxonomyEmbeddings:
    """Test taxonomy embedding computation."""

    def test_compute_genus_embeddings(self, trained_bpr_artifacts):
        """Should compute valid genus embeddings."""
        genus_emb = compute_taxonomy_embeddings(
            trained_bpr_artifacts, taxon_level="genus"
        )
        
        assert genus_emb is not None
        assert len(genus_emb) > 0

    def test_compute_family_embeddings(self, trained_bpr_artifacts):
        """Should compute valid family embeddings."""
        family_emb = compute_taxonomy_embeddings(
            trained_bpr_artifacts, taxon_level="family"
        )
        
        assert family_emb is not None
        assert len(family_emb) > 0

    def test_taxonomy_embeddings_have_correct_dimension(self, trained_bpr_artifacts):
        """Taxonomy embeddings should match model embedding dimension."""
        embedding_dim = trained_bpr_artifacts.model.embedding_dim
        genus_emb = compute_taxonomy_embeddings(
            trained_bpr_artifacts, taxon_level="genus"
        )
        
        for taxon_name, embedding in genus_emb.items():
            assert embedding.shape == (embedding_dim,), (
                f"Embedding for {taxon_name} has wrong shape: "
                f"{embedding.shape} vs expected ({embedding_dim},)"
            )

    def test_single_isolate_genus_embedding(self, trained_bpr_artifacts):
        """For single-isolate genus, embedding should match isolate embedding."""
        # Find a genus with only 1 isolate
        genus_counts = {}
        for org_id, tax in trained_bpr_artifacts.training_data.microbe_taxonomy.items():
            if tax.genus:
                genus_counts[tax.genus] = genus_counts.get(tax.genus, 0) + 1
        
        single_isolate_genus = None
        isolate_id = None
        for org_id, tax in trained_bpr_artifacts.training_data.microbe_taxonomy.items():
            if tax.genus and genus_counts[tax.genus] == 1:
                single_isolate_genus = tax.genus
                isolate_id = org_id
                break
        
        if single_isolate_genus:
            genus_emb = compute_taxonomy_embeddings(
                trained_bpr_artifacts, taxon_level="genus"
            )
            
            genus_embedding = genus_emb[single_isolate_genus]
            idx = trained_bpr_artifacts.training_data.microbe_to_idx[isolate_id]
            isolate_embedding = trained_bpr_artifacts.model.microbe_embeddings[idx]
            
            # Should be very close (accounting for floating point precision)
            assert np.allclose(genus_embedding, isolate_embedding, atol=1e-5)


class TestDistinctiveScores:
    """Test distinctive score computation."""

    def test_compute_distinctive_scores(self, trained_bpr_artifacts):
        """Should compute distinctive scores successfully."""
        microbe_id = list(
            trained_bpr_artifacts.training_data.microbe_to_idx.keys()
        )[0]
        
        distinctive = compute_distinctive_scores(
            trained_bpr_artifacts,
            microbe_id,
            taxon_level="genus",
            top_k=10,
        )
        
        assert distinctive is not None
        assert len(distinctive) > 0
        assert len(distinctive) <= 10

    def test_distinctive_scores_are_numeric(self, trained_bpr_artifacts):
        """Distinctive scores should be numeric and reasonable."""
        microbe_id = list(
            trained_bpr_artifacts.training_data.microbe_to_idx.keys()
        )[0]
        
        distinctive = compute_distinctive_scores(
            trained_bpr_artifacts,
            microbe_id,
            taxon_level="genus",
            top_k=5,
        )
        
        for metabolite_id, score in distinctive.items():
            assert isinstance(score, (int, float, np.number))
            assert not np.isnan(score), f"Score for {metabolite_id} is NaN"

    def test_distinctive_scores_respect_top_k(self, trained_bpr_artifacts):
        """Distinctive scores should respect top_k parameter."""
        microbe_id = list(
            trained_bpr_artifacts.training_data.microbe_to_idx.keys()
        )[0]
        
        for k in [1, 5, 10]:
            distinctive = compute_distinctive_scores(
                trained_bpr_artifacts,
                microbe_id,
                taxon_level="genus",
                top_k=k,
            )
            
            assert len(distinctive) <= k


class TestPredictionInterface:
    """Test unified prediction interface."""

    def test_predict_at_isolate_level(self, trained_bpr_artifacts):
        """Should predict scores at isolate level."""
        microbe_id = list(
            trained_bpr_artifacts.training_data.microbe_to_idx.keys()
        )[0]
        
        scores = trained_bpr_artifacts.predict_for_taxon("isolate", microbe_id)
        
        assert scores is not None
        assert len(scores) > 0

    def test_predict_at_genus_level(self, trained_bpr_artifacts):
        """Should predict scores at genus level."""
        # Get a genus from taxonomy
        organism_id = list(
            trained_bpr_artifacts.training_data.microbe_to_idx.keys()
        )[0]
        taxonomy = trained_bpr_artifacts.training_data.microbe_taxonomy[organism_id]
        
        if taxonomy.genus:
            scores = trained_bpr_artifacts.predict_for_taxon("genus", taxonomy.genus)
            
            assert scores is not None
            assert len(scores) > 0

    def test_predict_scores_are_reasonable(self, trained_bpr_artifacts):
        """Prediction scores should be reasonable values."""
        microbe_id = list(
            trained_bpr_artifacts.training_data.microbe_to_idx.keys()
        )[0]
        
        scores = trained_bpr_artifacts.predict_for_taxon("isolate", microbe_id)
        
        # Check for NaN and reasonable range
        assert not np.isnan(scores).any()
        assert scores.min() >= -10  # Reasonable bounds for scores
        assert scores.max() <= 10
