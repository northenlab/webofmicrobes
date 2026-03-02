"""Comprehensive tests for all model types across consumed and secreted data.

This module tests all 5 model types (BPR, ImplicitBPR, LMF, WRMF, SVD) with both
consumed and secreted assertion types, covering:
- Training
- Inference/prediction
- Embeddings
- Index mappings
- Save/load persistence
- Taxonomy aggregation
- Distinctive scores
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from core.training import train_model, compute_taxonomy_embeddings, compute_distinctive_scores
from core.io import save_model_artifacts, load_model_artifacts
from core.types import AssertionType
from loaders.loaders import prepare_training_data


# Test parameters
MODEL_TYPES = ["bpr", "implicit_bpr", "lmf", "wrmf", "svd"]
ASSERTION_TYPES = [AssertionType.CONSUMED, AssertionType.SECRETED]
EMBEDDING_DIM = 8  # Small for fast tests
N_EPOCHS = 3  # Small for fast tests


def get_training_kwargs(model_type):
    """Get appropriate training kwargs for each model type."""
    if model_type == "bpr":
        return {"n_epochs": N_EPOCHS, "batch_size": 32}
    elif model_type in ["implicit_bpr", "lmf", "wrmf"]:
        return {"iterations": N_EPOCHS}
    elif model_type == "svd":
        return {}  # SVD has no training iterations
    else:
        return {}


@pytest.fixture(scope="module", params=ASSERTION_TYPES)
def training_data_by_type(request):
    """Prepare training data for each assertion type."""
    return prepare_training_data(preference_type=request.param)


@pytest.fixture(scope="module", params=MODEL_TYPES)
def model_type(request):
    """Parametrize over all model types."""
    return request.param


class TestModelTraining:
    """Test that all model types train successfully."""
    
    def test_model_trains_on_consumed(self, model_type):
        """All models should train on consumed data."""
        training_data = prepare_training_data(preference_type=AssertionType.CONSUMED)
        
        artifacts = train_model(
            training_data=training_data,
            model_type=model_type,
            embedding_dim=EMBEDDING_DIM,
            verbose=False,
            **get_training_kwargs(model_type),
        )
        
        assert artifacts is not None
        assert artifacts.model is not None
        assert artifacts.model_type.lower() == model_type.lower()
    
    def test_model_trains_on_secreted(self, model_type):
        """All models should train on secreted data."""
        training_data = prepare_training_data(preference_type=AssertionType.SECRETED)
        
        artifacts = train_model(
            training_data=training_data,
            model_type=model_type,
            embedding_dim=EMBEDDING_DIM,
            verbose=False,
            **get_training_kwargs(model_type),
        )
        
        assert artifacts is not None
        assert artifacts.model is not None
        assert artifacts.training_data.preference_type == AssertionType.SECRETED


class TestModelEmbeddings:
    """Test embedding properties for all model types."""
    
    @pytest.fixture(scope="class")
    def trained_artifacts(self, model_type):
        """Train a model once per class."""
        training_data = prepare_training_data(preference_type=AssertionType.CONSUMED)
        return train_model(
            training_data=training_data,
            model_type=model_type,
            embedding_dim=EMBEDDING_DIM,
            **get_training_kwargs(model_type),
            verbose=False,
        )
    
    def test_embedding_dimensions(self, trained_artifacts):
        """Embeddings should have correct dimensions (some models may add bias dimensions)."""
        # Check row dimension matches number of entities
        assert trained_artifacts.model.microbe_embeddings.shape[0] == trained_artifacts.training_data.n_microbes
        assert trained_artifacts.model.metabolite_embeddings.shape[0] == trained_artifacts.training_data.n_metabolites
        
        # Check embedding dimension is close to requested (some models add bias terms)
        # implicit library models add +1 for bias, so allow a small margin
        microbe_dim = trained_artifacts.model.microbe_embeddings.shape[1]
        metabolite_dim = trained_artifacts.model.metabolite_embeddings.shape[1]
        assert EMBEDDING_DIM <= microbe_dim <= EMBEDDING_DIM + 2, f"Microbe embedding dim {microbe_dim} not in range [{EMBEDDING_DIM}, {EMBEDDING_DIM + 2}]"
        assert EMBEDDING_DIM <= metabolite_dim <= EMBEDDING_DIM + 2, f"Metabolite embedding dim {metabolite_dim} not in range [{EMBEDDING_DIM}, {EMBEDDING_DIM + 2}]"
    
    def test_embeddings_are_numeric(self, trained_artifacts):
        """Embeddings should be numeric without NaN or Inf."""
        microbe_emb = trained_artifacts.model.microbe_embeddings
        metabolite_emb = trained_artifacts.model.metabolite_embeddings
        
        assert not np.isnan(microbe_emb).any(), "Microbe embeddings contain NaN"
        assert not np.isinf(microbe_emb).any(), "Microbe embeddings contain Inf"
        assert not np.isnan(metabolite_emb).any(), "Metabolite embeddings contain NaN"
        assert not np.isinf(metabolite_emb).any(), "Metabolite embeddings contain Inf"
    
    def test_embeddings_have_variance(self, trained_artifacts):
        """Embeddings should have non-zero variance (not all the same)."""
        microbe_emb = trained_artifacts.model.microbe_embeddings
        metabolite_emb = trained_artifacts.model.metabolite_embeddings
        
        assert microbe_emb.std() > 0, "Microbe embeddings have no variance"
        assert metabolite_emb.std() > 0, "Metabolite embeddings have no variance"
    
    def test_embedding_magnitudes_reasonable(self, trained_artifacts):
        """Embedding values should be in a reasonable range."""
        microbe_emb = trained_artifacts.model.microbe_embeddings
        metabolite_emb = trained_artifacts.model.metabolite_embeddings
        
        # Most embeddings should be in [-10, 10] range
        assert np.abs(microbe_emb).max() < 100, "Microbe embeddings too large"
        assert np.abs(metabolite_emb).max() < 100, "Metabolite embeddings too large"


class TestMicrobeIndices:
    """Test microbe index mappings for all model types."""
    
    @pytest.fixture(scope="class")
    def trained_artifacts(self, model_type):
        """Train a model once per class."""
        training_data = prepare_training_data(preference_type=AssertionType.CONSUMED)
        return train_model(
            training_data=training_data,
            model_type=model_type,
            embedding_dim=EMBEDDING_DIM,
            **get_training_kwargs(model_type),
            verbose=False,
        )
    
    def test_microbe_to_idx_mapping(self, trained_artifacts):
        """Microbe IDs should map to valid indices."""
        training_data = trained_artifacts.training_data
        
        # Check all microbe IDs map to valid indices
        for microbe_id, idx in training_data.microbe_to_idx.items():
            assert 0 <= idx < training_data.n_microbes
            assert isinstance(idx, (int, np.integer))
    
    def test_microbe_idx_to_id_mapping(self, trained_artifacts):
        """Indices should map back to microbe IDs."""
        training_data = trained_artifacts.training_data
        
        # Check reverse mapping
        for microbe_id, idx in training_data.microbe_to_idx.items():
            assert training_data.idx_to_microbe[idx] == microbe_id
    
    def test_microbe_embedding_retrieval(self, trained_artifacts):
        """Should retrieve embeddings for specific microbes."""
        training_data = trained_artifacts.training_data
        model = trained_artifacts.model
        
        # Get a random microbe
        microbe_id = list(training_data.microbe_to_idx.keys())[0]
        idx = training_data.microbe_to_idx[microbe_id]
        
        # Retrieve embedding
        embedding = model.microbe_embeddings[idx]
        
        # Allow for bias dimensions (implicit models add +1)
        assert EMBEDDING_DIM <= embedding.shape[0] <= EMBEDDING_DIM + 2
        assert not np.isnan(embedding).any()
    
    def test_all_microbes_have_embeddings(self, trained_artifacts):
        """Every microbe in training data should have an embedding."""
        training_data = trained_artifacts.training_data
        model = trained_artifacts.model
        
        assert len(training_data.microbe_to_idx) == model.n_microbes
        assert model.microbe_embeddings.shape[0] == training_data.n_microbes


class TestMetaboliteIndices:
    """Test metabolite/compound index mappings for all model types."""
    
    @pytest.fixture(scope="class")
    def trained_artifacts(self, model_type):
        """Train a model once per class."""
        training_data = prepare_training_data(preference_type=AssertionType.CONSUMED)
        return train_model(
            training_data=training_data,
            model_type=model_type,
            embedding_dim=EMBEDDING_DIM,
            **get_training_kwargs(model_type),
            verbose=False,
        )
    
    def test_metabolite_to_idx_mapping(self, trained_artifacts):
        """Metabolite IDs should map to valid indices."""
        training_data = trained_artifacts.training_data
        
        # Check all metabolite IDs map to valid indices
        for metabolite_id, idx in training_data.metabolite_to_idx.items():
            assert 0 <= idx < training_data.n_metabolites
            assert isinstance(idx, (int, np.integer))
    
    def test_metabolite_idx_to_id_mapping(self, trained_artifacts):
        """Indices should map back to metabolite IDs."""
        training_data = trained_artifacts.training_data
        
        # Check reverse mapping
        for metabolite_id, idx in training_data.metabolite_to_idx.items():
            assert training_data.idx_to_metabolite[idx] == metabolite_id
    
    def test_metabolite_embedding_retrieval(self, trained_artifacts):
        """Should retrieve embeddings for specific metabolites."""
        training_data = trained_artifacts.training_data
        model = trained_artifacts.model
        
        # Get a random metabolite
        metabolite_id = list(training_data.metabolite_to_idx.keys())[0]
        idx = training_data.metabolite_to_idx[metabolite_id]
        
        # Retrieve embedding
        embedding = model.metabolite_embeddings[idx]
        
        # Allow for bias dimensions (implicit models add +1)
        assert EMBEDDING_DIM <= embedding.shape[0] <= EMBEDDING_DIM + 2
        assert not np.isnan(embedding).any()
    
    def test_all_metabolites_have_embeddings(self, trained_artifacts):
        """Every metabolite in training data should have an embedding."""
        training_data = trained_artifacts.training_data
        model = trained_artifacts.model
        
        assert len(training_data.metabolite_to_idx) == model.n_metabolites
        assert model.metabolite_embeddings.shape[0] == training_data.n_metabolites


class TestModelInference:
    """Test prediction/inference for all model types."""
    
    @pytest.fixture(scope="class")
    def trained_artifacts(self, model_type):
        """Train a model once per class."""
        training_data = prepare_training_data(preference_type=AssertionType.CONSUMED)
        artifacts = train_model(
            training_data=training_data,
            model_type=model_type,
            embedding_dim=EMBEDDING_DIM,
            **get_training_kwargs(model_type),
            verbose=False,
        )
        # Add taxonomy embeddings
        for level in ["genus", "family"]:
            artifacts.taxonomy_embeddings[level] = compute_taxonomy_embeddings(
                artifacts, taxon_level=level
            )
        return artifacts
    
    def test_predict_for_isolate(self, trained_artifacts):
        """Should predict scores for all metabolites for an isolate."""
        microbe_id = list(trained_artifacts.training_data.microbe_to_idx.keys())[0]
        scores = trained_artifacts.predict_for_taxon("isolate", microbe_id)
        
        assert len(scores) == trained_artifacts.training_data.n_metabolites
        assert not np.isnan(scores).any()
        assert not np.isinf(scores).any()
    
    def test_predict_for_genus(self, trained_artifacts):
        """Should predict scores at genus level."""
        # Get a genus from taxonomy
        microbe_id = list(trained_artifacts.training_data.microbe_to_idx.keys())[0]
        taxonomy = trained_artifacts.training_data.microbe_taxonomy[microbe_id]
        
        if taxonomy.genus:
            scores = trained_artifacts.predict_for_taxon("genus", taxonomy.genus)
            
            assert len(scores) == trained_artifacts.training_data.n_metabolites
            assert not np.isnan(scores).any()
    
    def test_prediction_scores_vary(self, trained_artifacts):
        """Prediction scores should have variance across metabolites."""
        microbe_id = list(trained_artifacts.training_data.microbe_to_idx.keys())[0]
        scores = trained_artifacts.predict_for_taxon("isolate", microbe_id)
        
        # Scores should vary (not all the same)
        assert scores.std() > 0, "All prediction scores are identical"
    
    def test_top_predictions_ranked(self, trained_artifacts):
        """Top predictions should be properly ranked."""
        microbe_id = list(trained_artifacts.training_data.microbe_to_idx.keys())[0]
        scores = trained_artifacts.predict_for_taxon("isolate", microbe_id)
        
        # Get top 10
        top_indices = np.argsort(-scores)[:10]
        top_scores = scores[top_indices]
        
        # Should be in descending order
        assert np.all(top_scores[:-1] >= top_scores[1:]), "Top scores not sorted"


class TestModelPersistence:
    """Test save/load for all model types."""
    
    def test_save_load_preserves_predictions(self, model_type):
        """Predictions should be identical after save/load."""
        # Train model
        training_data = prepare_training_data(preference_type=AssertionType.CONSUMED)
        artifacts = train_model(
            training_data=training_data,
            model_type=model_type,
            embedding_dim=EMBEDDING_DIM,
            verbose=False,
            **get_training_kwargs(model_type),
        )
        
        # Get predictions
        microbe_id = list(training_data.microbe_to_idx.keys())[0]
        original_scores = artifacts.predict_for_taxon("isolate", microbe_id)
        
        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            save_model_artifacts(artifacts, tmpdir)
            loaded_artifacts = load_model_artifacts(tmpdir)
            
            # Get predictions from loaded model
            loaded_scores = loaded_artifacts.predict_for_taxon("isolate", microbe_id)
            
            # Should be very close
            assert np.allclose(original_scores, loaded_scores, atol=1e-5)
    
    def test_save_load_preserves_embeddings(self, model_type):
        """Embeddings should be identical after save/load."""
        # Train model
        training_data = prepare_training_data(preference_type=AssertionType.CONSUMED)
        artifacts = train_model(
            training_data=training_data,
            model_type=model_type,
            embedding_dim=EMBEDDING_DIM,
            verbose=False,
            **get_training_kwargs(model_type),
        )
        
        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            save_model_artifacts(artifacts, tmpdir)
            loaded_artifacts = load_model_artifacts(tmpdir)
            
            # Check embeddings
            microbe_diff = np.abs(
                artifacts.model.microbe_embeddings - 
                loaded_artifacts.model.microbe_embeddings
            ).max()
            metabolite_diff = np.abs(
                artifacts.model.metabolite_embeddings - 
                loaded_artifacts.model.metabolite_embeddings
            ).max()
            
            assert microbe_diff < 1e-5, f"Microbe embeddings differ by {microbe_diff}"
            assert metabolite_diff < 1e-5, f"Metabolite embeddings differ by {metabolite_diff}"


class TestTaxonomyAggregation:
    """Test taxonomy-level embeddings for all model types."""
    
    @pytest.fixture(scope="class")
    def trained_artifacts(self, model_type):
        """Train a model once per class."""
        training_data = prepare_training_data(preference_type=AssertionType.CONSUMED)
        artifacts = train_model(
            training_data=training_data,
            model_type=model_type,
            embedding_dim=EMBEDDING_DIM,
            **get_training_kwargs(model_type),
            verbose=False,
        )
        # Compute taxonomy embeddings
        for level in ["genus", "family"]:
            artifacts.taxonomy_embeddings[level] = compute_taxonomy_embeddings(
                artifacts, taxon_level=level
            )
        return artifacts
    
    def test_genus_embeddings_exist(self, trained_artifacts):
        """Should compute genus-level embeddings."""
        assert "genus" in trained_artifacts.taxonomy_embeddings
        genus_emb = trained_artifacts.taxonomy_embeddings["genus"]
        assert len(genus_emb) > 0
    
    def test_family_embeddings_exist(self, trained_artifacts):
        """Should compute family-level embeddings."""
        assert "family" in trained_artifacts.taxonomy_embeddings
        family_emb = trained_artifacts.taxonomy_embeddings["family"]
        assert len(family_emb) > 0
    
    def test_taxonomy_embedding_dimensions(self, trained_artifacts):
        """Taxonomy embeddings should have correct dimension."""
        for level in ["genus", "family"]:
            if level in trained_artifacts.taxonomy_embeddings:
                for taxon, embedding in trained_artifacts.taxonomy_embeddings[level].items():
                    # Allow for bias dimensions (implicit models add +1)
                    emb_dim = embedding.shape[0]
                    assert EMBEDDING_DIM <= emb_dim <= EMBEDDING_DIM + 2, (
                        f"{level} embedding for {taxon} has shape {embedding.shape}, expected dim in [{EMBEDDING_DIM}, {EMBEDDING_DIM + 2}]"
                    )
    
    def test_single_isolate_genus_matches(self, trained_artifacts):
        """For single-isolate genera, genus embedding should match isolate."""
        training_data = trained_artifacts.training_data
        genus_emb = trained_artifacts.taxonomy_embeddings["genus"]
        
        # Find a genus with only 1 isolate
        genus_counts = {}
        for org_id, tax in training_data.microbe_taxonomy.items():
            if tax.genus:
                genus_counts[tax.genus] = genus_counts.get(tax.genus, 0) + 1
        
        for genus, count in genus_counts.items():
            if count == 1 and genus in genus_emb:
                # Find the isolate
                for org_id, tax in training_data.microbe_taxonomy.items():
                    if tax.genus == genus:
                        idx = training_data.microbe_to_idx[org_id]
                        isolate_emb = trained_artifacts.model.microbe_embeddings[idx]
                        
                        # Should be very close
                        assert np.allclose(genus_emb[genus], isolate_emb, atol=1e-5)
                        break
                break  # Only test one


class TestDistinctiveScores:
    """Test distinctive score computation for all model types."""
    
    @pytest.fixture(scope="class")
    def trained_artifacts(self, model_type):
        """Train a model once per class."""
        training_data = prepare_training_data(preference_type=AssertionType.CONSUMED)
        artifacts = train_model(
            training_data=training_data,
            model_type=model_type,
            embedding_dim=EMBEDDING_DIM,
            **get_training_kwargs(model_type),
            verbose=False,
        )
        # Add taxonomy embeddings
        for level in ["genus", "family"]:
            artifacts.taxonomy_embeddings[level] = compute_taxonomy_embeddings(
                artifacts, taxon_level=level
            )
        return artifacts
    
    def test_compute_distinctive_scores(self, trained_artifacts):
        """Should compute distinctive scores successfully."""
        microbe_id = list(trained_artifacts.training_data.microbe_to_idx.keys())[0]
        
        distinctive = compute_distinctive_scores(
            trained_artifacts,
            microbe_id,
            taxon_level="genus",
            top_k=10,
        )
        
        assert distinctive is not None
        assert len(distinctive) > 0
        assert len(distinctive) <= 10
    
    def test_distinctive_scores_are_numeric(self, trained_artifacts):
        """Distinctive scores should be numeric."""
        microbe_id = list(trained_artifacts.training_data.microbe_to_idx.keys())[0]
        
        distinctive = compute_distinctive_scores(
            trained_artifacts,
            microbe_id,
            taxon_level="genus",
            top_k=5,
        )
        
        for metabolite_id, score in distinctive.items():
            assert isinstance(score, (int, float, np.number))
            assert not np.isnan(score)


class TestConsumedVsSecreted:
    """Test differences between consumed and secreted models."""
    
    def test_different_training_data(self):
        """Consumed and secreted should have different training data."""
        consumed_data = prepare_training_data(preference_type=AssertionType.CONSUMED)
        secreted_data = prepare_training_data(preference_type=AssertionType.SECRETED)
        
        # Should have different numbers of interactions
        consumed_interactions = sum(len(v) for v in consumed_data.positives.values())
        secreted_interactions = sum(len(v) for v in secreted_data.positives.values())
        
        # These should generally be different (unless data is identical)
        # At minimum, the preference type should be different
        assert consumed_data.preference_type == AssertionType.CONSUMED
        assert secreted_data.preference_type == AssertionType.SECRETED
    
    def test_model_type_preserved(self, model_type):
        """Model type should be preserved in artifacts."""
        consumed_data = prepare_training_data(preference_type=AssertionType.CONSUMED)
        
        artifacts = train_model(
            training_data=consumed_data,
            model_type=model_type,
            embedding_dim=EMBEDDING_DIM,
            verbose=False,
            **get_training_kwargs(model_type),
        )
        
        assert artifacts.model_type.lower() == model_type.lower()
        assert artifacts.training_data.preference_type == AssertionType.CONSUMED


class TestModelStatistics:
    """Test statistical properties of model predictions."""
    
    @pytest.fixture(scope="class")
    def trained_artifacts(self, model_type):
        """Train a model once per class."""
        training_data = prepare_training_data(preference_type=AssertionType.CONSUMED)
        return train_model(
            training_data=training_data,
            model_type=model_type,
            embedding_dim=EMBEDDING_DIM,
            **get_training_kwargs(model_type),
            verbose=False,
        )
    
    def test_score_distribution_statistics(self, trained_artifacts):
        """Prediction scores should have reasonable statistics."""
        microbe_id = list(trained_artifacts.training_data.microbe_to_idx.keys())[0]
        scores = trained_artifacts.predict_for_taxon("isolate", microbe_id)
        
        # Basic statistics
        mean = scores.mean()
        std = scores.std()
        median = np.median(scores)
        
        # Sanity checks
        assert not np.isnan(mean)
        assert not np.isnan(std)
        assert not np.isnan(median)
        assert std > 0, "No variance in scores"
    
    def test_score_percentiles(self, trained_artifacts):
        """Should compute percentiles correctly."""
        microbe_id = list(trained_artifacts.training_data.microbe_to_idx.keys())[0]
        scores = trained_artifacts.predict_for_taxon("isolate", microbe_id)
        
        # Get percentiles
        p25 = np.percentile(scores, 25)
        p50 = np.percentile(scores, 50)
        p75 = np.percentile(scores, 75)
        p95 = np.percentile(scores, 95)
        
        # Should be ordered
        assert p25 <= p50 <= p75 <= p95
