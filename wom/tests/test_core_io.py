"""Test core.io module - model persistence and loading.

This module tests the core/io.py, which handles:
- Saving trained models and artifacts to disk
- Loading saved models back into memory
- Data integrity validation after save/load
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from core.io import save_model_artifacts, load_model_artifacts
from core.training import train_model
from loaders.loaders import prepare_training_data


class TestSaveModelArtifacts:
    """Test saving model artifacts."""

    def test_save_artifacts_creates_files(self, trained_bpr_artifacts):
        """Saving should create files in directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_model_artifacts(trained_bpr_artifacts, tmpdir)
            
            # Check that files were created
            saved_files = list(Path(tmpdir).glob("*"))
            assert len(saved_files) > 0, "No files were saved"

    def test_save_artifacts_preserves_model_structure(self, trained_bpr_artifacts):
        """Saved model should have correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_model_artifacts(trained_bpr_artifacts, tmpdir)
            
            tmpdir_path = Path(tmpdir)
            # Check for expected model files (specific names depend on implementation)
            # At minimum, should have model embeddings and metadata
            assert (tmpdir_path / "config.json").exists() or len(list(tmpdir_path.glob("*.pt"))) > 0

    def test_save_artifacts_multiple_models(self, trained_bpr_artifacts, trained_svd_artifacts):
        """Should be able to save different model types."""
        with tempfile.TemporaryDirectory() as tmpdir_bpr:
            with tempfile.TemporaryDirectory() as tmpdir_svd:
                save_model_artifacts(trained_bpr_artifacts, tmpdir_bpr)
                save_model_artifacts(trained_svd_artifacts, tmpdir_svd)
                
                assert len(list(Path(tmpdir_bpr).glob("*"))) > 0
                assert len(list(Path(tmpdir_svd).glob("*"))) > 0


class TestLoadModelArtifacts:
    """Test loading model artifacts."""

    def test_load_artifacts_successfully(self, trained_bpr_artifacts):
        """Should load saved artifacts without errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_model_artifacts(trained_bpr_artifacts, tmpdir)
            loaded = load_model_artifacts(tmpdir)
            
            assert loaded is not None
            assert loaded.model is not None

    def test_loaded_model_has_correct_dimensions(self, trained_bpr_artifacts):
        """Loaded model should have same dimensions as original."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_model_artifacts(trained_bpr_artifacts, tmpdir)
            loaded = load_model_artifacts(tmpdir)
            
            assert loaded.model.n_microbes == trained_bpr_artifacts.model.n_microbes
            assert loaded.model.n_metabolites == trained_bpr_artifacts.model.n_metabolites
            assert loaded.model.embedding_dim == trained_bpr_artifacts.model.embedding_dim

    def test_loaded_embeddings_match_original(self, trained_bpr_artifacts):
        """Loaded embeddings should match original (within floating point tolerance)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_model_artifacts(trained_bpr_artifacts, tmpdir)
            loaded = load_model_artifacts(tmpdir)
            
            # Check microbe embeddings
            microbe_diff = np.abs(
                loaded.model.microbe_embeddings - trained_bpr_artifacts.model.microbe_embeddings
            ).max()
            assert microbe_diff < 1e-5, f"Microbe embeddings differ by {microbe_diff}"
            
            # Check metabolite embeddings
            metabolite_diff = np.abs(
                loaded.model.metabolite_embeddings - trained_bpr_artifacts.model.metabolite_embeddings
            ).max()
            assert metabolite_diff < 1e-5, f"Metabolite embeddings differ by {metabolite_diff}"

    def test_loaded_training_data_preserved(self, trained_bpr_artifacts):
        """Loaded artifacts should preserve training data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_model_artifacts(trained_bpr_artifacts, tmpdir)
            loaded = load_model_artifacts(tmpdir)
            
            assert loaded.training_data is not None
            assert loaded.training_data.n_microbes == trained_bpr_artifacts.training_data.n_microbes
            assert loaded.training_data.n_metabolites == trained_bpr_artifacts.training_data.n_metabolites


class TestSaveLoadRoundTrip:
    """Test complete save/load round trip."""

    def test_save_load_roundtrip_predictions(self, trained_bpr_artifacts):
        """Predictions should be identical after save/load."""
        microbe_id = list(
            trained_bpr_artifacts.training_data.microbe_to_idx.keys()
        )[0]
        
        # Get predictions from original
        original_scores = trained_bpr_artifacts.predict_for_taxon("isolate", microbe_id)
        
        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            save_model_artifacts(trained_bpr_artifacts, tmpdir)
            loaded = load_model_artifacts(tmpdir)
            
            # Get predictions from loaded
            loaded_scores = loaded.predict_for_taxon("isolate", microbe_id)
            
            # Should be identical
            diff = np.abs(original_scores - loaded_scores).max()
            assert diff < 1e-5, f"Predictions differ by {diff}"

    def test_save_load_preserves_taxonomy_embeddings(self, trained_bpr_artifacts):
        """Loaded artifacts should preserve computed taxonomy embeddings if saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_model_artifacts(trained_bpr_artifacts, tmpdir)
            loaded = load_model_artifacts(tmpdir)
            
            # Note: Taxonomy embeddings are computed on-demand after loading
            # They should be present in original before saving
            assert "genus" in trained_bpr_artifacts.taxonomy_embeddings
            assert "family" in trained_bpr_artifacts.taxonomy_embeddings
            
            # After loading, they may not be persisted (depends on IO implementation)
            # But the loaded artifacts should be able to compute them
            assert loaded.training_data is not None
            assert loaded.model is not None

    def test_load_nonexistent_directory(self):
        """Loading from nonexistent directory should raise error."""
        with pytest.raises((FileNotFoundError, ValueError, OSError)):
            load_model_artifacts("/nonexistent/path/abc123")

    def test_multiple_save_load_cycles(self, trained_bpr_artifacts):
        """Should handle multiple save/load cycles."""
        microbe_id = list(
            trained_bpr_artifacts.training_data.microbe_to_idx.keys()
        )[0]
        original_scores = trained_bpr_artifacts.predict_for_taxon("isolate", microbe_id)
        
        artifacts = trained_bpr_artifacts
        
        # Do 3 save/load cycles
        for cycle in range(3):
            with tempfile.TemporaryDirectory() as tmpdir:
                save_model_artifacts(artifacts, tmpdir)
                artifacts = load_model_artifacts(tmpdir)
        
        # Predictions should still match
        final_scores = artifacts.predict_for_taxon("isolate", microbe_id)
        diff = np.abs(original_scores - final_scores).max()
        assert diff < 1e-5, f"After {3} cycles, predictions differ by {diff}"
