"""Shared pytest fixtures for wom tests."""

import tempfile
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from loaders.loaders import load_exometabolite, prepare_training_data, load_chebi
from core.training import train_model, compute_taxonomy_embeddings
from core.io import save_model_artifacts
from core.types import AssertionType
from api.query_engine import QueryEngine


@pytest.fixture(scope="session")
def chebi_data():
    """Load ChEBI database (session-scoped, loaded once)."""
    return load_chebi()


@pytest.fixture(scope="session")
def exometabolite_data():
    """Load exometabolite dataset (session-scoped, loaded once)."""
    return load_exometabolite()


@pytest.fixture(scope="function")
def training_data():
    """Prepare training data for tests."""
    return prepare_training_data(preference_type=AssertionType.CONSUMED)


@pytest.fixture(scope="function")
def trained_bpr_artifacts(training_data):
    """Train a small BPR model for testing."""
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
        artifacts.taxonomy_embeddings[level] = compute_taxonomy_embeddings(
            artifacts, taxon_level=level
        )
    
    return artifacts


@pytest.fixture(scope="function")
def trained_svd_artifacts(training_data):
    """Train a small SVD model for testing."""
    artifacts = train_model(
        training_data=training_data,
        model_type="svd",
        embedding_dim=8,
        verbose=False,
    )
    
    # Compute taxonomy embeddings
    for level in ["genus", "family"]:
        artifacts.taxonomy_embeddings[level] = compute_taxonomy_embeddings(
            artifacts, taxon_level=level
        )
    
    return artifacts


@pytest.fixture(scope="function")
def temp_model_dir(trained_bpr_artifacts):
    """Save trained model to temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_model_artifacts(trained_bpr_artifacts, tmpdir)
        yield tmpdir


@pytest.fixture(scope="function")
def trained_artifacts(temp_model_dir):
    """Compatibility fixture for new CLI/API integration tests.

    Exposes an object with `.training_data` as a proxy that supports both:
    - training-data attributes (e.g., microbe_to_idx)
    - query-engine methods used by execute_query()
    """
    engine = QueryEngine(temp_model_dir)
    training_data = engine.artifacts.training_data

    class _EngineTrainingDataProxy:
        def __init__(self, query_engine, td):
            self._query_engine = query_engine
            self._training_data = td

        def __getattr__(self, name):
            if hasattr(self._training_data, name):
                return getattr(self._training_data, name)
            return getattr(self._query_engine, name)

    class _CompatArtifacts:
        def __init__(self, query_engine, td):
            self.engine = query_engine
            self.training_data = _EngineTrainingDataProxy(query_engine, td)

    return _CompatArtifacts(engine, training_data)
