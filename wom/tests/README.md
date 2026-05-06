# WOM Test Suite

This directory contains formal tests for the Web of Microbes (WOM) project using pytest.

## Test Summary

**308 total tests** covering:

### Core Functionality (63 tests)
- `test_loaders.py`: 15 tests (data loading pipeline)
- `test_core_training.py`: 15 tests (model training, basic functionality)
- `test_core_io.py`: 11 tests (model persistence)
- `test_api_query_engine.py`: 22 tests (query interface)

### Comprehensive Model Validation (146 tests)
- `test_all_models.py`: **146 tests** (all 5 model types × both data types)
  * Tests training, embeddings, indices, inference, persistence, taxonomy, distinctive scores, statistics

### API & Interface Testing (81+ tests)
- `test_execute_query.py`: 37 tests (shared core query logic used by both CLI & Web API)
- `test_cli.py`: 14 tests (command-line interface argument parsing & output formatting)
- `test_web_api.py`: 30+ tests (Flask REST API endpoints)

## Quick Start

**Run tests using uv:**
```bash
cd /Users/bpb/repos/webofmicrobes/wom
uv run pytest tests/ -v
```

That's it! `uv` automatically:
- Creates a virtual environment (`.venv/`)
- Installs dependencies from `pyproject.toml`
- Runs the tests

First time setup:
```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh
# or: pip install uv
# or: mamba install uv

# Sync dependencies (first time only)
cd /Users/bpb/repos/webofmicrobes/wom
uv sync --extra dev
```

## Dependencies

All dependencies are managed via `pyproject.toml`:
- Core: numpy, pandas, torch, scikit-learn
- Data: rapidfuzz, pyarrow
- Testing: pytest

Works on macOS (Apple Silicon), Linux, and Windows.

## Testing on Linux (Docker)

To verify tests work on Linux:

```bash
# Build the Docker image
cd /Users/bpb/repos/webofmicrobes/wom
docker build -t wom-tests .

# Run tests in container
docker run --rm wom-tests
```

The Dockerfile uses the same `pyproject.toml` and `uv` setup, ensuring consistency across platforms.

## Test Organization

Tests are organized by the main subfolders of the wom package:

### `test_loaders.py`
Tests for the **loaders/** module:
- ChEBI chemical database loading
- Exometabolite dataset loading  
- Compound name mapping to ChEBI IDs
- Training data preparation and structure
- Organism preference queries

**Key Classes Tested:**
- `load_chebi()` - ChEBI database loader
- `load_exometabolite()` - Exometabolite data loader
- `prepare_training_data()` - Training data preparation
- `map_compound_name()` - Compound name mapping

### `test_core_training.py`
Tests for the **core/training.py** module:
- BPR and SVD model training
- Embedding computation and shape validation
- Taxonomy-level embeddings (genus, family)
- Distinctive score computation
- Unified prediction interface across taxonomy levels

**Key Functions Tested:**
- `train_model()` - Train BPR/SVD models
- `compute_taxonomy_embeddings()` - Compute embeddings at various taxonomy levels
- `compute_distinctive_scores()` - Compute distinctive metabolites
- `predict_for_taxon()` - Unified prediction interface

### `test_core_io.py`
Tests for the **core/io.py** module:
- Saving trained models and artifacts to disk
- Loading models from disk
- Data integrity validation across save/load cycles
- Embedding preservation accuracy

**Key Functions Tested:**
- `save_model_artifacts()` - Save models to directory
- `load_model_artifacts()` - Load models from directory
- Save/load round-trip accuracy

### `test_api_query_engine.py`
Tests for the **api/** module:
- QueryEngine initialization and model loading
- Various query types (compounds for microbes, selective metabolites, etc.)
- Result structure and validation
- JSON serialization of responses
- Query explanations and reasoning

**Key Query Types Tested:**
- `CompoundsForMicrobesQuery` - Get compounds consumed by microbe(s)
- `MicrobesForCompoundsQuery` - Get microbes that consume compound(s)
- `SelectiveMetabolitesQuery` - Find metabolites favoring one microbe over others
- `MetaboliteEnvironmentFitnessQuery` - Rank microbes for metabolite environment
- `NicheMetabolitesQuery` - Find distinctive metabolites for microbe
- `SearchCompoundQuery` - Search for compounds by name
- `ListTaxaQuery` - List taxa at given level
- `ModelInfoQuery` - Get model metadata and statistics

### `test_all_models.py`
**Comprehensive tests for all 5 model types** across both consumed and secreted data (146 tests total):

**Model Types Tested:**
- `bpr` - Bayesian Personalized Ranking (PyTorch implementation)
- `implicit_bpr` - BPR using implicit library (optimized C++)
- `lmf` - Logistic Matrix Factorization
- `wrmf` - Weighted Regularized Matrix Factorization
- `svd` - Singular Value Decomposition

**Test Coverage (parametrized across all models):**
- **TestModelTraining** (10 tests): Models train on consumed and secreted data
- **TestModelEmbeddings** (20 tests): Embedding dimensions, numeric validity, variance, magnitudes
- **TestMicrobeIndices** (20 tests): Microbe index mappings and embedding retrieval
- **TestMetaboliteIndices** (20 tests): Metabolite index mappings and embedding retrieval
- **TestModelInference** (20 tests): Predictions for isolates/genera, score variance, ranking
- **TestModelPersistence** (10 tests): Save/load round-trips preserve all data
- **TestTaxonomyAggregation** (20 tests): Genus/family embeddings, dimensions, single-isolate matching
- **TestDistinctiveScores** (10 tests): Distinctive metabolite scores are numeric and valid
- **TestConsumedVsSecreted** (2 tests): Models preserve data type and model type
- **TestModelStatistics** (10 tests): Score distributions and percentiles
- Helper: `get_training_kwargs()` - Returns correct training parameters per model type

**Note:** Implicit library models (implicit_bpr, lmf, wrmf) add bias dimensions internally, so embedding shapes may be `embedding_dim + 1`. Tests accommodate this.

### `test_execute_query.py`
Tests the **shared core query logic** (scripts/wom.py::execute_query):
- All 10 query commands: compounds, microbes, selective, fitness, niche, competition, rank-substrates, search, list, info
- Parameter validation and error handling
- Response structure and JSON serialization
- Both consumed and secreted data support
- Distinctive scores and aggregation methods

**Why it matters:** The `execute_query()` function is the **single source of truth** for query logic used by both CLI and Web API. Testing it ensures both interfaces always stay in sync.

### `test_cli.py`
Tests the **command-line interface** (scripts/wom.py):
- Argument parsing for all commands
- Output formatting (JSON vs table)
- stderr handling for warnings and errors
- CLI-specific option validation

### `test_web_api.py`
Tests the **REST API** (scripts/wom_api.py):
- Health check endpoint (`GET /api/health`)
- Schema endpoint (`GET /api/schema`)
- Query execution endpoint (`POST /api/query`)
- HTTP status codes and error handling
- CORS headers
- JSON content type validation
- All 10 commands via REST API

## Running Tests

### Run all tests
```bash
uv run pytest tests/
```

### Run tests for a specific module
```bash
uv run pytest tests/test_loaders.py
uv run pytest tests/test_core_training.py
uv run pytest tests/test_core_io.py
uv run pytest tests/test_api_query_engine.py
uv run pytest tests/test_all_models.py            # 146 comprehensive model tests
uv run pytest tests/test_execute_query.py         # Shared query logic
uv run pytest tests/test_cli.py                   # CLI interface tests
uv run pytest tests/test_web_api.py               # REST API tests
```

### Run specific test class
```bash
uv run pytest tests/test_loaders.py::TestChEBILoading
uv run pytest tests/test_all_models.py::TestModelTraining
uv run pytest tests/test_cli.py::TestArgumentParsing
uv run pytest tests/test_web_api.py::TestQueryEndpoint
```

### Run specific test
```bash
uv run pytest tests/test_loaders.py::TestChEBILoading::test_chebi_loads_successfully
# Run all tests for a specific model type using -k (keyword):
uv run pytest tests/test_all_models.py -k "implicit_bpr"
uv run pytest tests/test_cli.py::TestArgumentParsing::test_compounds_command_parsing
```

### Run tests by category
```bash
# Core model & data tests only
uv run pytest tests/test_loaders.py tests/test_core_training.py tests/test_core_io.py -q

# Interface tests (CLI + Web API)
uv run pytest tests/test_cli.py tests/test_web_api.py tests/test_execute_query.py -q

# Everything
uv run pytest tests/ -q
```bash
uv run pytest tests/test_loaders.py::TestChEBILoading::test_chebi_loads_successfully
# Run all tests for a specific model type using -k (keyword):
uv run pytest tests/test_all_models.py -k "implicit_bpr"  # 29 tests for implicit_bpr
uv run pytest tests/test_all_models.py -k "svd"  # 29 tests for svd
```

### Verbose output
```bash
uv run pytest tests/ -v
```

### Show print statements
```bash
uv run pytest tests/ -s
```

## Test Fixtures

Common fixtures are defined in `conftest.py`:

- `chebi_data` - Loaded ChEBI database (session-scoped, loaded once)
- `exometabolite_data` - Loaded exometabolite dataset (session-scoped)
- `training_data` - Prepared training data (function-scoped)
- `trained_bpr_artifacts` - Trained BPR model ready for testing
- `trained_svd_artifacts` - Trained SVD model ready for testing
- `temp_model_dir` - Temporary directory with saved model

## Dependencies

All dependencies are managed via `pyproject.toml`:
- **Core**: numpy, pandas, torch, scikit-learn
- **Models**: implicit (for optimized BPR, LMF, WRMF implementations)
- **Data**: rapidfuzz, pyarrow
- **Testing**: pytest

When you run `uv run pytest`, it automatically:
- Creates a virtual environment (`.venv/`)
- Installs all dependencies including torch and implicit
- Runs the tests

No manual environment setup needed!

## Coverage

These tests provide comprehensive coverage across all WOM functionality:

- **loaders/**: Data loading pipeline from raw files to training-ready data (15 tests)
- **core/training**: Basic model training for BPR and SVD (15 tests)
- **core/io**: Model persistence and serialization (11 tests)
- **api/**: User-facing query interface with 8 query types (22 tests)
- **All Models**: Comprehensive validation of all 5 model types across both consumed and secreted data (146 tests)

## Development

When adding new functionality:

1. Add corresponding test cases to the relevant test file
2. Use the existing fixtures to avoid duplication
3. Follow the test class organization (group related tests)
4. Document test purpose in docstrings
