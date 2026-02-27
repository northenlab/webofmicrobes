# Web of Microbes - Metabolite Preference Predictions

Machine learning system for predicting metabolite-microbe interactions using collaborative filtering.

## Development Workflow

```bash
# 1. Build training data
python scripts/prepare_data.py

# 2. Train models (all 6 variants: BPR/WR-MF/SVD × consumed/secreted)
python scripts/train_all_models.py

# 3. Generate AI agent system prompt (dynamically from code)
python scripts/generate_system_prompt.py > wom_agent_system_prompt.md
```

The system prompt is auto-generated from the codebase to ensure it always matches the implementation.

## Quick Start

```bash
# 1. Train a model (default: consumed + secreted)
python scripts/train_model.py

# 2. Query predictions  
python scripts/wom.py compounds --microbes genus:Bosea --top 20

# 3. See how it works
python scripts/wom.py compounds --microbes genus:Bosea --explain
```

## What It Does

- **Learns embeddings** for microbes and metabolites
- **Predicts at any taxonomic level** (isolate, genus, family, etc.)
- **Supports consumed AND secreted (produced) metabolites**
- **Computes distinctive scores** showing what makes each microbe unique

**Algorithm**: Score = dot(microbe_embedding, metabolite_embedding)

## Training

```bash
# Train all 6 models at once
# - BPR, WR-MF, SVD on consumed metabolites (decreased)
# - BPR, WR-MF, SVD on secreted metabolites (increased)
python scripts/train_all_models.py
```

**Optimal dimensions** (built into train_all_models.py):
- SVD: 8 dimensions (fast, good baseline)
- BPR: 32 dimensions (balanced performance)
- WR-MF: 128 dimensions (best for sparse data)

**Model outputs**: `trained_models/{model}_{consumed|secreted}/`

## Queries

```bash
# Query consumed metabolites (decreased)
python scripts/wom.py --model-dir trained_models/bpr_consumed compounds --microbes genus:Bosea --top 20

# Query secreted metabolites (increased/produced)
python scripts/wom.py --model-dir trained_models/bpr_secreted compounds --microbes genus:Bosea --top 20

# Compare models
python scripts/wom.py --model-dir trained_models/wrmf_consumed compounds --microbes genus:Bosea
python scripts/wom.py --model-dir trained_models/svd_consumed compounds --microbes genus:Bosea

# Other query types (work with any model)
python scripts/wom.py --model-dir trained_models/bpr_consumed microbes --compounds "glucose,pyruvate" --level genus
python scripts/wom.py --model-dir trained_models/bpr_consumed selective --target genus:Pseudomonas --competitors "genus:Burkholderia"
python scripts/wom.py --model-dir trained_models/bpr_consumed fitness --metabolites "arginine,lysine" --level genus
python scripts/wom.py --model-dir trained_models/bpr_consumed niche --microbe genus:Pseudomonas --top 20
python scripts/wom.py --model-dir trained_models/bpr_consumed competition --microbes "genus:Pseudomonas,genus:Burkholderia" --metabolites "glucose,glutamate"
python scripts/wom.py --model-dir trained_models/bpr_consumed rank-substrates --microbes "genus:Bacillus,genus:Pseudomonas" --substrates "glucose,glutamate,alanine"

# Search & info (model-independent)
python scripts/wom.py search "glutamic acid"
python scripts/wom.py list --level genus
python scripts/wom.py --model-dir trained_models/bpr_consumed info
```

## REST API

```bash
python scripts/wom_api.py --port 5000

curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"command": "compounds", "microbes": "genus:Bosea", "top": 10}'
```

## Explain Mode

```bash
# Show how any prediction works
python scripts/wom.py compounds --microbes genus:Bosea --explain
python scripts/wom.py fitness --metabolites "glutamate,alanine" --explain
python scripts/wom.py competition --microbes "genus:Pseudomonas,genus:Burkholderia" --metabolites "glucose" --explain
```

Shows embedding source, isolates averaged, prediction method, score statistics.

See [ALGORITHM_FLOW.md](ALGORITHM_FLOW.md) for detailed mechanics.

## Query Flexibility

All prediction commands support:
- `--explain`: Show detailed computation steps
- `--use-distinctive`: Use distinctive scores (unique preferences) instead of raw scores
- `--agg`: Choose aggregation method (mean, median, sum, max)

```bash
# Use distinctive scores for competition
python scripts/wom.py --model-dir trained_models/bpr_consumed competition \
  --microbes "genus:Pseudomonas,genus:Burkholderia" \
  --metabolites "glucose,glutamate" \
  --use-distinctive

# Use median aggregation  
python scripts/wom.py --model-dir trained_models/bpr_consumed microbes \
  --compounds "glucose,glutamate" \
  --agg median

# Distinctive fitness with explanation
python scripts/wom.py --model-dir trained_models/bpr_consumed fitness \
  --metabolites "glutamate,alanine" \
  --use-distinctive --explain
```

See [FLEXIBILITY_UPDATE.md](FLEXIBILITY_UPDATE.md) and [COMMANDS.md](COMMANDS.md) for complete details.

## Architecture

```
core/       # ML (types, models, training, io)
loaders/    # Data pipeline (loaders, compound_mapper)
api/        # Queries (query_types, query_engine)
scripts/    # CLI & training (wom.py, wom_api.py, train_model.py)
```

## Files

- **core/types.py** - TrainingData, ModelArtifacts, PredictionResult
- **core/models.py** - BPRModel, WRMFModel, SVDModel
- **loaders/loaders.py** - prepare_training_data()
- **api/query_engine.py** - QueryEngine
- **scripts/wom.py** - CLI (8 commands)
- **scripts/train_model.py** - Training

## Testing

```bash
python scripts/test_data_loading.py
python scripts/test_model_training.py
python scripts/test_query_system.py
python scripts/test_explain_mode.py
```

## Notes

- **Training produces 6 models**: BPR, WR-MF, SVD × (consumed, secreted)
- **Consumed** = decreased concentration (8,207 interactions)
- **Secreted** = increased concentration / produced (751 interactions)
- **More secreted data coming soon** - currently mostly consumed data
- **Model selection**: BPR best overall, WR-MF for sparse data, SVD for speed

## Citation

*Paper in preparation*
