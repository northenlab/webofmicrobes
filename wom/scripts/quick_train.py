#!/usr/bin/env python
"""Quick script to train a model and test --explain flag."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loaders.loaders import prepare_training_data
from core.training import train_model, compute_taxonomy_embeddings
from core.io import save_model_artifacts

# Train model
print("Training model...")
training_data = prepare_training_data()  # Default: both consumed and secreted
print(f"Loaded: {training_data.n_microbes} microbes, {training_data.n_metabolites} metabolites")

artifacts = train_model(
    training_data=training_data,
    model_type="bpr",
    embedding_dim=32,
    n_epochs=20,
    batch_size=64,
    verbose=True,
)

# Compute taxonomy embeddings
print("\nComputing taxonomy embeddings...")
for level in ["genus", "family", "order", "class", "kingdom"]:
    try:
        emb = compute_taxonomy_embeddings(artifacts, level)
        artifacts.taxonomy_embeddings[level] = emb
        print(f"  {level}: {len(emb)} taxa")
    except Exception as e:
        print(f"  {level}: skipped ({e})")

# Save
output_dir = Path("trained_models/latest")
output_dir.mkdir(parents=True, exist_ok=True)
save_model_artifacts(artifacts, output_dir)

print(f"\n✓ Model saved to {output_dir}")
print("\nNow try:")
print("  python scripts/wom.py compounds --microbes genus:Bosea --top 5 --explain")
