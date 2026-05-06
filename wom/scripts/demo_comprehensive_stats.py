"""Example showing comprehensive prediction statistics."""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from loaders.loaders import prepare_training_data
from core.training import train_model, compute_taxonomy_embeddings

print("=" * 70)
print("COMPREHENSIVE PREDICTION STATISTICS DEMO")
print("=" * 70)

# Train a quick model
print("\n1. Training model...")
training_data = prepare_training_data()
artifacts = train_model(
    training_data=training_data,
    model_type="bpr",
    embedding_dim=16,
    n_epochs=50,
    verbose=False,
)

# Compute taxonomy embeddings for background comparison
print("2. Computing taxonomy embeddings...")
genus_emb = compute_taxonomy_embeddings(artifacts, "genus")
family_emb = compute_taxonomy_embeddings(artifacts, "family")
artifacts.taxonomy_embeddings["genus"] = genus_emb
artifacts.taxonomy_embeddings["family"] = family_emb

# Get a sample isolate
isolate_id = list(artifacts.training_data.microbe_to_idx.keys())[0]
taxonomy = artifacts.training_data.microbe_taxonomy[isolate_id]

print(f"\n3. Comprehensive prediction for isolate: {isolate_id}")
print(f"   Taxonomy: genus={taxonomy.genus}, family={taxonomy.family}")

# Get full prediction with stats
result = artifacts.predict_with_stats(
    level="isolate",
    name=isolate_id,
    compute_distinctive=True,
    background_level="genus",
)

print("\n" + "=" * 70)
print("PREDICTION STATISTICS")
print("=" * 70)
print(f"\nBasic Statistics:")
print(f"  Mean score:     {result.mean_score:8.4f}")
print(f"  Median score:   {result.median_score:8.4f}")
print(f"  Std deviation:  {result.std_score:8.4f}")
print(f"  Std error:      {result.stderr_score:8.4f}" if result.stderr_score else "  Std error:      N/A")

print(f"\nCounts:")
print(f"  Total metabolites:        {result.n_metabolites}")
print(f"  Known positives (train):  {result.n_positive_training}")
print(f"  Background isolates:      {result.n_isolates_aggregated}")

print(f"\nBackground Comparison:")
print(f"  Background mean:  {result.background_mean:8.4f}" if result.background_mean else "  Background mean:  N/A")
print(f"  Background std:   {result.background_std:8.4f}" if result.background_std else "  Background std:   N/A")

print("\n" + "=" * 70)
print("TOP 20 METABOLITES (by Distinctive Score)")
print("=" * 70)
print(f"\n{'Rank':<6} {'Metabolite':<40} {'Raw':>8} {'Dist':>8} {'%ile':>8}")
print("-" * 70)

top20 = result.top_k(20, by="distinctive")
for i, name in enumerate(top20.metabolite_names):
    print(f"{i+1:<6} {name:<40} {top20.raw_scores[i]:8.3f} {top20.distinctive_scores[i]:8.3f} {top20.percentiles[i]:8.1f}")

print("\n" + "=" * 70)
print("COMPARISON: Raw vs Distinctive Scores")
print("=" * 70)

print("\nTop 10 by RAW score:")
top10_raw = result.top_k(10, by="raw")
for i, name in enumerate(top10_raw.metabolite_names):
    print(f"  {i+1:2}. {name[:50]:<50} {top10_raw.raw_scores[i]:.3f}")

print("\nTop 10 by DISTINCTIVE score:")
top10_dist = result.top_k(10, by="distinctive")
for i, name in enumerate(top10_dist.metabolite_names):
    print(f"  {i+1:2}. {name[:50]:<50} {top10_dist.distinctive_scores[i]:.3f}")

print("\n" + "=" * 70)
print("JSON EXPORT EXAMPLE")
print("=" * 70)

result_dict = top20.to_dict()
print(f"\nExported {len(result_dict['metabolites'])} metabolite predictions")
print(f"Statistics included: {list(result_dict['statistics'].keys())}")
print(f"\nFirst metabolite entry:")
print(json.dumps(result_dict['metabolites'][0], indent=2))

print("\n" + "=" * 70)
print("✓ Demo complete!")
print("=" * 70)
