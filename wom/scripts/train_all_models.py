#!/usr/bin/env python
"""Train all model types with optimal hyperparameters.

Trains 12 models total:
- Ordinal BPR, BPR, Implicit BPR, LMF, WR-MF, SVD on consumed metabolites (decreased)
- Ordinal BPR, BPR, Implicit BPR, LMF, WR-MF, SVD on secreted metabolites (increased)

Uses optimal dimensions and hyperparameters for each model type:
- Ordinal BPR: 32 dimensions, 1000 epochs, 256 batch_size, 0.01 lr, hard_neg_ratio=0.5, lambda_ord=0
  (lambda_ord=0.1 for the submitted manuscript; that model is archived in
  trained_models/archive_submitted_lambda0.1/)
- SVD: 8 dimensions
- WR-MF: 128 dimensions, 50 iterations, 0.1 regularization
- BPR: 32 dimensions, 1000 epochs, 256 batch_size, 0.01 learning_rate
- Implicit BPR: 32 dimensions, 100 iterations, 0.01 learning_rate, 0.01 regularization
- LMF: 32 dimensions, 30 iterations, 1.0 learning_rate, 0.1 regularization
Usage:
    python scripts/train_all_models.py                                # all 12 models
    python scripts/train_all_models.py --models ordinal_bpr --seed 0  # only Ordinal-BPR
"""

import sys
import json
import random
import argparse
from pathlib import Path
import logging

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from loaders.loaders import prepare_training_data
from core.training import train_model, compute_taxonomy_embeddings
from core.io import save_model_artifacts
from core.types import AssertionType

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def train_all_models(model_types: list[str] | None = None, seed: int | None = None):
    """Train model types (default: all) for both consumed and secreted metabolites.

    If seed is given, random/numpy/torch are seeded before each fit.
    """
    
    # Model configurations: (model_type, embedding_dim, kwargs)
    model_configs = [
        ("ordinal_bpr", 32, {"n_epochs": 1000, "batch_size": 256, "learning_rate": 0.01, "hard_neg_ratio": 0.5, "lambda_ord": 0.0}),
        ("svd", 8, {}),
        ("bpr", 32, {"n_epochs": 1000, "batch_size": 256, "learning_rate": 0.01}),
        ("implicit_bpr", 32, {"iterations": 500, "learning_rate": 0.01, "regularization": 0.0}),
        ("lmf", 32, {"iterations": 30, "learning_rate": 1.0, "regularization": 0.1}),
        ("wrmf", 128, {"iterations": 50, "regularization": 0.1}),
    ]
    if model_types is not None:
        model_configs = [c for c in model_configs if c[0] in model_types]
    
    # Data types: consumed (decreased) and secreted (increased)
    data_configs = [
        ("consumed", AssertionType.CONSUMED),
        ("secreted", AssertionType.SECRETED),
    ]
    
    results = []
    
    for data_name, assertion_type in data_configs:
        logger.info("=" * 80)
        logger.info(f"TRAINING MODELS FOR: {data_name.upper()} metabolites")
        logger.info("=" * 80)
        
        # Load training data
        logger.info(f"\nLoading {data_name} training data...")
        training_data = prepare_training_data(
            preference_type=assertion_type,
            include_secreted=False,  # Train separately
        )
        
        for model_type, embedding_dim, model_kwargs in model_configs:
            logger.info(f"\n--- Training {model_type.upper()} ({embedding_dim}D) on {data_name} ---")
            
            try:
                if seed is not None:
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)

                # Train model
                artifacts = train_model(
                    training_data=training_data,
                    model_type=model_type,
                    embedding_dim=embedding_dim,
                    verbose=True,
                    **model_kwargs,
                )
                
                # Compute taxonomy embeddings
                logger.info("Computing taxonomy embeddings...")
                for level in ["genus", "family", "order", "class", "kingdom"]:
                    try:
                        emb = compute_taxonomy_embeddings(artifacts, level)
                        artifacts.taxonomy_embeddings[level] = emb
                    except Exception as e:
                        logger.warning(f"  Could not compute {level}: {e}")
                
                # Save
                output_dir = Path(f"trained_models/{model_type}_{data_name}")
                output_dir.mkdir(parents=True, exist_ok=True)
                save_model_artifacts(artifacts, output_dir)
                with open(output_dir / "training_config.json", "w") as f:
                    json.dump({"model_type": model_type, "embedding_dim": embedding_dim,
                               "assertion_type": data_name, "seed": seed, **model_kwargs,
                               "trained_by": "scripts/train_all_models.py"}, f, indent=2)
                
                logger.info(f"✓ Saved to {output_dir}")
                results.append({
                    "model": model_type,
                    "data": data_name,
                    "dim": embedding_dim,
                    "dir": str(output_dir),
                    "success": True,
                })
                
            except Exception as e:
                logger.error(f"✗ Failed to train {model_type} on {data_name}: {e}")
                results.append({
                    "model": model_type,
                    "data": data_name,
                    "dim": embedding_dim,
                    "success": False,
                    "error": str(e),
                })
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)
    
    for result in results:
        status = "✓" if result["success"] else "✗"
        model_info = f"{result['model'].upper()} ({result['dim']}D)"
        data_info = result['data']
        
        if result["success"]:
            logger.info(f"{status} {model_info:20s} on {data_info:10s} -> {result['dir']}")
        else:
            logger.info(f"{status} {model_info:20s} on {data_info:10s} -> FAILED: {result['error']}")
    
    # Print usage instructions
    logger.info("\n" + "=" * 80)
    logger.info("USAGE")
    logger.info("=" * 80)
    logger.info("\nQuery consumed metabolites:")
    logger.info("  python scripts/wom.py --model-dir trained_models/bpr_consumed compounds --microbes genus:Bosea")
    logger.info("  python scripts/wom.py --model-dir trained_models/implicit_bpr_consumed compounds --microbes genus:Bosea")
    logger.info("  python scripts/wom.py --model-dir trained_models/lmf_consumed compounds --microbes genus:Bosea")
    logger.info("  python scripts/wom.py --model-dir trained_models/wrmf_consumed compounds --microbes genus:Bosea")
    logger.info("  python scripts/wom.py --model-dir trained_models/svd_consumed compounds --microbes genus:Bosea")
    
    logger.info("\nQuery secreted metabolites:")
    logger.info("  python scripts/wom.py --model-dir trained_models/bpr_secreted compounds --microbes genus:Bosea")
    logger.info("  python scripts/wom.py --model-dir trained_models/implicit_bpr_secreted compounds --microbes genus:Bosea")
    logger.info("  python scripts/wom.py --model-dir trained_models/lmf_secreted compounds --microbes genus:Bosea")
    logger.info("  python scripts/wom.py --model-dir trained_models/wrmf_secreted compounds --microbes genus:Bosea")
    logger.info("  python scripts/wom.py --model-dir trained_models/svd_secreted compounds --microbes genus:Bosea")
    
    logger.info("\nSet default model for wom.py:")
    logger.info("  ln -sf ordinal_bpr_consumed trained_models/latest")
    
    # Count successes
    successes = sum(1 for r in results if r["success"])
    logger.info(f"\n✓ {successes}/{len(results)} models trained successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train WoM recommendation models")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model types to train (default: all), e.g. --models ordinal_bpr")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed random/numpy/torch before each fit (default: unseeded)")
    args = parser.parse_args()
    train_all_models(args.models, args.seed)
