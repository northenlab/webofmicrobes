# build.py
"""Web of Microbes - main entry point."""

import logging
import sys
from pathlib import Path

from model import (
    BPRModel,
    WRMFModel,
    SVDModel,
    train_model,
    save_model,
)
from config import CACHE_DIR
from gtdb import get_gtdb

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("wom")


# Model configurations to build
MODEL_CONFIGS = {
    "BPR": {
        "class": BPRModel,
        "kwargs": {"n_epochs": 1000, "batch_size": 256, "learning_rate": 0.01},
    },
    "WR-MF": {
        "class": WRMFModel,
        "kwargs": {"iterations": 50, "regularization": 0.1},
    },
    "SVD": {
        "class": SVDModel,
        "kwargs": {},
    },
}

PREFERENCE_TYPES = ["consumed", "secreted"]
DEFAULT_EMBEDDING_DIM = 32


def build_gtdb():
    """Pre-download and cache GTDB taxonomy."""
    logger.info("Building GTDB taxonomy cache...")
    gtdb = get_gtdb()
    gtdb.load(force_download=False)
    logger.info(f"  {len(gtdb.df)} genomes loaded")


def build_models(
    exo,
    model_names: list[str] = None,
    preference_types: list[str] = None,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    output_dir: Path = None,
) -> dict[str, dict[str, "ModelWrapper"]]:
    """Build and save all configured models.

    Args:
        exo: Exometabolite DataFrame
        model_names: Which models to build (default: all)
        preference_types: Which preference types (default: consumed, secreted)
        embedding_dim: Embedding dimension
        output_dir: Where to save models (default: CACHE_DIR)

    Returns:
        Nested dict: model_name -> preference_type -> ModelWrapper
    """
    if model_names is None:
        model_names = list(MODEL_CONFIGS.keys())
    if preference_types is None:
        preference_types = PREFERENCE_TYPES
    if output_dir is None:
        output_dir = CACHE_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_models = {}

    for model_name in model_names:
        config = MODEL_CONFIGS[model_name]
        all_models[model_name] = {}

        for pref_type in preference_types:
            logger.info(f"Training {model_name} for {pref_type} (dim={embedding_dim})...")

            wrapper = train_model(
                model_class=config["class"],
                df=exo,
                preference_type=pref_type,
                embedding_dim=embedding_dim,
                verbose=True,
                **config["kwargs"],
            )

            logger.info(
                f"  {model_name}/{pref_type}: "
                f"{wrapper.model.n_microbes} microbes, "
                f"{wrapper.model.n_metabolites} metabolites"
            )

            # Save model
            safe_name = model_name.lower().replace("-", "_")
            save_path = output_dir / f"{safe_name}_{pref_type}"
            save_model(wrapper, save_path)

            all_models[model_name][pref_type] = wrapper

    return all_models


def main():
    logger.info("Initializing Web of Microbes")

    import kg
    import pref
    import chebi_mapping

    # Step 0: Build GTDB
    logger.info("=" * 60)
    logger.info("Step 0: Building GTDB graph")
    logger.info("=" * 60)
    build_gtdb()

    # Step 2: Build ChEBI cache
    logger.info("=" * 60)
    logger.info("Step 2: Building ChEBI cache")
    logger.info("=" * 60)
    try:
        chebi = chebi_mapping.build_chebi(force=False)
        logger.info(
            f"ChEBI loaded: {len(chebi):,} name entries, "
            f"{chebi['compound_id'].nunique():,} unique compounds"
        )
        with_smiles = chebi["smiles"].notna().sum()
        with_inchi_key = chebi["inchi_key"].notna().sum()
        logger.info(f"  Entries with SMILES: {with_smiles:,}")
        logger.info(f"  Entries with InChIKey: {with_inchi_key:,}")
    except Exception as e:
        logger.error(f"Failed to build ChEBI: {e}")
        raise

    # Step 4: Build exometabolite data
    logger.info("=" * 60)
    logger.info("Step 4: Building exometabolite data")
    logger.info("=" * 60)
    try:
        exo = pref.build_exometabolite(force=False)
        logger.info(f"Exometabolite data loaded: {len(exo):,} observations")
    except Exception as e:
        logger.error(f"Failed to build exometabolite data: {e}")
        raise

    # Step 5: Build recommendation models
    logger.info("=" * 60)
    logger.info("Step 5: Building recommendation models")
    logger.info("=" * 60)
    try:
        all_models = build_models(exo)

        logger.info("Models built:")
        for model_name, pref_models in all_models.items():
            for pref_type, wrapper in pref_models.items():
                logger.info(
                    f"  {model_name}/{pref_type}: "
                    f"{wrapper.model.n_microbes} microbes, "
                    f"{wrapper.model.n_metabolites} metabolites, "
                    f"dim={wrapper.model.embedding_dim}"
                )
    except Exception as e:
        logger.error(f"Failed to build models: {e}")
        raise

    logger.info("=" * 60)
    logger.info("Initialization complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()