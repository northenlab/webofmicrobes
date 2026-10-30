"""Leave-one-isolate-out (LOIO) cross-validation for the Ordinal-BPR model.

Runs LOIO on every training isolate (default) or a random sample (--n).
Results are cached per-isolate and the loop is resumable: if the script is
interrupted, re-run with the same arguments and it will pick up where it left
off.  Use --force to blow away the cache and start from scratch.

For each held-out isolate:
  1. Remove it from the training corpus
  2. Retrain Ordinal-BPR from scratch with standard hyperparameters
  3. Predict at the best available taxonomic level (genus → family → order →
     class → kingdom) using the mean embedding of remaining training isolates
  4. Evaluate ROC-AUC against the isolate's consumed/neutral observations
  5. Bootstrap AUC standard error (500 resamplings of compound observations)

GCS and Embedding Diversity are computed from the full model (trained once at
startup) and used as X-axis variables in the output scatter figure.

Outputs:
  - CSV:    loio_auc_results.csv        (one row per isolate, grows as cache)
  - Figure: figure_loio_validation.{pdf,png}   (2-panel scatter: AUC vs GCS
            and AUC vs Embedding Diversity)

Usage:
    python loio_validation.py                      # all isolates, resume cache
    python loio_validation.py --force              # start over
    python loio_validation.py --n 20 --seed 7     # random 20-isolate sample
"""

import sys
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from loaders.loaders import load_exometabolite, prepare_training_data
from core.training import train_model

logger = logging.getLogger(__name__)

OUTPUT_DIR = (
    Path(__file__).parent.parent
    / 'manuscript_figures_nat_microbiology_first_submission'
)
CACHE_FILE = OUTPUT_DIR / 'loio_auc_results.csv'
STATS_CSV   = OUTPUT_DIR / 'stats_gcs_emb_diversity.csv'

TRAIN_KWARGS = dict(
    model_type='ordinal_bpr',
    embedding_dim=32,
    n_epochs=1000,
    batch_size=256,
    learning_rate=0.01,
    hard_neg_ratio=0.5,
    lambda_ord=0.0,  # 0.1 in the submitted analysis
)

TAX_LEVELS = ['genus', 'family', 'order', 'class', 'kingdom']
CONSUMED_THRESHOLD = -2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bootstrap_auc(true_labels, scores, n_boot=500, seed=42):
    """Bootstrap SE for ROC-AUC by resampling compound observations."""
    rng = np.random.default_rng(seed)
    true_labels = np.asarray(true_labels)
    scores = np.asarray(scores)
    n = len(true_labels)
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        tl, sc = true_labels[idx], scores[idx]
        if tl.sum() == 0 or tl.sum() == n:
            continue
        try:
            boot_aucs.append(roc_auc_score(tl, sc))
        except Exception:
            pass
    return float(np.std(boot_aucs)) if len(boot_aucs) > 1 else None


def compute_predictions_all_levels(model, training_data):
    """Mean BPR scores aggregated at every taxonomic level.

    Returns dict: level -> DataFrame[level, metabolite, mean_score, n_isolates].
    """
    idx_to_microbe   = training_data.idx_to_microbe
    idx_to_metabolite = training_data.idx_to_metabolite

    records = []
    for microbe_idx in range(training_data.n_microbes):
        microbe_id = idx_to_microbe[microbe_idx]
        scores     = model.predict_scores(microbe_idx)
        taxonomy   = training_data.microbe_taxonomy.get(microbe_id)
        if taxonomy is None:
            continue
        tax_vals = {
            'genus':   taxonomy.genus,
            'family':  taxonomy.family,
            'order':   taxonomy.order,
            'class':   taxonomy.class_,
            'kingdom': taxonomy.kingdom,
        }
        for met_idx, score in enumerate(scores):
            row = {'metabolite': idx_to_metabolite[met_idx], 'score': float(score)}
            row.update(tax_vals)
            records.append(row)

    df = pd.DataFrame(records)
    result = {}
    for level in TAX_LEVELS:
        if level not in df.columns:
            continue
        agg = (
            df.dropna(subset=[level])
            .groupby([level, 'metabolite'])['score']
            .agg(mean_score='mean', n_isolates='count')
            .reset_index()
        )
        result[level] = agg
    return result


def compute_embedding_diversity_all_levels(model, training_data):
    """Mean pairwise cosine distance between BPR embeddings per taxon.

    Returns dict: (taxon_name, level) -> (mean_div, se).
    se is None when K < 3.
    """
    E_all  = model.microbe_embeddings
    norms  = np.linalg.norm(E_all, axis=1, keepdims=True).clip(1e-12)
    E_norm = E_all / norms

    # Group isolate indices by (taxon, level)
    groups = {}   # (taxon, level) -> [microbe_idx, ...]
    for org_id, tax in training_data.microbe_taxonomy.items():
        if org_id not in training_data.microbe_to_idx:
            continue
        idx = training_data.microbe_to_idx[org_id]
        level_vals = {
            'genus': tax.genus, 'family': tax.family, 'order': tax.order,
            'class': tax.class_, 'kingdom': tax.kingdom,
        }
        for level, taxon in level_vals.items():
            if taxon is None:
                continue
            key = (taxon, level)
            groups.setdefault(key, []).append(idx)

    result = {}
    rng = np.random.default_rng(42)
    for (taxon, level), indices in groups.items():
        K = len(indices)
        if K < 2:
            continue
        E = E_norm[np.array(indices)]
        sims = E @ E.T
        mask = np.triu(np.ones((K, K), dtype=bool), k=1)
        mean_div = 1.0 - float(sims[mask].mean())
        if K < 3:
            result[(taxon, level)] = (mean_div, None)
            continue
        boot_vals = []
        for _ in range(500):
            bi = rng.choice(K, size=K, replace=True)
            Eb = E[bi]
            sb = Eb @ Eb.T
            boot_vals.append(1.0 - float(sb[mask].mean()))
        result[(taxon, level)] = (mean_div, float(np.std(boot_vals)))

    return result


def load_gcs_lookup(output_dir: Path, full_df: pd.DataFrame,
                    needed_keys: set = None):
    """GCS per (taxon, level): precomputed CSV for genus, on-the-fly otherwise.

    Args:
        needed_keys: if provided, only compute GCS for these (taxon, level) pairs.
                     Pass the set of pairs that actually appear in the results so
                     we don't bootstrap every possible taxon at every level.

    Returns dict: (taxon, level) -> (gcs, gcs_se).
    """
    from scipy.stats import spearmanr as _sr

    lookup = {}

    # --- genus from precomputed CSV ---
    stats_csv = output_dir / 'stats_gcs_emb_diversity.csv'
    if stats_csv.exists():
        gcs_df = pd.read_csv(stats_csv)
        for _, row in gcs_df.iterrows():
            ci_half = (row['ci_high'] - row['ci_low']) / 2.0
            se = ci_half / 1.96 if ci_half > 0 else None
            lookup[(row['genus'], 'genus')] = (float(row['gcs']), se)

    # --- other levels from exometabolite data, only for needed taxa ---
    def _compute(taxon, level, df, min_isolates=3, min_shared=5):
        col = 'class' if level == 'class' else level
        sub = df[df[col] == taxon] if col in df.columns else pd.DataFrame()
        if sub.empty:
            return None, None
        wide = sub.pivot_table(
            index='organism_id', columns='representative_compound',
            values='assertion_value'
        )
        if len(wide) < min_isolates:
            return None, None
        orgs = wide.index.tolist()
        rhos = []
        for i in range(len(orgs)):
            for j in range(i + 1, len(orgs)):
                shared = wide.iloc[i].notna() & wide.iloc[j].notna()
                if shared.sum() < min_shared:
                    continue
                rho, _ = _sr(wide.iloc[i][shared], wide.iloc[j][shared])
                rhos.append(rho)
        if not rhos:
            return None, None
        mean_rho = float(np.mean(rhos))
        return mean_rho, None

    # Only compute for keys we actually need and don't already have
    keys_to_compute = needed_keys - lookup.keys() if needed_keys is not None else set()
    if needed_keys is None:
        # Fallback: compute for all taxa at non-genus levels (original behaviour)
        for level in ['family', 'order', 'class', 'kingdom']:
            col = 'class' if level == 'class' else level
            if col not in full_df.columns:
                continue
            for taxon in full_df[col].dropna().unique():
                keys_to_compute.add((taxon, level))

    logger.info(f"  Computing GCS for {len(keys_to_compute)} (taxon, level) pairs on-the-fly")
    for taxon, level in sorted(keys_to_compute):
        if (taxon, level) not in lookup:
            lookup[(taxon, level)] = _compute(taxon, level, full_df)

    return lookup


# ---------------------------------------------------------------------------
# LOIO loop with caching
# ---------------------------------------------------------------------------

def run_loio(full_df, all_isolates, cache_path: Path):
    """Run LOIO for any isolate not already in the cache.

    Results are appended to cache_path row-by-row so the loop is resumable.
    Returns the complete results DataFrame (cached + newly computed).
    """
    cached_ids = set()
    if cache_path.exists():
        try:
            cached_df = pd.read_csv(cache_path)
            cached_ids = set(cached_df['isolate'].astype(str).tolist())
            logger.info(f"  Resuming: {len(cached_ids)} isolates already cached")
        except Exception:
            logger.warning("  Cache unreadable — starting fresh")

    todo = [iso for iso in all_isolates if str(iso) not in cached_ids]
    logger.info(f"  {len(todo)} isolates to process")

    for i, isolate_id in enumerate(todo):
        isolate_id = str(isolate_id)
        logger.info(f"\n[{len(cached_ids)+i+1}/{len(all_isolates)}] Holding out: {isolate_id}")

        iso_rows = full_df[full_df['organism_id'] == isolate_id].copy()
        if iso_rows.empty:
            logger.warning(f"  Not found — skipping")
            continue

        def _get(col):
            val = iso_rows[col].iloc[0] if col in iso_rows.columns else None
            return None if (val is not None and pd.isna(val)) else val

        genus   = _get('genus')
        family  = _get('family')
        order   = _get('order')
        class_  = _get('class')
        kingdom = _get('kingdom')
        tax_map = {
            'genus': genus, 'family': family, 'order': order,
            'class': class_, 'kingdom': kingdom,
        }

        loio_df       = full_df[full_df['organism_id'] != isolate_id]
        training_data = prepare_training_data(loio_df)
        artifacts     = train_model(training_data, verbose=True, **TRAIN_KWARGS)
        predictions   = compute_predictions_all_levels(artifacts.model, training_data)

        pred_level, pred_df = None, None
        for level in TAX_LEVELS:
            taxon = tax_map.get(level)
            if taxon is None:
                continue
            if level not in predictions:
                continue
            rows = predictions[level][predictions[level][level] == taxon]
            if rows.empty:
                continue
            logger.info(f"  {level}-level: '{taxon}' "
                        f"({int(rows['n_isolates'].iloc[0])} training isolates)")
            pred_level, pred_df = level, rows
            break

        if pred_df is None:
            logger.warning(f"  No predictions at any level — skipping")
            continue

        iso_cn = iso_rows[
            iso_rows['assertion_value'].notna() &
            (
                (iso_rows['assertion_value'] <= CONSUMED_THRESHOLD) |
                ((iso_rows['assertion_value'] > CONSUMED_THRESHOLD) &
                 (iso_rows['assertion_value'] < -CONSUMED_THRESHOLD))
            )
        ]
        if iso_cn.empty:
            logger.warning(f"  No consumed/neutral observations — skipping")
            continue

        merged = pd.merge(
            iso_cn[['representative_compound', 'assertion_value']],
            pred_df[['metabolite', 'mean_score']],
            left_on='representative_compound', right_on='metabolite',
            how='inner',
        )
        if merged.empty:
            logger.warning(f"  No compound overlap — skipping")
            continue

        true_consumed = (merged['assertion_value'] <= CONSUMED_THRESHOLD).astype(int)
        n_consumed = int(true_consumed.sum())
        n_total    = len(merged)

        if n_consumed == 0 or n_consumed == n_total:
            logger.warning(f"  Degenerate labels ({n_consumed}/{n_total}) — no AUC")
            auc, auc_se = float('nan'), None
        else:
            try:
                auc    = float(roc_auc_score(true_consumed, merged['mean_score'].values))
                auc_se = _bootstrap_auc(true_consumed.values, merged['mean_score'].values)
            except Exception as e:
                logger.warning(f"  AUC error: {e}")
                auc, auc_se = float('nan'), None

        _se_str = f"{auc_se:.3f}" if auc_se is not None else "n/a"
        logger.info(f"  AUC={auc:.3f}  SE={_se_str}  "
                    f"n_consumed={n_consumed}/{n_total}  level={pred_level}")

        row = {
            'isolate': isolate_id, 'genus': genus, 'family': family,
            'order': order, 'class': class_, 'kingdom': kingdom,
            'tax_level': pred_level, 'auc': auc, 'auc_se': auc_se,
            'n_consumed': n_consumed, 'n_total': n_total,
        }
        # Append to cache immediately so progress survives interruption
        pd.DataFrame([row]).to_csv(
            cache_path, mode='a',
            header=not cache_path.exists(),
            index=False,
        )

    return pd.read_csv(cache_path)


# ---------------------------------------------------------------------------
# Figure: 2-panel scatter (AUC vs GCS  |  AUC vs Embedding Diversity)
# ---------------------------------------------------------------------------

def make_scatter_figure(results_df, gcs_lookup, emb_div_lookup, output_dir: Path):
    """2-panel scatter styled like figure 8 scatter."""
    df = results_df[results_df['auc'].notna()].copy()
    if df.empty:
        logger.warning("No valid AUC results — skipping figure")
        return

    # Attach GCS and Emb.Div. per isolate (looked up by its prediction taxon+level)
    def _lookup(row, d):
        key = (row['taxon'], row['tax_level'])
        val = d.get(key)
        if val is None:
            return None, None
        return val

    df['taxon'] = df.apply(
        lambda r: r.get(r['tax_level']) if r['tax_level'] in r.index else None,
        axis=1,
    )
    # safer extraction: pick the column named by tax_level for each row
    taxon_vals = []
    for _, r in df.iterrows():
        col = r['tax_level']
        taxon_vals.append(r[col] if col in df.columns else None)
    df['taxon'] = taxon_vals

    gcs_vals, gcs_se_vals, ed_vals, ed_se_vals = [], [], [], []
    for _, r in df.iterrows():
        key = (r['taxon'], r['tax_level'])
        g = gcs_lookup.get(key, (None, None))
        e = emb_div_lookup.get(key, (None, None))
        gcs_vals.append(g[0]);    gcs_se_vals.append(g[1])
        ed_vals.append(e[0]);     ed_se_vals.append(e[1])

    df['gcs']      = gcs_vals
    df['gcs_se']   = gcs_se_vals
    df['emb_div']  = ed_vals
    df['emb_div_se'] = ed_se_vals

    level_colors = {'genus': '#2196F3', 'family': '#FF9800',
                    'order': '#9C27B0', 'class': '#4CAF50', 'kingdom': '#F44336'}

    def _panel(ax, x_col, x_se_col, xlabel):
        from scipy.stats import linregress as _linreg
        sub = df.dropna(subset=[x_col, 'auc']).copy()
        if sub.empty:
            ax.set_visible(False)
            return

        for _, row in sub.iterrows():
            col  = level_colors.get(row['tax_level'], '#888888')
            xerr = row[x_se_col] if pd.notna(row.get(x_se_col)) else 0
            yerr = row['auc_se']  if pd.notna(row.get('auc_se'))  else 0
            ax.errorbar(
                row[x_col], row['auc'],
                xerr=[[xerr], [xerr]], yerr=[[yerr], [yerr]],
                fmt='o', color=col, markersize=5, capsize=3,
                elinewidth=1.0, ecolor=col, alpha=0.5, zorder=3,
            )

        # Linear trendline + 95% CI via bootstrap (1000 resamplings of isolates)
        x_arr = sub[x_col].values
        y_arr = sub['auc'].values
        slope, intercept, *_ = _linreg(x_arr, y_arr)
        x_fit = np.linspace(x_arr.min(), x_arr.max(), 200)
        y_fit = slope * x_fit + intercept

        rng_boot = np.random.default_rng(42)
        n_sub = len(sub)
        boot_lines = []
        for _ in range(1000):
            bi = rng_boot.choice(n_sub, size=n_sub, replace=True)
            xb, yb = x_arr[bi], y_arr[bi]
            if xb.std() == 0:
                continue
            s, i, *_ = _linreg(xb, yb)
            boot_lines.append(s * x_fit + i)
        if boot_lines:
            boot_arr = np.array(boot_lines)
            ci_lo = np.percentile(boot_arr, 2.5, axis=0)
            ci_hi = np.percentile(boot_arr, 97.5, axis=0)
            ax.fill_between(x_fit, ci_lo, ci_hi, color='#333333', alpha=0.15, zorder=1)
        ax.plot(x_fit, y_fit, color='#333333', lw=1.5, alpha=1.0, zorder=4)

        # Spearman on taxon-averaged values (avoid pseudoreplication)
        avg = sub.groupby('taxon', as_index=False)[['auc', x_col]].mean(numeric_only=True)
        if len(avg) >= 3:
            rho, p = spearmanr(avg[x_col], avg['auc'])
            p_str = 'p<0.001' if p < 0.001 else f'p={p:.2f}'
            n_str = f'n={len(avg)} taxa'
        else:
            rho, p_str, n_str = float('nan'), '', ''
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('LOIO AUC', fontsize=12)
        ax.set_title(f'Spearman ρ={rho:.2f}, {p_str} ({n_str})', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    _panel(ax_a, 'gcs', 'gcs_se',
           'GCS (mean within-taxon pairwise Spearman ρ)')
    ax_a.set_title('a   AUC vs. GCS\n' + ax_a.get_title(), fontsize=12)

    _panel(ax_b, 'emb_div', 'emb_div_se',
           'Embedding Diversity (mean pairwise cosine distance)')
    ax_b.set_title('b   AUC vs. Embedding Diversity\n' + ax_b.get_title(), fontsize=12)

    # Legend
    present_levels = df['tax_level'].dropna().unique()
    for lv in TAX_LEVELS:
        if lv in present_levels:
            ax_a.scatter([], [], color=level_colors.get(lv, '#888'), s=40,
                         label=f'{lv}-level', alpha=0.8)
    ax_a.legend(title='Prediction level', fontsize=10, title_fontsize=10, frameon=False)

    valid_aucs = df['auc'].dropna()
    n_above = (valid_aucs > 0.8).sum()
    fig.suptitle(
        f'Leave-one-isolate-out validation  '
        f'(n={len(valid_aucs)} isolates)  |  '
        f'Mean AUC={valid_aucs.mean():.2f}  |  '
        f'{n_above}/{len(valid_aucs)} > 0.8',
        fontsize=13, y=1.01,
    )

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(output_dir / f'figure_loio_validation.{ext}',
                    dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved figure_loio_validation.{pdf,png}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)s  %(message)s',
        datefmt='%H:%M:%S',
    )

    parser = argparse.ArgumentParser(
        description='LOIO cross-validation for Ordinal-BPR — all training isolates'
    )
    parser.add_argument(
        '--n', type=int, default=None,
        help='Random subsample size (default: all isolates)',
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--force', action='store_true', default=False,
        help='Delete the cache and recompute everything from scratch',
    )
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / 'loio_auc_results.csv'

    if args.force and cache_path.exists():
        cache_path.unlink()
        logger.info("Cache cleared (--force)")

    logger.info("Loading exometabolite data...")
    full_df = load_exometabolite()
    all_isolates = full_df['organism_id'].unique()
    logger.info(f"  {len(all_isolates)} training isolates available")

    if args.n is not None:
        rng = np.random.default_rng(args.seed)
        all_isolates = rng.choice(all_isolates, size=min(args.n, len(all_isolates)),
                                  replace=False)
        logger.info(f"  Subsampled to {len(all_isolates)} isolates (seed={args.seed})")

    # --- LOIO loop (with resume) ---
    results_df = run_loio(full_df, all_isolates, cache_path)

    valid = results_df['auc'].notna()
    valid_aucs = results_df.loc[valid, 'auc']
    logger.info(
        f"\nLOIO Summary:\n"
        f"  Isolates evaluated: {valid.sum()}\n"
        f"  Mean AUC:   {valid_aucs.mean():.3f}\n"
        f"  Median AUC: {valid_aucs.median():.3f}\n"
        f"  Min / Max:  {valid_aucs.min():.3f} / {valid_aucs.max():.3f}\n"
        f"  AUC > 0.8:  {(valid_aucs > 0.8).sum()}/{valid.sum()}"
    )

    # --- Taxon-level GCS and Embedding Diversity for scatter figure ---
    # Only compute GCS for the (taxon, level) pairs that appear in the results
    needed_keys = set()
    for _, row in results_df.iterrows():
        level = row.get('tax_level')
        taxon = row.get(level) if level and level in results_df.columns else None
        if taxon and pd.notna(taxon):
            needed_keys.add((taxon, level))

    logger.info(f"\nComputing GCS lookup for {len(needed_keys)} needed (taxon, level) pairs...")
    gcs_lookup = load_gcs_lookup(args.output_dir, full_df, needed_keys=needed_keys)
    logger.info(f"  {len(gcs_lookup)} (taxon, level) GCS entries")

    logger.info("Training full model for Embedding Diversity calculation...")
    full_training_data = prepare_training_data(full_df)
    full_artifacts = train_model(full_training_data, verbose=True, **TRAIN_KWARGS)
    emb_div_lookup = compute_embedding_diversity_all_levels(
        full_artifacts.model, full_training_data
    )
    logger.info(f"  {len(emb_div_lookup)} (taxon, level) Embedding Diversity entries")

    # --- Figure ---
    make_scatter_figure(results_df, gcs_lookup, emb_div_lookup, args.output_dir)


if __name__ == '__main__':
    main()
