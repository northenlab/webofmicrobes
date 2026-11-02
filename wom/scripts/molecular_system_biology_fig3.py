"""Molecular Systems Biology Figure 3 - taxonomic hold-out cross-validation.

Reworked from wom/scripts/figure3_holdout_cv/. Two messages, two panels:

  (a) QUALITY: on the compounds a naive taxonomic baseline can actually score,
      does the learned Ordinal-BPR model rank consumed-vs-neutral better?
      Side-by-side violins (median + 25/75 quartiles) at four hold-out levels of
      increasing phylogenetic distance. Paired Wilcoxon annotated per level.
  (c) COVERAGE: BPR assigns a score to every metabolite; the naive baseline can
      only score compounds some relative was tested on. Stacked to the full
      panel of 311 metabolites, highlighting how many only BPR can score.

Key fix vs. the previous figure: the naive baseline used to score 0 for compounds
no relative tested and KEEP those in the AUC, which (a) contradicted the Methods
text ("where both methods scored a compound") and (b) double-counted the coverage
gap into the quality gap. Here the head-to-head AUC (panel a) is computed for BOTH
methods on the baseline-scoreable set only; coverage is reported separately (panel c).

Design faithful to the preprint Methods: for each level in {isolate, genus, family,
order}, every group at that rank is withheld in turn, a fresh Ordinal-BPR (dim 32,
400 epochs, SEED=0) is retrained on the remainder, and each held-out isolate is
predicted from the MEAN embedding of its nearest remaining relatives one rank higher:
  isolate held out -> other isolates in the same genus
  genus   held out -> other genera in the same family
  family  held out -> other families in the same order
  order   held out -> other orders in the same class

Slow compute (~30-40 min, retrains one model per held-out group) is cached to
molecular_system_biology_fig3_results.csv; rendering reads that cache and is fast.

Usage:
  python molecular_system_biology_fig3.py             # render from cache (compute if missing)
  python molecular_system_biology_fig3.py --recompute # force the full CV
  python molecular_system_biology_fig3.py --smoke     # ~1-2 min pipeline check to /tmp
"""
import sys
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon

HERE = Path(__file__).parent
RESULTS_CSV = HERE / 'molecular_system_biology_fig3_results.csv'
FIG_STEM = HERE / 'molecular_system_biology_fig3'

SEED = 0
N_EPOCHS = 400
LAMBDA_ORD = 0.0  # 0.1 (OrdinalBPRModel.fit default) for the submitted Fig. 3
EMB_DIM = 32
LEVELS = ['isolate', 'genus', 'family', 'order']

# palette (teal/orange: strong hue + lightness separation, CVD-safe; identity also
# carried by position + legend, never color alone)
TEAL, ORANGE, GREY, INK = '#237c7c', '#d98a34', '#9aa0a8', '#222222'


def auc(pos_scores, neg_scores):
    """P(score(consumed) > score(neutral)); ties count 0.5. NaN if either side empty."""
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return np.nan
    p = np.asarray(pos_scores)[:, None]
    n = np.asarray(neg_scores)[None, :]
    return float((np.sum(p > n) + 0.5 * np.sum(p == n)) / (len(pos_scores) * len(neg_scores)))


def average_precision(pos_scores, neg_scores):
    """PR-AUC (average precision) of consumed (positive) vs neutral; NaN if either side empty.
    A random ranking scores the prevalence, len(pos) / (len(pos) + len(neg))."""
    from sklearn.metrics import average_precision_score
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return np.nan
    y = np.r_[np.ones(len(pos_scores)), np.zeros(len(neg_scores))]
    return float(average_precision_score(y, np.r_[np.asarray(pos_scores, float), np.asarray(neg_scores, float)]))


# --------------------------------------------------------------------------- compute
def compute(out_csv=RESULTS_CSV, n_epochs=N_EPOCHS, isolate_limit=None, levels=LEVELS,
            lambda_ord=LAMBDA_ORD):
    """Run the hierarchical hold-out CV, computing BPR and naive AUCs in one pass
    (inherently paired, identical relative sets) and coverage. Writes out_csv.

    Resumable: each finished group's rows go to <out_csv stem>.partial.csv and its
    key to <out_csv stem>.done_groups.tsv; a rerun skips finished groups. Every
    model is seeded with SEED, so resumed and uninterrupted runs give the same rows."""
    sys.path.insert(0, str(HERE.parent))  # wom/ on path for `loaders` and `core`
    from loaders.loaders import prepare_training_data
    from core.models import OrdinalBPRModel
    import torch

    td = prepare_training_data(include_secreted=False)
    N, M = td.n_microbes, td.n_metabolites
    idx_of = td.microbe_to_idx
    ranks = ['genus', 'family', 'order', 'class']
    def taxon(m, L):  # missing ranks load as NaN or None; treat both as absent
        value = td.microbe_taxonomy[m].get_level(L)
        return value if isinstance(value, str) and value.strip() else None

    lvl = {m: {L: taxon(m, L) for L in ranks} for m in idx_of}
    members = {L: defaultdict(list) for L in ranks}
    for m in idx_of:
        for L in ranks:
            if lvl[m][L]:
                members[L][lvl[m][L]].append(m)

    def train(positives_sub, known_neg_sub):
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        model = OrdinalBPRModel(embedding_dim=EMB_DIM)
        model.fit(positives=positives_sub, n_microbes=N, n_metabolites=M,
                  known_negatives=known_neg_sub, n_epochs=n_epochs, lambda_ord=lambda_ord,
                  verbose=False)
        return model.microbe_embeddings, model.metabolite_embeddings

    out_csv = Path(out_csv)
    partial_csv = out_csv.with_name(out_csv.stem + '.partial.csv')
    done_log = out_csv.with_name(out_csv.stem + '.done_groups.tsv')
    done = set()
    if done_log.exists():
        done = {tuple(line.rstrip('\n').split('\t', 1)) for line in open(done_log) if line.strip()}
    rows = []
    if partial_csv.exists():
        cached = pd.read_csv(partial_csv, dtype={'group': str, 'isolate': str})
        # keep only rows of groups recorded as finished
        keep = [(lv, g) in done for lv, g in zip(cached.level, cached.group)]
        rows = cached[keep].to_dict('records')
        pd.DataFrame(rows, columns=cached.columns).to_csv(partial_csv, index=False)
    if done:
        print(f"resuming: {len(done)} groups cached, {len(rows)} rows", flush=True)

    def evaluate(level, group_key, held_ids, parent_level, parent_key,
                 exclude_child_level, exclude_child_key):
        # relatives = members of parent_key at parent_level, minus the held subtree
        # (same selection ladder as figure3_holdout_cv, for both methods)
        relatives = [m for m in members[parent_level][parent_key]
                     if lvl[m][exclude_child_level] != exclude_child_key]
        held_set = set(held_ids)
        relatives = [m for m in relatives if m not in held_set]
        if not relatives:
            return 0

        # retrain BPR on everything except the held-out group
        positives_sub = {idx_of[m]: td.positives[idx_of[m]] for m in idx_of
                         if m not in held_set and idx_of[m] in td.positives and td.positives[idx_of[m]]}
        known_sub = {idx_of[m]: td.known_negatives.get(idx_of[m], set())
                     for m in idx_of if m not in held_set}
        mic_emb, met_emb = train(positives_sub, known_sub)
        rel_idx = [idx_of[m] for m in relatives]
        pred_emb = mic_emb[rel_idx].mean(axis=0)
        bpr_scores = pred_emb @ met_emb.T  # length M, defined for every metabolite

        # naive baseline: consumption prevalence among the same relatives
        consumed_ct = defaultdict(int)
        tested_ct = defaultdict(int)
        for x in relatives:
            xi = idx_of[x]
            for met in td.positives.get(xi, set()):
                consumed_ct[met] += 1
                tested_ct[met] += 1
            for met in td.known_negatives.get(xi, set()):
                tested_ct[met] += 1
        scoreable = {met for met in range(M) if tested_ct.get(met, 0) > 0}

        def base_score(met):  # only valid on `scoreable`
            return consumed_ct[met] / tested_ct[met]

        n_eval = 0
        for m in held_ids:
            i = idx_of[m]
            pos = list(td.positives.get(i, set()))
            neg = list(td.known_negatives.get(i, set()))
            if not pos or not neg:
                continue
            pos_s = [c for c in pos if c in scoreable]
            neg_s = [c for c in neg if c in scoreable]
            both_s = bool(pos_s) and bool(neg_s)
            rows.append(dict(
                level=level, group=group_key, isolate=m,
                genus=lvl[m]['genus'], family=lvl[m]['family'], order=lvl[m]['order'],
                n_relatives=len(relatives),
                n_consumed=len(pos), n_neutral=len(neg),
                n_consumed_scoreable=len(pos_s), n_neutral_scoreable=len(neg_s),
                n_scoreable=len(scoreable), n_metabolites=M, coverage=len(scoreable) / M,
                # reference: BPR over ALL compounds (should match old auc_raw)
                auc_bpr_all=auc(bpr_scores[pos], bpr_scores[neg]),
                # head-to-head QUALITY: both methods on the scoreable set only
                auc_bpr_scoreable=(auc(bpr_scores[pos_s], bpr_scores[neg_s]) if both_s else np.nan),
                auc_baseline_scoreable=(auc([base_score(c) for c in pos_s],
                                            [base_score(c) for c in neg_s]) if both_s else np.nan),
                # PR-AUC (Reviewer 1); a random ranking scores the prevalence
                prevalence=len(pos) / (len(pos) + len(neg)),
                prevalence_scoreable=(len(pos_s) / (len(pos_s) + len(neg_s)) if both_s else np.nan),
                ap_bpr_all=average_precision(bpr_scores[pos], bpr_scores[neg]),
                ap_bpr_scoreable=(average_precision(bpr_scores[pos_s], bpr_scores[neg_s]) if both_s else np.nan),
                ap_baseline_scoreable=(average_precision([base_score(c) for c in pos_s],
                                                         [base_score(c) for c in neg_s]) if both_s else np.nan),
            ))
            n_eval += 1
        return n_eval

    def run_group(level, group_key, *args):
        key = (level, str(group_key))
        if key in done:
            return
        start = len(rows)
        evaluate(level, group_key, *args)
        if len(rows) > start:
            pd.DataFrame(rows[start:]).to_csv(partial_csv, mode='a', index=False,
                                              header=not partial_csv.exists())
        with open(done_log, 'a') as f:
            f.write(f"{level}\t{group_key}\n")
        done.add(key)

    t0 = time.time()

    if 'isolate' in levels:
        todo = [m for m in idx_of if lvl[m]['genus'] and len(members['genus'][lvl[m]['genus']]) >= 2]
        if isolate_limit:
            todo = todo[:isolate_limit]
        for k, m in enumerate(todo):
            run_group('isolate', m, [m], 'genus', lvl[m]['genus'], 'genus', '__none__')
            if (k + 1) % 25 == 0:
                print(f"isolate {k+1}/{len(todo)} ({time.time()-t0:.0f}s) rows={len(rows)}", flush=True)
        print(f"ISOLATE done ({time.time()-t0:.0f}s) rows={len(rows)}", flush=True)

    if 'genus' in levels:
        for g in [x for x in members['genus'] if x]:
            held = members['genus'][g]
            fam = lvl[held[0]]['family']
            if fam:
                run_group('genus', g, held, 'family', fam, 'genus', g)
        print(f"GENUS done ({time.time()-t0:.0f}s) rows={len(rows)}", flush=True)

    if 'family' in levels:
        for f in [x for x in members['family'] if x]:
            held = members['family'][f]
            od = lvl[held[0]]['order']
            if od:
                run_group('family', f, held, 'order', od, 'family', f)
        print(f"FAMILY done ({time.time()-t0:.0f}s) rows={len(rows)}", flush=True)

    if 'order' in levels:
        for o in [x for x in members['order'] if x]:
            held = members['order'][o]
            cl = lvl[held[0]]['class']
            if cl:
                run_group('order', o, held, 'class', cl, 'order', o)
        print(f"ORDER done ({time.time()-t0:.0f}s) rows={len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    partial_csv.unlink(missing_ok=True)
    done_log.unlink(missing_ok=True)
    print(f"wrote {out_csv}  ({len(df)} rows, {time.time()-t0:.0f}s)", flush=True)
    return df


# ---------------------------------------------------------------------------- render
def _violin(ax, datasets, positions, color, rng, width=0.34):
    """Violin backdrop + jittered points (show the data) + inner IQR bar + median dot."""
    vp = ax.violinplot(datasets, positions=positions, widths=width, showextrema=False)
    for body in vp['bodies']:
        body.set_facecolor(color)
        body.set_alpha(0.22)
        body.set_edgecolor(color)
        body.set_linewidth(0.8)
    for pos, d in zip(positions, datasets):
        x = pos + rng.uniform(-0.12, 0.12, size=len(d))
        ax.scatter(x, d, s=5, color=color, alpha=0.55, edgecolor='none', zorder=3)
        q25, med, q75 = np.percentile(d, [25, 50, 75])
        ax.vlines(pos, q25, q75, color=INK, lw=3.0, zorder=5)
        ax.scatter(pos, med, s=20, color='white', edgecolor=INK, linewidth=1.0, zorder=6)


def render(df, fig_stem=FIG_STEM):
    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 15, 'xtick.labelsize': 13,
                         'ytick.labelsize': 13, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': 0.8, 'font.family': 'sans-serif'})
    levels = [L for L in LEVELS if (df.level == L).any()]
    XLAB = {'isolate': 'Isolate\ngenus ref.', 'genus': 'Genus\nfamily ref.',
            'family': 'Family\norder ref.', 'order': 'Order\nclass ref.'}
    base = np.arange(len(levels)) * 1.0
    rng = np.random.default_rng(0)  # reproducible point jitter

    fig, (axA, axC) = plt.subplots(1, 2, figsize=(13.0, 6.0),
                                   gridspec_kw={'width_ratios': [1.7, 1.0]})

    # ---------- Panel a: quality on scoreable compounds ----------
    bpr_by, base_by, ns, advs, ps = [], [], [], [], []
    for L in levels:
        sub = df[df.level == L].dropna(subset=['auc_bpr_scoreable', 'auc_baseline_scoreable'])
        bpr_by.append(sub.auc_bpr_scoreable.values)
        base_by.append(sub.auc_baseline_scoreable.values)
        ns.append(len(sub))
        d = (sub.auc_bpr_scoreable - sub.auc_baseline_scoreable).values
        advs.append(np.median(d) if len(d) else np.nan)
        ps.append(wilcoxon(sub.auc_bpr_scoreable, sub.auc_baseline_scoreable).pvalue if len(sub) > 1 else np.nan)

    _violin(axA, bpr_by, base - 0.20, TEAL, rng)
    _violin(axA, base_by, base + 0.20, ORANGE, rng)
    axA.axhline(0.5, ls='--', lw=0.9, color=GREY)
    for i in range(len(levels)):
        p = ps[i]
        pstr = 'p<0.001' if p < 1e-3 else (f'p={p:.3f}' if p == p else '')
        axA.text(base[i], 1.085, f"n={ns[i]}", ha='center', fontsize=11, color='#444')
        axA.text(base[i], 1.045, f"$\\Delta${advs[i]:+.3f}", ha='center', fontsize=11, color='#333')
        axA.text(base[i], 1.008, pstr, ha='center', fontsize=11, color='#333')
    axA.set_xticks(base)
    axA.set_xticklabels([XLAB[L] for L in levels], fontsize=13)
    axA.set_ylabel('Hold-out AUC on scoreable metabolites\n(consumed vs. neutral)')
    axA.set_ylim(0.3, 1.13)
    axA.legend(handles=[Patch(facecolor=TEAL, alpha=0.55, label='BPR (learned)'),
                        Patch(facecolor=ORANGE, alpha=0.55, label='Prevalence in reference set (naive)')],
               loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=13, frameon=False)
    axA.text(-0.11, 1.03, 'a', transform=axA.transAxes, fontsize=24, fontweight='bold',
             va='bottom', ha='left')

    # ---------- Panel b: coverage spread (BPR scores every metabolite) ----------
    # Coverage is a property of the held-out GROUP's relative set (all isolates in a
    # held group share it), so one point per group -- not per isolate (pseudoreplication).
    M_total = int(df.n_metabolites.iloc[0])
    grp = df.groupby(['level', 'group'], as_index=False)['n_scoreable'].first()
    naive_by = [grp.loc[grp.level == L, 'n_scoreable'].values for L in levels]
    _violin(axC, naive_by, base, ORANGE, rng, width=0.5)
    axC.axhline(M_total, ls='-', lw=2.0, color=TEAL, zorder=2)
    axC.set_xticks(base)
    axC.set_xticklabels([L.capitalize() for L in levels], fontsize=13)
    axC.set_xlabel('hold-out level', fontsize=15)
    axC.set_ylabel(f'Metabolites scoreable (of {M_total})')
    axC.set_ylim(0, M_total * 1.15)
    axC.legend(handles=[Line2D([0], [0], color=TEAL, lw=2.5, label=f'BPR (all {M_total})'),
                        Patch(facecolor=ORANGE, alpha=0.55, label='Naive (prevalence in reference set)')],
               loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=13, frameon=False)
    axC.text(-0.16, 1.03, 'b', transform=axC.transAxes, fontsize=24, fontweight='bold',
             va='bottom', ha='left')

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f"{fig_stem}.{ext}", bbox_inches='tight', dpi=300)
    print(f"SAVED {fig_stem}.pdf/.png", flush=True)
    _summary(df, levels)
    _write_summary_json(df, levels, Path(f"{fig_stem}_summary.json"))


def _summary(df, levels=LEVELS):
    print("=== per-level summary ===", flush=True)
    for L in levels:
        sub = df[df.level == L]
        paired = sub.dropna(subset=['auc_bpr_scoreable', 'auc_baseline_scoreable'])
        print(f"{L:8s} n_all={len(sub):4d} n_paired={len(paired):4d} | "
              f"BPR(scoreable) med={paired.auc_bpr_scoreable.median():.3f} "
              f"naive med={paired.auc_baseline_scoreable.median():.3f} "
              f"Δmed={ (paired.auc_bpr_scoreable-paired.auc_baseline_scoreable).median():+.3f} | "
              f"BPR(all) med={sub.auc_bpr_all.median():.3f} | "
              f"coverage={100*sub.coverage.mean():.0f}%", flush=True)


def _write_summary_json(df, levels, path):
    """Per-level numbers quoted in the manuscript (n, medians, median advantage, Wilcoxon p)."""
    out = {}
    for L in levels:
        sub = df[df.level == L]
        paired = sub.dropna(subset=['auc_bpr_scoreable', 'auc_baseline_scoreable'])
        d = paired.auc_bpr_scoreable - paired.auc_baseline_scoreable
        out[L] = {
            'n_paired': int(len(paired)),
            'n_groups': int(sub.group.nunique()),
            'median_auc_bpr_scoreable': float(paired.auc_bpr_scoreable.median()),
            'median_auc_baseline_scoreable': float(paired.auc_baseline_scoreable.median()),
            'median_delta': float(d.median()),
            'wilcoxon_p': float(wilcoxon(paired.auc_bpr_scoreable, paired.auc_baseline_scoreable).pvalue),
            'median_auc_bpr_all': float(sub.auc_bpr_all.median()),
            'n_all': int(len(sub)),
        }
    path.write_text(json.dumps(out, indent=2))
    print(f"SAVED {path}", flush=True)


# ------------------------------------------------------------------------------ main
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--recompute', action='store_true', help='force the full CV even if cache exists')
    ap.add_argument('--smoke', action='store_true', help='fast end-to-end pipeline check to /tmp')
    ap.add_argument('--results-csv', type=Path, default=RESULTS_CSV,
                    help='CV results cache (default: next to this script = the submitted results)')
    ap.add_argument('--fig-stem', type=Path, default=FIG_STEM,
                    help='figure path without extension (default: the submitted figure)')
    ap.add_argument('--lambda-ord', type=float, default=LAMBDA_ORD,
                    help='ordinal-term weight (0.1 reproduces the submitted Fig. 3)')
    args = ap.parse_args()

    if args.smoke:
        out = Path('/tmp/msb_fig3_smoke_results.csv')
        df = compute(out_csv=out, n_epochs=30, isolate_limit=5, levels=['isolate'],
                     lambda_ord=args.lambda_ord)
        render(df, fig_stem=Path('/tmp/msb_fig3_smoke'))
    else:
        args.results_csv.parent.mkdir(parents=True, exist_ok=True)
        args.fig_stem.parent.mkdir(parents=True, exist_ok=True)
        if args.recompute:
            for suffix in ('.partial.csv', '.done_groups.tsv'):
                args.results_csv.with_name(args.results_csv.stem + suffix).unlink(missing_ok=True)
        if args.recompute or not args.results_csv.exists():
            df = compute(out_csv=args.results_csv, lambda_ord=args.lambda_ord)
        else:
            df = pd.read_csv(args.results_csv)
            print(f"loaded cache {args.results_csv} ({len(df)} rows); use --recompute to regenerate", flush=True)
        render(df, fig_stem=args.fig_stem)
