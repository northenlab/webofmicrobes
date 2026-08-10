"""New main Figure 3: taxonomic hold-out CV, BPR vs naive genus/parent-mean baseline.

(a) BPR and the naive baseline at four hold-out levels of increasing phylogenetic
    distance. Both decline; BPR's margin over naive averaging GROWS with distance.
(b) Per-isolate paired advantage (BPR - naive) at each level (zero line, paired
    Wilcoxon p). Positive and growing = BPR's learned structure helps most when
    predicting across taxonomic gaps.
(c) Coverage: fraction of the 311 metabolites a method can score for a held-out
    isolate. Naive averaging is undefined for compounds untested in the peer set;
    BPR scores all -- the model's central value-add.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import wilcoxon

HERE = Path(__file__).parent
cv = pd.read_csv(HERE / 'holdout_cv_results.csv')          # BPR (auc_raw) per level+isolate
bl = pd.read_csv(HERE / 'baseline_all_levels_results.csv') # naive baseline per level+isolate

LEVELS = ['isolate', 'genus', 'family', 'order']
XLAB = ['Isolate\n(genus peers)', 'Genus\n(family peers)', 'Family\n(order peers)', 'Order\n(class peers)']
XLAB_SHORT = ['Isolate', 'Genus', 'Family', 'Order']
TEAL, ORANGE, GREY = '#237c7c', '#d98a34', '#9aa0a8'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.linewidth': 0.8, 'font.family': 'sans-serif'})
fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(11.4, 3.7), gridspec_kw={'width_ratios': [1.5, 1.25, 0.9]})

# ---------- Panel A: grouped BPR vs naive, per level ----------
def box(ax, data, pos, color, w=0.34):
    bp = ax.boxplot(data, positions=pos, widths=w, patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', lw=1.3),
                    boxprops=dict(facecolor=color, alpha=0.35, edgecolor=color),
                    whiskerprops=dict(color=color), capprops=dict(color=color))
    for i, d in enumerate(data):
        ax.scatter(np.random.normal(pos[i], 0.05, len(d)), d, s=5, color=color, alpha=0.4, edgecolor='none', zorder=3)
bpr_by = [cv.loc[cv.level == L, 'auc_raw'].dropna().values for L in LEVELS]
bl_by = [bl.loc[bl.level == L, 'auc_baseline'].dropna().values for L in LEVELS]
base = np.arange(len(LEVELS)) * 1.2
box(axA, bpr_by, base - 0.21, TEAL)
box(axA, bl_by, base + 0.21, ORANGE)
for i, L in enumerate(LEVELS):
    axA.text(base[i], 1.05, f"n={len(bpr_by[i])}", ha='center', fontsize=6.8, color='#444')
axA.axhline(0.5, ls='--', lw=0.9, color=GREY); axA.text(base[-1]+0.4, 0.505, 'chance', color=GREY, fontsize=7, va='bottom', ha='right')
axA.set_xticks(base); axA.set_xticklabels(XLAB, fontsize=7.5)
axA.set_ylabel('Hold-out AUC (consumed vs. neutral)'); axA.set_ylim(0.3, 1.12)
axA.legend(handles=[Patch(facecolor=TEAL, alpha=0.4, label='Ordinal-BPR (learned)'),
                    Patch(facecolor=ORANGE, alpha=0.4, label='Genus-mean profile (naive)')],
           loc='lower left', fontsize=7.3, frameon=False)
axA.set_title('a  Generalization across phylogenetic distance', loc='left', fontsize=9.3, weight='bold')

# ---------- Panel B: paired advantage BPR - naive ----------
diffs, ps = [], []
for L in LEVELS:
    m = cv[cv.level == L][['isolate', 'auc_raw']].merge(bl[bl.level == L][['isolate', 'auc_baseline']], on='isolate')
    d = (m.auc_raw - m.auc_baseline).dropna().values
    diffs.append(d)
    try: ps.append(wilcoxon(m.auc_raw, m.auc_baseline).pvalue)
    except Exception: ps.append(np.nan)
bpx = axB.boxplot(diffs, positions=base, widths=0.5, patch_artist=True, showfliers=False,
                  medianprops=dict(color='black', lw=1.3),
                  boxprops=dict(facecolor='#4c6ea5', alpha=0.35, edgecolor='#4c6ea5'),
                  whiskerprops=dict(color='#4c6ea5'), capprops=dict(color='#4c6ea5'))
for i, d in enumerate(diffs):
    axB.scatter(np.random.normal(base[i], 0.06, len(d)), d, s=5, color='#4c6ea5', alpha=0.4, edgecolor='none', zorder=3)
    star = 'p<0.001' if ps[i] < 1e-3 else (f'p={ps[i]:.3f}' if ps[i] == ps[i] else '')
    axB.text(base[i], 0.60, f"+{np.median(d):.03f}\n{star}", ha='center', fontsize=6.6, color='#333')
axB.axhline(0, ls='-', lw=1.0, color='#333')
axB.set_xticks(base); axB.set_xticklabels(XLAB_SHORT, fontsize=8)
axB.set_xlabel('hold-out level', fontsize=8)
axB.set_ylabel('AUC advantage (BPR - naive), per isolate'); axB.set_ylim(-0.5, 0.72)
axB.set_title('b  BPR vs. naive averaging (paired)', loc='left', fontsize=9.3, weight='bold')

# ---------- Panel C: coverage ----------
cov = [100*bl.loc[bl.level == L, 'coverage_all_metabolites'].mean() for L in LEVELS]
axC.bar(base, cov, width=0.62, color=ORANGE, alpha=0.55, label='Naive (peer-tested only)')
axC.axhline(100, ls='-', lw=1.6, color=TEAL)
axC.text(base[-1]+0.2, 100.5, 'BPR: 100%', color=TEAL, fontsize=8, va='bottom', ha='right', weight='bold')
for i, c in enumerate(cov):
    axC.text(base[i], c+1.5, f"{c:.0f}%", ha='center', fontsize=7, color='#7a4a15')
axC.set_xticks(base); axC.set_xticklabels(XLAB_SHORT, fontsize=8)
axC.set_xlabel('hold-out level', fontsize=8)
axC.set_ylabel('% of 311 metabolites scoreable'); axC.set_ylim(0, 112)
axC.set_title('c  Coverage of metabolite space', loc='left', fontsize=9.3, weight='bold')

fig.tight_layout()
for ext in ['pdf', 'png']:
    fig.savefig(HERE / f'Figure3_holdout_CV.{ext}', bbox_inches='tight', dpi=300)
print("SAVED Figure3_holdout_CV.pdf/.png")
print("=== numbers ===")
for i, L in enumerate(LEVELS):
    print(f"{L:8s} BPR med={np.median(bpr_by[i]):.3f} mean={np.mean(bpr_by[i]):.3f} | "
          f"naive med={np.median(bl_by[i]):.3f} mean={np.mean(bl_by[i]):.3f} | "
          f"diff med=+{np.median(diffs[i]):.3f} p={ps[i]:.2g} | coverage={cov[i]:.0f}%")
