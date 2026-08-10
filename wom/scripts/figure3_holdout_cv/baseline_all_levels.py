"""Naive genus/parent-mean baseline at ALL hold-out levels (no learned embedding),
mirroring the peer-selection of holdout_cv.py so BPR and baseline are compared on
identical information at each level. Also computes COVERAGE: of the 311 metabolites,
what fraction can each method score for a held-out isolate.
  isolate held out -> peers = same genus            (predict from genus peers)
  genus  held out -> peers = same family, diff genus
  family held out -> peers = same order,  diff family
  order  held out -> peers = same class,  diff order
Baseline score(metabolite) = consumption prevalence among peers (consumed/tested);
untested-by-peers -> 0 (cannot predict). AUC over the isolate's consumed vs neutral.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from loaders.loaders import prepare_training_data

def auc(pos, neg):
    if not len(pos) or not len(neg):
        return np.nan
    p = np.asarray(pos)[:, None]; n = np.asarray(neg)[None, :]
    return float((np.sum(p > n) + 0.5*np.sum(p == n)) / (len(pos)*len(neg)))

td = prepare_training_data(include_secreted=False)
M = td.n_metabolites
idx_of = td.microbe_to_idx
lvl = {m: {L: td.microbe_taxonomy[m].get_level(L) for L in ['genus','family','order','class']} for m in idx_of}
members = {L: defaultdict(list) for L in ['genus','family','order','class']}
for m in idx_of:
    for L in members:
        if lvl[m][L]:
            members[L][lvl[m][L]].append(m)

# level -> (parent_level, exclude_child_level)
LADDER = {'isolate': ('genus', None), 'genus': ('family', 'genus'),
          'family': ('order', 'family'), 'order': ('class', 'order')}

def peers_for(level, m):
    parent, exch = LADDER[level]
    pkey = lvl[m][parent]
    if pkey is None:
        return []
    if level == 'isolate':
        return [x for x in members['genus'][lvl[m]['genus']] if x != m]
    exkey = lvl[m][exch]
    return [x for x in members[parent][pkey] if lvl[x][exch] != exkey]

rows = []
for level in ['isolate', 'genus', 'family', 'order']:
    for m in idx_of:
        # only evaluate isolates that had a BPR result at this level (genus>=2 for isolate, etc.)
        peers = peers_for(level, m)
        if not peers:
            continue
        i = idx_of[m]
        pos_i = td.positives.get(i, set()); neg_i = td.known_negatives.get(i, set())
        if not pos_i or not neg_i:
            continue
        consumed_ct = defaultdict(int); tested_ct = defaultdict(int)
        for x in peers:
            xi = idx_of[x]
            for met in td.positives.get(xi, set()):
                consumed_ct[met] += 1; tested_ct[met] += 1
            for met in td.known_negatives.get(xi, set()):
                tested_ct[met] += 1
        def sc(met):
            return consumed_ct[met]/tested_ct[met] if tested_ct.get(met, 0) > 0 else 0.0
        s_pos = [sc(x) for x in pos_i]; s_neg = [sc(x) for x in neg_i]
        # coverage of ALL metabolites (fraction the peer set can score at all)
        cover_all = len([1 for met in range(M) if tested_ct.get(met, 0) > 0]) / M
        rows.append(dict(level=level, isolate=m, genus=lvl[m]['genus'],
                         n_peers=len(peers), auc_baseline=auc(s_pos, s_neg),
                         coverage_all_metabolites=cover_all))

df = pd.DataFrame(rows)
df.to_csv(Path(__file__).parent / 'baseline_all_levels_results.csv', index=False)
for L in ['isolate','genus','family','order']:
    s = df[df.level == L]
    print(f"{L:8s} n={len(s):4d} baseline median AUC={s.auc_baseline.median():.3f} mean={s.auc_baseline.mean():.3f}"
          f"  mean coverage of 311 mets={100*s.coverage_all_metabolites.mean():.1f}%")
print("wrote baseline_all_levels_results.csv")
