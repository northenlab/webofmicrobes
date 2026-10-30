"""Hierarchical taxonomic hold-out cross-validation for Ordinal-BPR.

Faithful to the preprint Methods ('Taxonomic Holdout Cross-Validation'):
for each level in {isolate, genus, family, order}, every group at that rank is
withheld in turn, a fresh Ordinal-BPR (dim 32) is retrained on the remainder,
and each held-out isolate is predicted from the MEAN embedding of its nearest
remaining relatives one rank higher:
  isolate held out -> predicted from other isolates in the same genus
  genus   held out -> predicted from other genera in the same family
  family  held out -> predicted from other families in the same order
  order   held out -> predicted from other orders in the same class
Two scorings: raw (prediction embedding) and distinctive (global-popularity
direction projected out). Per held-out isolate we score consumed vs neutral
metabolites (AUC).
"""
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from loaders.loaders import prepare_training_data
from core.models import OrdinalBPRModel

SEED = 0
N_EPOCHS = 400
LAMBDA_ORD = 0.0  # 0.1 (fit() default) in the submitted analysis
OUT = str(Path(__file__).parent / 'holdout_cv_results.csv')

def auc(pos_scores, neg_scores):
    # P(score(consumed) > score(neutral)); ties count 0.5
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return np.nan
    p = np.asarray(pos_scores)[:, None]
    n = np.asarray(neg_scores)[None, :]
    return float((np.sum(p > n) + 0.5 * np.sum(p == n)) / (len(pos_scores) * len(neg_scores)))

def train(positives_sub, known_neg_sub, n_mic, n_met):
    np.random.seed(SEED)
    import torch; torch.manual_seed(SEED)
    m = OrdinalBPRModel(embedding_dim=32)
    m.fit(positives=positives_sub, n_microbes=n_mic, n_metabolites=n_met,
          known_negatives=known_neg_sub, n_epochs=N_EPOCHS, lambda_ord=LAMBDA_ORD, verbose=False)
    return m.microbe_embeddings, m.metabolite_embeddings

def scores_for(pred_emb, met_emb, gdir=None):
    e = pred_emb
    if gdir is not None:
        e = e - np.dot(e, gdir) * gdir
    return e @ met_emb.T

def main():
    td = prepare_training_data(include_secreted=False)
    N, M = td.n_microbes, td.n_metabolites
    idx_of = td.microbe_to_idx
    lvl = {m: {L: td.microbe_taxonomy[m].get_level(L) for L in ['genus','family','order','class']}
           for m in idx_of}
    # membership maps by taxon
    members = {L: defaultdict(list) for L in ['genus','family','order','class']}
    for m in idx_of:
        for L in ['genus','family','order','class']:
            if lvl[m][L]:
                members[L][lvl[m][L]].append(m)

    rows = []
    def evaluate(level, group_key, held_ids, parent_level, parent_key, exclude_child_level, exclude_child_key):
        # peers = microbes in parent_key at parent_level, excluding those whose exclude_child_level==exclude_child_key
        peers = [m for m in members[parent_level][parent_key]
                 if lvl[m][exclude_child_level] != exclude_child_key]
        peers = [m for m in peers if m not in held_ids]
        if not peers:
            return 0
        held_set = set(held_ids)
        positives_sub = {idx_of[m]: td.positives[idx_of[m]] for m in idx_of
                         if m not in held_set and idx_of[m] in td.positives and td.positives[idx_of[m]]}
        known_sub = {idx_of[m]: td.known_negatives.get(idx_of[m], set()) for m in idx_of
                     if m not in held_set}
        mic_emb, met_emb = train(positives_sub, known_sub, N, M)
        peer_idx = [idx_of[m] for m in peers]
        pred_emb = mic_emb[peer_idx].mean(axis=0)
        # global direction from trained (non-held-out) microbes that actually got updated
        train_idx = [idx_of[m] for m in idx_of if m not in held_set]
        gmean = mic_emb[train_idx].mean(axis=0)
        gdir = gmean / (np.linalg.norm(gmean) + 1e-9)
        s_raw = pred_emb @ met_emb.T
        s_dis = scores_for(pred_emb, met_emb, gdir)
        n_eval = 0
        for m in held_ids:
            i = idx_of[m]
            pos = list(td.positives.get(i, set()))
            neg = list(td.known_negatives.get(i, set()))
            if not pos or not neg:
                continue
            rows.append(dict(level=level, group=group_key, isolate=m, genus=lvl[m]['genus'],
                             family=lvl[m]['family'], order=lvl[m]['order'], class_=lvl[m]['class'],
                             n_consumed=len(pos), n_neutral=len(neg), n_peers=len(peers),
                             auc_raw=auc(s_raw[pos], s_raw[neg]),
                             auc_distinctive=auc(s_dis[pos], s_dis[neg])))
            n_eval += 1
        return n_eval

    t0 = time.time()
    # ISOLATE level: hold out one isolate, predict from same-genus peers
    todo = [m for m in idx_of if lvl[m]['genus'] and len(members['genus'][lvl[m]['genus']]) >= 2]
    for k, m in enumerate(todo):
        evaluate('isolate', m, [m], 'genus', lvl[m]['genus'], 'genus', '__none__')
        if (k+1) % 25 == 0:
            print(f"isolate {k+1}/{len(todo)}  ({time.time()-t0:.0f}s)  rows={len(rows)}", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"ISOLATE done {len(todo)} groups, {time.time()-t0:.0f}s", flush=True)

    # GENUS level: hold out a whole genus, predict from other genera in same family
    genera = [g for g in members['genus'] if g]
    gk = 0
    for g in genera:
        held = members['genus'][g]
        fam = lvl[held[0]]['family']
        if not fam:
            continue
        n = evaluate('genus', g, held, 'family', fam, 'genus', g)
        gk += 1
        if gk % 20 == 0:
            print(f"genus {gk}/{len(genera)}  ({time.time()-t0:.0f}s)  rows={len(rows)}", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"GENUS done, {time.time()-t0:.0f}s", flush=True)

    # FAMILY level: hold out a whole family, predict from other families in same order
    for f in [x for x in members['family'] if x]:
        held = members['family'][f]
        od = lvl[held[0]]['order']
        if not od:
            continue
        evaluate('family', f, held, 'order', od, 'family', f)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"FAMILY done, {time.time()-t0:.0f}s", flush=True)

    # ORDER level: hold out a whole order, predict from other orders in same class
    for o in [x for x in members['order'] if x]:
        held = members['order'][o]
        cl = lvl[held[0]]['class']
        if not cl:
            continue
        evaluate('order', o, held, 'class', cl, 'order', o)
    pd.DataFrame(rows).to_csv(OUT, index=False)

    df = pd.DataFrame(rows)
    print("=== SUMMARY (median AUC per level) ===", flush=True)
    for L in ['isolate','genus','family','order']:
        sub = df[df.level == L]
        if len(sub):
            print(f"{L:8s} n={len(sub):4d}  raw median={sub.auc_raw.median():.3f} mean={sub.auc_raw.mean():.3f}"
                  f"  distinctive median={sub.auc_distinctive.median():.3f} mean={sub.auc_distinctive.mean():.3f}",
                  flush=True)
    print(f"TOTAL_TIME {time.time()-t0:.0f}s  wrote {OUT}", flush=True)

if __name__ == '__main__':
    main()
