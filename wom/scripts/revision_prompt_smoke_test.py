# revision_prompt_smoke_test.py
"""Smoke test: how much do the SI agent prompts change with the revised model? (mSystems revision)

Replays the tool calls that the WoM agent made in the runs reproduced in the SI (chat logs of
Prompts 1-3) through the MCP server's own tool functions, once with the submitted model
(archive_submitted_lambda0.1) and once with the revised model (trained_models/ordinal_bpr_*), and
compares what the agent would see. The submitted-model replay must reproduce the numbers quoted in
the transcripts. Supplementary Fig. 4 statistics are recomputed from the experimental data in the
SI workbook with each model's scores. No LLM is called; the narrative an LLM writes from these
tool outputs can still vary between runs.

fastmcp and requests are not needed to call the tools, so minimal stand-ins are put on the path.

Usage (from the repository root):
    python wom/scripts/revision_prompt_smoke_test.py --workbook <SI tables .xlsx> --output-dir <dir>
"""

import os
import re
import sys
import json
import argparse
import tempfile
import importlib.util
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[2]
MODELS = {"submitted": REPO / "wom/trained_models/archive_submitted_lambda0.1",
          "revised": REPO / "wom/trained_models"}
RUNS = {"Prompt 1": "24ec6051-2afd-4e32-a7e3-5968d8b0ceb9",
        "Prompt 2": "defb77d6-455b-4cea-998a-f1f9a2746190",
        "Prompt 3": "3fba3c05-8166-464e-80a0-6037f0fa9d76"}
MODEL_TOOLS = {"rank_substrates", "compounds_for_microbes", "microbes_for_compounds", "analyze_substrate_gaps"}
THRESHOLD = 4.0  # Supplementary Fig. 4: predicted consumed if BPR score > 4.0
STUB = {"fastmcp/__init__.py": "class FastMCP:\n    def __init__(self, *a, **k): pass\n"
                               "    def tool(self, *a, **k):\n        return (lambda f: f)\n",
        "requests.py": ""}


def load_server():
    stub = Path(tempfile.mkdtemp(prefix="mcp_stub_"))
    for name, text in STUB.items():
        (stub / name).parent.mkdir(parents=True, exist_ok=True)
        (stub / name).write_text(text)
    sys.path.insert(0, str(stub))
    os.chdir(REPO)  # the server adds ./wom and ./growth_rate to sys.path
    spec = importlib.util.spec_from_file_location("wom_mcp_server", REPO / "mcp/wom_mcp_server.py")
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    return server


@contextmanager
def use_model(server, root: Path):
    """Point the server's trained_models/latest at the ordinal_bpr models under `root`."""
    tmp = Path(tempfile.mkdtemp(prefix="models_"))
    for kind in ("consumed", "secreted"):
        (tmp / f"ordinal_bpr_{kind}").symlink_to(root / f"ordinal_bpr_{kind}")
    saved = server.trained_models_dir
    server.trained_models_dir = tmp
    server._ENGINE_CACHE.clear()
    try:
        yield
    finally:
        server.trained_models_dir = saved
        server._ENGINE_CACHE.clear()


def tool_calls(run_id: str) -> tuple[list, str]:
    log = json.loads((REPO / "chat" / f"{run_id}.json").read_text())
    calls, text = [], ""
    for msg in log:
        if msg["role"] == "assistant" and isinstance(msg["content"], list):
            for part in msg["content"]:
                if part["type"] == "tool_call":
                    calls.append((part["tool_name"], part["args"]))
                elif part["type"] == "text":
                    text += part["content"] + "\n"
    return calls, text


def replay(server, run_id: str) -> tuple[dict, str]:
    calls, text = tool_calls(run_id)
    out = {"submitted": [], "revised": []}
    for model, root in MODELS.items():
        with use_model(server, root):
            for name, args in calls:
                if name in MODEL_TOOLS:
                    out[model].append((name, args, json.loads(getattr(server, name)(**args))))
    return out, text


def rank_table(response: dict) -> pd.DataFrame:
    rows = []
    for res in response["results"]:
        for r in res["rankings"]:
            rows.append({"genus": res["microbe"]["name"], "substrate": r["substrate"], "matched": r["matched"],
                         "score": r["score"], "sd": r.get("score_sd")})
    df = pd.DataFrame(rows)
    df["rank"] = df.groupby("genus").score.rank(ascending=False, method="min").astype(int)
    return df


def top_table(response: dict, k: int) -> pd.DataFrame:
    rows = []
    for res in response["results"]:
        for c in res["compounds"][:k]:
            rows.append({"genus": res["microbe"]["name"], "compound": c["name"], "score": c["raw_score"],
                         "sd": c.get("score_sd")})
    return pd.DataFrame(rows)


def quoted_scores(text: str) -> dict:
    """{(genus, substrate): score} from the transcript's markdown tables; a blank genus cell
    continues the genus of the row above, and bold/italic markers are ignored."""
    found, genus = {}, None
    for line in text.splitlines():
        cells = [c.strip().strip("*").strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3 or set(cells[0]) <= set(":- "):
            continue
        genus = cells[0] or genus
        m = re.match(r"(-?\d+\.\d\d)", cells[2])
        if genus and m and cells[1]:
            found[(genus, cells[1].lower())] = float(m.group(1))
    return found


def classification(score: pd.Series, consumed: pd.Series) -> dict:
    pred = score > THRESHOLD
    obs = consumed.astype(bool)
    rng = np.random.default_rng(0)
    r = pearsonr(score, consumed).statistic
    acc = float((pred == obs).mean())
    perm_r = [abs(pearsonr(score, rng.permutation(consumed.to_numpy())).statistic) for _ in range(10000)]
    perm_acc = [float((pred.to_numpy() == rng.permutation(obs.to_numpy())).mean()) for _ in range(10000)]
    return {"TP": int((pred & obs).sum()), "FN": int((~pred & obs).sum()), "TN": int((~pred & ~obs).sum()),
            "FP": int((pred & ~obs).sum()), "r": float(r), "r_perm_p": float(np.mean(np.array(perm_r) >= abs(r))),
            "accuracy": acc, "accuracy_perm_p": float(np.mean(np.array(perm_acc) >= acc)), "n": int(len(score))}


def read_s4(workbook: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    import openpyxl
    wb = openpyxl.load_workbook(workbook, data_only=True)
    a = []
    for r in wb["Data for Figure S4a and S4b"].iter_rows(min_row=10, values_only=True):
        if r[0] is None:  # the block of 22 metabolites ends at the first blank row
            break
        a.append(r[:4])
    a = pd.DataFrame(a, columns=["metabolite", "consumed", "time_h", "score_sheet"])
    a = a[a.time_h.notna()].reset_index(drop=True)
    a[["consumed", "time_h", "score_sheet"]] = a[["consumed", "time_h", "score_sheet"]].astype(float)
    c = [r[:5] for r in wb["Data for Figure S4c and S4d"].iter_rows(min_row=8, values_only=True)]
    c = pd.DataFrame([r for r in c if r[0] and isinstance(r[2], (int, float))],
                     columns=["isolate", "compound", "score_sheet", "wom_score", "consumed"])
    c[["score_sheet", "wom_score", "consumed"]] = c[["score_sheet", "wom_score", "consumed"]].astype(float)
    return a, c


def main(workbook: Path, output_dir: Path):
    server = load_server()
    output_dir.mkdir(parents=True, exist_ok=True)
    report, summary = [], {}
    s4a, s4c = read_s4(workbook)

    # ---- Prompt 1: rank 23 substrates for Pseudomonas
    out, text = replay(server, RUNS["Prompt 1"])
    t = {m: rank_table(out[m][0][2]) for m in out}
    p1 = t["submitted"].merge(t["revised"], on=["genus", "substrate", "matched"], suffixes=("_old", "_new"))
    quoted = quoted_scores(text)
    p1["quoted"] = [quoted.get(("Pseudomonas", s.lower())) for s in p1.substrate]
    check1 = (p1.quoted - p1.score_old.round(2)).abs().max()
    rho = spearmanr(p1.score_old, p1.score_new).statistic
    p1["rank_change"] = p1.rank_old - p1.rank_new
    a = s4a.assign(key=s4a.metabolite.str.lower()).merge(
        p1.assign(key=p1.substrate.str.lower())[["key", "score_old", "score_new"]], on="key", how="left")
    s4_old_a, s4_new_a = classification(a.score_old, a.consumed), classification(a.score_new, a.consumed)
    s4b = {m: {"pearson_r": float(pearsonr(a[f"score_{m}"], a.time_h).statistic),
               "pearson_p": float(pearsonr(a[f"score_{m}"], a.time_h).pvalue),
               "spearman_rho": float(spearmanr(a[f"score_{m}"], a.time_h).statistic)} for m in ("old", "new")}
    summary["prompt1"] = {"max_abs_diff_quoted_vs_replay": float(check1), "spearman_old_new": float(rho),
                          "s4a": {"old": s4_old_a, "new": s4_new_a}, "s4b": s4b}
    p1.to_csv(output_dir / "prompt1_pseudomonas_ranking.csv", index=False, float_format="%.3f")

    # ---- Prompt 2: top-3 substrates for 8 gut genera, then a genus x metabolite table
    out, text = replay(server, RUNS["Prompt 2"])
    top_call = next(r for n, a_, r in out["submitted"] if n == "compounds_for_microbes")
    top = {m: top_table(next(r for n, a_, r in out[m] if n == "compounds_for_microbes"), 3) for m in out}
    genera = list(dict.fromkeys(top["submitted"].genus))
    p2_rows = []
    for g in genera:
        old = top["submitted"][top["submitted"].genus == g]
        new = top["revised"][top["revised"].genus == g]
        p2_rows.append({"genus": g, "old_top3": "; ".join(f"{c} {s:.2f}" for c, s in zip(old.compound, old.score)),
                        "new_top3": "; ".join(f"{c} {s:.2f}" for c, s in zip(new.compound, new.score)),
                        "same_set": set(old.compound) == set(new.compound),
                        "same_order": list(old.compound) == list(new.compound),
                        "n_shared": len(set(old.compound) & set(new.compound))})
    p2 = pd.DataFrame(p2_rows)
    quoted = quoted_scores(text)
    q2 = [(abs(quoted[(g, c.lower())] - round(s, 2))) for g, c, s in
          zip(top["submitted"].genus, top["submitted"].compound, top["submitted"].score) if (g, c.lower()) in quoted]
    union_old = list(dict.fromkeys(top["submitted"].compound))
    union_new = list(dict.fromkeys(top["revised"].compound))
    table = {m: rank_table(next(r for n, a_, r in reversed(out[m]) if n == "rank_substrates")) for m in out}
    def by_name(tab, col):  # requested and resolved names both point to the score
        long = pd.concat([tab[["genus", "substrate", "score"]].rename(columns={"substrate": "compound"}),
                          tab[["genus", "matched", "score"]].rename(columns={"matched": "compound"})])
        return long.drop_duplicates(["genus", "compound"]).rename(columns={"score": col})
    c = s4c.assign(genus=s4c.isolate.str.split().str[0]).merge(by_name(table["submitted"], "score_old"),
        on=["genus", "compound"], how="left").merge(by_name(table["revised"], "score_new"), on=["genus", "compound"],
        how="left")
    assert c[["score_old", "score_new"]].notna().all().all(), c[c.score_old.isna()]
    s4_old_c, s4_new_c = classification(c.score_old, c.consumed), classification(c.score_new, c.consumed)
    kept = c[c.compound.isin(union_new)]  # pairs whose compound the revised agent would still select
    s4_new_c_selected = classification(kept.score_new, kept.consumed)
    by_compound = {m: c.assign(ok=(c[f"score_{m}"] > THRESHOLD) == c.consumed.astype(bool)).groupby("compound").ok.mean()
                   for m in ("old", "new")}
    by_isolate = {m: c.assign(ok=(c[f"score_{m}"] > THRESHOLD) == c.consumed.astype(bool)).groupby("isolate").ok.mean()
                  for m in ("old", "new")}
    summary["prompt2"] = {"max_abs_diff_quoted_vs_replay": float(max(q2)) if q2 else None, "n_quoted_checked": len(q2),
                          "genera_same_top3_set": int(p2.same_set.sum()), "genera_same_order": int(p2.same_order.sum()),
                          "n_genera": len(p2), "union_old": union_old, "union_new": union_new,
                          "new_compounds_without_s4_data": sorted(set(union_new) - set(s4c.compound)),
                          "s4c": {"old": s4_old_c, "new": s4_new_c, "new_selected_compounds_only": s4_new_c_selected,
                                  "compounds_dropped_from_selection": sorted(set(s4c.compound) - set(union_new))},
                          "sheet_scores_match_old_replay": float((c.score_sheet - c.score_old.round(2)).abs().max())}
    p2.to_csv(output_dir / "prompt2_top3.csv", index=False)
    pd.concat([by_compound["old"].rename("old"), by_compound["new"].rename("new")], axis=1).to_csv(
        output_dir / "prompt2_s4d_by_compound.csv")
    pd.concat([by_isolate["old"].rename("old"), by_isolate["new"].rename("new")], axis=1).to_csv(
        output_dir / "prompt2_s4d_by_isolate.csv")
    c.to_csv(output_dir / "prompt2_s4c_pairs.csv", index=False, float_format="%.3f")

    # ---- Prompt 3: Acinetobacter carbon sources in NLDM
    out, text = replay(server, RUNS["Prompt 3"])
    nldm = {m: rank_table(next(r for n, a_, r in out[m] if n == "rank_substrates")) for m in out}
    p3 = nldm["submitted"].merge(nldm["revised"], on=["genus", "substrate", "matched"], suffixes=("_old", "_new"))
    quoted = quoted_scores(text)
    q3 = [abs(quoted[("Acinetobacter", s.lower())] - round(v, 2)) for s, v in zip(p3.matched, p3.score_old)
          if ("Acinetobacter", s.lower()) in quoted]
    mfc = {m: next(r for n, a_, r in out[m] if n == "microbes_for_compounds") for m in out}
    glucose = {m: [(x["taxon"], round(x["compound_scores"]["D-glucose"], 2)) for x in
                   sorted(resp["results"], key=lambda x: -x["compound_scores"].get("D-glucose", -99))][:10]
               for m, resp in mfc.items()}
    summary["prompt3"] = {"max_abs_diff_quoted_vs_replay": float(max(q3)) if q3 else None, "n_quoted_checked": len(q3),
                          "spearman_old_new": float(spearmanr(p3.score_old, p3.score_new).statistic),
                          "top8_old": p3.sort_values("rank_old").head(8)[["matched", "score_old"]].values.tolist(),
                          "top8_new": p3.sort_values("rank_new").head(8)[["matched", "score_new"]].values.tolist(),
                          "bottom3_old": p3.sort_values("rank_old").tail(3)[["matched", "score_old"]].values.tolist(),
                          "bottom3_new": p3.sort_values("rank_new").tail(3)[["matched", "score_new"]].values.tolist(),
                          "glucose_top_genera": glucose}
    p3.to_csv(output_dir / "prompt3_acinetobacter_nldm.csv", index=False, float_format="%.3f")

    (output_dir / "smoke_test_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    main(args.workbook.resolve(), args.output_dir.resolve())
