# revision_figS3_scatter_stats.py
"""Per-taxon ROC-AUC, GCS and embedding diversity for the 12 held-out isolates (mSystems revision).

Writes lambda0/figS3/figS3_scatter_per_taxon.csv with the values behind the
Supplementary Fig. 3 text (AUC vs genus consistency score and vs embedding
diversity), computed with revision_lambda0_numbers.s3_scatter for the adopted
model, so the Word-editing scripts can read them without loading the model.

Usage:
    python scripts/revision_figS3_scatter_stats.py
"""

import sys
from pathlib import Path

WOM_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(WOM_DIR))

import pandas as pd

import scripts.revision_lambda0_numbers as rn

OUT = rn.REV / "lambda0"

if __name__ == "__main__":
    table = pd.read_csv(OUT / "figS3" / "table_S_validation_12_isolates.csv")
    scatter = rn.s3_scatter(table, rn.NEW_MODEL, OUT / "fig2" / "stats_gcs_emb_diversity.csv")
    out = OUT / "figS3" / "figS3_scatter_per_taxon.csv"
    scatter["per_taxon"].to_csv(out, index=False, float_format="%.6f")
    print(out)
    for col in ["gcs", "emb_div"]:
        print(f"AUC vs {col}: Spearman rho = {scatter[col][0]:.3f}, p = {scatter[col][1]:.3f}, n = {scatter[col][2]}")
