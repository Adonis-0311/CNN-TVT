"""Recompute the Fig. 4 five-seed matched Pareto inputs from sealed predictions."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
SEALED = ROOT / "artifacts" / "tvt_v4r_headline_composite" / "models"
SIDECAR = ROOT / "artifacts" / "tier2_iq_sidecar_v1" / "models"
PARETO = ROOT / "analysis_zero_compute" / "outputs" / "csv" / "a8_pareto.csv"
OUT = ROOT / "analysis_zero_compute" / "outputs" / "csv" / "a15_v41_pareto_matched.csv"
SEEDS = (17, 29, 43, 71, 101)


def score(path: Path) -> float:
    with np.load(path) as data:
        return float(f1_score(data["labels"], data["probabilities"].argmax(1), average="macro"))


def main() -> None:
    source = pd.read_csv(PARETO).query("regime == 'hard_interference'")
    rows = []
    for row in source.itertuples(index=False):
        scores = [score(SEALED / f"{row.model}_seed{seed}" / "predictions_hard_interference.npz") for seed in SEEDS]
        rows.append({"model": row.model, "model_short": row.model_short, "parameters": int(row.parameters), "evidence_class": "sealed_matched_subset", "seed_count": 5, "macro_f1_mean": float(np.mean(scores)), "macro_f1_std": float(np.std(scores, ddof=1))})
    scores = [score(SIDECAR / f"tier2_h2_f2_iq_sidecar_seed{seed}" / "predictions_hard_interference.npz") for seed in SEEDS]
    rows.append({"model": "tier2_h2_f2_iq_sidecar", "model_short": "I/Q sidecar", "parameters": 46794, "evidence_class": "exploratory_matched_subset", "seed_count": 5, "macro_f1_mean": float(np.mean(scores)), "macro_f1_std": float(np.std(scores, ddof=1))})
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(OUT)


if __name__ == "__main__":
    main()
