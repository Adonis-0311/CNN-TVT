"""Pair a Tier-2 exploratory run against the sealed A5 fits, seed by seed.

Reads prediction bundles from a *new* run directory and from the sealed
composite, checks that the test source order is identical, and reports the
preregistered Tier-2 decision quantities.  Nothing is written back into the
composite.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import zc_core as z  # noqa: E402

DECISION_SPLITS = (
    "clean_retention",
    "hard_interference",
    "id_test",
    "unseen_jammer",
    "unseen_speed",
    "heldout_channel",
    "combined_ood",
    "adc_10bit_agc",
    "adc_12bit_agc",
    "per_emitter_sync",
)
DRAWS = 10000


def _load_run(root: Path, model: str, seed: int, split: str):
    path = root / "models" / f"{model}_seed{seed}" / f"predictions_{split}.npz"
    with np.load(path) as data:
        return (
            np.asarray(data["probabilities"], dtype=np.float32).argmax(axis=1).astype(np.int8),
            np.asarray(data["labels"], dtype=np.int64),
            np.asarray(data["source_ids"], dtype=np.int64),
        )


def _load_external_preds(root: Path, model: str, seeds: tuple[int, ...], split: str):
    predictions, labels, source_ids = [], None, None
    for seed in seeds:
        pred, current_labels, current_sources = _load_run(root, model, seed, split)
        if labels is None:
            labels, source_ids = current_labels, current_sources
        elif not np.array_equal(current_sources, source_ids) or not np.array_equal(current_labels, labels):
            raise SystemExit(f"baseline source order or labels differ for {split}/seed{seed}")
        predictions.append(pred)
    metadata = z.load_preds(z.PROPOSED, split, seeds=seeds)
    if not np.array_equal(metadata.source_ids, source_ids) or not np.array_equal(metadata.labels, labels):
        raise SystemExit(f"external baseline is not aligned to the sealed cache for {split}")
    return z.Preds(
        model, split, labels, source_ids, metadata.snr_db, metadata.sir_db,
        metadata.profile, np.stack(predictions), np.zeros_like(np.stack(predictions), dtype=np.float32), seeds,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new", type=Path, required=True, help="new run directory")
    parser.add_argument("--new-model", required=True)
    parser.add_argument("--baseline-model", default=z.PROPOSED, help="sealed model to pair against")
    parser.add_argument("--baseline-root", type=Path, default=None, help="optional Tier-2 baseline run root")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in z.SEEDS))
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    seeds = tuple(int(value) for value in arguments.seeds.split(","))
    rows = []
    for split in DECISION_SPLITS:
        sealed = (
            z.load_preds(arguments.baseline_model, split, seeds=seeds)
            if arguments.baseline_root is None
            else _load_external_preds(arguments.baseline_root, arguments.baseline_model, seeds, split)
        )
        new_pred = []
        for seed in seeds:
            pred, labels, source_ids = _load_run(arguments.new, arguments.new_model, seed, split)
            if not np.array_equal(source_ids, sealed.source_ids):
                raise SystemExit(f"test source order differs for {split}/seed{seed}")
            if not np.array_equal(labels, sealed.labels):
                raise SystemExit(f"labels differ for {split}/seed{seed}")
            new_pred.append(pred)
        new_pred = np.stack(new_pred)
        stats = z.hierarchical_paired_diff(sealed.labels, sealed.pred, new_pred, draws=arguments.draws)
        sealed_f1 = float(np.mean([z.macro_f1(sealed.labels, sealed.pred[s]) for s in range(len(seeds))]))
        new_f1 = float(np.mean([z.macro_f1(sealed.labels, new_pred[s]) for s in range(len(seeds))]))
        rows.append(
            {
                "split": split,
                "sealed_model": arguments.baseline_model,
                "new_model": arguments.new_model,
                "sealed_macro_f1": sealed_f1,
                "new_macro_f1": new_f1,
                "evidence_class": "exploratory_tier2_outside_frozen_family",
                **stats,
            }
        )
        print(
            f"{split:20s} sealed={sealed_f1:.4f} new={new_f1:.4f} "
            f"delta={stats['macro_f1_difference']*100:+.2f} pp "
            f"[{stats['macro_f1_ci95_low']*100:+.2f},{stats['macro_f1_ci95_high']*100:+.2f}]"
        )

        if split == "hard_interference":
            severe = np.isclose(sealed.sir_db, -15.0)
            if not severe.any():
                raise SystemExit("hard_interference has no SIR=-15 dB rows")
            severe_stats = z.hierarchical_paired_diff(
                sealed.labels[severe],
                sealed.pred[:, severe],
                np.stack(new_pred)[:, severe],
                draws=arguments.draws,
            )
            rows.append(
                {
                    "split": "hard_interference_sir_minus15",
                    "sealed_model": arguments.baseline_model,
                    "new_model": arguments.new_model,
                    "sealed_macro_f1": float(
                        np.mean(
                            [z.macro_f1(sealed.labels[severe], sealed.pred[s, severe]) for s in range(len(seeds))]
                        )
                    ),
                    "new_macro_f1": float(
                        np.mean(
                            [z.macro_f1(sealed.labels[severe], np.stack(new_pred)[s, severe]) for s in range(len(seeds))]
                        )
                    ),
                    "evidence_class": "exploratory_tier2_outside_frozen_family",
                    **severe_stats,
                }
            )

    import pandas as pd

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(arguments.out, index=False)

    frame = pd.DataFrame(rows).set_index("split")
    clean = frame.loc["clean_retention"]
    hard = frame.loc["hard_interference"]
    print("\nPreregistered Tier-2 decisions")
    print(f"  H2a clean gain >= +6 pp and CI low > 0 : "
          f"{clean.macro_f1_difference >= 0.06 and clean.macro_f1_ci95_low > 0}")
    print(f"  H2b hard non-inferior at 1 pp          : {hard.macro_f1_ci95_low > -0.01}")
    if arguments.baseline_model == z.BACKBONE:
        axes = (
            "id_test", "unseen_jammer", "unseen_speed", "heldout_channel",
            "combined_ood", "per_emitter_sync",
        )
        print(
            "  H2d six OOD/receiver axes positive     : "
            f"{all(float(frame.loc[axis].macro_f1_difference) > 0 for axis in axes)}"
        )


if __name__ == "__main__":
    main()
