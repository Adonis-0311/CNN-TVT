"""A8: accuracy-cost Pareto frontier and exploratory non-inferiority tests.

The Pareto view is the honest headline this evidence base supports: the
proposed model is not the most accurate system, it is the cheapest system on
the accuracy-cost frontier under structured interference.

The TOST-style tests are exploratory.  Margins are declared here, after the
confirmatory family was frozen and evaluated, so they may not be presented as
preregistered decisions.
"""

from __future__ import annotations

import numpy as np

import zc_core as z

MARGINS_PP = (1.0, 2.0)
CONTRASTS = (
    ("heldout_channel", z.REFERENCE),
    ("unseen_speed", z.REFERENCE),
    ("id_test", z.REFERENCE),
    ("hard_interference", z.REFERENCE),
    ("hard_interference", "mcldnn_reimplementation"),
    ("hard_interference", "iqformer_inspired"),
    ("clean_retention", z.BACKBONE),
)


def _complexity() -> dict[str, dict]:
    out = {}
    for model in z.MODELS:
        complexity = z.result_json(model, z.SEEDS[0])["complexity"]
        out[model] = {
            "parameters": float(complexity["parameters"]),
            "macs": float(complexity.get("conv_linear_recurrent_macs_excluding_stft", np.nan)),
            "latency_ms_p50": float(complexity.get("latency_ms_p50", np.nan)),
            "latency_ms_p95": float(complexity.get("latency_ms_p95", np.nan)),
            "state_dict_megabytes": float(complexity.get("state_dict_megabytes", np.nan)),
        }
    return out


def run() -> None:
    import pandas as pd

    aggregates = pd.read_csv(z.COMPOSITE / "seed_aggregates.csv")
    metrics = pd.read_csv(z.COMPOSITE / "metrics.csv")
    complexity = _complexity()

    rows: list[dict] = []
    for regime in ("hard_interference", "id_test", "clean_retention", "combined_ood"):
        subset = aggregates[aggregates.regime == regime]
        for _, record in subset.iterrows():
            info = complexity[record.model]
            rows.append(
                {
                    "regime": regime,
                    "model": record.model,
                    "model_short": z.SHORT.get(record.model, record.model),
                    "macro_f1_mean": float(record.macro_f1_mean),
                    "macro_f1_std": float(record.macro_f1_std),
                    **info,
                    "macro_f1_per_kparameter": float(record.macro_f1_mean) / (info["parameters"] / 1e3),
                    "macro_f1_per_mmac": float(record.macro_f1_mean) / (info["macs"] / 1e6),
                }
            )
    frame = pd.DataFrame(rows)

    # Pareto frontier per regime and cost axis (maximise macro-F1, minimise cost)
    for cost in ("parameters", "macs", "latency_ms_p50"):
        flags = []
        for _, record in frame.iterrows():
            same = frame[frame.regime == record.regime]
            dominated = (
                (same[cost] <= record[cost])
                & (same.macro_f1_mean >= record.macro_f1_mean)
                & ((same[cost] < record[cost]) | (same.macro_f1_mean > record.macro_f1_mean))
            ).any()
            flags.append(not dominated)
        frame[f"pareto_optimal_{cost}"] = flags
    z.write_csv("a8_pareto.csv", frame.to_dict("records"))

    tost_rows: list[dict] = []
    for regime, reference in CONTRASTS:
        table = metrics[metrics.regime == regime].pivot_table(
            index="seed", columns="model", values="macro_f1"
        )
        delta = (table[z.PROPOSED] - table[reference]).to_numpy()
        mean, low, high = z.seed_level_ci(delta)
        row = {
            "regime": regime,
            "reference": reference,
            "reference_short": z.SHORT.get(reference, reference),
            "seed_count": len(delta),
            "delta_mean_pp": 100 * mean,
            "delta_seed_ci_low_pp": 100 * low,
            "delta_seed_ci_high_pp": 100 * high,
        }
        for margin in MARGINS_PP:
            result = z.tost_noninferiority(delta, margin / 100.0)
            row[f"noninferior_margin_{margin:g}pp"] = result["non_inferior_at_0_05"]
            row[f"p_value_margin_{margin:g}pp"] = result["p_value_noninferiority"]
        tost_rows.append(row)
    z.write_csv("a8_noninferiority.csv", tost_rows)

    # SIR-resolved severe-interference corner, taken from the A1 product
    try:
        levels = pd.read_csv(z.CSV / "a1_paired_by_level.csv")
        severe = levels[
            (levels.regime == "hard_interference")
            & (levels.axis == "sir_db")
            & (levels.level_db == levels.level_db.min())
        ]
        z.write_csv("a8_severe_corner.csv", severe.to_dict("records"))
    except FileNotFoundError:
        pass

    _figure(frame)


def _figure(frame) -> None:
    plt = z.mpl()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, cost, label in zip(
        axes,
        ("parameters", "macs", "latency_ms_p50"),
        ("parameters", "MACs (excl. STFT)", "CUDA p50 latency (ms)"),
    ):
        subset = frame[frame.regime == "hard_interference"]
        for _, record in subset.iterrows():
            marker = "*" if record[f"pareto_optimal_{cost}"] else "o"
            size = 130 if record.model == z.PROPOSED else 45
            ax.scatter(record[cost], record.macro_f1_mean, marker=marker, s=size)
            ax.annotate(
                z.SHORT.get(record.model, record.model),
                (record[cost], record.macro_f1_mean),
                fontsize=6,
                xytext=(4, 3),
                textcoords="offset points",
            )
        if cost != "latency_ms_p50":
            ax.set_xscale("log")
        ax.set_xlabel(label)
        ax.set_title(f"Hard interference vs {label}", fontsize=9)
    axes[0].set_ylabel("macro-F1")
    fig.suptitle("Accuracy-cost frontier (stars: Pareto-optimal)", fontsize=10)
    z.save_fig(fig, "figA8_pareto")
    plt.close(fig)


if __name__ == "__main__":
    run()
