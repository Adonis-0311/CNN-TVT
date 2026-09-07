"""A4: post-hoc calibration and selective prediction (risk-coverage).

Temperature is fitted per (model, algorithm seed) on the *validation* split
only and then applied unchanged to the test regimes, so no test information
leaks into the calibrator.  Selective prediction turns the modest absolute
accuracy into an operationally meaningful statement: what accuracy is
available if the receiver is allowed to abstain on the hardest windows.
"""

from __future__ import annotations

import numpy as np

import zc_core as z

MODELS = (z.BACKBONE, z.PROPOSED, z.REFERENCE, "mcldnn_reimplementation", "iqformer_inspired")
REGIMES = ("hard_interference", "id_test", "combined_ood", "clean_retention")
COVERAGES = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)


def _fit_temperature(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Minimise validation NLL over a log-spaced temperature grid."""

    log_probabilities = np.log(np.clip(probabilities, 1e-12, None))
    grid = np.exp(np.linspace(np.log(0.25), np.log(6.0), 96))
    best_temperature, best_nll = 1.0, np.inf
    rows = np.arange(len(labels))
    for temperature in grid:
        scaled = log_probabilities / temperature
        scaled -= scaled.max(axis=1, keepdims=True)
        normaliser = np.log(np.exp(scaled).sum(axis=1))
        nll = float((normaliser - scaled[rows, labels]).mean())
        if nll < best_nll:
            best_nll, best_temperature = nll, float(temperature)
    return best_temperature


def _apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.log(np.clip(probabilities, 1e-12, None)) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exponentiated = np.exp(scaled)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def _ece(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    score = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence >= lower) & (confidence < upper if upper < 1.0 else confidence <= upper)
        if selected.any():
            score += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return float(score)


def _risk_coverage(confidence: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(-confidence)
    ordered = correct[order]
    coverage = np.arange(1, len(ordered) + 1) / len(ordered)
    accuracy = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    aurc = float(np.trapezoid(1.0 - accuracy, coverage))
    return coverage, accuracy, aurc


def run() -> None:
    temperature_rows: list[dict] = []
    curve_rows: list[dict] = []
    point_rows: list[dict] = []

    for model in MODELS:
        temperatures = {}
        for seed in z.SEEDS:
            probabilities = z.load_probabilities(model, seed, "validation")
            with np.load(z.fit_dir(model, seed) / "predictions_validation.npz") as data:
                labels = np.asarray(data["labels"], dtype=np.int64)
            temperature = _fit_temperature(probabilities, labels)
            temperatures[seed] = temperature
            temperature_rows.append(
                {
                    "model": model,
                    "model_short": z.SHORT.get(model, model),
                    "algorithm_seed": seed,
                    "temperature": temperature,
                    "validation_ece_raw": _ece(probabilities, labels),
                    "validation_ece_calibrated": _ece(_apply_temperature(probabilities, temperature), labels),
                }
            )

        for regime in REGIMES:
            accuracies = {coverage: [] for coverage in COVERAGES}
            aurc_values, ece_raw, ece_cal = [], [], []
            grid = np.linspace(0.02, 1.0, 50)
            curves = []
            for seed in z.SEEDS:
                probabilities = z.load_probabilities(model, seed, regime)
                with np.load(z.fit_dir(model, seed) / f"predictions_{regime}.npz") as data:
                    labels = np.asarray(data["labels"], dtype=np.int64)
                calibrated = _apply_temperature(probabilities, temperatures[seed])
                correct = (calibrated.argmax(axis=1) == labels).astype(float)
                confidence = calibrated.max(axis=1)
                ece_raw.append(_ece(probabilities, labels))
                ece_cal.append(_ece(calibrated, labels))
                coverage, accuracy, aurc = _risk_coverage(confidence, correct)
                aurc_values.append(aurc)
                curves.append(np.interp(grid, coverage, accuracy))
                for target in COVERAGES:
                    index = max(int(round(target * len(correct))) - 1, 0)
                    accuracies[target].append(float(accuracy[index]))
            curves = np.stack(curves)
            for index, coverage in enumerate(grid):
                curve_rows.append(
                    {
                        "model": model,
                        "model_short": z.SHORT.get(model, model),
                        "regime": regime,
                        "coverage": float(coverage),
                        "accuracy_mean": float(curves[:, index].mean()),
                        "accuracy_std": float(curves[:, index].std(ddof=1)),
                    }
                )
            for target in COVERAGES:
                values = np.array(accuracies[target])
                mean, low, high = z.seed_level_ci(values)
                point_rows.append(
                    {
                        "model": model,
                        "model_short": z.SHORT.get(model, model),
                        "regime": regime,
                        "coverage": target,
                        "accuracy_mean": mean,
                        "accuracy_seed_ci_low": low,
                        "accuracy_seed_ci_high": high,
                        "aurc_mean": float(np.mean(aurc_values)),
                        "ece_raw_mean": float(np.mean(ece_raw)),
                        "ece_calibrated_mean": float(np.mean(ece_cal)),
                    }
                )

    z.write_csv("a4_temperature.csv", temperature_rows)
    z.write_csv("a4_risk_coverage_curves.csv", curve_rows)
    z.write_csv("a4_selective_points.csv", point_rows)
    _figure(curve_rows)


def _figure(curve_rows: list[dict]) -> None:
    import pandas as pd

    plt = z.mpl()
    curves = pd.DataFrame(curve_rows)
    fig, axes = plt.subplots(1, len(REGIMES), figsize=(3.4 * len(REGIMES), 3.0), sharey=True)
    for ax, regime in zip(np.atleast_1d(axes), REGIMES):
        for model in MODELS:
            subset = curves[(curves.regime == regime) & (curves.model == model)].sort_values("coverage")
            ax.plot(subset.coverage, subset.accuracy_mean, lw=1.4, label=z.SHORT.get(model, model))
            ax.fill_between(
                subset.coverage,
                subset.accuracy_mean - subset.accuracy_std,
                subset.accuracy_mean + subset.accuracy_std,
                alpha=0.12,
            )
        ax.set_title(regime.replace("_", " "), fontsize=9)
        ax.set_xlabel("coverage")
    np.atleast_1d(axes)[0].set_ylabel("selective accuracy")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle("Risk-coverage after validation-only temperature scaling", fontsize=10)
    z.save_fig(fig, "figA4_risk_coverage")
    plt.close(fig)


if __name__ == "__main__":
    run()
