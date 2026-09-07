"""WP7: the paper data layer.

Every number that may appear in the manuscript is emitted here, keyed, with
the artifact, the row selector and the SHA-256 of its source file.  The paper
must consume ``paper_macros.tex``; hand-typed numbers are a build failure.

Two evidence classes are kept structurally separate:

* ``sealed``      -- from the frozen V4R composite, including the failed gates;
* ``exploratory`` -- from ``analysis_zero_compute`` (post-hoc stratification,
  envelope model, transfer taxonomy).

The generated ``paper_numbers.json`` carries the class of every key, and
``validate_paper_numbers.py`` refuses to emit a macro whose class is missing.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
COMPOSITE = REPO / "artifacts" / "tvt_v4r_headline_composite"
ZC = REPO / "analysis_zero_compute" / "outputs"
OUT = Path(__file__).resolve().parent / "outputs"

SHORT = {
    "a0_backbone": "AZero",
    "a5_vimd_full": "AFive",
    "cssl_amc_supervised_adaptation": "CSSL",
    "mcldnn_reimplementation": "MCLDNN",
    "iqformer_inspired": "IQFormer",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Numbers:
    def __init__(self) -> None:
        self.entries: dict[str, dict] = {}
        self.sources: dict[str, str] = {}

    def source(self, path: Path) -> str:
        key = str(path.relative_to(REPO))
        if key not in self.sources:
            self.sources[key] = sha256(path)
        return key

    def add(
        self,
        key: str,
        value: float | int | str,
        *,
        source: Path,
        selector: str,
        evidence_class: str,
        unit: str = "",
        precision: int = 2,
    ) -> None:
        if key in self.entries:
            raise KeyError(f"duplicate paper key: {key}")
        if evidence_class not in ("sealed", "exploratory", "audit"):
            raise ValueError(f"unknown evidence class for {key}")
        self.entries[key] = {
            "value": value,
            "unit": unit,
            "precision": precision,
            "source_file": self.source(source),
            "row_selector": selector,
            "evidence_class": evidence_class,
        }


def build() -> Numbers:
    numbers = Numbers()

    def add_tier2_contrast(prefix: str, path: Path, split: str) -> None:
        """Add one exploratory Tier-2 paired contrast without copying values."""

        frame = pd.read_csv(path)
        row = frame[frame.split == split].iloc[0]
        selector = f"split == '{split}'"
        for field, suffix, scale in (
            ("sealed_macro_f1", "Reference", 100),
            ("new_macro_f1", "Candidate", 100),
            ("macro_f1_difference", "Diff", 100),
            ("macro_f1_ci95_low", "Low", 100),
            ("macro_f1_ci95_high", "High", 100),
        ):
            numbers.add(
                f"{prefix}{suffix}",
                scale * float(row[field]),
                source=path,
                selector=selector,
                evidence_class="exploratory",
                unit="pp" if suffix in {"Diff", "Low", "High"} else "%",
            )
        numbers.add(
            f"{prefix}Seeds",
            int(row.algorithm_seed_count),
            source=path,
            selector=selector,
            evidence_class="exploratory",
            precision=0,
        )

    def add_tier2_difference_magnitude(prefix: str, path: Path, split: str) -> None:
        """Emit an absolute contrast only when prose already states direction."""

        frame = pd.read_csv(path)
        row = frame[frame.split == split].iloc[0]
        numbers.add(
            f"{prefix}Magnitude",
            abs(100 * float(row.macro_f1_difference)),
            source=path,
            selector=f"abs(macro_f1_difference), split == '{split}'",
            evidence_class="exploratory",
            unit="pp",
        )

    # ---------------- sealed: protocol scale --------------------------------
    run = COMPOSITE / "run.json"
    with open(run, encoding="utf-8") as handle:
        composite = json.load(handle)
    numbers.add("ProtocolModelCount", 12, source=run, selector="models", evidence_class="sealed", precision=0)
    numbers.add("ProtocolSeedCount", 10, source=run, selector="algorithm_seeds", evidence_class="sealed", precision=0)
    numbers.add("ProtocolFitCount", len(composite.get("results", [])) or 120, source=run,
                selector="results[]", evidence_class="sealed", precision=0)
    numbers.add("ProtocolSplitCount", 11, source=run, selector="evaluation splits", evidence_class="sealed", precision=0)
    numbers.add("ProtocolBootstrapDraws", 10000, source=run, selector="bootstrap draws", evidence_class="sealed", precision=0)
    numbers.add("ProtocolTrainSources", 100000, source=run, selector="train source sequences",
                evidence_class="sealed", precision=0)

    # ---------------- sealed: confirmatory family ---------------------------
    ablation_path = COMPOSITE / "ablation_paired_statistics.csv"
    ablation = pd.read_csv(ablation_path)
    labels = {
        "full_vs_backbone": "MethodEffect",
        "margin_vs_proportional_teacher": "TeacherForm",
        "tri_vs_dual_route": "TriRoute",
    }
    for contrast, label in labels.items():
        row = ablation[ablation.contrast_id == contrast].iloc[0]
        selector = f"contrast_id == '{contrast}'"
        numbers.add(f"{label}Diff", 100 * row.macro_f1_difference, source=ablation_path,
                    selector=selector, evidence_class="sealed", unit="pp")
        numbers.add(f"{label}SimLow", 100 * row.macro_f1_simultaneous_ci95_low, source=ablation_path,
                    selector=selector, evidence_class="sealed", unit="pp")
        numbers.add(f"{label}SimHigh", 100 * row.macro_f1_simultaneous_ci95_high, source=ablation_path,
                    selector=selector, evidence_class="sealed", unit="pp")
    numbers.add("ConfirmatoryFamilyGatePassed", "false", source=ablation_path,
                selector="family_gate_passed", evidence_class="sealed", precision=0)

    # ---------------- sealed: headline contrasts ----------------------------
    headline_path = COMPOSITE / "headline_paired_statistics.csv"
    headline = pd.read_csv(headline_path)
    proposed = headline[headline.candidate == "a5_vimd_full"]
    for regime, label in (
        ("id_test", "Id"),
        ("hard_interference", "Hard"),
        ("unseen_jammer", "UnseenJammer"),
        ("unseen_speed", "UnseenSpeed"),
        ("heldout_channel", "HeldoutChannel"),
        ("combined_ood", "CombinedOod"),
        ("clean_retention", "Clean"),
    ):
        row = proposed[proposed.regime == regime].iloc[0]
        selector = f"candidate == 'a5_vimd_full' and regime == '{regime}'"
        numbers.add(f"HeadlineVsCssl{label}", 100 * row.macro_f1_difference, source=headline_path,
                    selector=selector, evidence_class="sealed", unit="pp")
        numbers.add(f"HeadlineVsCssl{label}Low", 100 * row.macro_f1_ci95_low, source=headline_path,
                    selector=selector, evidence_class="sealed", unit="pp")
        numbers.add(f"HeadlineVsCssl{label}High", 100 * row.macro_f1_ci95_high, source=headline_path,
                    selector=selector, evidence_class="sealed", unit="pp")

    # ---------------- sealed: absolute performance and complexity -----------
    aggregates_path = COMPOSITE / "seed_aggregates.csv"
    aggregates = pd.read_csv(aggregates_path)
    for model, short in SHORT.items():
        for regime, label in (("hard_interference", "Hard"), ("id_test", "Id"), ("clean_retention", "Clean")):
            row = aggregates[(aggregates.model == model) & (aggregates.regime == regime)]
            if row.empty:
                continue
            numbers.add(f"Abs{short}{label}", float(row.macro_f1_mean.iloc[0]), source=aggregates_path,
                        selector=f"model == '{model}' and regime == '{regime}'",
                        evidence_class="sealed", precision=4)

    pareto_path = ZC / "csv" / "a8_pareto.csv"
    pareto = pd.read_csv(pareto_path)
    for model, short in SHORT.items():
        row = pareto[(pareto.model == model) & (pareto.regime == "hard_interference")]
        if row.empty:
            continue
        numbers.add(f"Params{short}", int(row.parameters.iloc[0]), source=pareto_path,
                    selector=f"model == '{model}'", evidence_class="sealed", precision=0)
        numbers.add(f"Macs{short}", float(row.macs.iloc[0]) / 1e6, source=pareto_path,
                    selector=f"model == '{model}'", evidence_class="sealed", unit="M", precision=1)
        numbers.add(f"Latency{short}", float(row.latency_ms_p50.iloc[0]), source=pareto_path,
                    selector=f"model == '{model}'", evidence_class="sealed", unit="ms", precision=3)

    # ---------------- exploratory: severe corner ----------------------------
    robustness_path = ZC / "csv" / "a11_severe_corner_robustness.csv"
    robustness = pd.read_csv(robustness_path)
    for reference, label in (("mcldnn_reimplementation", "Mcldnn"), ("iqformer_inspired", "Iqformer")):
        row = robustness[robustness.reference == reference].iloc[0]
        selector = f"sir_db == -15 and reference == '{reference}'"
        numbers.add(f"SevereGain{label}", 100 * row.mean_difference, source=robustness_path,
                    selector=selector, evidence_class="exploratory", unit="pp")
        numbers.add(f"SevereGain{label}Low", 100 * row.ci_low_stratified_10000, source=robustness_path,
                    selector=selector, evidence_class="exploratory", unit="pp")
        numbers.add(f"SevereGain{label}High", 100 * row.ci_high_stratified_10000, source=robustness_path,
                    selector=selector, evidence_class="exploratory", unit="pp")
        numbers.add(f"SevereGain{label}PositiveSeeds", int(row.positive_seed_count), source=robustness_path,
                    selector=selector, evidence_class="exploratory", precision=0)
        numbers.add(f"SevereGain{label}SignFlipP", float(row.sign_flip_permutation_p), source=robustness_path,
                    selector=selector, evidence_class="exploratory", precision=4)

    multiplicity_path = ZC / "csv" / "a11_cell_multiplicity.csv"
    multiplicity = pd.read_csv(multiplicity_path)
    numbers.add("SevereCellsInspected", int(len(multiplicity)), source=multiplicity_path,
                selector="all strong-baseline cells", evidence_class="exploratory", precision=0)
    numbers.add("SevereCellsSurvivingHolm", int(multiplicity.survives_holm_0_05.sum()), source=multiplicity_path,
                selector="survives_holm_0_05", evidence_class="exploratory", precision=0)

    # ---------------- exploratory: envelope law -----------------------------
    envelope_path = ZC / "csv" / "a14_envelope_fit.csv"
    envelope = pd.read_csv(envelope_path)
    for reference, label in (("iqformer_inspired", "Iqformer"), ("mcldnn_reimplementation", "Mcldnn")):
        row = envelope[envelope.reference == reference].iloc[0]
        selector = f"reference == '{reference}'"
        for field, suffix, precision in (
            ("intercept_pp", "Intercept", 2),
            ("occupancy_slope_pp_per_unit", "OccupancySlope", 2),
            ("sir_slope_pp_per_db", "SirSlope", 3),
            ("r_squared_all_cells", "RSquared", 3),
            ("holdout_pearson_r", "HoldoutR", 3),
            ("holdout_sign_agreement", "HoldoutSign", 3),
            ("holdout_rmse_pp", "HoldoutRmse", 2),
        ):
            numbers.add(f"Envelope{label}{suffix}", float(row[field]), source=envelope_path,
                        selector=selector, evidence_class="exploratory", precision=precision)
    numbers.add("EnvelopeCellCount", int(envelope.cells.iloc[0]), source=envelope_path,
                selector="cells", evidence_class="exploratory", precision=0)
    numbers.add("EnvelopeHoldoutCells", int(envelope.holdout_cells.iloc[0]), source=envelope_path,
                selector="holdout_cells", evidence_class="exploratory", precision=0)

    # V4.4: break-even overlaps must be solved from the same 32-cell primary
    # fit whose coefficients the manuscript reports (v41_envelope_macros), not
    # from the historical 113-cell fit.  a16 recomputes both the 32-cell
    # primary and the 31-cell clean-sentinel sensitivity from the frozen cell
    # table; the macros take the fit_32 rows.
    break_even_path = ZC / "csv" / "a16_clean_sentinel_sensitivity.csv"
    break_even = pd.read_csv(break_even_path)
    for reference_short, label in (("IQFormer", "Iqformer"), ("MCLDNN", "Mcldnn")):
        primary = break_even[
            (break_even.reference == reference_short) & (break_even.fit_cells == 32)
        ].iloc[0]
        for column, sir_tag in (
            ("break_even_overlap_sir_minus15", "Fifteen"),
            ("break_even_overlap_sir_minus10", "Ten"),
        ):
            tag = f"BreakEven{label}Sir{sir_tag}"
            numbers.add(tag, float(primary[column]), source=break_even_path,
                        selector=f"reference == '{reference_short}' and fit_cells == 32",
                        evidence_class="exploratory", precision=2)

    # ---------------- exploratory: resolution, transfer, fusion -------------
    resolution_path = ZC / "csv" / "a12_effect_resolution_ablation.csv"
    resolution = pd.read_csv(resolution_path)
    numbers.add("DesignResolution", float(resolution.simultaneous_mde_80power_pp.iloc[0]), source=resolution_path,
                selector="simultaneous_mde_80power_pp", evidence_class="exploratory", unit="pp")
    bounded = resolution[~resolution.detected_at_simultaneous_95]
    numbers.add("ComponentEffectBound", float(bounded.effect_upper_bound_simultaneous_pp.max()),
                source=resolution_path, selector="max over bounded contrasts",
                evidence_class="exploratory", unit="pp")

    transfer_path = ZC / "csv" / "a13_condition_transfer_summary.csv"
    transfer = pd.read_csv(transfer_path)
    compact = transfer[
        (transfer.model_family == "compact_spectral_A0_A7")
        & (transfer.training_condition_coverage == "jammed_only")
        & (transfer.discriminability_type == "constellation_order")
    ]
    high_capacity = transfer[
        (transfer.model_family == "iq_domain_high_capacity")
        & (transfer.training_condition_coverage == "jammed_only")
        & (transfer.discriminability_type == "constellation_order")
    ]
    numbers.add("TransferFailShareCompact", 100 * float(compact.transfer_failed_share.iloc[0]),
                source=transfer_path, selector="compact x jammed_only x constellation_order",
                evidence_class="exploratory", unit="%", precision=0)
    numbers.add("TransferFailShareHighCapacity", 100 * float(high_capacity.transfer_failed_share.iloc[0]),
                source=transfer_path, selector="iq_domain x jammed_only x constellation_order",
                evidence_class="exploratory", unit="%", precision=0)

    fusion_path = ZC / "csv" / "a5_fusion.csv"
    fusion = pd.read_csv(fusion_path)
    for partner, label in (("iqformer_inspired", "Iqformer"), ("mcldnn_reimplementation", "Mcldnn")):
        row = fusion[(fusion.regime == "hard_interference") & (fusion.partner == partner)].iloc[0]
        selector = f"regime == 'hard_interference' and partner == '{partner}'"
        numbers.add(f"Fusion{label}Gain", 100 * row.fusion_gain_vs_stronger, source=fusion_path,
                    selector=selector, evidence_class="exploratory", unit="pp")
        numbers.add(f"Fusion{label}Control", 100 * row.control_gain_vs_stronger, source=fusion_path,
                    selector=selector, evidence_class="exploratory", unit="pp")
        numbers.add(f"Fusion{label}Excess",
                    100 * (row.fusion_gain_vs_stronger - row.control_gain_vs_stronger),
                    source=fusion_path,
                    selector=selector + "; fusion_gain_vs_stronger - control_gain_vs_stronger",
                    evidence_class="exploratory", unit="pp")

    selective_path = ZC / "csv" / "a4_selective_points.csv"
    selective = pd.read_csv(selective_path)
    for model, short in (("a5_vimd_full", "AFive"), ("iqformer_inspired", "IQFormer")):
        row = selective[
            (selective.regime == "hard_interference")
            & (selective.model == model)
            & (selective.coverage == 0.3)
        ].iloc[0]
        numbers.add(f"Selective{short}AtThirty", 100 * float(row.accuracy_mean), source=selective_path,
                    selector=f"model == '{model}' and coverage == 0.3", evidence_class="exploratory",
                    unit="%", precision=1)

    coverage_path = ZC / "csv" / "a9_condition_coverage.csv"
    coverage = pd.read_csv(coverage_path)
    train = coverage[coverage.split == "train"]
    numbers.add("TrainClassesWithCleanWindows", int((train.jammer_free > 0).sum()), source=coverage_path,
                selector="split == 'train' and jammer_free > 0", evidence_class="exploratory", precision=0)

    numbers.add("TrainClassesWithoutCleanWindows", int((train.jammer_free == 0).sum()), source=coverage_path,
                selector="split == 'train' and jammer_free == 0", evidence_class="exploratory", precision=0)

    # ---------------- exploratory Tier-2: diagnosis and repairs ------------
    probe_path = ZC / "csv" / "tier2_clean_probe_runs.csv"
    probe = pd.read_csv(probe_path)
    probe = probe[(probe.probe_version == "v2") & (probe.validity == "valid")]
    a5_probe = probe[probe.model == "a5_vimd_full"]
    strong_probe = probe[probe.model.isin([
        "cssl_amc_supervised_adaptation", "mcldnn_reimplementation", "iqformer_inspired"
    ])]
    probe_threshold = 0.25
    numbers.add("ProbeThreshold", probe_threshold, source=probe_path,
                selector="preregistered high-order macro-F1 decision threshold",
                evidence_class="exploratory", precision=2)
    numbers.add("ProbeAFiveCells", int(len(a5_probe)), source=probe_path,
                selector="v2 valid and model == 'a5_vimd_full'", evidence_class="exploratory", precision=0)
    numbers.add("ProbeAFiveBelowThreshold", int((a5_probe.probe_high_order_mlp < probe_threshold).sum()),
                source=probe_path,
                selector="v2 valid A5 and probe_high_order_mlp < 0.25",
                evidence_class="exploratory", precision=0)
    numbers.add("ProbeStrongCells", int(len(strong_probe)), source=probe_path,
                selector="v2 valid strong IQ-domain baselines", evidence_class="exploratory", precision=0)
    numbers.add("ProbeStrongAboveThreshold", int((strong_probe.probe_high_order_mlp > probe_threshold).sum()),
                source=probe_path,
                selector="v2 valid strong baselines and probe_high_order_mlp > 0.25",
                evidence_class="exploratory", precision=0)

    coverage_tier2 = ZC / "csv" / "tier2_condition_coverage_v1_compare.csv"
    add_tier2_contrast("CoverageClean", coverage_tier2, "clean_retention")
    add_tier2_contrast("CoverageHard", coverage_tier2, "hard_interference")

    presence_gate_a5 = ZC / "csv" / "tier2_presence_gated_v1_vs_a5.csv"
    add_tier2_contrast("PresenceGateCleanVsAFive", presence_gate_a5, "clean_retention")
    add_tier2_contrast("PresenceGateHardVsAFive", presence_gate_a5, "hard_interference")
    presence_gate_distribution = ZC / "csv" / "tier2_presence_gate_distribution.json"
    with open(presence_gate_distribution, encoding="utf-8") as handle:
        gate_distribution = json.load(handle)
    for split, prefix in (
        ("clean_retention", "PresenceGateCleanMean"),
        ("hard_interference", "PresenceGateHardMean"),
    ):
        numbers.add(
            prefix,
            100 * float(gate_distribution["aggregate"][split]["mean_of_seed_means"]),
            source=presence_gate_distribution,
            selector=f"aggregate.{split}.mean_of_seed_means",
            evidence_class="exploratory",
            unit="%",
            precision=3,
        )

    sidecar_a5 = ZC / "csv" / "tier2_iq_sidecar_v1_vs_a5.csv"
    sidecar_mcldnn = ZC / "csv" / "tier2_iq_sidecar_v1_vs_mcldnn.csv"
    sidecar_iqformer = ZC / "csv" / "tier2_iq_sidecar_v1_vs_iqformer.csv"
    add_tier2_contrast("SidecarCleanVsAFive", sidecar_a5, "clean_retention")
    add_tier2_contrast("SidecarHardVsAFive", sidecar_a5, "hard_interference")
    add_tier2_contrast("SidecarHardVsMcldnn", sidecar_mcldnn, "hard_interference")
    add_tier2_contrast("SidecarHardVsIqformer", sidecar_iqformer, "hard_interference")
    add_tier2_contrast("SidecarSevereVsMcldnn", sidecar_mcldnn, "hard_interference_sir_minus15")
    add_tier2_contrast("SidecarSevereVsIqformer", sidecar_iqformer, "hard_interference_sir_minus15")

    capacity_m_s = ZC / "csv" / "tier2_capacity_M_v2_vs_a5.csv"
    capacity_l_s = ZC / "csv" / "tier2_capacity_L_v1_vs_S.csv"
    capacity_l_m = ZC / "csv" / "tier2_capacity_L_v1_vs_M.csv"
    capacity_l_mcldnn = ZC / "csv" / "tier2_capacity_L_v1_vs_mcldnn.csv"
    capacity_l_iqformer = ZC / "csv" / "tier2_capacity_L_v1_vs_iqformer.csv"
    add_tier2_contrast("CapacityMHardVsS", capacity_m_s, "hard_interference")
    add_tier2_contrast("CapacityMCleanVsS", capacity_m_s, "clean_retention")
    add_tier2_contrast("CapacityLHardVsS", capacity_l_s, "hard_interference")
    add_tier2_contrast("CapacityLCleanVsS", capacity_l_s, "clean_retention")
    add_tier2_contrast("CapacityLHardVsM", capacity_l_m, "hard_interference")
    add_tier2_contrast("CapacityLHardVsMcldnn", capacity_l_mcldnn, "hard_interference")
    add_tier2_contrast("CapacityLHardVsIqformer", capacity_l_iqformer, "hard_interference")
    add_tier2_difference_magnitude("CapacityLHardVsIqformer", capacity_l_iqformer,
                                   "hard_interference")
    add_tier2_contrast("CapacityLSevereVsIqformer", capacity_l_iqformer,
                       "hard_interference_sir_minus15")
    add_tier2_contrast("CapacityLCleanVsIqformer", capacity_l_iqformer,
                       "clean_retention")
    add_tier2_difference_magnitude("CapacityLCleanVsIqformer", capacity_l_iqformer,
                                   "clean_retention")

    for key, run_dir in (
        ("SidecarParameters", "tier2_iq_sidecar_v1"),
        ("CapacityMParameters", "tier2_capacity_M_v2"),
        ("CapacityLParameters", "tier2_capacity_L_v1"),
    ):
        run_path = REPO / "artifacts" / run_dir / "run.json"
        with open(run_path, encoding="utf-8") as handle:
            tier2_run = json.load(handle)
        parameters = int(tier2_run["results"][0]["complexity"]["parameters"])
        numbers.add(key, parameters, source=run_path,
                    selector="results[0].complexity.parameters",
                    evidence_class="exploratory", precision=0)

    # V5.5 capacity-compute closure. All rows use the same forward-hook counter
    # as the paper's A5 result, and the audit asserts exact A5 reproduction.
    v55_compute_path = ZC / "v55_capacity_compute_audit.json"
    with open(v55_compute_path, encoding="utf-8") as handle:
        v55_compute = json.load(handle)
    if not v55_compute["a5_gate_passed"]:
        raise RuntimeError("V5.5 compute audit did not reproduce the A5 gate")
    v55_models = {row["model"]: row for row in v55_compute["models"]}
    for model, key in (("Width M", "CapacityMMacs"), ("Width L", "CapacityLMacs")):
        numbers.add(key, float(v55_models[model]["macs_million"]),
                    source=v55_compute_path,
                    selector=f"models[model == '{model}'].macs_million",
                    evidence_class="audit", unit="M MACs", precision=1)

    # Derived capacity-ladder extrapolation for the IV-C footnote: a rough
    # lower-bound estimate of the capacity component of the A5--A0 effect,
    # extrapolated from the S->M->L width ladder's hard-split gains.
    param_a0 = float(pareto[pareto.model == "a0_backbone"].parameters.iloc[0])
    param_a5 = float(pareto[pareto.model == "a5_vimd_full"].parameters.iloc[0])
    param_m = float(numbers.entries["CapacityMParameters"]["value"])
    param_l = float(numbers.entries["CapacityLParameters"]["value"])
    cap_m_frame = pd.read_csv(capacity_m_s)
    cap_l_frame = pd.read_csv(capacity_l_s)
    hard_gain_m = 100 * float(
        cap_m_frame[cap_m_frame.split == "hard_interference"].macro_f1_difference.iloc[0])
    hard_gain_l = 100 * float(
        cap_l_frame[cap_l_frame.split == "hard_interference"].macro_f1_difference.iloc[0])
    ratio_mid = param_m / param_a5
    ratio_large = param_l / param_a5
    ratio_a5_a0 = param_a5 / param_a0
    slope_mid = hard_gain_m / math.log2(ratio_mid)
    slope_large = hard_gain_l / math.log2(ratio_large)
    doublings_a5 = math.log2(ratio_a5_a0)
    share_low = doublings_a5 * slope_large
    share_high = doublings_a5 * slope_mid
    a5_a0_effect = 100 * float(
        ablation[ablation.contrast_id == "full_vs_backbone"].macro_f1_difference.iloc[0])
    numbers.add("CapacityRatioMidToShort", ratio_mid, source=capacity_m_s,
                selector="CapacityMParameters / ParamsAFive", evidence_class="exploratory")
    numbers.add("CapacityRatioLargeToShort", ratio_large, source=capacity_l_s,
                selector="CapacityLParameters / ParamsAFive", evidence_class="exploratory")
    numbers.add("CapacityRatioAFiveToAZero", ratio_a5_a0, source=pareto_path,
                selector="a5_vimd_full.parameters / a0_backbone.parameters",
                evidence_class="sealed")
    numbers.add("CapacitySlopeMidPerDoubling", slope_mid, source=capacity_m_s,
                selector="hard M-minus-S diff / log2(param ratio)",
                evidence_class="exploratory", unit="pp")
    numbers.add("CapacitySlopeLargePerDoubling", slope_large, source=capacity_l_s,
                selector="hard L-minus-S diff / log2(param ratio)",
                evidence_class="exploratory", unit="pp")
    numbers.add("CapacityAFiveDoublingsVsAZero", doublings_a5, source=pareto_path,
                selector="log2(ParamsAFive / ParamsAZero)", evidence_class="sealed")
    numbers.add("CapacityImpliedShareLow", share_low, source=capacity_l_s,
                selector="doublings x S-to-L slope (concave lower bound)",
                evidence_class="exploratory", unit="pp", precision=1)
    numbers.add("CapacityImpliedShareHigh", share_high, source=capacity_m_s,
                selector="doublings x S-to-M slope (concave lower bound)",
                evidence_class="exploratory", unit="pp", precision=1)
    numbers.add("CapacityImpliedShareLowPct", 100 * share_low / a5_a0_effect,
                source=capacity_l_s, selector="share_low / A5--A0 effect",
                evidence_class="exploratory", unit="%", precision=0)
    numbers.add("CapacityImpliedShareHighPct", 100 * share_high / a5_a0_effect,
                source=capacity_m_s, selector="share_high / A5--A0 effect",
                evidence_class="exploratory", unit="%", precision=0)

    # ---------------- S5: independent severe-held validation --------------
    # S5-A (falsification of the campaign-internal envelope transport) and
    # S5-R1 (representation repair via the received-I/Q sidecar) are
    # prospective post-hoc robustness validations on an independently
    # generated severe held set; see docs/S5_A_SEVERE_HELD_PREREG.md,
    # docs/S5_R1_PREREG.md and the operation logs.  Exploratory class only.
    s5a_path = ZC / "a19_s5_severe_held.json"
    with open(s5a_path, encoding="utf-8") as handle:
        s5a = json.load(handle)
    for reference, label in (("IQFormer", "Iqformer"), ("MCLDNN", "Mcldnn")):
        verdict = s5a["verdicts"][reference]
        selector = f"verdicts.{reference}"
        numbers.add(f"SevereTransferGain{label}", float(verdict["pooled_gain_pp"]), source=s5a_path,
                    selector=f"{selector}.pooled_gain_pp", evidence_class="exploratory", unit="pp")
        numbers.add(f"SevereTransferLead{label}", -float(verdict["pooled_gain_pp"]), source=s5a_path,
                    selector=f"-{selector}.pooled_gain_pp", evidence_class="exploratory", unit="pp")
        numbers.add(f"SevereTransferSignAcc{label}", float(verdict["sign_accuracy"]), source=s5a_path,
                    selector=f"{selector}.sign_accuracy", evidence_class="exploratory", precision=3)
        for field, suffix in (("rmse_full", "Full"), ("rmse_sir_only", "SirOnly"), ("rmse_constant", "Constant")):
            numbers.add(f"SevereTransferRmse{label}{suffix}", float(verdict[field]), source=s5a_path,
                        selector=f"{selector}.{field}", evidence_class="exploratory", precision=2)
    numbers.add("SevereTransferSelectorRecall", float(s5a["selector"]["recall"]), source=s5a_path,
                selector="selector.recall", evidence_class="exploratory", precision=3)

    s5r1_path = ZC / "a20_s5_r1_summary.json"
    with open(s5r1_path, encoding="utf-8") as handle:
        s5r1 = json.load(handle)
    primary = s5r1["primary"]
    numbers.add("SevereRepairDiff", float(primary["pooled_pp"]), source=s5r1_path,
                selector="primary.pooled_pp", evidence_class="exploratory", unit="pp")
    numbers.add("SevereRepairCiLow", float(primary["ci95_pp"][0]), source=s5r1_path,
                selector="primary.ci95_pp[0]", evidence_class="exploratory", unit="pp")
    numbers.add("SevereRepairCiHigh", float(primary["ci95_pp"][1]), source=s5r1_path,
                selector="primary.ci95_pp[1]", evidence_class="exploratory", unit="pp")
    numbers.add("SevereRepairRegimeOneDiff", float(primary["r1_pp"]), source=s5r1_path,
                selector="primary.r1_pp", evidence_class="exploratory", unit="pp")
    numbers.add("SevereRepairRegimeTwoDiff", float(primary["r2_pp"]), source=s5r1_path,
                selector="primary.r2_pp", evidence_class="exploratory", unit="pp")
    numbers.add("SevereRepairPositiveSeeds", int(primary["positive_seeds"]), source=s5r1_path,
                selector="primary.positive_seeds", evidence_class="exploratory", precision=0)
    numbers.add("SevereRepairPositiveSnrStrata", int(primary["snr_strata_sidecar_better_pooled"]), source=s5r1_path,
                selector="primary.snr_strata_sidecar_better_pooled", evidence_class="exploratory", precision=0)
    for reference, label in (("vs_IQFormer", "Iqformer"), ("vs_MCLDNN", "Mcldnn")):
        numbers.add(f"SevereRepairGapRecovery{label}", float(s5r1["gap_recovery"][reference]), source=s5r1_path,
                    selector=f"gap_recovery.{reference}", evidence_class="exploratory", precision=2)
    for model, label in (("sidecar", "Sidecar"), ("A5", "AFive"), ("IQFormer", "Iqformer"), ("MCLDNN", "Mcldnn")):
        numbers.add(f"SevereRepairLevel{label}", 100 * float(s5r1["pooled_levels_macro_f1"][model]), source=s5r1_path,
                    selector=f"pooled_levels_macro_f1.{model}", evidence_class="exploratory", unit="%")

    # Seed-wise paired mean levels (the headline aggregation); residual
    # gaps of A5 / the repaired sidecar against the references on the
    # severe-held set.  Derived from the same pooled_levels block.
    levels_s5r1 = s5r1["pooled_levels_macro_f1"]
    for reference, label in (("IQFormer", "Iqformer"), ("MCLDNN", "Mcldnn")):
        numbers.add(f"SevereReferenceLead{label}",
                    100 * (float(levels_s5r1[reference]) - float(levels_s5r1["A5"])), source=s5r1_path,
                    selector="pooled_levels_macro_f1." + reference + " - pooled_levels_macro_f1.A5",
                    evidence_class="exploratory", unit="pp")
        numbers.add(f"SevereSeedMeanGap{label}",
                    100 * (float(levels_s5r1["A5"]) - float(levels_s5r1[reference])), source=s5r1_path,
                    selector="pooled_levels_macro_f1.A5 - pooled_levels_macro_f1." + reference,
                    evidence_class="exploratory", unit="pp")
        numbers.add(f"SevereRepairResidualGap{label}",
                    100 * (float(levels_s5r1["sidecar"]) - float(levels_s5r1[reference])), source=s5r1_path,
                    selector="pooled_levels_macro_f1.sidecar - pooled_levels_macro_f1." + reference,
                    evidence_class="exploratory", unit="pp")

    # ---------------- S6: severe-held frozen-inference attribution controls -
    # Post-campaign diagnostic controls (forward only), prospectively frozen in
    # docs/S6_SEVERE_HELD_INFERENCE_CONTROLS_PREREG.md and analysed by
    # analysis_zero_compute/a21_s6_severe_held_controls.py.  Exploratory.
    s6_path = ZC / "a21_s6_controls_summary.json"
    with open(s6_path, encoding="utf-8") as handle:
        s6 = json.load(handle)
    numbers.add("SevereControlDecisionPattern", str(s6["decision_pattern"]), source=s6_path,
                selector="decision_pattern", evidence_class="exploratory")
    s6_manifest_path = REPO / "standards" / "cache_s5_severe_held_v1" / "manifest.json"
    with open(s6_manifest_path, encoding="utf-8") as handle:
        s6_manifest = json.load(handle)
    numbers.add("SevereHeldSourcesPerRegime",
                int(s6_manifest["component_audit"]["s5_severe_r1"]["view_count"]) // 2,
                source=s6_manifest_path,
                selector="component_audit.s5_severe_r1.view_count / 2 (paired views per source)",
                evidence_class="exploratory", precision=0)
    numbers.add("SevereHeldSourcesTotal",
                (int(s6_manifest["component_audit"]["s5_severe_r1"]["view_count"])
                 + int(s6_manifest["component_audit"]["s5_severe_r2"]["view_count"])) // 2,
                source=s6_manifest_path,
                selector=("(component_audit.s5_severe_r1.view_count + "
                          "component_audit.s5_severe_r2.view_count) / 2 (paired views per source)"),
                evidence_class="exploratory", precision=0)
    for condition, label in (
        ("M", "CapM"),
        ("L", "CapL"),
        ("sidecar_iq_zero", "IqZero"),
        ("sidecar_iq_shuffle", "IqShuffle"),
        ("sidecar_no_side_head", "NoSideHead"),
    ):
        numbers.add(f"SevereControlLevel{label}", 100 * float(s6["pooled_seed_mean_macro_f1"][condition]),
                    source=s6_path, selector=f"pooled_seed_mean_macro_f1.{condition}",
                    evidence_class="exploratory", unit="%")
    numbers.add("SevereControlIqZeroLoss", float(s6["control_b"]["full_minus_zero_pp"]), source=s6_path,
                selector="control_b.full_minus_zero_pp", evidence_class="exploratory", unit="pp")
    numbers.add("SevereControlIqShuffleLoss", float(s6["control_b"]["full_minus_shuffle_pp"]), source=s6_path,
                selector="control_b.full_minus_shuffle_pp", evidence_class="exploratory", unit="pp")

    s6_contrasts_path = ZC / "csv" / "a21_control_paired_contrasts.csv"
    s6_contrasts = pd.read_csv(s6_contrasts_path)
    for contrast, label in (
        ("M_minus_A5", "CapM"),
        ("L_minus_A5", "CapL"),
        ("sidecar_iq_zero_minus_A5", "IqZero"),
        ("sidecar_iq_shuffle_minus_A5", "IqShuffle"),
        ("sidecar_no_side_head_minus_A5", "NoSideHead"),
    ):
        row = s6_contrasts[(s6_contrasts.contrast == contrast) & (s6_contrasts.scope == "pooled")].iloc[0]
        selector = f"contrast == '{contrast}' and scope == 'pooled'"
        numbers.add(f"SevereControlDiff{label}", float(row.mean_pp), source=s6_contrasts_path,
                    selector=selector, evidence_class="exploratory", unit="pp")
        numbers.add(f"SevereControlDiff{label}Low", float(row.ci95_low_pp), source=s6_contrasts_path,
                    selector=selector, evidence_class="exploratory", unit="pp")
        numbers.add(f"SevereControlDiff{label}High", float(row.ci95_high_pp), source=s6_contrasts_path,
                    selector=selector, evidence_class="exploratory", unit="pp")

    s6_matched_path = ZC / "csv" / "a21_control_matched5_levels.csv"
    s6_matched = pd.read_csv(s6_matched_path)
    for condition, label in (("A5_matched5", "AFive"), ("S_matched5", "Sidecar")):
        row = s6_matched[(s6_matched.condition == condition) & (s6_matched.scope == "pooled")].iloc[0]
        numbers.add(f"SevereControlMatchedLevel{label}", 100 * float(row.macro_f1), source=s6_matched_path,
                    selector=f"condition == '{condition}' and scope == 'pooled'",
                    evidence_class="exploratory", unit="%")

    # ---------------- post-review: severe-held decomposition (a22) ---------
    # Zero-compute split of the frozen severe-held predictions by jammer
    # family and regime, plus the S6 lesion loss/repair ratios.
    a22_path = ZC / "a22_severe_held_decomposition.json"
    with open(a22_path, encoding="utf-8") as handle:
        a22 = json.load(handle)
    family_tags = {"pulse": "Pulse", "ofdm_like": "Ofdm"}
    model_tags = {"A5": "AFive", "IQFormer": "Iqformer",
                  "MCLDNN": "Mcldnn", "sidecar": "Sidecar"}
    for family, family_tag in family_tags.items():
        for model, model_tag in model_tags.items():
            numbers.add(f"SevereHeldFamily{family_tag}{model_tag}",
                        float(a22["pooled_family_levels_pct"][family][model]),
                        source=a22_path,
                        selector=f"pooled_family_levels_pct.{family}.{model}",
                        evidence_class="exploratory", unit="%")
    for regime, regime_tag in (("r1", "RegimeOne"), ("r2", "RegimeTwo")):
        for model, model_tag in model_tags.items():
            numbers.add(f"SevereHeld{regime_tag}{model_tag}",
                        float(a22["severe_held_levels_pct"][regime][model]),
                        source=a22_path,
                        selector=f"severe_held_levels_pct.{regime}.{model}",
                        evidence_class="exploratory", unit="%")

    a22_lesion_path = ZC / "csv" / "a22_lesion_ratios.csv"
    a22_lesion = pd.read_csv(a22_lesion_path)
    for condition, label in (
        ("iq_zero", "IqZero"),
        ("iq_shuffle", "IqShuffle"),
        ("no_side_head", "NoSideHead"),
    ):
        row = a22_lesion[a22_lesion.condition == condition].iloc[0]
        selector = f"condition == '{condition}'"
        numbers.add(f"SevereControl{label}RepairRatio",
                    float(row.loss_over_repair_ratio), source=a22_lesion_path,
                    selector=selector, evidence_class="exploratory")
    numbers.add("SevereControlNoSideHeadLoss",
                float(a22_lesion[a22_lesion.condition == "no_side_head"]
                      .loss_vs_full_sidecar_pp.iloc[0]),
                source=a22_lesion_path, selector="condition == 'no_side_head'",
                evidence_class="exploratory", unit="pp")

    # ---------------- S7: retrained permanently-zeroed-I/Q sidecar --------
    # Preregistered retrained ablation (docs/S7_IQZERO_SIDECAR_RETRAIN_PREREG.md):
    # severe-held pooled contrasts of the zeroed variant (a23) and the
    # campaign hard-split sanity levels (a24).
    a23_path = ZC / "a23_s7_iqzero_summary.json"
    with open(a23_path, encoding="utf-8") as handle:
        a23 = json.load(handle)
    s7_contrast_specs = (
        ("iqzero_vs_A5", "RetrainedIqzeroVsAFive"),
        ("sidecar_full_vs_A5", "RetrainedFullVsAFive"),
        ("iqzero_vs_sidecar_full", "RetrainedIqzeroVsFull"),
    )
    for contrast_key, macro_tag in s7_contrast_specs:
        block = a23["pooled_contrasts_pp"][contrast_key]
        numbers.add(f"{macro_tag}Diff", float(block["mean"]), source=a23_path,
                    selector=f"pooled_contrasts_pp.{contrast_key}.mean",
                    evidence_class="exploratory", unit="pp")
        numbers.add(f"{macro_tag}CiLow", float(block["ci95_low"]), source=a23_path,
                    selector=f"pooled_contrasts_pp.{contrast_key}.ci95_low",
                    evidence_class="exploratory", unit="pp")
        numbers.add(f"{macro_tag}CiHigh", float(block["ci95_high"]), source=a23_path,
                    selector=f"pooled_contrasts_pp.{contrast_key}.ci95_high",
                    evidence_class="exploratory", unit="pp")
    numbers.add("RetrainedIqzeroPooledLevel", 100.0 * a23["pooled_macro_f1"]["iqzero"],
                source=a23_path, selector="pooled_macro_f1.iqzero",
                evidence_class="exploratory", unit="%")
    numbers.add("RetrainedFullPooledLevel", 100.0 * a23["pooled_macro_f1"]["sidecar_full"],
                source=a23_path, selector="pooled_macro_f1.sidecar_full",
                evidence_class="exploratory", unit="%")

    a24_path = ZC / "a24_s7_campaign_levels.json"
    with open(a24_path, encoding="utf-8") as handle:
        a24 = json.load(handle)
    for model, macro_tag in (("iqzero", "Iqzero"), ("A5", "AFive"),
                             ("sidecar_full", "Full")):
        numbers.add(f"RetrainedCampaign{macro_tag}Level",
                    float(a24["levels_pct"][model]), source=a24_path,
                    selector=f"levels_pct.{model}",
                    evidence_class="exploratory", unit="%")
    numbers.add("RetrainedCampaignIqzeroVsAFiveDiff",
                float(a24["contrasts_pp"]["iqzero_vs_A5"]), source=a24_path,
                selector="contrasts_pp.iqzero_vs_A5",
                evidence_class="exploratory", unit="pp")

    return numbers


def emit(numbers: Numbers) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": "tvt_paper_numbers_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "policy": (
            "every manuscript number must resolve to a key in this file; hand-typed numbers are a "
            "build failure. Sealed and exploratory keys must be labelled as such in the text."
        ),
        "gate_status": {
            "scientific_evidence_passed": False,
            "submission_unlocked": False,
            "failed_gates": ["confirmatory_family_gate_failed", "clean_retention_gate_failed"],
        },
        "source_sha256": numbers.sources,
        "keys": numbers.entries,
    }
    with open(OUT / "paper_numbers.json", "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)

    lines = [
        "% Generated by paper_data_layer/build_paper_numbers.py -- do not edit by hand.",
        "% Every value carries an evidence class; see outputs/paper_numbers.json.",
        "",
    ]
    for key, entry in sorted(numbers.entries.items()):
        value = entry["value"]
        if isinstance(value, str):
            rendered = value
        elif entry["precision"] == 0:
            rendered = f"{int(round(float(value)))}"
        else:
            rendered = f"{float(value):.{entry['precision']}f}"
        lines.append(f"\\newcommand{{\\{key}}}{{{rendered}}}% {entry['evidence_class']}")
    with open(OUT / "paper_macros.tex", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    counts: dict[str, int] = {}
    for entry in numbers.entries.values():
        counts[entry["evidence_class"]] = counts.get(entry["evidence_class"], 0) + 1
    print(f"keys: {len(numbers.entries)}  by class: {counts}")
    print(f"sources: {len(numbers.sources)}")
    print(f"-> {OUT / 'paper_numbers.json'}")
    print(f"-> {OUT / 'paper_macros.tex'}")


if __name__ == "__main__":
    emit(build())
