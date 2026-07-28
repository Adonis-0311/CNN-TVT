"""Run bounded, reproducible proxy experiments and emit auditable artifacts."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vimd_amc.data.dataset import PairedAMCDataset, Regime  # noqa: E402
from vimd_amc.data.split import assert_disjoint_source_ids  # noqa: E402
from vimd_amc.data.synthesis import SignalSynthesizer, SynthesisConfig  # noqa: E402
from vimd_amc.ablation import (  # noqa: E402
    MODEL_ALIASES,
    PAPER_ABLATION_NAMES,
    PAPER_ABLATION_PROTOCOLS,
    canonical_model_name,
)
from vimd_amc.evaluation import (  # noqa: E402
    complexity_metrics,
    mechanism_metrics,
    predict,
)
from vimd_amc.metrics import holm_adjust, paired_bundle_statistics  # noqa: E402
from vimd_amc.models.baselines import (  # noqa: E402
    BackboneClassifier,
    CSSLAMCSupervisedAdaptation,
    IQFormerInspiredClassifier,
    MCLDNNReimplementation,
    SingleMaskClassifier,
)
from vimd_amc.models.common import ModelConfig  # noqa: E402
from vimd_amc.models.vimd import (  # noqa: E402
    DualMaskVIMDNet,
    PhysicalDualMaskTeacher,
    PhysicalTriMaskTeacher,
    VIMDNet,
)
from vimd_amc.reproducibility import source_tree_record  # noqa: E402
from vimd_amc.losses import VIMDLossWeights  # noqa: E402
from vimd_amc.training import (  # noqa: E402
    TrainingConfig,
    TrainingObjective,
    seed_everything,
    train_model,
)


PROFILES: dict[str, dict[str, Any]] = {
    "smoke": {
        "sample_length": 128,
        "modulations": ("BPSK", "QPSK", "8PSK", "16QAM"),
        "train_size": 192,
        "validation_size": 96,
        "test_size": 128,
        "cache_in_memory": True,
        "model": ModelConfig(
            feature_channels=32,
            embedding_dim=48,
            spectral_channels=24,
            n_fft=32,
            hop_length=8,
            dropout=0.05,
        ),
        "training": TrainingConfig(
            epochs=3,
            batch_size=32,
            mask_start_epoch=1,
            contrastive_start_epoch=2,
            mask_ramp_epochs=1,
            contrastive_ramp_epochs=1,
            minimum_full_stage_epochs=1,
            patience=3,
            use_amp=False,
        ),
    },
    "dev": {
        "sample_length": 256,
        "modulations": ("BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "GMSK"),
        "train_size": 4_096,
        "validation_size": 1_024,
        "test_size": 1_024,
        "cache_in_memory": True,
        "model": ModelConfig(
            feature_channels=64,
            embedding_dim=96,
            spectral_channels=48,
            n_fft=64,
            hop_length=16,
            dropout=0.10,
        ),
        "training": TrainingConfig(
            epochs=14,
            batch_size=64,
            mask_start_epoch=2,
            contrastive_start_epoch=6,
            patience=5,
            use_amp=True,
        ),
    },
    "pilot": {
        "sample_length": 512,
        "modulations": (
            "BPSK",
            "PI2BPSK",
            "QPSK",
            "8PSK",
            "16QAM",
            "64QAM",
            "GMSK",
            "CPFSK",
            "4FSK",
        ),
        "train_size": 20_000,
        "validation_size": 4_000,
        "test_size": 4_000,
        "cache_in_memory": False,
        "model": ModelConfig(
            feature_channels=96,
            embedding_dim=128,
            spectral_channels=64,
            n_fft=64,
            hop_length=16,
            dropout=0.10,
        ),
        "training": TrainingConfig(
            epochs=30,
            batch_size=64,
            mask_start_epoch=4,
            contrastive_start_epoch=12,
            patience=8,
            use_amp=True,
        ),
    },
}

CANONICAL_MODELS = (
    *PAPER_ABLATION_NAMES,
    "mcldnn_reimplementation",
    "iqformer_inspired",
    "cssl_amc_supervised_adaptation",
)


@dataclass
class BuiltExperimentModel:
    model: nn.Module
    teacher: nn.Module | None
    objective: TrainingObjective
    loss_weights: VIMDLossWeights
    protocol: dict[str, Any]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=tuple(PROFILES), default="smoke")
    parser.add_argument(
        "--models",
        default="backbone,single_mask,vimd",
        help=(
            "Comma-separated model names. Canonical A0--A7 names: "
            + ",".join(CANONICAL_MODELS)
            + ". Legacy aliases backbone,single_mask,dual_mask,vimd are retained."
        ),
    )
    parser.add_argument("--seeds", default="17", help="Comma-separated model seeds")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--data-seed", type=int, default=20260727)
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=1,
        help="PyTorch intra-op CPU threads; one is fastest for the bounded profiles",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def build_model(
    name: str,
    *,
    classes: int,
    jammers: int,
    config: ModelConfig,
) -> BuiltExperimentModel:
    canonical = canonical_model_name(name)
    if canonical not in CANONICAL_MODELS:
        choices = ", ".join((*CANONICAL_MODELS, *MODEL_ALIASES))
        raise ValueError(f"Unknown model {name!r}; available names: {choices}")

    weights = VIMDLossWeights()
    classification = TrainingObjective(
        name="paired_view_modulation_ce",
        loss_family="classification",
    )
    internal_source = {
        "claim_level": "internal controlled ablation",
        "official_code_url": None,
        "paper_url": None,
    }
    if canonical == "a0_backbone":
        model = BackboneClassifier(classes, config)
        return BuiltExperimentModel(
            model,
            None,
            classification,
            weights,
            {
                **internal_source,
                "ablation_id": "A0",
                "display_name": "shared spectral backbone",
                "mask_routes": 0,
                "enabled_components": ["modulation_ce"],
            },
        )
    if canonical == "a1_single_mask":
        model = SingleMaskClassifier(classes, config)
        return BuiltExperimentModel(
            model,
            None,
            classification,
            weights,
            {
                **internal_source,
                "ablation_id": "A1",
                "display_name": "single target-mask ablation",
                "mask_routes": 1,
                "enabled_components": ["modulation_ce"],
            },
        )
    if canonical == "mcldnn_reimplementation":
        model = MCLDNNReimplementation(classes)
        return BuiltExperimentModel(
            model,
            None,
            classification,
            weights,
            dict(model.provenance),
        )
    if canonical == "iqformer_inspired":
        model = IQFormerInspiredClassifier(classes)
        return BuiltExperimentModel(
            model,
            None,
            classification,
            weights,
            dict(model.provenance),
        )
    if canonical == "cssl_amc_supervised_adaptation":
        model = CSSLAMCSupervisedAdaptation(classes)
        return BuiltExperimentModel(
            model,
            None,
            classification,
            weights,
            dict(model.provenance),
        )

    if canonical == "a6_dual_full":
        model = DualMaskVIMDNet(classes, jammers, config)
        teacher: nn.Module | None = PhysicalDualMaskTeacher(config)
    else:
        model = VIMDNet(
            classes,
            jammers,
            config,
            use_residual=canonical != "a7_vimd_no_residual",
        )
        teacher = None
    zero_auxiliary = replace(
        weights,
        jammer=0.0,
        quality=0.0,
        contrastive=0.0,
        orthogonality=0.0,
    )
    if canonical == "a2_tri_no_teacher":
        return BuiltExperimentModel(
            model,
            None,
            TrainingObjective(
                name="tri_mask_no_teacher_no_mtl_no_xcc_with_residual",
                loss_family="vimd",
            ),
            replace(zero_auxiliary, mask=0.0),
            {
                **internal_source,
                "ablation_id": "A2",
                "display_name": "tri-mask without teacher/MTL/XCC, with residual",
                "mask_routes": 3,
                "residual_path": True,
                "enabled_components": ["modulation_ce"],
            },
        )
    if canonical == "a3_tri_teacher":
        teacher = PhysicalTriMaskTeacher(config)
        return BuiltExperimentModel(
            model,
            teacher,
            TrainingObjective(
                name="tri_mask_fixed_teacher",
                loss_family="vimd",
                use_mask_supervision=True,
            ),
            zero_auxiliary,
            {
                **internal_source,
                "ablation_id": "A3",
                "display_name": "A2 plus fixed physical teacher",
                "mask_routes": 3,
                "residual_path": True,
                "enabled_components": ["modulation_ce", "fixed_teacher"],
            },
        )
    if canonical == "a4_tri_teacher_mtl":
        teacher = PhysicalTriMaskTeacher(config)
        return BuiltExperimentModel(
            model,
            teacher,
            TrainingObjective(
                name="tri_mask_teacher_mtl_no_xcc",
                loss_family="vimd",
                use_mask_supervision=True,
                use_jammer_auxiliary=True,
                use_quality_auxiliary=True,
                use_orthogonality=True,
            ),
            replace(weights, contrastive=0.0),
            {
                **internal_source,
                "ablation_id": "A4",
                "display_name": "A3 plus multitask bundle, no XCC",
                "mask_routes": 3,
                "residual_path": True,
                "enabled_components": [
                    "modulation_ce",
                    "fixed_teacher",
                    "jammer_auxiliary",
                    "quality_auxiliary",
                    "branch_orthogonality",
                ],
            },
        )
    if canonical in {"a5_vimd_full", "a7_vimd_no_residual"}:
        teacher = PhysicalTriMaskTeacher(config)
    ablation_id = {
        "a5_vimd_full": "A5",
        "a6_dual_full": "A6",
        "a7_vimd_no_residual": "A7",
    }[canonical]
    route_count = 2 if canonical == "a6_dual_full" else 3
    residual_path = canonical != "a7_vimd_no_residual"
    return BuiltExperimentModel(
        model,
        teacher,
        TrainingObjective(
            name="full_vimd",
            loss_family="vimd",
            use_mask_supervision=True,
            use_jammer_auxiliary=True,
            use_quality_auxiliary=True,
            use_cross_condition_contrastive=True,
            use_orthogonality=True,
        ),
        weights,
        {
            **internal_source,
            "ablation_id": ablation_id,
            "display_name": {
                "a5_vimd_full": "full tri-mask VIMD-Net with residual",
                "a6_dual_full": "dual-mask full objective with residual",
                "a7_vimd_no_residual": "full tri-mask VIMD-Net without residual",
            }[canonical],
            "mask_routes": route_count,
            "residual_path": residual_path,
            "enabled_components": [
                "modulation_ce",
                "fixed_teacher",
                "jammer_auxiliary",
                "quality_auxiliary",
                "branch_orthogonality",
                "cross_condition_contrastive",
            ],
        },
    )


def build_datasets(
    profile: dict[str, Any],
    synthesizer: SignalSynthesizer,
    data_seed: int,
) -> tuple[PairedAMCDataset, PairedAMCDataset, dict[str, PairedAMCDataset]]:
    common = {
        "synthesizer": synthesizer,
        "master_seed": data_seed,
        "modulations": profile["modulations"],
        "cache_in_memory": profile["cache_in_memory"],
    }
    train = PairedAMCDataset(
        split="train",
        size=profile["train_size"],
        regime=Regime.train(),
        **common,
    )
    validation = PairedAMCDataset(
        split="validation",
        size=profile["validation_size"],
        regime=Regime.validation(),
        **common,
    )
    test_regimes = {
        "in_distribution": PairedAMCDataset(
            split="test",
            size=profile["test_size"],
            regime=Regime.in_distribution_test(),
            **common,
        ),
        "hard_interference": PairedAMCDataset(
            split="hard",
            size=profile["test_size"],
            regime=Regime.hard_interference(),
            **common,
        ),
        "unseen_jammer": PairedAMCDataset(
            split="unseen_jammer",
            size=profile["test_size"],
            regime=Regime.unseen_jammer(),
            **common,
        ),
        "unseen_speed": PairedAMCDataset(
            split="unseen_speed",
            size=profile["test_size"],
            regime=Regime.unseen_speed(),
            **common,
        ),
        "unseen_channel": PairedAMCDataset(
            split="unseen_channel",
            size=profile["test_size"],
            regime=Regime.unseen_channel(),
            **common,
        ),
        "unseen_speed_and_channel": PairedAMCDataset(
            split="unseen_speed_and_channel",
            size=profile["test_size"],
            regime=Regime.unseen_speed_and_channel(),
            **common,
        ),
        "clean_high_snr": PairedAMCDataset(
            split="clean",
            size=profile["test_size"],
            regime=Regime.clean_high_snr(),
            **common,
        ),
    }
    assert_disjoint_source_ids(
        train.source_ids(),
        validation.source_ids(),
        *(dataset.source_ids() for dataset in test_regimes.values()),
    )
    return train, validation, test_regimes


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def environment_record(device: torch.device) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "proxy_evidence_only": True,
        "channel_evidence_label": "controlled V2X-motivated heuristic proxy",
        "source_tree": source_tree_record(ROOT, Path(__file__)),
    }


def main() -> None:
    arguments = parse_arguments()
    if arguments.cpu_threads <= 0:
        raise ValueError("--cpu-threads must be positive")
    torch.set_num_threads(arguments.cpu_threads)
    torch.set_num_interop_threads(1)
    profile = PROFILES[arguments.profile]
    models = [name.strip() for name in arguments.models.split(",") if name.strip()]
    seeds = [int(seed) for seed in arguments.seeds.split(",") if seed.strip()]
    device = resolve_device(arguments.device)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_directory = arguments.output / f"{arguments.profile}_{run_id}"
    run_directory.mkdir(parents=True, exist_ok=False)

    synthesis_config = SynthesisConfig(sample_length=profile["sample_length"])
    synthesizer = SignalSynthesizer(synthesis_config)
    train_dataset, validation_dataset, test_regimes = build_datasets(
        profile,
        synthesizer,
        arguments.data_seed,
    )
    write_json(run_directory / "manifests" / "train.json", train_dataset.manifest())
    write_json(
        run_directory / "manifests" / "validation.json",
        validation_dataset.manifest(),
    )
    for regime_name, dataset in test_regimes.items():
        write_json(run_directory / "manifests" / f"{regime_name}.json", dataset.manifest())

    run_record = {
        "run_id": run_id,
        "profile": arguments.profile,
        "profile_configuration": {
            **{
                key: value
                for key, value in profile.items()
                if key not in {"model", "training"}
            },
            "model": asdict(profile["model"]),
            "training": asdict(profile["training"]),
        },
        "models": models,
        "seeds": seeds,
        "data_seed": arguments.data_seed,
        "environment": environment_record(device),
        "comparison_protocol": {
            "data_realizations_shared_across_models": True,
            "optimizer_and_schedule_shared_across_models": True,
            "checkpoint_rule_shared_across_models": True,
            "model_initialization_seed_shared_by_reported_seed": True,
            "parameter_matched": False,
            "complexity_reported_per_model": True,
            "architecture_specific_hyperparameter_sweep_performed": False,
            "interpretation": (
                "Fixed-protocol comparison. Complexity is reported rather than "
                "claiming parameter matching; literature baselines require a "
                "separate tuned-protocol sensitivity check before final claims."
            ),
        },
        "paper_ablation_protocols": PAPER_ABLATION_PROTOCOLS,
        "results": [],
    }
    flat_rows: list[dict[str, Any]] = []
    prediction_bundles: dict[tuple[str, int, str], Any] = {}

    for model_name in models:
        for seed in seeds:
            # Model construction must occur after seeding; otherwise a reported
            # model seed controls minibatch order but not initialization.
            seed_everything(seed)
            built = build_model(
                model_name,
                classes=len(profile["modulations"]),
                jammers=len(synthesis_config.jammer_types),
                config=profile["model"],
            )
            model = built.model
            teacher = built.teacher
            start = time.perf_counter()
            training_result = train_model(
                model=model,
                teacher=teacher,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                device=device,
                seed=seed,
                config=profile["training"],
                objective=built.objective,
                loss_weights=built.loss_weights,
            )
            training_seconds = time.perf_counter() - start
            model_directory = run_directory / "models" / f"{model_name}_seed{seed}"
            model_directory.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_directory / "model.pt")
            if teacher is not None:
                torch.save(teacher.state_dict(), model_directory / "teacher.pt")

            result: dict[str, Any] = {
                "model": model_name,
                "canonical_model": canonical_model_name(model_name),
                "seed": seed,
                "protocol": {
                    **PAPER_ABLATION_PROTOCOLS.get(
                        canonical_model_name(model_name),
                        {},
                    ),
                    **built.protocol,
                },
                "objective": asdict(built.objective),
                "loss_weights": asdict(built.loss_weights),
                "training_seconds": training_seconds,
                "training": training_result,
                "complexity": complexity_metrics(
                    model,
                    sample_length=profile["sample_length"],
                    device=device,
                    latency_runs=30 if arguments.profile == "smoke" else 80,
                ),
                "regimes": {},
            }
            for regime_name, dataset in test_regimes.items():
                bundle, metrics = predict(
                    model,
                    dataset,
                    device=device,
                    batch_size=profile["training"].batch_size,
                )
                result["regimes"][regime_name] = metrics
                prediction_bundles[(model_name, seed, regime_name)] = bundle
                np.savez_compressed(
                    model_directory / f"predictions_{regime_name}.npz",
                    probabilities=bundle.probabilities,
                    labels=bundle.labels,
                    source_ids=bundle.source_ids,
                    snr_db=bundle.snr_db,
                    sir_db=bundle.sir_db,
                )
                flat_rows.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "regime": regime_name,
                        **{
                            key: value
                            for key, value in metrics.items()
                            if isinstance(value, (int, float))
                        },
                    }
                )
            if (
                built.objective.loss_family == "vimd"
                and teacher is not None
                and bool(getattr(model, "supports_tri_mechanism", False))
            ):
                result["mechanism"] = mechanism_metrics(
                    model,
                    teacher,
                    test_regimes["hard_interference"],
                    device=device,
                    batch_size=profile["training"].batch_size,
                    maximum_samples=profile["test_size"],
                )
            write_json(model_directory / "result.json", result)
            run_record["results"].append(result)

    metrics_frame = pd.DataFrame(flat_rows)
    metrics_frame.to_csv(run_directory / "metrics.csv", index=False)
    if len(seeds) > 1 and not metrics_frame.empty:
        aggregate_rows: list[dict[str, Any]] = []
        metric_names = (
            "accuracy",
            "macro_f1",
            "worst_recall",
            "nll",
            "ece",
        )
        for (model_name, regime_name), group in metrics_frame.groupby(["model", "regime"]):
            row: dict[str, Any] = {
                "model": model_name,
                "regime": regime_name,
                "seed_count": int(group["seed"].nunique()),
            }
            for metric_name in metric_names:
                values = group[metric_name].astype(float)
                row[f"{metric_name}_mean"] = float(values.mean())
                row[f"{metric_name}_std"] = float(values.std(ddof=1))
            aggregate_rows.append(row)
        pd.DataFrame(aggregate_rows).to_csv(
            run_directory / "seed_aggregates.csv",
            index=False,
        )

    paired_rows: list[dict[str, Any]] = []
    reference_model = None
    if "a0_backbone" in models:
        reference_model = "a0_backbone"
    elif "backbone" in models:
        reference_model = "backbone"
    if reference_model is not None:
        for candidate_name in models:
            if candidate_name == reference_model:
                continue
            for seed in seeds:
                for regime_name in test_regimes:
                    reference_key = (reference_model, seed, regime_name)
                    candidate_key = (candidate_name, seed, regime_name)
                    if reference_key not in prediction_bundles or candidate_key not in prediction_bundles:
                        continue
                    statistics = paired_bundle_statistics(
                        prediction_bundles[reference_key],
                        prediction_bundles[candidate_key],
                        draws=2_000 if arguments.profile == "smoke" else 10_000,
                        seed=arguments.data_seed + seed,
                    )
                    paired_rows.append(
                        {
                            "reference": reference_model,
                            "candidate": candidate_name,
                            "seed": seed,
                            "regime": regime_name,
                            **statistics,
                        }
                    )
    if paired_rows:
        adjusted = holm_adjust(
            np.asarray([row["exact_p_value"] for row in paired_rows], dtype=np.float64)
        )
        for row, adjusted_value in zip(paired_rows, adjusted):
            row["holm_adjusted_p_value"] = float(adjusted_value)
        pd.DataFrame(paired_rows).to_csv(
            run_directory / "paired_statistics.csv",
            index=False,
        )
    write_json(run_directory / "run.json", run_record)
    print(run_directory.resolve())


if __name__ == "__main__":
    main()
