"""Recover one missing V4 CSSL fit in an isolated staging directory.

This worker is deliberately not a replacement for the formal runner.  It is a
technical-retry tool for the native CUDA abort recorded after 110/120 V4 fits.
It trains exactly one preregistered CSSL model/seed, writes no cross-model
statistics, and publishes nothing: the caller owns validation and the atomic
rename of the staging directory.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterator

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CANONICAL_RECOVERY_CONFIG = (
    ROOT
    / "tvt_submission"
    / "configs"
    / "formal_tvt_recovery_v4r_110plus10.json"
).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments import run_standard_experiment as runner  # noqa: E402
import vimd_amc.training as training_module  # noqa: E402
from vimd_amc.reproducibility import (  # noqa: E402
    TVT_V2_SOURCE_PROFILE,
    source_tree_record,
)


WORKER_SCHEMA = "vimd_amc.tvt.recovery_worker.v4r"
RECOVERY_CONFIG_SCHEMA = "vimd_amc.tvt.formal_recovery.v4r_110plus10"
EXPECTED_RECOVERY_CONFIG_SHA256 = (
    "e88926f3c4d0958074524665b9543734fc89455d0b463a967ae70da382cbb916"
)
RECOVERY_MODEL = "cssl_amc_supervised_adaptation"
EXPECTED_PREDICTION_SPLITS = (
    "validation",
    "id_test",
    "hard_interference",
    "unseen_jammer",
    "unseen_speed",
    "heldout_channel",
    "combined_ood",
    "clean_retention",
    "adc_10bit_agc",
    "adc_12bit_agc",
    "per_emitter_sync",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            value[key] = item
        return value

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            runner.json_safe(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _root_path(relative: str, label: str) -> Path:
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository root") from error
    return path


def _artifact(path: Path, staging: Path) -> dict[str, Any]:
    resolved = path.resolve()
    relative = resolved.relative_to(staging.resolve()).as_posix()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise RuntimeError(f"worker artifact is absent or empty: {relative}")
    return {
        "path": relative,
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: actual={actual!r}, expected={expected!r}")


def _current_source_record() -> dict[str, Any]:
    return source_tree_record(
        ROOT,
        ROOT / "experiments" / "run_standard_experiment.py",
        profile=TVT_V2_SOURCE_PROFILE,
    )


def _scientific_configs(
    config: dict[str, Any],
    source_run: dict[str, Any],
) -> tuple[Any, Any]:
    invariant = config["scientific_invariants"]
    model_config = runner.make_model_config(
        sample_length=int(source_run["sample_length"]),
        n_fft=int(invariant["n_fft"]),
        hop_length=int(invariant["hop_length"]),
        spectral_channels=int(invariant["spectral_channels"]),
        embedding_dim=int(invariant["embedding_dim"]),
        environment_dim=int(invariant["environment_dim"]),
        dropout=float(invariant["dropout"]),
    )
    training_config = runner.TrainingConfig(
        epochs=int(invariant["epochs"]),
        batch_size=int(invariant["training_batch_size"]),
        learning_rate=float(invariant["learning_rate"]),
        weight_decay=float(invariant["weight_decay"]),
        mask_start_epoch=int(invariant["mask_start_epoch"]),
        contrastive_start_epoch=int(invariant["contrastive_start_epoch"]),
        mask_ramp_epochs=int(invariant["mask_ramp_epochs"]),
        contrastive_ramp_epochs=int(invariant["contrastive_ramp_epochs"]),
        minimum_full_stage_epochs=int(invariant["minimum_full_stage_epochs"]),
        patience=int(invariant["patience"]),
        use_amp=bool(invariant["use_amp"]),
    )
    runner.validate_training_config(training_config)
    _require_equal(
        asdict(model_config), source_run["model_configuration"],
        "model configuration",
    )
    _require_equal(
        asdict(training_config), source_run["training_configuration"],
        "training configuration",
    )
    _require_equal(
        int(invariant["bootstrap_draws"]),
        int(source_run["comparison_protocol"]["bootstrap_draws"]),
        "bootstrap draws",
    )
    _require_equal(
        int(invariant["bootstrap_seed"]),
        int(source_run["comparison_protocol"]["bootstrap_seed_base"]),
        "bootstrap seed",
    )
    return model_config, training_config


def _validate_bindings(
    config_path: Path,
    seed: int,
) -> tuple[dict[str, Any], str, Path, dict[str, Any], str, dict[str, Any]]:
    _require_equal(
        config_path.resolve(),
        CANONICAL_RECOVERY_CONFIG,
        "canonical V4R recovery config path",
    )
    config = _read_json(config_path, "V4R recovery config")
    _require_equal(config.get("schema_version"), RECOVERY_CONFIG_SCHEMA, "config schema")
    config_sha = _sha256(config_path)
    _require_equal(
        config_sha,
        EXPECTED_RECOVERY_CONFIG_SHA256,
        "canonical V4R recovery config SHA-256",
    )
    source_spec = config["source_run"]
    run_path = _root_path(str(source_spec["run_json"]), "source run.json")
    run_sha = _sha256(run_path)
    _require_equal(
        run_sha,
        str(source_spec["run_json_sha256_at_seal"]),
        "sealed source run.json SHA-256",
    )
    source_run = _read_json(run_path, "sealed source run.json")
    _require_equal(source_run.get("run_id"), source_spec["run_id"], "source run id")
    _require_equal(len(source_run.get("results", [])), int(source_spec["completed_fit_count"]), "completed fit count")
    _require_equal(source_run.get("cache_digest"), source_spec["cache_digest"], "source cache digest")

    planned_model = str(config["missing_fit_plan"]["model"])
    _require_equal(planned_model, RECOVERY_MODEL, "recovery model")
    planned_seeds = [int(value) for value in config["missing_fit_plan"]["seeds"]]
    if seed not in planned_seeds:
        raise ValueError(f"seed {seed} is not in the frozen recovery plan")
    observed = {
        (str(result.get("model")), int(result.get("seed")))
        for result in source_run["results"]
    }
    if (RECOVERY_MODEL, seed) in observed:
        raise ValueError(f"source V4 run already contains {RECOVERY_MODEL}/seed{seed}")
    expected_all = {
        (str(model), int(item_seed))
        for model in source_run["models"]
        for item_seed in source_run["seeds"]
    }
    missing = expected_all.difference(observed)
    expected_missing = {(RECOVERY_MODEL, item_seed) for item_seed in planned_seeds}
    _require_equal(missing, expected_missing, "source-run missing-fit set")

    source_record = _current_source_record()
    bound_record = source_run["environment"]["source_tree"]
    _require_equal(source_record, bound_record, "current TVT source-tree record")
    _require_equal(
        source_record["aggregate_digest"],
        source_spec["source_tree_aggregate_digest"],
        "source-tree aggregate digest",
    )
    for label, path_key, sha_key in (
        ("V4 freeze", "v4_freeze", "v4_freeze_sha256"),
        ("learning evidence", "learning_evidence", "learning_evidence_sha256"),
        ("headline cache manifest", "headline_cache_manifest", "headline_cache_manifest_sha256"),
    ):
        bound_path = _root_path(str(config["bindings"][path_key]), label)
        _require_equal(_sha256(bound_path), str(config["bindings"][sha_key]), f"{label} SHA-256")
    return config, config_sha, run_path, source_run, run_sha, source_record


@contextmanager
def _unpinned_training_loaders() -> Iterator[None]:
    """Force only training.py loaders to use pageable host memory.

    V4's scientific parameters and minibatch order are unchanged.  This is a
    recorded technical stability measure for the observed CachingHostAllocator
    native abort.  Evaluation already uses unpinned loaders in the bound source.
    """

    original = training_module.DataLoader

    def unpinned_loader(*args: Any, **kwargs: Any) -> Any:
        kwargs["pin_memory"] = False
        return original(*args, **kwargs)

    training_module.DataLoader = unpinned_loader
    try:
        yield
    finally:
        training_module.DataLoader = original


def _validate_worker_outputs(
    staging: Path,
    result: dict[str, Any],
    model: torch.nn.Module,
    expected_splits: tuple[str, ...],
    cache_digest: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    _require_equal(result["model"], RECOVERY_MODEL, "result model")
    _require_equal(int(result["seed"]), seed, "result seed")
    _require_equal(set(result["regimes"]), set(expected_splits), "result regimes")
    checkpoint_path = staging / str(result["checkpoint"])
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected_state = model.state_dict()
    _require_equal(set(state), set(expected_state), "checkpoint state keys")
    for key in state:
        _require_equal(tuple(state[key].shape), tuple(expected_state[key].shape), f"checkpoint tensor shape {key}")

    predictions: dict[str, dict[str, Any]] = {}
    for split in expected_splits:
        path = staging / "model" / f"predictions_{split}.npz"
        with np.load(path, allow_pickle=False) as bundle:
            required = {
                "probabilities", "labels", "source_ids", "snr_db", "sir_db",
                "target_profile_index", "cache_digest", "split",
            }
            _require_equal(set(bundle.files), required, f"prediction fields for {split}")
            _require_equal(str(bundle["cache_digest"].item()), cache_digest, f"prediction cache digest for {split}")
            _require_equal(str(bundle["split"].item()), split, f"prediction split label for {split}")
            if int(bundle["labels"].shape[0]) <= 0:
                raise RuntimeError(f"prediction bundle is empty for {split}")
            _require_equal(int(bundle["probabilities"].shape[0]), int(bundle["labels"].shape[0]), f"prediction row count for {split}")
        predictions[split] = _artifact(path, staging)
    return _artifact(checkpoint_path, staging), predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--staging-directory", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config_path = arguments.config.resolve()
    (
        config,
        config_sha,
        source_run_path,
        source_run,
        source_run_sha,
        source_record,
    ) = _validate_bindings(config_path, arguments.seed)
    model_config, training_config = _scientific_configs(config, source_run)
    factories = runner.available_model_factories()
    if RECOVERY_MODEL not in factories:
        raise RuntimeError(
            f"bound recovery model factory is unavailable: {RECOVERY_MODEL}"
        )
    cache_root = _root_path(str(source_run["cache_root"]), "headline cache root")
    contract = runner.inspect_cache_contract(cache_root)
    _require_equal(contract.cache_digest, source_run["cache_digest"], "live cache digest")
    expected_splits = tuple(runner.evaluation_split_names(contract))
    _require_equal(set(expected_splits), set(EXPECTED_PREDICTION_SPLITS), "formal prediction split set")

    preflight = {
        "schema_version": WORKER_SCHEMA,
        "status": "preflight_passed",
        "seed": int(arguments.seed),
        "model": RECOVERY_MODEL,
        "run_id": config["missing_fit_plan"]["worker_run_id_template"].format(seed=arguments.seed),
        "recovery_config_sha256": config_sha,
        "source_run_json_sha256": source_run_sha,
        "cache_digest": contract.cache_digest,
        "source_tree_aggregate_digest": source_record["aggregate_digest"],
        "prediction_splits": list(expected_splits),
        "technical_stability_measure": {
            "training_dataloader_pin_memory": False,
            "reason": "fresh-process retry after native CachingHostAllocator abort",
            "scientific_hyperparameters_changed": False,
        },
    }
    if arguments.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    staging = arguments.staging_directory.resolve()
    try:
        staging.relative_to(ROOT)
    except ValueError as error:
        raise ValueError("staging directory must be inside the repository") from error
    if staging.exists():
        if not staging.is_dir() or any(staging.iterdir()):
            raise FileExistsError(f"staging directory is not empty: {staging}")
    else:
        staging.mkdir(parents=True, exist_ok=False)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = runner.resolve_device("cuda")
    datasets = runner.load_cache_datasets(contract, verify_checksums=True)
    try:
        support = runner.jammer_training_support_contract(contract, datasets["train"])
        runner.seed_everything(arguments.seed)
        built = factories[RECOVERY_MODEL](
            len(contract.modulations), contract.num_jammers, model_config
        )
        model = built.model
        started = time.perf_counter()
        with _unpinned_training_loaders():
            training_result = runner.train_model(
                model=model,
                teacher=built.teacher,
                train_dataset=datasets["train"],
                validation_dataset=datasets["validation"],
                device=device,
                seed=arguments.seed,
                config=training_config,
                objective=built.objective,
                loss_weights=built.loss_weights,
                jammer_support_mask=torch.tensor(support["support_mask"], dtype=torch.float32),
            )
        training_seconds = time.perf_counter() - started
        model_directory = staging / "model"
        model_directory.mkdir(parents=True, exist_ok=False)
        checkpoint_path = model_directory / "model.pt"
        torch.save(model.state_dict(), checkpoint_path)
        result: dict[str, Any] = {
            "model": RECOVERY_MODEL,
            "model_class": model.__class__.__name__,
            "model_provenance": runner.model_provenance(model),
            "ablation_protocol": runner.paper_ablation_protocol(RECOVERY_MODEL),
            "objective": asdict(built.objective),
            "loss_weights": asdict(built.loss_weights),
            "jammer_auxiliary_training_support": (
                support if built.objective.use_jammer_auxiliary else {
                    "status": "not_applicable",
                    "reason": "objective has no jammer auxiliary loss",
                }
            ),
            "seed": int(arguments.seed),
            "training_seconds": training_seconds,
            "training": training_result,
            "checkpoint": "model/model.pt",
            "teacher_checkpoint": None,
            "complexity": runner.complexity_metrics(
                model,
                sample_length=contract.sample_length,
                device=device,
                latency_runs=30,
            ),
            "regimes": {},
        }
        for split in expected_splits:
            bundle, metrics = runner.predict(
                model,
                datasets[split],
                device=device,
                batch_size=training_config.batch_size,
            )
            if bundle.target_profile_index is None:
                raise RuntimeError(f"prediction bundle lacks target_profile_index: {split}")
            result["regimes"][split] = metrics
            np.savez_compressed(
                model_directory / f"predictions_{split}.npz",
                probabilities=bundle.probabilities,
                labels=bundle.labels,
                source_ids=bundle.source_ids,
                snr_db=bundle.snr_db,
                sir_db=bundle.sir_db,
                target_profile_index=bundle.target_profile_index,
                cache_digest=np.asarray(contract.cache_digest),
                split=np.asarray(split),
            )
        probe_x = datasets["validation"][0]["view1"]["x"].unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            probe_output = model(probe_x)
        available_heads = [
            name for name, key in (("jammer_multilabel", "jam_logits"), ("quality", "quality"))
            if key in probe_output
        ]
        if available_heads:
            raise RuntimeError("CSSL recovery unexpectedly exposed auxiliary heads")
        result["auxiliary_metrics"] = {
            "status": "unavailable",
            "reason": "model output has neither jam_logits nor quality auxiliary head",
            "used_for_checkpoint_or_model_selection": False,
            "available_heads": [],
            "regimes": {},
        }
        result_path = model_directory / "result.json"
        _write_json(result_path, result)
        checkpoint_artifact, predictions = _validate_worker_outputs(
            staging,
            result,
            model,
            expected_splits,
            contract.cache_digest,
            arguments.seed,
        )
    finally:
        for dataset in datasets.values():
            dataset.close()

    _require_equal(_sha256(config_path), config_sha, "recovery config stability")
    _require_equal(_sha256(source_run_path), source_run_sha, "source run.json stability")
    _require_equal(_current_source_record(), source_record, "source-tree stability")
    worker_manifest = {
        **preflight,
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "result_json": _artifact(result_path, staging),
        "checkpoint": checkpoint_artifact,
        "predictions": predictions,
        "scientific_parameters": {
            "model_configuration": asdict(model_config),
            "training_configuration": asdict(training_config),
            "bootstrap_draws": int(config["scientific_invariants"]["bootstrap_draws"]),
            "bootstrap_seed": int(config["scientific_invariants"]["bootstrap_seed"]),
        },
        "publication": "staging_only_atomic_commit_owned_by_scheduler",
    }
    _write_json(staging / "worker_manifest.json", worker_manifest)
    print(json.dumps(worker_manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
