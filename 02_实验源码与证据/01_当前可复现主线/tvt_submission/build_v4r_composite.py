"""Build and validate the V4R 110+10 composite evidence artifact.

The aborted V4 directory is an immutable input.  This tool will not create the
composite output until all ten independently sealed CSSL recovery workers pass
preflight.  A successful build materializes a new, atomic directory, derives
every per-fit metric from the 120 prediction grids, recomputes all runner
cross-model statistics, and invokes the locked V2 scientific gate over the
composite view.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments import run_standard_experiment as runner  # noqa: E402
from experiments.run_formal_tvt_v2 import bind_runner_contract  # noqa: E402
from tvt_submission import validate_v2_release as release  # noqa: E402
from tvt_submission.formal_v2_contract import load_contract  # noqa: E402
from vimd_amc.metrics import (  # noqa: E402
    classification_metrics,
    headline_paired_bootstrap,
    holm_adjust,
    paired_accuracy_macro_f1_bootstrap,
    paired_bundle_statistics,
)
from vimd_amc.reproducibility import source_tree_record  # noqa: E402


DEFAULT_CONFIG = (
    ROOT
    / "tvt_submission"
    / "configs"
    / "formal_tvt_recovery_v4r_110plus10.json"
)
COMPOSITE_SCHEMA = "vimd_amc.tvt.composite_recovery.v4r"
INVENTORY_SCHEMA = "vimd_amc.tvt.composite_recovery.inventory.v1"
SEAL_SCHEMA = "vimd_amc.tvt.composite_recovery.seal.v1"
WORKER_SCHEMA = "vimd_amc.tvt.recovery_worker.v4r"
REQUIRED_NPZ_KEYS = {
    "probabilities",
    "labels",
    "source_ids",
    "snr_db",
    "sir_db",
    "target_profile_index",
    "cache_digest",
    "split",
}


class CompositeError(RuntimeError):
    """Fail-closed recovery/composite contract violation."""


def _read_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise CompositeError(f"duplicate JSON key {key!r}: {path}")
            value[key] = item
        return value

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                CompositeError(f"non-finite JSON value {token}: {path}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CompositeError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise CompositeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(
            runner.json_safe(dict(payload)),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _workspace_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CompositeError(f"{label} must be a nonempty workspace path")
    path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise CompositeError(f"{label} escapes workspace: {value}") from error
    return path


def _contained_path(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CompositeError(f"{label}.path must be a nonempty string")
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise CompositeError(f"{label}.path escapes worker root: {value}") from error
    if not candidate.is_file() or candidate.is_symlink():
        raise CompositeError(f"{label}.path is not a regular file: {candidate}")
    return candidate


def _descriptor_file(
    root: Path,
    descriptor: Any,
    *,
    label: str,
) -> tuple[Path, str]:
    expected_keys = {"path", "sha256", "size_bytes"}
    if not isinstance(descriptor, dict) or set(descriptor) != expected_keys:
        raise CompositeError(
            f"{label} must contain exactly path, sha256, and size_bytes"
        )
    path = _contained_path(root, descriptor["path"], label=label)
    expected = descriptor["sha256"]
    if not isinstance(expected, str) or len(expected) != 64:
        raise CompositeError(f"{label}.sha256 is malformed")
    actual = _sha256(path)
    if actual != expected:
        raise CompositeError(f"{label} SHA-256 drift: {actual} != {expected}")
    size = descriptor["size_bytes"]
    if (
        not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 0
        or size != path.stat().st_size
    ):
        raise CompositeError(f"{label}.size_bytes does not bind the file")
    return path, actual


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _expected_grid(run: Mapping[str, Any]) -> list[tuple[str, int]]:
    models = run.get("models")
    seeds = run.get("seeds")
    if not isinstance(models, list) or not all(isinstance(x, str) and x for x in models):
        raise CompositeError("source run model grid is malformed")
    if len(models) != len(set(models)):
        raise CompositeError("source run model grid contains duplicates")
    if not isinstance(seeds, list) or not all(
        isinstance(x, int) and not isinstance(x, bool) for x in seeds
    ):
        raise CompositeError("source run seed grid is malformed")
    if len(seeds) != len(set(seeds)):
        raise CompositeError("source run seed grid contains duplicates")
    return [(model, seed) for model in models for seed in seeds]


def _formal_ablation_seeds(headline_seeds: Sequence[int]) -> list[int]:
    """Return the exact freeze-bound confirmatory seeds from a headline grid."""

    available = [int(seed) for seed in headline_seeds]
    selected = [int(seed) for seed in runner.FORMAL_ABLATION_ALGORITHM_SEEDS]
    missing = [seed for seed in selected if seed not in available]
    if missing:
        raise CompositeError(
            "headline grid is missing preregistered ablation seeds: "
            f"{missing}"
        )
    return selected


def _index_results(results: Any, *, label: str) -> dict[tuple[str, int], dict[str, Any]]:
    if not isinstance(results, list):
        raise CompositeError(f"{label} results is not a list")
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            raise CompositeError(f"{label} result {index} is not an object")
        try:
            key = (str(result["model"]), int(result["seed"]))
        except (KeyError, TypeError, ValueError) as error:
            raise CompositeError(f"{label} result {index} has invalid identity") from error
        if key in indexed:
            raise CompositeError(f"{label} duplicate result {key[0]}/seed{key[1]}")
        indexed[key] = result
    return indexed


def _evaluation_splits(run: Mapping[str, Any]) -> list[str]:
    splits = run.get("splits")
    if not isinstance(splits, dict):
        raise CompositeError("source run splits is malformed")
    names = [str(split) for split in splits if str(split) != "train"]
    if not names or "validation" not in names:
        raise CompositeError("source run evaluation split grid is incomplete")
    return names


def _validate_prediction(
    path: Path,
    *,
    split: str,
    cache_digest: str,
) -> None:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != REQUIRED_NPZ_KEYS:
                raise CompositeError(
                    f"prediction key drift {path}: {sorted(archive.files)}"
                )
            observed_digest = str(np.asarray(archive["cache_digest"]).item())
            observed_split = str(np.asarray(archive["split"]).item())
            if observed_digest != cache_digest or observed_split != split:
                raise CompositeError(f"prediction binding mismatch: {path}")
            row_count = int(np.asarray(archive["labels"]).shape[0])
            if row_count <= 0:
                raise CompositeError(f"prediction file is empty: {path}")
            for key in (
                "probabilities",
                "source_ids",
                "snr_db",
                "sir_db",
                "target_profile_index",
            ):
                if int(np.asarray(archive[key]).shape[0]) != row_count:
                    raise CompositeError(f"prediction row mismatch {key}: {path}")
    except (OSError, ValueError, KeyError) as error:
        if isinstance(error, CompositeError):
            raise
        raise CompositeError(f"invalid prediction archive {path}: {error}") from error


def _verify_bound_file(config: Mapping[str, Any], section: str, path_key: str, sha_key: str) -> Path:
    payload = config.get(section)
    if not isinstance(payload, dict):
        raise CompositeError(f"recovery config section {section} is malformed")
    path = _workspace_path(payload.get(path_key), label=f"{section}.{path_key}")
    expected = payload.get(sha_key)
    if not path.is_file() or _sha256(path) != expected:
        raise CompositeError(f"bound artifact drift: {section}.{path_key}")
    return path


def preflight(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Validate immutable inputs.  This function never creates output."""

    config_path = config_path.resolve()
    config = _read_json(config_path)
    if config.get("schema_version") != "vimd_amc.tvt.formal_recovery.v4r_110plus10":
        raise CompositeError("unexpected recovery config schema")
    source = config.get("source_run")
    plan = config.get("missing_fit_plan")
    bindings = config.get("bindings")
    composite = config.get("composite_evidence")
    if not all(isinstance(item, dict) for item in (source, plan, bindings, composite)):
        raise CompositeError("recovery config sections are malformed")

    source_json = _workspace_path(source.get("run_json"), label="source_run.run_json")
    if not source_json.is_file():
        raise CompositeError(f"source run is missing: {source_json}")
    source_sha = _sha256(source_json)
    if source_sha != source.get("run_json_sha256_at_seal"):
        raise CompositeError("sealed source run.json SHA-256 drifted")
    source_run = _read_json(source_json)
    if source_run.get("run_id") != source.get("run_id"):
        raise CompositeError("source run identity drifted")
    if source_run.get("cache_digest") != source.get("cache_digest"):
        raise CompositeError("source run cache digest drifted")
    if source_run.get("execution_status") == "complete":
        raise CompositeError("source V4 unexpectedly claims complete")

    freeze = _verify_bound_file(config, "bindings", "v4_freeze", "v4_freeze_sha256")
    # The standard runner module carries generic defaults.  Formal V4 values
    # (designation, ten seeds, and the three-contrast confirmatory family)
    # must be installed before any eligibility or family statistic is built.
    formal_contract = load_contract(freeze)
    bind_runner_contract(formal_contract)
    learning = _verify_bound_file(
        config, "bindings", "learning_evidence", "learning_evidence_sha256"
    )
    cache_manifest = _verify_bound_file(
        config,
        "bindings",
        "headline_cache_manifest",
        "headline_cache_manifest_sha256",
    )
    manifest = _read_json(cache_manifest)
    if manifest.get("cache_digest") != source.get("cache_digest"):
        raise CompositeError("current cache manifest is not the sealed source cache")

    source_tree = source_run.get("environment", {}).get("source_tree")
    if not isinstance(source_tree, dict):
        raise CompositeError("source run lacks source-tree binding")
    if source_tree.get("aggregate_digest") != source.get("source_tree_aggregate_digest"):
        raise CompositeError("source-tree aggregate binding drifted")
    for relative, expected in source_tree.get("files", {}).items():
        current = _workspace_path(relative, label="source_tree.files")
        if not current.is_file() or _sha256(current) != expected:
            raise CompositeError(f"controlled source drift: {relative}")

    grid = _expected_grid(source_run)
    indexed = _index_results(source_run.get("results"), label="source")
    missing_model = plan.get("model")
    missing_seeds = plan.get("seeds")
    if not isinstance(missing_model, str) or not isinstance(missing_seeds, list):
        raise CompositeError("missing fit plan is malformed")
    expected_missing = {(missing_model, int(seed)) for seed in missing_seeds}
    expected_completed = set(grid).difference(expected_missing)
    if set(indexed) != expected_completed:
        raise CompositeError(
            "sealed source completed grid drift: "
            f"observed={len(indexed)} expected={len(expected_completed)}"
        )
    if len(indexed) != int(source.get("completed_fit_count", -1)):
        raise CompositeError("sealed source completed fit count drifted")

    source_root = source_json.parent
    splits = _evaluation_splits(source_run)
    input_records: list[dict[str, Any]] = []
    for role, path in (
        ("recovery_config", config_path),
        ("sealed_source_run_json", source_json),
        ("formal_freeze", freeze),
        ("learning_curve_evidence", learning),
        ("cache_manifest", cache_manifest),
    ):
        input_records.append(
            {
                "origin": "recovery_control_binding",
                "role": role,
                "source_path": _relative(path),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    for model, seed in grid:
        if (model, seed) not in indexed:
            continue
        result = indexed[(model, seed)]
        result_path = source_root / "models" / f"{model}_seed{seed}" / "result.json"
        disk_result = _read_json(result_path)
        if disk_result != result:
            raise CompositeError(f"embedded/source result mismatch: {model}/seed{seed}")
        required = [result_path]
        checkpoint = result.get("checkpoint")
        if not isinstance(checkpoint, str):
            raise CompositeError(f"checkpoint missing: {model}/seed{seed}")
        required.append(source_root / checkpoint)
        teacher_checkpoint = result.get("teacher_checkpoint")
        if teacher_checkpoint is not None:
            if not isinstance(teacher_checkpoint, str):
                raise CompositeError(f"teacher checkpoint malformed: {model}/seed{seed}")
            required.append(source_root / teacher_checkpoint)
        model_root = result_path.parent
        for split in splits:
            prediction = model_root / f"predictions_{split}.npz"
            _validate_prediction(
                prediction,
                split=split,
                cache_digest=str(source_run["cache_digest"]),
            )
            required.append(prediction)
        for path in required:
            if not path.is_file() or path.is_symlink():
                raise CompositeError(f"sealed source artifact missing: {path}")
            input_records.append(
                {
                    "origin": "sealed_v4_completed_fit",
                    "model": model,
                    "seed": seed,
                    "source_path": _relative(path),
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )

    for path in sorted((source_root / "manifests").glob("*.json")):
        input_records.append(
            {
                "origin": "sealed_v4_manifest",
                "source_path": _relative(path),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )

    worker_root = _workspace_path(
        plan.get("worker_output_root"), label="missing_fit_plan.worker_output_root"
    )
    worker_records: dict[int, dict[str, Any]] = {}
    missing_workers: list[int] = []
    invalid_workers: list[dict[str, Any]] = []
    config_sha = _sha256(config_path)
    for raw_seed in missing_seeds:
        seed = int(raw_seed)
        run_id = str(plan.get("worker_run_id_template", "")).format(seed=seed)
        root = worker_root / run_id
        manifest_path = root / "worker_manifest.json"
        if not manifest_path.is_file():
            missing_workers.append(seed)
            continue
        try:
            worker = _read_json(manifest_path)
            exact = {
                "schema_version": WORKER_SCHEMA,
                "status": "complete",
                "seed": seed,
                "model": missing_model,
                "run_id": run_id,
                "recovery_config_sha256": config_sha,
                "source_run_json_sha256": source_sha,
                "cache_digest": source_run["cache_digest"],
            }
            for key, expected in exact.items():
                if worker.get(key) != expected:
                    raise CompositeError(
                        f"worker seed{seed} {key} drift: {worker.get(key)!r} != {expected!r}"
                    )
            result_path, result_sha = _descriptor_file(
                root, worker.get("result_json"), label=f"seed{seed}.result_json"
            )
            checkpoint_path, checkpoint_sha = _descriptor_file(
                root, worker.get("checkpoint"), label=f"seed{seed}.checkpoint"
            )
            predictions = worker.get("predictions")
            if not isinstance(predictions, dict) or set(predictions) != set(splits):
                raise CompositeError(f"worker seed{seed} prediction split grid drifted")
            prediction_paths: dict[str, Path] = {}
            prediction_hashes: dict[str, str] = {}
            for split in splits:
                path, digest = _descriptor_file(
                    root,
                    predictions[split],
                    label=f"seed{seed}.predictions.{split}",
                )
                _validate_prediction(
                    path, split=split, cache_digest=str(source_run["cache_digest"])
                )
                prediction_paths[split] = path
                prediction_hashes[split] = digest
            result = _read_json(result_path)
            if result.get("model") != missing_model or result.get("seed") != seed:
                raise CompositeError(f"worker seed{seed} result identity drifted")
            if set(result.get("regimes", {})) != set(splits):
                raise CompositeError(f"worker seed{seed} result regime grid drifted")
            worker_manifest_sha = _sha256(manifest_path)
            worker_records[seed] = {
                "root": root,
                "manifest": worker,
                "manifest_path": manifest_path,
                "manifest_sha256": worker_manifest_sha,
                "result": result,
                "result_path": result_path,
                "result_sha256": result_sha,
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": checkpoint_sha,
                "prediction_paths": prediction_paths,
                "prediction_hashes": prediction_hashes,
            }
            expected_worker_files = {
                manifest_path.resolve(),
                result_path.resolve(),
                checkpoint_path.resolve(),
                *(path.resolve() for path in prediction_paths.values()),
                (root / "stdout.log").resolve(),
                (root / "stderr.log").resolve(),
            }
            observed_worker_files = {
                path.resolve()
                for path in root.rglob("*")
                if path.is_file() and not path.is_symlink()
            }
            if observed_worker_files != expected_worker_files:
                raise CompositeError(
                    f"worker seed{seed} file inventory drift: "
                    f"missing={sorted(str(path) for path in expected_worker_files - observed_worker_files)!r}, "
                    f"extra={sorted(str(path) for path in observed_worker_files - expected_worker_files)!r}"
                )
            if any(path.is_symlink() for path in root.rglob("*")):
                raise CompositeError(f"worker seed{seed} contains a symbolic link")
            known_roles = {
                manifest_path.resolve(): "worker_manifest",
                result_path.resolve(): "result_json",
                checkpoint_path.resolve(): "checkpoint",
            }
            known_roles.update(
                {path.resolve(): "prediction" for path in prediction_paths.values()}
            )
            for path in sorted(observed_worker_files):
                role = known_roles.get(path, "worker_process_log")
                input_records.append(
                    {
                        "origin": "sealed_recovery_worker",
                        "role": role,
                        "model": missing_model,
                        "seed": seed,
                        "source_path": _relative(path),
                        "sha256": _sha256(path),
                        "size_bytes": path.stat().st_size,
                    }
                )
        except (CompositeError, OSError, ValueError) as error:
            invalid_workers.append({"seed": seed, "reason": str(error)})

    output_root = _workspace_path(composite.get("output_root"), label="composite_evidence.output_root")
    ready = not missing_workers and not invalid_workers and len(worker_records) == len(missing_seeds)
    return {
        "schema_version": "vimd_amc.tvt.composite_recovery.preflight.v1",
        "status": "ready" if ready else "not_ready",
        "ready": ready,
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": config_path,
        "config_sha256": config_sha,
        "config": config,
        "source_run_path": source_json,
        "source_run_sha256": source_sha,
        "source_run": source_run,
        "source_root": source_root,
        "freeze_path": freeze,
        "formal_contract": formal_contract,
        "learning_evidence_path": learning,
        "cache_manifest_path": cache_manifest,
        "output_root": output_root,
        "expected_fit_count": len(grid),
        "sealed_completed_fit_count": len(indexed),
        "expected_worker_count": len(missing_seeds),
        "valid_worker_count": len(worker_records),
        "missing_workers": missing_workers,
        "invalid_workers": invalid_workers,
        "workers": worker_records,
        "input_records": input_records,
        "splits": splits,
        "grid": grid,
    }


def _public_preflight(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: runner.json_safe(value)
        for key, value in state.items()
        if key not in {"config", "source_run", "workers", "input_records", "grid"}
    }


def _link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise CompositeError(f"destination already exists: {destination}")
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _materialize_inputs(state: Mapping[str, Any], staging: Path) -> list[dict[str, Any]]:
    source_run = state["source_run"]
    source_root: Path = state["source_root"]
    workers: Mapping[int, Mapping[str, Any]] = state["workers"]
    missing_model = state["config"]["missing_fit_plan"]["model"]
    records: list[dict[str, Any]] = []

    for manifest in sorted((source_root / "manifests").glob("*.json")):
        destination = staging / "manifests" / manifest.name
        method = _link_or_copy(manifest, destination)
        records.append(
            {
                "origin": "sealed_v4_manifest",
                "source_path": _relative(manifest),
                "source_sha256": _sha256(manifest),
                "composite_path": destination.relative_to(staging).as_posix(),
                "composite_sha256": _sha256(destination),
                "materialization": method,
            }
        )

    source_index = _index_results(source_run["results"], label="source")
    for model, seed in state["grid"]:
        destination_root = staging / "models" / f"{model}_seed{seed}"
        if (model, seed) in source_index:
            result = source_index[(model, seed)]
            source_model_root = source_root / "models" / f"{model}_seed{seed}"
            files = [source_model_root / "model.pt"]
            if (source_model_root / "teacher.pt").is_file():
                files.append(source_model_root / "teacher.pt")
            files.extend(
                source_model_root / f"predictions_{split}.npz"
                for split in state["splits"]
            )
            origin = "sealed_v4_completed_fit"
            source_result_path = source_model_root / "result.json"
        else:
            worker = workers[seed]
            result = worker["result"]
            files = [worker["checkpoint_path"]]
            files.extend(worker["prediction_paths"][split] for split in state["splits"])
            origin = "sealed_recovery_worker"
            source_result_path = worker["result_path"]
        for source_file in files:
            destination_name = (
                "model.pt"
                if source_file == files[0]
                else source_file.name
            )
            destination = destination_root / destination_name
            method = _link_or_copy(source_file, destination)
            source_hash = _sha256(source_file)
            destination_hash = _sha256(destination)
            if source_hash != destination_hash:
                raise CompositeError(f"materialized hash mismatch: {destination}")
            records.append(
                {
                    "origin": origin,
                    "model": model,
                    "seed": seed,
                    "source_path": _relative(source_file),
                    "source_sha256": source_hash,
                    "composite_path": destination.relative_to(staging).as_posix(),
                    "composite_sha256": destination_hash,
                    "materialization": method,
                }
            )
        # The source result.json is already hash-bound in ``source_inputs``.
        # The composite result.json is a newly derived artifact recorded in
        # ``derived_artifacts`` below, so it must not be represented as a
        # materialized file before it exists.
        if model == missing_model and seed not in workers:
            raise CompositeError(f"internal preflight inconsistency: missing worker seed{seed}")
    return records


def _recompute_results_and_metrics(
    state: Mapping[str, Any],
    staging: Path,
) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[tuple[str, int, str], Any]]:
    source_index = _index_results(state["source_run"]["results"], label="source")
    workers: Mapping[int, Mapping[str, Any]] = state["workers"]
    missing_model = state["config"]["missing_fit_plan"]["model"]
    cache_digest = str(state["source_run"]["cache_digest"])
    classes = int(state["source_run"]["num_classes"])
    results: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    bundles: dict[tuple[str, int, str], Any] = {}
    for model, seed in state["grid"]:
        input_result = (
            source_index[(model, seed)]
            if (model, seed) in source_index
            else workers[seed]["result"]
        )
        result = json.loads(json.dumps(input_result))
        result["checkpoint"] = f"models/{model}_seed{seed}/model.pt"
        teacher = staging / "models" / f"{model}_seed{seed}" / "teacher.pt"
        result["teacher_checkpoint"] = (
            f"models/{model}_seed{seed}/teacher.pt" if teacher.is_file() else None
        )
        result["regimes"] = {}
        result["composite_metric_provenance"] = {
            "schema_version": 1,
            "source": "recomputed_from_bound_prediction_npz",
            "source_run_kind": (
                "recovery_worker" if model == missing_model else "sealed_interrupted_v4"
            ),
        }
        for split in state["splits"]:
            bundle = release._load_bundle(staging, model, seed, split, cache_digest)
            metrics = classification_metrics(bundle, classes)
            result["regimes"][split] = metrics
            bundles[(model, seed, split)] = bundle
            flat_rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "regime": split,
                    "cache_digest": cache_digest,
                    "standards_evidence_label": state["source_run"][
                        "standards_evidence_label"
                    ],
                    **{
                        key: value
                        for key, value in metrics.items()
                        if isinstance(value, (int, float))
                    },
                }
            )
        result_path = staging / "models" / f"{model}_seed{seed}" / "result.json"
        _write_json(result_path, result)
        results.append(result)
    return results, pd.DataFrame(flat_rows), bundles


def _write_cross_model_statistics(
    state: Mapping[str, Any],
    staging: Path,
    metrics_frame: pd.DataFrame,
    bundles: Mapping[tuple[str, int, str], Any],
) -> dict[str, Any]:
    source = state["source_run"]
    models = list(source["models"])
    seeds = [int(seed) for seed in source["seeds"]]
    splits = list(state["splits"])
    comparison = source["comparison_protocol"]
    reference_model = str(comparison["reference_model"])
    reference_selection = str(comparison["reference_selection"])
    holm_candidates = list(comparison["holm_candidate_family"])
    draws = int(comparison["bootstrap_draws"])
    seed_base = int(comparison["bootstrap_seed_base"])
    cache_digest = str(source["cache_digest"])

    metrics_frame.to_csv(staging / "metrics.csv", index=False)
    runner._aggregate_seed_metrics(metrics_frame).to_csv(
        staging / "seed_aggregates.csv", index=False
    )

    paired_rows: list[dict[str, Any]] = []
    for candidate in models:
        if candidate == reference_model:
            continue
        for seed in seeds:
            for split in splits:
                reference = bundles[(reference_model, seed, split)]
                candidate_bundle = bundles[(candidate, seed, split)]
                bootstrap_seed = runner.analysis_seed(
                    seed_base,
                    "single_seed",
                    reference_model,
                    candidate,
                    seed,
                    split,
                )
                preregistered = split != "validation" and candidate in holm_candidates
                if preregistered:
                    statistics = paired_bundle_statistics(
                        reference, candidate_bundle, draws=draws, seed=bootstrap_seed
                    )
                else:
                    statistics = paired_accuracy_macro_f1_bootstrap(
                        reference, candidate_bundle, draws=draws, seed=bootstrap_seed
                    )
                    statistics.update(
                        {
                            "reference_only_correct": None,
                            "candidate_only_correct": None,
                            "discordant_pairs": None,
                            "exact_p_value": None,
                            "inference_scope": "descriptive_paired_ci",
                            "mcnemar_scope": "not_performed",
                        }
                    )
                paired_rows.append(
                    {
                        "reference": reference_model,
                        "reference_selection": reference_selection,
                        "reference_strength_claimed": False,
                        "candidate": candidate,
                        "seed": seed,
                        "regime": split,
                        "comparison_role": (
                            "validation_descriptive"
                            if split == "validation"
                            else "heldout_single_seed"
                        ),
                        "mcnemar_preregistered": preregistered,
                        "cache_digest": cache_digest,
                        **statistics,
                        "holm_included": False,
                        "holm_family_id": None,
                        "holm_family_size": None,
                        "holm_adjusted_p_value": None,
                        "holm_exclusion_reason": (
                            "validation_is_excluded_from_all_holm_families"
                            if split == "validation"
                            else (
                                "candidate_not_in_predeclared_holm_family"
                                if candidate not in holm_candidates
                                else None
                            )
                        ),
                    }
                )
    holm_groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in paired_rows:
        if row["regime"] != "validation" and row["candidate"] in holm_candidates:
            holm_groups.setdefault((str(row["regime"]), int(row["seed"])), []).append(row)
    for (regime, algorithm_seed), family_rows in holm_groups.items():
        ordered_rows = [
            next(row for row in family_rows if row["candidate"] == candidate)
            for candidate in holm_candidates
        ]
        adjusted = holm_adjust(
            np.asarray([row["exact_p_value"] for row in ordered_rows], dtype=np.float64)
        )
        family_id = (
            f"regime={regime}|seed={algorithm_seed}|reference={reference_model}|"
            f"candidates={','.join(holm_candidates)}"
        )
        for row, value in zip(ordered_rows, adjusted, strict=True):
            row.update(
                {
                    "holm_included": True,
                    "holm_family_id": family_id,
                    "holm_family_size": len(ordered_rows),
                    "holm_adjusted_p_value": float(value),
                    "holm_exclusion_reason": None,
                }
            )
    paired_columns = [
        "reference", "reference_selection", "reference_strength_claimed",
        "candidate", "seed", "regime", "comparison_role", "cache_digest",
        "difference", "ci95_low", "ci95_high", "accuracy_difference",
        "accuracy_ci95_low", "accuracy_ci95_high", "macro_f1_difference",
        "macro_f1_ci95_low", "macro_f1_ci95_high", "bootstrap_draws",
        "bootstrap_seed", "bootstrap_cluster_count", "bootstrap_stratified_by_class",
        "bootstrap_unit", "reference_only_correct", "candidate_only_correct",
        "discordant_pairs", "exact_p_value", "inference_scope", "mcnemar_scope",
        "mcnemar_preregistered", "holm_included", "holm_family_id",
        "holm_family_size", "holm_adjusted_p_value", "holm_exclusion_reason",
    ]
    pd.DataFrame(paired_rows, columns=paired_columns).to_csv(
        staging / "paired_statistics.csv", index=False
    )

    headline_rows: list[dict[str, Any]] = []
    for candidate in models:
        if candidate == reference_model:
            continue
        for split in splits:
            if split == "validation":
                continue
            bootstrap_seed = runner.analysis_seed(
                seed_base, "hierarchical", reference_model, candidate, split
            )
            headline_rows.append(
                {
                    "reference": reference_model,
                    "reference_selection": reference_selection,
                    "reference_strength_claimed": False,
                    "candidate": candidate,
                    "regime": split,
                    "parent_regime": split,
                    "target_profile_indices": "all",
                    "cache_digest": cache_digest,
                    **headline_paired_bootstrap(
                        {seed: bundles[(reference_model, seed, split)] for seed in seeds},
                        {seed: bundles[(candidate, seed, split)] for seed in seeds},
                        draws=draws,
                        seed=bootstrap_seed,
                    ),
                }
            )
    if runner.FORMAL_METHOD_MODEL in models and "clean_retention" in splits:
        for stratum, profiles in runner.CLEAN_RETENTION_PROFILE_STRATA.items():
            references: dict[int, Any] = {}
            candidates: dict[int, Any] = {}
            for seed in seeds:
                reference = bundles[(reference_model, seed, "clean_retention")]
                candidate = bundles[(runner.FORMAL_METHOD_MODEL, seed, "clean_retention")]
                if reference.target_profile_index is None:
                    raise CompositeError("clean-retention prediction lacks target profile")
                selected = np.isin(
                    reference.target_profile_index, np.asarray(profiles, dtype=np.int64)
                )
                references[seed] = reference.subset(selected)
                candidates[seed] = candidate.subset(selected)
            bootstrap_seed = runner.analysis_seed(
                seed_base,
                "hierarchical",
                reference_model,
                runner.FORMAL_METHOD_MODEL,
                stratum,
            )
            headline_rows.append(
                {
                    "reference": reference_model,
                    "reference_selection": reference_selection,
                    "reference_strength_claimed": False,
                    "candidate": runner.FORMAL_METHOD_MODEL,
                    "regime": stratum,
                    "parent_regime": "clean_retention",
                    "target_profile_indices": repr(list(profiles)),
                    "cache_digest": cache_digest,
                    **headline_paired_bootstrap(
                        references, candidates, draws=draws, seed=bootstrap_seed
                    ),
                }
            )
    headline_columns = [
        "reference", "reference_selection", "reference_strength_claimed",
        "candidate", "regime", "parent_regime", "target_profile_indices",
        "cache_digest", "difference", "ci95_low", "ci95_high",
        "accuracy_difference", "accuracy_ci95_low", "accuracy_ci95_high",
        "macro_f1_difference", "macro_f1_ci95_low", "macro_f1_ci95_high",
        "accuracy_test_source_only_ci95_low", "accuracy_test_source_only_ci95_high",
        "macro_f1_test_source_only_ci95_low", "macro_f1_test_source_only_ci95_high",
        "accuracy_algorithm_seed_only_ci95_low", "accuracy_algorithm_seed_only_ci95_high",
        "macro_f1_algorithm_seed_only_ci95_low", "macro_f1_algorithm_seed_only_ci95_high",
        "bootstrap_draws", "bootstrap_seed", "algorithm_seed_count",
        "algorithm_seed_ids", "test_source_cluster_count",
        "bootstrap_stratified_by_class", "bootstrap_hierarchy",
        "mcnemar_exact_test_performed", "mcnemar_reason",
    ]
    pd.DataFrame(headline_rows, columns=headline_columns).to_csv(
        staging / "headline_paired_statistics.csv", index=False
    )

    # ``preflight`` binds the effective V4 freeze into the generic runner.
    # Select that exact ordered family here and fail closed if the materialized
    # headline grid cannot supply it.
    ablation_seeds = _formal_ablation_seeds(seeds)
    ablation_rows, ablation_summary = runner.build_ablation_paired_rows(
        prediction_bundles=bundles,
        seeds=ablation_seeds,
        cache_digest=cache_digest,
        bootstrap_draws=draws,
        bootstrap_seed_base=seed_base,
    )
    pd.DataFrame(ablation_rows, columns=runner.ABLATION_PAIRED_COLUMNS).to_csv(
        staging / "ablation_paired_statistics.csv", index=False
    )
    return {
        "single_seed_pairs": "paired_statistics.csv",
        "multi_seed_headline_pairs": "headline_paired_statistics.csv",
        "ablation_pairs": "ablation_paired_statistics.csv",
        "ablation_family": ablation_summary,
        "clean_retention_profile_strata": {
            name: {
                "parent_regime": "clean_retention",
                "target_profile_indices": list(indices),
                "reference": reference_model,
                "candidate": runner.FORMAL_METHOD_MODEL,
            }
            for name, indices in runner.CLEAN_RETENTION_PROFILE_STRATA.items()
        },
        "holm_families": [
            {
                "family_id": family_rows[0]["holm_family_id"],
                "regime": regime,
                "algorithm_seed": algorithm_seed,
                "reference": reference_model,
                "candidates": [row["candidate"] for row in family_rows],
                "validation_included": False,
            }
            for (regime, algorithm_seed), family_rows in holm_groups.items()
        ],
        "pooled_multi_seed_mcnemar_performed": False,
        "derivation": "recomputed_from_120_bound_prediction_grids",
    }


def _source_audit(source_run: Mapping[str, Any]) -> dict[str, Any]:
    start = source_run.get("environment", {}).get("source_tree")
    if not isinstance(start, dict):
        raise CompositeError("source tree start record is missing")
    end = source_tree_record(
        ROOT,
        Path(runner.__file__),
        profile=str(start.get("profile")),
    )
    unchanged = (
        start.get("aggregate_digest") == end.get("aggregate_digest")
        and start.get("files") == end.get("files")
    )
    return {
        "unchanged": unchanged,
        "reason": "unchanged" if unchanged else "controlled_source_drift",
        "start": start,
        "end": end,
        "recovery_tool_outside_controlled_profile": _relative(Path(__file__)),
    }


def build(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    state = preflight(config_path)
    if not state["ready"]:
        return _public_preflight(state)
    output: Path = state["output_root"]
    if output.exists():
        raise CompositeError(f"composite output already exists: {output}")
    staging = output.with_name(f".{output.name}.staging-{os.getpid()}")
    if staging.exists():
        raise CompositeError(f"staging path already exists: {staging}")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        materialization = _materialize_inputs(state, staging)
        results, metrics_frame, bundles = _recompute_results_and_metrics(state, staging)
        statistical_outputs = _write_cross_model_statistics(
            state, staging, metrics_frame, bundles
        )
        source_run = state["source_run"]
        source_audit = _source_audit(source_run)
        if not source_audit["unchanged"]:
            raise CompositeError("controlled source drift detected during composite")
        contract = runner.inspect_cache_contract(Path(source_run["cache_root"]))
        eligibility = runner.assess_evidence_eligibility(
            contract=contract,
            support_audit=source_run["dataset_support_audit"],
            checksums_verified=True,
            component_validation=source_run["component_validation"],
            models=list(source_run["models"]),
            seeds=[int(seed) for seed in source_run["seeds"]],
            execution_status="complete",
            explicit_reference_model=source_run["comparison_protocol"]["reference_model"],
            holm_candidates=list(source_run["comparison_protocol"]["holm_candidate_family"]),
            source_tree_unchanged=True,
            training_configuration=source_run["training_configuration"],
            training_results=results,
            jammer_training_support=source_run["jammer_auxiliary_training_support"],
        )
        run = json.loads(json.dumps(source_run))
        run.update(
            {
                "schema_version": COMPOSITE_SCHEMA,
                "artifact_id": output.name,
                "runner": _relative(Path(__file__)),
                "status": "complete",
                "execution_status": "complete",
                "results": results,
                "statistical_outputs": statistical_outputs,
                "source_tree_execution_audit": source_audit,
                "evidence_eligibility": eligibility,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "recovery_provenance": {
                    "schema_version": COMPOSITE_SCHEMA,
                    "execution_kind": "composite_recovery",
                    "release_label": state["config"]["composite_evidence"]["release_label"],
                    "original_v4_run_interrupted": True,
                    "original_run_completion_claimed": False,
                    "original_run_status_observed": source_run.get("status"),
                    "original_run_execution_status_observed": source_run.get("execution_status"),
                    "original_run_json": _relative(state["source_run_path"]),
                    "original_run_json_sha256": state["source_run_sha256"],
                    "sealed_original_fit_count": state["sealed_completed_fit_count"],
                    "fresh_recovery_fit_count": state["valid_worker_count"],
                    "composite_fit_count": len(results),
                    "recovery_config": _relative(state["config_path"]),
                    "recovery_config_sha256": state["config_sha256"],
                    "worker_manifest_sha256": {
                        str(seed): worker["manifest_sha256"]
                        for seed, worker in state["workers"].items()
                    },
                    "all_per_fit_metrics_recomputed_from_npz": True,
                    "all_cross_model_statistics_recomputed": True,
                    "old_statistical_csv_reused": False,
                },
            }
        )
        run["submission_release"] = runner.submission_release_source_gate(run)
        _write_json(staging / "run.json", run)

        gate = release.derive_gate(
            run_json=staging / "run.json",
            learning_evidence=state["learning_evidence_path"],
            freeze=state["freeze_path"],
            write=True,
        )
        # ``derive_gate`` runs against the still-private staging tree.  Store
        # the eventual atomic destination, not a path that disappears at
        # commit, while preserving every recomputed value and run hash.
        gate["run_json"] = str((output / "run.json").resolve())
        _write_json(staging / "v2_scientific_release_gate.json", gate)
        derived_files = [
            "metrics.csv",
            "seed_aggregates.csv",
            "paired_statistics.csv",
            "headline_paired_statistics.csv",
            "ablation_paired_statistics.csv",
            "run.json",
            "v2_scientific_release_gate.json",
        ]
        inventory = {
            "schema_version": INVENTORY_SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_id": output.name,
            "source_run_json": _relative(state["source_run_path"]),
            "source_run_json_sha256": state["source_run_sha256"],
            "recovery_config": _relative(state["config_path"]),
            "recovery_config_sha256": state["config_sha256"],
            "source_inputs": state["input_records"],
            "materialized_artifacts": materialization,
            "derived_artifacts": [
                {
                    "composite_path": name,
                    "sha256": _sha256(staging / name),
                    "size_bytes": (staging / name).stat().st_size,
                    "derivation": "recomputed_or_composite_metadata",
                }
                for name in derived_files
            ]
            + [
                {
                    "composite_path": f"models/{model}_seed{seed}/result.json",
                    "sha256": _sha256(
                        staging / "models" / f"{model}_seed{seed}" / "result.json"
                    ),
                    "size_bytes": (
                        staging / "models" / f"{model}_seed{seed}" / "result.json"
                    ).stat().st_size,
                    "derivation": "metrics_recomputed_from_bound_prediction_npz",
                }
                for model, seed in state["grid"]
            ],
            "fit_grid_order": [
                {"model": model, "seed": seed} for model, seed in state["grid"]
            ],
        }
        _write_json(staging / "recovery_inventory.json", inventory)
        seal = {
            "schema_version": SEAL_SCHEMA,
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_id": output.name,
            "original_v4_run_interrupted": True,
            "original_run_completion_claimed": False,
            "composite_recovery_complete": True,
            "run_json_sha256": _sha256(staging / "run.json"),
            "inventory_sha256": _sha256(staging / "recovery_inventory.json"),
            "scientific_gate_sha256": _sha256(
                staging / "v2_scientific_release_gate.json"
            ),
            "scientific_evidence_passed": gate.get("scientific_evidence_passed") is True,
            "submission_unlocked": gate.get("submission_unlocked") is True,
            "recovery_config_sha256": state["config_sha256"],
            "builder_sha256": _sha256(Path(__file__)),
        }
        _write_json(staging / "composite_seal.json", seal)
        os.replace(staging, output)
        return validate_composite(output, config_path=config_path)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def validate_composite(
    output: Path,
    *,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    output = output.resolve()
    if not output.is_dir():
        raise CompositeError(f"composite output does not exist: {output}")
    seal = _read_json(output / "composite_seal.json")
    inventory = _read_json(output / "recovery_inventory.json")
    run = _read_json(output / "run.json")
    reasons: list[str] = []
    if seal.get("schema_version") != SEAL_SCHEMA or seal.get("status") != "complete":
        reasons.append("seal_schema_or_status_invalid")
    if inventory.get("schema_version") != INVENTORY_SCHEMA:
        reasons.append("inventory_schema_invalid")
    if run.get("schema_version") != COMPOSITE_SCHEMA:
        reasons.append("run_schema_invalid")
    if run.get("recovery_provenance", {}).get("original_run_completion_claimed") is not False:
        reasons.append("original_run_completion_claim_not_explicitly_false")
    expected_hashes = {
        "run_json_sha256": output / "run.json",
        "inventory_sha256": output / "recovery_inventory.json",
        "scientific_gate_sha256": output / "v2_scientific_release_gate.json",
    }
    for key, path in expected_hashes.items():
        if not path.is_file() or seal.get(key) != _sha256(path):
            reasons.append(f"seal_hash_mismatch:{key}")
    if seal.get("recovery_config_sha256") != _sha256(config_path.resolve()):
        reasons.append("recovery_config_hash_mismatch")
    for record in inventory.get("source_inputs", []):
        try:
            source = _workspace_path(record["source_path"], label="inventory.source_path")
            if _sha256(source) != record["sha256"]:
                reasons.append(f"source_input_hash_mismatch:{record['source_path']}")
        except (CompositeError, KeyError, OSError) as error:
            reasons.append(f"source_input_invalid:{error}")
    for record in inventory.get("materialized_artifacts", []):
        try:
            path = _contained_path(
                output, record["composite_path"], label="inventory.composite_path"
            )
            if _sha256(path) != record["composite_sha256"]:
                reasons.append(f"materialized_hash_mismatch:{record['composite_path']}")
        except (CompositeError, KeyError, OSError) as error:
            reasons.append(f"materialized_artifact_invalid:{error}")
    for record in inventory.get("derived_artifacts", []):
        try:
            path = _contained_path(
                output, record["composite_path"], label="inventory.derived_path"
            )
            if _sha256(path) != record["sha256"]:
                reasons.append(f"derived_hash_mismatch:{record['composite_path']}")
        except (CompositeError, KeyError, OSError) as error:
            reasons.append(f"derived_artifact_invalid:{error}")
    expected_files = {
        str(record.get("composite_path"))
        for record in inventory.get("materialized_artifacts", [])
    } | {
        str(record.get("composite_path"))
        for record in inventory.get("derived_artifacts", [])
    } | {"recovery_inventory.json", "composite_seal.json"}
    observed_files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    if observed_files != expected_files:
        missing = sorted(expected_files.difference(observed_files))
        extra = sorted(observed_files.difference(expected_files))
        reasons.append(
            "composite_inventory_not_exact:"
            f"missing={missing!r}:extra={extra!r}"
        )
    grid = _expected_grid(run)
    indexed = _index_results(run.get("results"), label="composite")
    if list(indexed) != grid:
        reasons.append("result_order_or_grid_drift")
    gate = _read_json(output / "v2_scientific_release_gate.json")
    if gate.get("run_json_sha256") != _sha256(output / "run.json"):
        reasons.append("scientific_gate_run_binding_mismatch")
    return {
        "schema_version": "vimd_amc.tvt.composite_recovery.validation.v1",
        "status": "valid" if not reasons else "invalid",
        "valid": not reasons,
        "artifact": output,
        "fit_count": len(indexed),
        "original_run_completion_claimed": run.get("recovery_provenance", {}).get(
            "original_run_completion_claimed"
        ),
        "scientific_evidence_passed": gate.get("scientific_evidence_passed") is True,
        "submission_unlocked": gate.get("submission_unlocked") is True,
        "reasons": reasons,
    }


def repair_inventory_metadata(
    output: Path,
    *,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Remove only legacy non-materialized result provenance records.

    Retry-1 completed every expensive statistic before discovering that source
    result provenance had also been appended to ``materialized_artifacts``.
    Those same source result files are already present in ``source_inputs``.
    This repair is deliberately narrow and fails closed on any other drift.
    """

    output = output.resolve()
    inventory_path = output / "recovery_inventory.json"
    seal_path = output / "composite_seal.json"
    run_path = output / "run.json"
    gate_path = output / "v2_scientific_release_gate.json"
    inventory_bytes = inventory_path.read_bytes()
    seal_bytes = seal_path.read_bytes()
    inventory = _read_json(inventory_path)
    seal = _read_json(seal_path)
    run = _read_json(run_path)
    if seal.get("schema_version") != SEAL_SCHEMA or seal.get("status") != "complete":
        raise CompositeError("inventory repair requires a complete composite seal")
    for key, path in {
        "run_json_sha256": run_path,
        "inventory_sha256": inventory_path,
        "scientific_gate_sha256": gate_path,
    }.items():
        if seal.get(key) != _sha256(path):
            raise CompositeError(f"inventory repair refused seal drift: {key}")
    records = inventory.get("materialized_artifacts")
    if not isinstance(records, list):
        raise CompositeError("inventory repair requires materialized_artifacts")
    real = [record for record in records if isinstance(record, dict) and "composite_path" in record]
    legacy = [record for record in records if not (isinstance(record, dict) and "composite_path" in record)]
    grid = _expected_grid(run)
    if len(legacy) != len(grid):
        raise CompositeError(
            "inventory repair legacy record count mismatch: "
            f"{len(legacy)} != {len(grid)}"
        )
    expected_keys = {
        (str(model), int(seed)) for model, seed in grid
    }
    observed_keys: set[tuple[str, int]] = set()
    source_bindings = {
        (str(record.get("source_path")), str(record.get("sha256")))
        for record in inventory.get("source_inputs", [])
        if isinstance(record, dict)
    }
    for record in legacy:
        if record.get("role") != "result_json_input_for_recomputation" or record.get(
            "materialization"
        ) != "derived_result_written_later":
            raise CompositeError("inventory repair encountered an unexpected legacy record")
        key = (str(record.get("model")), int(record.get("seed")))
        if key in observed_keys:
            raise CompositeError(f"inventory repair found duplicate legacy fit: {key}")
        observed_keys.add(key)
        binding = (str(record.get("source_path")), str(record.get("source_sha256")))
        if binding not in source_bindings:
            raise CompositeError(f"legacy source result is not independently bound: {key}")
    if observed_keys != expected_keys:
        raise CompositeError("inventory repair legacy fit grid mismatch")
    expected_files = {
        str(record["composite_path"]) for record in real
    } | {
        str(record["composite_path"])
        for record in inventory.get("derived_artifacts", [])
    } | {"recovery_inventory.json", "composite_seal.json"}
    observed_files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    if observed_files != expected_files:
        raise CompositeError("inventory repair refused non-metadata file drift")
    for record in real:
        path = _contained_path(output, record["composite_path"], label="repair.materialized")
        if _sha256(path) != record["composite_sha256"]:
            raise CompositeError(f"inventory repair refused materialized drift: {record['composite_path']}")
    for record in inventory.get("derived_artifacts", []):
        path = _contained_path(output, record["composite_path"], label="repair.derived")
        if _sha256(path) != record["sha256"]:
            raise CompositeError(f"inventory repair refused derived drift: {record['composite_path']}")

    previous_inventory_sha256 = _sha256(inventory_path)
    previous_seal_sha256 = _sha256(seal_path)
    repaired_utc = datetime.now(timezone.utc).isoformat()
    inventory["materialized_artifacts"] = real
    inventory["metadata_repair"] = {
        "status": "applied",
        "repaired_utc": repaired_utc,
        "reason": "removed_duplicate_non_materialized_result_provenance",
        "removed_record_count": len(legacy),
        "source_result_bindings_retained": True,
        "statistical_artifacts_modified": False,
        "previous_inventory_sha256": previous_inventory_sha256,
    }
    try:
        _write_json(inventory_path, inventory)
        seal["inventory_sha256"] = _sha256(inventory_path)
        seal["builder_sha256"] = _sha256(Path(__file__))
        seal["metadata_repair"] = {
            "repaired_utc": repaired_utc,
            "previous_seal_sha256": previous_seal_sha256,
            "statistical_artifacts_modified": False,
        }
        _write_json(seal_path, seal)
        result = validate_composite(output, config_path=config_path)
        if not result["valid"]:
            raise CompositeError(
                "inventory metadata repair did not validate: "
                + repr(result["reasons"])
            )
        return result
    except Exception:
        inventory_path.write_bytes(inventory_bytes)
        seal_path.write_bytes(seal_bytes)
        raise


def rederive_v4_protocol(
    output: Path,
    *,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Re-derive the freeze-bound V4 family and release gate in place.

    The first composite pass used generic runner defaults.  This routine first
    validates the sealed composite, binds the canonical V4 contract, then
    replaces only derived protocol artifacts.  All touched files are restored
    byte-for-byte if any computation or final validation fails.
    """

    output = output.resolve()
    current = validate_composite(output, config_path=config_path)
    if not current["valid"]:
        raise CompositeError("V4 protocol re-derivation requires a valid composite")
    state = preflight(config_path)
    if not state["ready"]:
        raise CompositeError("V4 protocol re-derivation inputs are not ready")
    paths = {
        name: output / name
        for name in (
            "ablation_paired_statistics.csv",
            "run.json",
            "v2_scientific_release_gate.json",
            "recovery_inventory.json",
            "composite_seal.json",
        )
    }
    backups = {name: path.read_bytes() for name, path in paths.items()}
    previous_hashes = {name: _sha256(path) for name, path in paths.items()}
    try:
        run = _read_json(paths["run.json"])
        seeds = [int(seed) for seed in run["seeds"]]
        cache_digest = str(run["cache_digest"])
        bundles: dict[tuple[str, int, str], Any] = {}
        required_models = {
            str(model)
            for record in runner.FORMAL_ABLATION_CONTRASTS
            for model in (record["reference"], record["candidate"])
        }
        for model in sorted(required_models):
            for seed in seeds:
                bundles[(model, seed, runner.FORMAL_ABLATION_REGIME)] = (
                    release._load_bundle(
                        output,
                        model,
                        seed,
                        runner.FORMAL_ABLATION_REGIME,
                        cache_digest,
                    )
                )
        ablation_rows, ablation_summary = runner.build_ablation_paired_rows(
            prediction_bundles=bundles,
            seeds=_formal_ablation_seeds(seeds),
            cache_digest=cache_digest,
            bootstrap_draws=int(run["comparison_protocol"]["bootstrap_draws"]),
            bootstrap_seed_base=int(
                run["comparison_protocol"]["bootstrap_seed_base"]
            ),
        )
        temporary_csv = paths["ablation_paired_statistics.csv"].with_suffix(".csv.tmp")
        pd.DataFrame(ablation_rows, columns=runner.ABLATION_PAIRED_COLUMNS).to_csv(
            temporary_csv,
            index=False,
        )
        os.replace(temporary_csv, paths["ablation_paired_statistics.csv"])

        source_audit = _source_audit(state["source_run"])
        if not source_audit["unchanged"]:
            raise CompositeError("controlled source drift during V4 re-derivation")
        contract = runner.inspect_cache_contract(Path(run["cache_root"]))
        eligibility = runner.assess_evidence_eligibility(
            contract=contract,
            support_audit=run["dataset_support_audit"],
            checksums_verified=True,
            component_validation=run["component_validation"],
            models=list(run["models"]),
            seeds=seeds,
            execution_status="complete",
            explicit_reference_model=run["comparison_protocol"]["reference_model"],
            holm_candidates=list(
                run["comparison_protocol"]["holm_candidate_family"]
            ),
            source_tree_unchanged=True,
            training_configuration=run["training_configuration"],
            training_results=list(run["results"]),
            jammer_training_support=run["jammer_auxiliary_training_support"],
        )
        run["evidence_eligibility"] = eligibility
        run["submission_release"] = runner.submission_release_source_gate(run)
        run["statistical_outputs"]["ablation_family"] = ablation_summary
        run["protocol_rederivation"] = {
            "status": "complete",
            "rederived_utc": datetime.now(timezone.utc).isoformat(),
            "formal_freeze": _relative(state["freeze_path"]),
            "formal_freeze_sha256": _sha256(state["freeze_path"]),
            "confirmatory_seed_ids": seeds,
            "confirmatory_contrast_ids": [
                str(record["contrast_id"])
                for record in runner.FORMAL_ABLATION_CONTRASTS
            ],
            "previous_derived_hashes": previous_hashes,
            "training_or_prediction_artifacts_modified": False,
        }
        _write_json(paths["run.json"], run)

        gate = release.derive_gate(
            run_json=paths["run.json"],
            learning_evidence=state["learning_evidence_path"],
            freeze=state["freeze_path"],
            write=True,
        )
        gate["run_json"] = str(paths["run.json"].resolve())
        _write_json(paths["v2_scientific_release_gate.json"], gate)

        inventory = _read_json(paths["recovery_inventory.json"])
        derived_by_path = {
            str(record["composite_path"]): record
            for record in inventory["derived_artifacts"]
        }
        for name in (
            "ablation_paired_statistics.csv",
            "run.json",
            "v2_scientific_release_gate.json",
        ):
            record = derived_by_path[name]
            record["sha256"] = _sha256(paths[name])
            record["size_bytes"] = paths[name].stat().st_size
        inventory["protocol_rederivation"] = run["protocol_rederivation"]
        _write_json(paths["recovery_inventory.json"], inventory)

        seal = _read_json(paths["composite_seal.json"])
        seal.update(
            {
                "run_json_sha256": _sha256(paths["run.json"]),
                "inventory_sha256": _sha256(paths["recovery_inventory.json"]),
                "scientific_gate_sha256": _sha256(
                    paths["v2_scientific_release_gate.json"]
                ),
                "scientific_evidence_passed": gate.get(
                    "scientific_evidence_passed"
                )
                is True,
                "submission_unlocked": gate.get("submission_unlocked") is True,
                "builder_sha256": _sha256(Path(__file__)),
                "protocol_rederivation": {
                    "status": "complete",
                    "rederived_utc": run["protocol_rederivation"]["rederived_utc"],
                    "training_or_prediction_artifacts_modified": False,
                },
            }
        )
        _write_json(paths["composite_seal.json"], seal)
        result = validate_composite(output, config_path=config_path)
        if not result["valid"]:
            raise CompositeError(
                "V4 protocol re-derivation did not validate: "
                + repr(result["reasons"])
            )
        result["scientific_gate_reasons"] = gate.get("reasons", [])
        return result
    except Exception:
        for name, content in backups.items():
            paths[name].write_bytes(content)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--build", action="store_true")
    action.add_argument("--validate", type=Path)
    action.add_argument("--repair-inventory", type=Path)
    action.add_argument("--rederive-v4-protocol", type=Path)
    arguments = parser.parse_args(argv)
    try:
        if arguments.rederive_v4_protocol is not None:
            payload = rederive_v4_protocol(
                arguments.rederive_v4_protocol,
                config_path=arguments.config,
            )
            code = 0 if payload["valid"] else 2
        elif arguments.repair_inventory is not None:
            payload = repair_inventory_metadata(
                arguments.repair_inventory,
                config_path=arguments.config,
            )
            code = 0 if payload["valid"] else 2
        elif arguments.validate is not None:
            payload = validate_composite(arguments.validate, config_path=arguments.config)
            code = 0 if payload["valid"] else 2
        elif arguments.build:
            payload = build(arguments.config)
            code = 0 if payload.get("valid") is True else 3
        else:
            payload = _public_preflight(preflight(arguments.config))
            code = 0 if payload["ready"] else 3
    except Exception as error:
        payload = {
            "schema_version": "vimd_amc.tvt.composite_recovery.error.v1",
            "status": "error",
            "ready": False,
            "valid": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        code = 2
    print(json.dumps(runner.json_safe(payload), ensure_ascii=False, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
