"""Run and bind the preregistered 10k/30k/100k TVT v2 learning curve."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tvt_submission.formal_v2_contract import (  # noqa: E402
    DEFAULT_FREEZE,
    EXPECTED_CACHE_DESIGNATION,
    EXPECTED_CACHE_MASTER_SEED,
    EXPECTED_STRESS_AXES,
    load_contract,
    loaded_freeze_identity,
)
from vimd_amc.reproducibility import (  # noqa: E402
    TVT_V2_SOURCE_PROFILE,
    execution_environment_reasons,
    resolve_frozen_device,
    source_tree_audit_reasons,
    source_tree_record,
    source_tree_record_matches,
)


EVIDENCE_SCHEMA = "vimd_amc.tvt.learning_curve_evidence.v2"
LEARNING_CACHE_DESIGNATION = "learning_curve_v2_not_formal_evidence"


def _read_json(path: Path) -> dict[str, Any]:
    def reject_duplicate_keys(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            value[key] = item
        return value

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON value in {path}: {token}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _manifest_cache_digest(manifest: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in manifest.items()
        if key != "cache_digest"
    }
    canonical = json.dumps(
        [payload],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _split_sizes(configuration: Mapping[str, Any], label: str) -> dict[str, int]:
    raw = configuration.get("split_sizes")
    if not isinstance(raw, list):
        raise ValueError(f"{label} split_sizes must be a list")
    sizes: dict[str, int] = {}
    for item in raw:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or isinstance(item[1], bool)
            or not isinstance(item[1], int)
            or item[1] <= 0
        ):
            raise ValueError(f"{label} split_sizes are malformed")
        split, size = item
        if split in sizes:
            raise ValueError(f"{label} split_sizes contain duplicate {split}")
        sizes[split] = size
    return sizes


def _expected_learning_split_counts(
    contract: Mapping[str, Any],
    source_count: int,
) -> dict[str, int]:
    counts = {
        str(split): int(size)
        for split, size in contract["cache"][
            "expected_split_source_counts"
        ].items()
    }
    counts["train"] = source_count
    if source_count != int(
        contract["learning_curve"]["fixed_formal_scale_regardless_of_curve"]
    ):
        for split in EXPECTED_STRESS_AXES:
            counts.pop(split)
    return counts


def _resolve_run_artifact(
    run_directory: Path,
    relative: Any,
    label: str,
) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} path is missing")
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError(f"{label} must be relative to its run directory")
    root = run_directory.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes its run directory") from error
    if not resolved.is_file():
        raise ValueError(f"{label} file is missing: {resolved}")
    return resolved


def _finite_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def validate_learning_caches(
    cache_roots: list[Path],
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    learning = contract["learning_curve"]
    expected_counts = list(learning["training_source_counts"])
    if len(cache_roots) != len(expected_counts):
        raise ValueError("learning curve requires exactly three cache roots")
    resolved_roots = [path.resolve() for path in cache_roots]
    if len(set(resolved_roots)) != len(resolved_roots):
        raise ValueError("learning-curve cache roots contain duplicates")
    validation_source_ids: list[list[int]] = []
    bindings: list[dict[str, Any]] = []
    for cache_root, expected_train_count in zip(
        resolved_roots,
        expected_counts,
        strict=True,
    ):
        manifest_path = cache_root / "manifest.json"
        manifest = _read_json(manifest_path)
        configuration = manifest.get("configuration", {})
        if not isinstance(configuration, Mapping):
            raise ValueError(
                f"{cache_root} manifest configuration is malformed"
            )
        sizes = _split_sizes(configuration, str(cache_root))
        expected_sizes = _expected_learning_split_counts(
            contract,
            expected_train_count,
        )
        if sizes != expected_sizes:
            raise ValueError(
                f"{cache_root} split sizes drifted from the "
                f"{expected_train_count}-source learning contract"
            )
        expected_designation = (
            EXPECTED_CACHE_DESIGNATION
            if expected_train_count
            == learning["fixed_formal_scale_regardless_of_curve"]
            else LEARNING_CACHE_DESIGNATION
        )
        if not (
            configuration.get("master_seed")
            == EXPECTED_CACHE_MASTER_SEED
            and configuration.get("sample_length")
            == contract["cache"]["sample_length"]
            and configuration.get("guard_samples")
            == contract["cache"]["guard_samples"]
            and configuration.get("evidence_designation")
            == expected_designation
        ):
            raise ValueError(
                f"{cache_root} seed, dimensions, or designation drifted"
            )
        declared_digest = manifest.get("cache_digest")
        if (
            not _is_sha256(declared_digest)
            or declared_digest != _manifest_cache_digest(manifest)
        ):
            raise ValueError(
                f"{cache_root} cache manifest digest is invalid"
            )
        source_ids = manifest.get("source_ids")
        files = manifest.get("files")
        if (
            not isinstance(source_ids, Mapping)
            or set(source_ids) != set(expected_sizes)
            or not isinstance(files, Mapping)
            or set(files) != set(expected_sizes)
        ):
            raise ValueError(
                f"{cache_root} cache split artifacts are incomplete"
            )
        all_source_ids: set[int] = set()
        for split, expected_size in expected_sizes.items():
            identifiers = source_ids.get(split)
            if (
                not isinstance(identifiers, list)
                or len(identifiers) != expected_size
                or any(
                    isinstance(identifier, bool)
                    or not isinstance(identifier, int)
                    for identifier in identifiers
                )
                or len(set(identifiers)) != len(identifiers)
            ):
                raise ValueError(
                    f"{cache_root} {split} source IDs are malformed"
                )
            overlap = all_source_ids.intersection(identifiers)
            if overlap:
                raise ValueError(
                    f"{cache_root} source IDs overlap across splits"
                )
            all_source_ids.update(identifiers)
        current_validation = source_ids.get("validation")
        if (
            not isinstance(current_validation, list)
            or len(current_validation) != sizes.get("validation")
        ):
            raise ValueError(
                f"{cache_root} validation source IDs are malformed"
            )
        validation_source_ids.append(
            [int(value) for value in current_validation]
        )
        bindings.append(
            {
                "training_source_count": expected_train_count,
                "cache_root": str(cache_root),
                "cache_digest": declared_digest,
                "cache_manifest": str(manifest_path),
                "cache_manifest_sha256": _sha256(manifest_path),
                "split_source_counts": sizes,
                "validation_source_ids_sha256": hashlib.sha256(
                    json.dumps(
                        current_validation,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ).encode("utf-8")
                ).hexdigest(),
            }
        )
    if any(
        current != validation_source_ids[0]
        for current in validation_source_ids[1:]
    ):
        raise ValueError(
            "learning-curve caches do not share identical validation sources"
        )
    return bindings


def build_evidence(
    *,
    contract: Mapping[str, Any],
    run_directories: list[Path],
    cache_bindings: list[dict[str, Any]],
    output: Path,
    expected_source_tree: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    learning = contract["learning_curve"]
    expected_models = list(learning["models"])
    expected_seeds = list(learning["algorithm_seeds"])
    expected_counts = list(learning["training_source_counts"])
    if (
        len(run_directories) != len(expected_counts)
        or len(cache_bindings) != len(expected_counts)
    ):
        raise ValueError(
            "learning evidence requires exactly one run and cache per scale"
        )
    freeze_path, freeze_digest = loaded_freeze_identity(contract)
    if _sha256(freeze_path) != freeze_digest:
        raise ValueError(
            "formal v2 freeze changed after the contract was loaded"
        )
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite learning evidence: {output}"
        )
    expected_device = resolve_frozen_device(
        contract["experiment"]["device"]
    )
    standard_entrypoint = (
        ROOT / "experiments" / "run_standard_experiment.py"
    )
    bound_source_tree = (
        dict(expected_source_tree)
        if expected_source_tree is not None
        else source_tree_record(
            ROOT,
            standard_entrypoint,
            profile=TVT_V2_SOURCE_PROFILE,
        )
    )
    if not source_tree_record_matches(
        bound_source_tree,
        ROOT,
        standard_entrypoint,
        profile=TVT_V2_SOURCE_PROFILE,
    ):
        raise ValueError(
            "learning-curve source binding differs from current TVT-v2 sources"
        )
    required_epoch_metrics = tuple(learning["required_epoch_metrics"])
    required_scale_metrics = tuple(learning["required_scale_metrics"])
    protocol_tag = str(learning.get("protocol_tag", "v2"))
    run_digests: dict[str, str] = {}
    run_paths: dict[str, str] = {}
    cache_evidence: dict[str, Any] = {}
    fit_artifacts: dict[str, Any] = {}
    curves: dict[str, Any] = {}
    runtime_environments: dict[str, Any] = {}
    for source_count, run_directory, cache_binding in zip(
        expected_counts,
        run_directories,
        cache_bindings,
        strict=True,
    ):
        token = str(source_count)
        if cache_binding.get("training_source_count") != source_count:
            raise ValueError(
                f"cache binding is assigned to the wrong scale: {source_count}"
            )
        manifest_path = Path(
            str(cache_binding.get("cache_manifest", ""))
        ).resolve()
        if (
            not manifest_path.is_file()
            or _sha256(manifest_path)
            != cache_binding.get("cache_manifest_sha256")
        ):
            raise ValueError(
                f"cache manifest binding changed for {source_count}"
            )
        manifest = _read_json(manifest_path)
        if not (
            manifest.get("cache_digest")
            == cache_binding.get("cache_digest")
            and _manifest_cache_digest(manifest)
            == cache_binding.get("cache_digest")
        ):
            raise ValueError(
                f"cache digest binding changed for {source_count}"
            )
        run_directory = run_directory.resolve()
        run_path = run_directory / "run.json"
        run = _read_json(run_path)
        if (
            run.get("execution_status") != "complete"
            or run.get("status") != "complete"
            or run.get("runner")
            != "experiments/run_standard_experiment.py"
            or run.get("run_id")
            != f"tvt_learning_curve_{source_count}_{protocol_tag}"
            or run.get("models") != expected_models
            or run.get("seeds") != expected_seeds
            or run.get("checksums_verified") is not True
        ):
            raise ValueError(
                f"learning-curve run is incomplete or drifted: {run_path}"
            )
        run_cache_root = Path(
            str(run.get("cache_root", ""))
        ).expanduser().resolve()
        run_splits = run.get("splits")
        component_validation = run.get("component_validation")
        source_audit = run.get("source_tree_execution_audit")
        source_reasons = source_tree_audit_reasons(
            source_audit,
            ROOT,
            standard_entrypoint,
            expected_record=bound_source_tree,
            profile=TVT_V2_SOURCE_PROFILE,
        )
        environment = run.get("environment")
        environment_reasons = execution_environment_reasons(
            environment,
            expected_device=expected_device,
            expected_source_tree=bound_source_tree,
        )
        if not (
            run_cache_root
            == Path(str(cache_binding["cache_root"])).resolve()
            and run.get("cache_digest") == cache_binding["cache_digest"]
            and isinstance(run_splits, dict)
            and run_splits == cache_binding["split_source_counts"]
            and isinstance(component_validation, dict)
            and set(component_validation)
            == set(cache_binding["split_source_counts"])
            and not source_reasons
            and not environment_reasons
        ):
            raise ValueError(
                f"learning-curve run/cache/execution binding failed: "
                f"{run_path}; "
                + ",".join((*source_reasons, *environment_reasons))
            )
        expected_pairs = {
            (model, int(seed))
            for model in expected_models
            for seed in expected_seeds
        }
        results = run.get("results")
        if not isinstance(results, list):
            raise ValueError(
                f"learning-curve run lacks a result list: {run_path}"
            )
        observed_pairs: set[tuple[str, int]] = set()
        scale_curves: dict[str, Any] = {}
        scale_artifacts: dict[str, Any] = {}
        for result in results:
            if not isinstance(result, dict):
                raise ValueError(f"malformed result in {run_path}")
            seed_value = result.get("seed")
            if isinstance(seed_value, bool) or not isinstance(seed_value, int):
                raise ValueError(f"invalid fit seed in {run_path}")
            pair = (str(result.get("model")), seed_value)
            if pair in observed_pairs:
                raise ValueError(
                    f"duplicate learning-curve fit: {pair[0]}/seed{pair[1]}"
                )
            if pair not in expected_pairs:
                raise ValueError(
                    f"unexpected learning-curve fit: {pair[0]}/seed{pair[1]}"
                )
            observed_pairs.add(pair)
            pair_token = f"{pair[0]}/seed{pair[1]}"
            expected_result_path = (
                run_directory
                / "models"
                / f"{pair[0]}_seed{pair[1]}"
                / "result.json"
            ).resolve()
            if not expected_result_path.is_file():
                raise ValueError(
                    f"per-fit result.json is missing for {pair_token}"
                )
            per_fit_result = _read_json(expected_result_path)
            if per_fit_result != result:
                raise ValueError(
                    f"per-fit result.json disagrees with run.json for "
                    f"{pair_token}"
                )
            checkpoint_path = _resolve_run_artifact(
                run_directory,
                result.get("checkpoint"),
                f"checkpoint for {pair_token}",
            )
            expected_checkpoint = (
                expected_result_path.parent / "model.pt"
            ).resolve()
            if (
                checkpoint_path != expected_checkpoint
                or checkpoint_path.stat().st_size <= 0
            ):
                raise ValueError(
                    f"checkpoint is not the expected nonempty artifact for "
                    f"{pair_token}"
                )
            training = result.get("training")
            if not isinstance(training, dict):
                raise ValueError(f"training result is missing for {pair_token}")
            history = training.get("history")
            if not isinstance(history, list) or not history:
                raise ValueError(
                    f"training history is missing for {pair_token}"
                )
            observed_epochs: set[int] = set()
            for row in history:
                if not isinstance(row, dict):
                    raise ValueError(
                        f"malformed history row for {pair_token}"
                    )
                epoch_value = _finite_number(
                    row.get("epoch"),
                    f"history epoch for {pair_token}",
                )
                if (
                    not epoch_value.is_integer()
                    or int(epoch_value) <= 0
                    or int(epoch_value) in observed_epochs
                ):
                    raise ValueError(
                        f"history epochs are invalid or duplicated for "
                        f"{pair_token}"
                    )
                observed_epochs.add(int(epoch_value))
                for metric in required_epoch_metrics:
                    _finite_number(
                        row.get(metric),
                        f"{metric} for {pair_token}",
                    )
            selection = training.get("checkpoint_selection")
            if not isinstance(selection, dict):
                raise ValueError(
                    f"checkpoint selection is missing for {pair_token}"
                )
            selected_epoch = selection.get("selected_epoch")
            if (
                isinstance(selected_epoch, bool)
                or not isinstance(selected_epoch, int)
                or selection.get("status")
                != "eligible_validation_checkpoint_selected"
                or selection.get("selected_checkpoint_eligible") is not True
                or selection.get("fallback_used") is not False
                or isinstance(
                    selection.get("eligible_checkpoint_count"),
                    bool,
                )
                or not isinstance(
                    selection.get("eligible_checkpoint_count"),
                    int,
                )
                or selection["eligible_checkpoint_count"] <= 0
                or training.get("checkpoint_fallback_used") is not False
            ):
                raise ValueError(
                    f"fallback or ineligible checkpoint for {pair_token}"
                )
            selected_rows = [
                row
                for row in history
                if int(float(row["epoch"])) == selected_epoch
            ]
            if (
                len(selected_rows) != 1
                or _finite_number(
                    selected_rows[0].get(
                        "checkpoint_selection_eligible"
                    ),
                    f"selected checkpoint eligibility for {pair_token}",
                )
                <= 0.5
            ):
                raise ValueError(
                    f"selected learning-curve epoch is ineligible or "
                    f"ambiguous for {pair_token}"
                )
            selected_validation = _finite_number(
                selected_rows[0].get("validation_macro_f1"),
                f"selected validation macro-F1 for {pair_token}",
            )
            selected_loss = _finite_number(
                selected_rows[0].get("validation_loss"),
                f"selected validation loss for {pair_token}",
            )
            recorded_loss = _finite_number(
                selection.get("selected_validation_loss"),
                f"recorded selected validation loss for {pair_token}",
            )
            if not math.isclose(
                selected_loss,
                recorded_loss,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"selected checkpoint loss disagrees with history for "
                    f"{pair_token}"
                )
            regimes = result.get("regimes")
            if not isinstance(regimes, dict):
                raise ValueError(
                    f"evaluation results are missing for {pair_token}"
                )
            scale_metrics = {
                "selected_validation_macro_f1": selected_validation,
                "id_test_macro_f1": _finite_number(
                    (
                        regimes.get("id_test", {}).get("macro_f1")
                        if isinstance(regimes.get("id_test"), dict)
                        else None
                    ),
                    f"id_test macro-F1 for {pair_token}",
                ),
                "hard_interference_macro_f1": _finite_number(
                    (
                        regimes.get("hard_interference", {}).get("macro_f1")
                        if isinstance(
                            regimes.get("hard_interference"),
                            dict,
                        )
                        else None
                    ),
                    f"hard_interference macro-F1 for {pair_token}",
                ),
            }
            if tuple(scale_metrics) != required_scale_metrics:
                raise ValueError(
                    "learning-curve required scale metrics drifted"
                )
            scale_curves[pair_token] = {
                "history": [
                    {
                        metric: row[metric]
                        for metric in (
                            "epoch",
                            *learning["required_epoch_metrics"],
                        )
                    }
                    for row in history
                ],
                "selected_epoch": selected_epoch,
                **scale_metrics,
            }
            scale_artifacts[pair_token] = {
                "result_json": str(expected_result_path),
                "result_json_sha256": _sha256(expected_result_path),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": _sha256(checkpoint_path),
                "checkpoint_selection_status": selection["status"],
                "selected_epoch": selected_epoch,
            }
        if observed_pairs != expected_pairs:
            raise ValueError(
                f"learning-curve run has an incomplete fit grid: {run_path}"
            )
        run_digests[token] = _sha256(run_path)
        run_paths[token] = str(run_path.resolve())
        cache_evidence[token] = {
            key: value
            for key, value in cache_binding.items()
            if key != "training_source_count"
        }
        fit_artifacts[token] = scale_artifacts
        curves[token] = scale_curves
        runtime_environments[token] = environment
    evidence = {
        "schema_version": EVIDENCE_SCHEMA,
        "status": "complete",
        "freeze": str(freeze_path),
        "freeze_sha256": freeze_digest,
        "training_source_counts": expected_counts,
        "models": expected_models,
        "algorithm_seeds": expected_seeds,
        "required_epoch_metrics_present": True,
        "validation_source_ids_identical": True,
        "posthoc_scale_selected": False,
        "formal_scale": learning["fixed_formal_scale_regardless_of_curve"],
        "execution_device": expected_device,
        "source_tree_binding": bound_source_tree,
        "source_tree_unchanged_across_scales": True,
        "runtime_environments": runtime_environments,
        "run_json_sha256": run_digests,
        "run_json_paths": run_paths,
        "cache_bindings": cache_evidence,
        "fit_artifacts": fit_artifacts,
        "curves": curves,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            evidence,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--cache-10k", type=Path)
    parser.add_argument("--cache-30k", type=Path)
    parser.add_argument("--cache-100k", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    parser.add_argument(
        "--evidence-output",
        type=Path,
        default=(
            ROOT
            / "artifacts"
            / "tvt_learning_curve_v2"
            / "learning_curve_evidence.json"
        ),
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--validate-caches-only",
        action="store_true",
        help="validate exact cache sizes and shared validation IDs, then exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    contract = load_contract(arguments.freeze)
    freeze_path, freeze_digest = loaded_freeze_identity(contract)
    learning = contract["learning_curve"]
    experiment = contract["experiment"]
    protocol_tag = str(learning.get("protocol_tag", "v2"))
    if not protocol_tag or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
        for character in protocol_tag
    ):
        raise ValueError("learning_curve.protocol_tag is malformed")
    device = resolve_frozen_device(
        experiment["device"],
        arguments.device,
    )
    source_tree_start = source_tree_record(
        ROOT,
        ROOT / "experiments" / "run_standard_experiment.py",
        profile=TVT_V2_SOURCE_PROFILE,
    )
    if arguments.preflight_only:
        print(
            json.dumps(
                {
                    "ok": True,
                    "execution_started": False,
                    "training_source_counts": learning[
                        "training_source_counts"
                    ],
                    "models": learning["models"],
                    "algorithm_seeds": learning["algorithm_seeds"],
                    "device_contract": {
                        "frozen": experiment["device"],
                        "selected": device,
                        "matches": True,
                    },
                    "freeze": str(freeze_path),
                    "freeze_sha256": freeze_digest,
                    "cuda_execution": contract.get(
                        "execution_amendment", {}
                    ).get("cuda_execution"),
                    "source_tree_binding": source_tree_start,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    cache_roots = [
        arguments.cache_10k,
        arguments.cache_30k,
        arguments.cache_100k,
    ]
    if any(path is None for path in cache_roots):
        raise ValueError(
            "--cache-10k, --cache-30k, and --cache-100k are all required"
        )
    resolved_caches = [path.resolve() for path in cache_roots if path]
    cache_bindings = validate_learning_caches(
        resolved_caches,
        contract,
    )
    if arguments.validate_caches_only:
        print(
            json.dumps(
                {
                    "ok": True,
                    "execution_started": False,
                    "cache_validation": "passed",
                    "freeze_sha256": freeze_digest,
                    "cache_bindings": cache_bindings,
                },
                sort_keys=True,
            )
        )
        return 0
    training = experiment["training"]
    model = experiment["model"]
    runner_path = ROOT / "experiments" / "run_standard_experiment.py"
    output_root = arguments.output.resolve()
    run_directories = [
        output_root / f"tvt_learning_curve_{source_count}_{protocol_tag}"
        for source_count in learning["training_source_counts"]
    ]
    existing_runs = [
        str(path)
        for path in run_directories
        if path.exists()
    ]
    if existing_runs:
        raise FileExistsError(
            "refusing duplicate learning-curve fits; run directories "
            "already exist: " + ",".join(existing_runs)
        )
    evidence_output = arguments.evidence_output.resolve()
    if evidence_output.exists():
        raise FileExistsError(
            "refusing to overwrite existing learning evidence: "
            f"{evidence_output}"
        )
    commands: list[list[str]] = []
    for source_count, cache_root in zip(
        learning["training_source_counts"],
        resolved_caches,
        strict=True,
    ):
        run_id = f"tvt_learning_curve_{source_count}_{protocol_tag}"
        command = [
            sys.executable,
            str(runner_path),
            "--cache-root",
            str(cache_root),
            "--models",
            ",".join(learning["models"]),
            "--seeds",
            ",".join(str(value) for value in learning["algorithm_seeds"]),
            "--reference-model",
            "a0_backbone",
            "--device",
            device,
            "--source-binding-profile",
            TVT_V2_SOURCE_PROFILE,
            "--output",
            str(output_root),
            "--run-id",
            run_id,
            "--verify-checksums",
            "--validate-components",
            "--epochs",
            str(training["epochs"]),
            "--batch-size",
            str(training["batch_size"]),
            "--learning-rate",
            str(training["learning_rate"]),
            "--weight-decay",
            str(training["weight_decay"]),
            "--mask-start-epoch",
            str(training["mask_start_epoch"]),
            "--contrastive-start-epoch",
            str(training["contrastive_start_epoch"]),
            "--mask-ramp-epochs",
            str(training["mask_ramp_epochs"]),
            "--contrastive-ramp-epochs",
            str(training["contrastive_ramp_epochs"]),
            "--minimum-full-stage-epochs",
            str(training["minimum_full_stage_epochs"]),
            "--patience",
            str(training["patience"]),
            "--n-fft",
            str(model["n_fft"]),
            "--hop-length",
            str(model["hop_length"]),
            "--spectral-channels",
            str(model["spectral_channels"]),
            "--embedding-dim",
            str(model["embedding_dim"]),
            "--environment-dim",
            str(model["environment_dim"]),
            "--dropout",
            str(model["dropout"]),
            "--bootstrap-draws",
            str(experiment["statistics"]["bootstrap_draws"]),
            "--bootstrap-seed",
            str(experiment["statistics"]["bootstrap_seed"]),
        ]
        if training["use_amp"]:
            command.append("--use-amp")
        commands.append(command)
    cuda_execution = contract.get("execution_amendment", {}).get(
        "cuda_execution", {}
    )
    fit_concurrency = int(cuda_execution.get("cuda_fit_concurrency", 1))
    if fit_concurrency < 1:
        raise ValueError("cuda_fit_concurrency must be positive")
    for offset in range(0, len(commands), fit_concurrency):
        batch = commands[offset : offset + fit_concurrency]
        processes = [subprocess.Popen(command, cwd=ROOT) for command in batch]
        for process in processes:
            if process.wait() != 0:
                failed_process = process
                for sibling in processes:
                    if sibling.poll() is None:
                        sibling.terminate()
                for sibling in processes:
                    sibling.wait()
                raise subprocess.CalledProcessError(
                    failed_process.returncode or 1,
                    failed_process.args,
                )
    build_evidence(
        contract=contract,
        run_directories=run_directories,
        cache_bindings=cache_bindings,
        output=evidence_output,
        expected_source_tree=source_tree_start,
    )
    print(evidence_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
