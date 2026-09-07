"""Execute the immutable TVT v2 contract through the standard runner.

This wrapper changes no v1 constant at import time.  It loads the prospective
v2 freeze, verifies the exact cache and completed learning-curve prerequisite,
then binds the generic runner's formal globals for this process only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tvt_submission.formal_v2_contract import (  # noqa: E402
    DEFAULT_FREEZE,
    EXPECTED_STRESS_AXES,
    FormalV2ContractError,
    freeze_sha256,
    load_contract,
    loaded_freeze_identity,
)
from vimd_amc.standards import (  # noqa: E402
    TVT_V2_FACTOR_SPLITS,
    factor_isolated_split_policies_v2,
)
from vimd_amc.reproducibility import (  # noqa: E402
    TVT_V2_SOURCE_PROFILE,
    execution_environment_reasons,
    resolve_frozen_device,
    source_tree_audit_reasons,
    source_tree_record,
    source_tree_record_matches,
)

from experiments import run_standard_experiment as runner  # noqa: E402


LEARNING_EVIDENCE_SCHEMA = "vimd_amc.tvt.learning_curve_evidence.v2"


def _read_json(path: Path, label: str) -> dict[str, Any]:
    def reject_duplicate_keys(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{label} contains duplicate key {key}")
            value[key] = item
        return value

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(
                    f"{label} contains non-finite JSON value {token}"
                )
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
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


def _manifest_cache_digest(manifest: dict[str, Any]) -> str:
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


def _manifest_split_sizes(
    manifest: dict[str, Any],
    label: str,
) -> dict[str, int]:
    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError(f"{label} configuration is missing")
    raw_sizes = configuration.get("split_sizes")
    if not isinstance(raw_sizes, list):
        raise ValueError(f"{label} split sizes are malformed")
    sizes: dict[str, int] = {}
    for item in raw_sizes:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or isinstance(item[1], bool)
            or not isinstance(item[1], int)
            or item[1] <= 0
            or item[0] in sizes
        ):
            raise ValueError(f"{label} split sizes are malformed")
        sizes[item[0]] = item[1]
    return sizes


def _bound_file(path_value: Any, digest: Any, label: str) -> Path:
    if not isinstance(path_value, str) or not _is_sha256(digest):
        raise ValueError(f"{label} path or digest is malformed")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file() or _sha256(path) != digest:
        raise ValueError(f"{label} SHA-256 binding failed")
    return path


def validate_learning_prerequisite(
    path: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    evidence = _read_json(path, "learning-curve evidence")
    freeze_path, expected_freeze_digest = loaded_freeze_identity(contract)
    if freeze_sha256(freeze_path) != expected_freeze_digest:
        raise ValueError("formal v2 freeze changed after it was loaded")
    expected_counts = contract["learning_curve"]["training_source_counts"]
    expected_device = resolve_frozen_device(
        contract["experiment"]["device"]
    )
    bound_source_tree = evidence.get("source_tree_binding")
    standard_entrypoint = (
        ROOT / "experiments" / "run_standard_experiment.py"
    )
    if not (
        evidence.get("schema_version") == LEARNING_EVIDENCE_SCHEMA
        and evidence.get("status") == "complete"
        and evidence.get("freeze_sha256") == expected_freeze_digest
        and evidence.get("training_source_counts") == expected_counts
        and evidence.get("models") == contract["learning_curve"]["models"]
        and evidence.get("algorithm_seeds")
        == contract["learning_curve"]["algorithm_seeds"]
        and evidence.get("required_epoch_metrics_present") is True
        and evidence.get("validation_source_ids_identical") is True
        and evidence.get("posthoc_scale_selected") is False
        and evidence.get("formal_scale")
        == contract["learning_curve"][
            "fixed_formal_scale_regardless_of_curve"
        ]
        and evidence.get("execution_device") == expected_device
        and evidence.get("source_tree_unchanged_across_scales") is True
        and source_tree_record_matches(
            bound_source_tree,
            ROOT,
            standard_entrypoint,
            profile=TVT_V2_SOURCE_PROFILE,
        )
    ):
        raise ValueError(
            "learning-curve evidence is incomplete or drifted from the v2 freeze"
        )
    run_digests = evidence.get("run_json_sha256")
    run_paths = evidence.get("run_json_paths")
    cache_bindings = evidence.get("cache_bindings")
    fit_artifacts = evidence.get("fit_artifacts")
    curves = evidence.get("curves")
    runtime_environments = evidence.get("runtime_environments")
    expected_tokens = {str(value) for value in expected_counts}
    if (
        not isinstance(run_digests, dict)
        or set(run_digests) != expected_tokens
        or not isinstance(run_paths, dict)
        or set(run_paths) != expected_tokens
        or not isinstance(cache_bindings, dict)
        or set(cache_bindings) != expected_tokens
        or not isinstance(fit_artifacts, dict)
        or set(fit_artifacts) != expected_tokens
        or not isinstance(curves, dict)
        or set(curves) != expected_tokens
        or not isinstance(runtime_environments, dict)
        or set(runtime_environments) != expected_tokens
        or any(not _is_sha256(value) for value in run_digests.values())
    ):
        raise ValueError(
            "learning-curve evidence lacks the exact bound scale grid"
        )
    expected_models = list(contract["learning_curve"]["models"])
    expected_seeds = list(contract["learning_curve"]["algorithm_seeds"])
    expected_pairs = {
        (model, int(seed))
        for model in expected_models
        for seed in expected_seeds
    }
    expected_pair_tokens = {
        f"{model}/seed{seed}" for model, seed in expected_pairs
    }
    validation_id_digests: set[str] = set()
    for source_count in expected_counts:
        token = str(source_count)
        binding = cache_bindings[token]
        if not isinstance(binding, dict):
            raise ValueError(
                f"learning cache binding is malformed for {source_count}"
            )
        manifest_path = _bound_file(
            binding.get("cache_manifest"),
            binding.get("cache_manifest_sha256"),
            f"learning cache manifest {source_count}",
        )
        cache_root = Path(
            str(binding.get("cache_root", ""))
        ).expanduser().resolve()
        if manifest_path != cache_root / "manifest.json":
            raise ValueError(
                f"learning cache path binding failed for {source_count}"
            )
        manifest = _read_json(
            manifest_path,
            f"learning cache manifest {source_count}",
        )
        cache_digest = binding.get("cache_digest")
        if not (
            _is_sha256(cache_digest)
            and manifest.get("cache_digest") == cache_digest
            and _manifest_cache_digest(manifest) == cache_digest
        ):
            raise ValueError(
                f"learning cache digest binding failed for {source_count}"
            )
        expected_split_counts = {
            str(split): int(count)
            for split, count in contract["cache"][
                "expected_split_source_counts"
            ].items()
        }
        expected_split_counts["train"] = int(source_count)
        if source_count != contract["learning_curve"][
            "fixed_formal_scale_regardless_of_curve"
        ]:
            for split in EXPECTED_STRESS_AXES:
                expected_split_counts.pop(split)
        if not (
            binding.get("split_source_counts") == expected_split_counts
            and _manifest_split_sizes(
                manifest,
                f"learning cache manifest {source_count}",
            )
            == expected_split_counts
        ):
            raise ValueError(
                f"learning cache split counts drifted for {source_count}"
            )
        source_ids = manifest.get("source_ids")
        validation_ids = (
            source_ids.get("validation")
            if isinstance(source_ids, dict)
            else None
        )
        if not isinstance(validation_ids, list):
            raise ValueError(
                f"learning cache validation IDs missing for {source_count}"
            )
        validation_digest = hashlib.sha256(
            json.dumps(
                validation_ids,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
        if (
            validation_digest
            != binding.get("validation_source_ids_sha256")
        ):
            raise ValueError(
                f"learning cache validation IDs drifted for {source_count}"
            )
        validation_id_digests.add(validation_digest)

        run_path = _bound_file(
            run_paths[token],
            run_digests[token],
            f"learning-curve run {source_count}",
        )
        protocol_tag = str(
            contract["learning_curve"].get("protocol_tag", "v2")
        )
        expected_run_id = (
            f"tvt_learning_curve_{source_count}_{protocol_tag}"
        )
        if (
            run_path.name != "run.json"
            or run_path.parent.name != expected_run_id
        ):
            raise ValueError(
                f"learning-curve run path drifted for {source_count}"
            )
        run = _read_json(run_path, f"learning-curve run {source_count}")
        source_reasons = source_tree_audit_reasons(
            run.get("source_tree_execution_audit"),
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
            run.get("execution_status") == "complete"
            and run.get("status") == "complete"
            and run.get("run_id") == expected_run_id
            and run.get("models") == expected_models
            and run.get("seeds") == expected_seeds
            and run.get("cache_digest") == cache_digest
            and Path(str(run.get("cache_root", ""))).resolve()
            == cache_root
            and run.get("splits") == expected_split_counts
            and run.get("checksums_verified") is True
            and isinstance(run.get("component_validation"), dict)
            and set(run["component_validation"])
            == set(expected_split_counts)
            and environment == runtime_environments[token]
            and not source_reasons
            and not environment_reasons
        ):
            raise ValueError(
                f"learning-curve run drifted for {source_count}: "
                + ",".join((*source_reasons, *environment_reasons))
            )
        results = run.get("results")
        if not isinstance(results, list):
            raise ValueError(
                f"learning-curve fit results missing for {source_count}"
            )
        observed: set[tuple[str, int]] = set()
        scale_artifacts = fit_artifacts[token]
        scale_curves = curves[token]
        if (
            not isinstance(scale_artifacts, dict)
            or set(scale_artifacts) != expected_pair_tokens
            or not isinstance(scale_curves, dict)
            or set(scale_curves) != expected_pair_tokens
        ):
            raise ValueError(
                f"learning-curve per-fit evidence is incomplete for "
                f"{source_count}"
            )
        for result in results:
            if not isinstance(result, dict):
                raise ValueError(
                    f"malformed learning fit for {source_count}"
                )
            seed = result.get("seed")
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError(
                    f"invalid learning fit seed for {source_count}"
                )
            pair = (str(result.get("model")), seed)
            if pair in observed:
                raise ValueError(
                    f"duplicate learning fit: {pair[0]}/seed{pair[1]}"
                )
            observed.add(pair)
            if pair not in expected_pairs:
                raise ValueError(
                    f"unexpected learning fit: {pair[0]}/seed{pair[1]}"
                )
            pair_token = f"{pair[0]}/seed{pair[1]}"
            selection = (
                result.get("training", {}).get("checkpoint_selection")
                if isinstance(result.get("training"), dict)
                else None
            )
            if not (
                isinstance(selection, dict)
                and selection.get("status")
                == "eligible_validation_checkpoint_selected"
                and selection.get("selected_checkpoint_eligible") is True
                and selection.get("fallback_used") is False
                and isinstance(selection.get("selected_epoch"), int)
                and not isinstance(selection.get("selected_epoch"), bool)
                and isinstance(
                    selection.get("eligible_checkpoint_count"),
                    int,
                )
                and not isinstance(
                    selection.get("eligible_checkpoint_count"),
                    bool,
                )
                and selection["eligible_checkpoint_count"] > 0
            ):
                raise ValueError(
                    f"fallback or ineligible learning checkpoint: "
                    f"{pair_token}"
                )
            artifact = scale_artifacts[pair_token]
            if not isinstance(artifact, dict):
                raise ValueError(
                    f"learning fit artifact binding malformed: {pair_token}"
                )
            result_path = _bound_file(
                artifact.get("result_json"),
                artifact.get("result_json_sha256"),
                f"learning result {pair_token}",
            )
            checkpoint_path = _bound_file(
                artifact.get("checkpoint"),
                artifact.get("checkpoint_sha256"),
                f"learning checkpoint {pair_token}",
            )
            expected_model_root = (
                run_path.parent
                / "models"
                / f"{pair[0]}_seed{pair[1]}"
            ).resolve()
            recorded_checkpoint = Path(
                str(result.get("checkpoint", ""))
            )
            if not (
                result_path == expected_model_root / "result.json"
                and checkpoint_path == expected_model_root / "model.pt"
                and not recorded_checkpoint.is_absolute()
                and (run_path.parent / recorded_checkpoint).resolve()
                == checkpoint_path
                and _read_json(
                    result_path,
                    f"learning result {pair_token}",
                )
                == result
                and artifact.get("checkpoint_selection_status")
                == selection["status"]
                and artifact.get("selected_epoch")
                == selection["selected_epoch"]
            ):
                raise ValueError(
                    f"learning fit artifact binding failed: {pair_token}"
                )
            history = result["training"].get("history")
            required_metrics = contract["learning_curve"][
                "required_epoch_metrics"
            ]
            if not isinstance(history, list) or not history:
                raise ValueError(
                    f"learning fit history is missing: {pair_token}"
                )
            try:
                projected_history = [
                    {
                        metric: row[metric]
                        for metric in ("epoch", *required_metrics)
                    }
                    for row in history
                    if isinstance(row, dict)
                ]
                selected_rows = [
                    row
                    for row in history
                    if isinstance(row, dict)
                    and int(float(row["epoch"]))
                    == selection["selected_epoch"]
                ]
                expected_curve = {
                    "history": projected_history,
                    "selected_epoch": selection["selected_epoch"],
                    "selected_validation_macro_f1": float(
                        selected_rows[0]["validation_macro_f1"]
                    ),
                    "id_test_macro_f1": float(
                        result["regimes"]["id_test"]["macro_f1"]
                    ),
                    "hard_interference_macro_f1": float(
                        result["regimes"]["hard_interference"]["macro_f1"]
                    ),
                }
            except (
                IndexError,
                KeyError,
                TypeError,
                ValueError,
                OverflowError,
            ) as error:
                raise ValueError(
                    f"learning fit curve is not derivable: {pair_token}"
                ) from error
            if (
                len(projected_history) != len(history)
                or len(selected_rows) != 1
                or scale_curves[pair_token] != expected_curve
            ):
                raise ValueError(
                    f"learning fit curve binding failed: {pair_token}"
                )
        if observed != expected_pairs:
            raise ValueError(
                f"learning fit grid is incomplete for {source_count}"
            )
    if len(validation_id_digests) != 1:
        raise ValueError(
            "learning-curve caches no longer share validation source IDs"
        )
    return evidence


def validate_cache(cache_root: Path, contract: dict[str, Any]) -> None:
    manifest = _read_json(cache_root / "manifest.json", "cache manifest")
    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("cache manifest configuration is missing")
    expected_cache = contract["cache"]
    observed_sizes = {
        str(split): int(size)
        for split, size in configuration.get("split_sizes", ())
    }
    if observed_sizes != expected_cache["expected_split_source_counts"]:
        raise ValueError("cache split sizes differ from formal v2 freeze")
    if (
        configuration.get("master_seed") != expected_cache["master_seed"]
        or configuration.get("sample_length")
        != expected_cache["sample_length"]
        or configuration.get("guard_samples")
        != expected_cache["guard_samples"]
        or configuration.get("evidence_designation")
        != expected_cache["expected_designation"]
    ):
        raise ValueError(
            "cache seed, dimensions, or evidence designation drifted"
        )
    policy = manifest.get("preregistered_split_policy")
    if not isinstance(policy, dict) or set(policy) != set(TVT_V2_FACTOR_SPLITS):
        raise ValueError("cache lacks the exact v2 split policy")
    expected_policies = {
        record.split: record
        for record in factor_isolated_split_policies_v2(observed_sizes)
    }
    for split in TVT_V2_FACTOR_SPLITS:
        record = policy.get(split)
        expected = expected_policies[split]
        if not isinstance(record, dict):
            raise ValueError(f"cache policy missing for {split}")
        for key in (
            "adc_bits",
            "adc_full_scale",
            "per_emitter_cfo_norm_max",
            "per_emitter_timing_offset_max_samples",
        ):
            defaults = {
                "adc_bits": None,
                "adc_full_scale": 3.0,
                "per_emitter_cfo_norm_max": 0.0,
                "per_emitter_timing_offset_max_samples": 0,
            }
            if record.get(key, defaults[key]) != getattr(
                expected,
                key,
                defaults[key],
            ):
                raise ValueError(
                    f"cache receiver-stress policy drifted: {split}.{key}"
                )


def bind_runner_contract(contract: dict[str, Any]) -> None:
    experiment = contract["experiment"]
    family = experiment["confirmatory_family"]
    gates = experiment["scientific_release_gates"]
    runner.FORMAL_RELEASE_DESIGNATION = contract["cache"][
        "expected_designation"
    ]
    runner.FORMAL_METHOD_MODEL = gates["method_model"]
    runner.FORMAL_PRIMARY_REFERENCE_MODEL = gates[
        "primary_reference_model"
    ]
    runner.FORMAL_REQUIRED_NONORACLE_BASELINES = tuple(
        gates["required_nonoracle_baselines"]
    )
    runner.FORMAL_HOLM_CANDIDATES = tuple(experiment["holm_candidates"])
    runner.FORMAL_ABLATION_FAMILY_ID = family["family_id"]
    runner.FORMAL_ABLATION_REGIME = family["regime"]
    runner.FORMAL_ABLATION_METRIC = family["metric"]
    runner.FORMAL_ABLATION_DIRECTION = family["direction"]
    runner.FORMAL_ABLATION_CONFIDENCE_LEVEL = family["confidence_level"]
    runner.FORMAL_ABLATION_MULTIPLICITY_METHOD = family[
        "multiplicity_method"
    ]
    runner.FORMAL_ABLATION_CONTRASTS = tuple(family["contrasts"])
    runner.FORMAL_ABLATION_GATE_THRESHOLD = family[
        "simultaneous_ci95_low_strictly_greater_than"
    ]
    runner.FORMAL_ABLATION_ALGORITHM_SEEDS = tuple(experiment["seeds"])
    runner.SCIENTIFIC_RELEASE_THRESHOLDS = gates
    runner.HEADLINE_DESIGNATIONS = frozenset(
        {
            *runner.HEADLINE_DESIGNATIONS,
            runner.FORMAL_RELEASE_DESIGNATION,
        }
    )
    runner.PREREGISTERED_MODEL_SUITES = {
        **runner.PREREGISTERED_MODEL_SUITES,
        "headline": tuple(experiment["models"]),
    }
    headline_minimums = dict(runner.EVIDENCE_MINIMUMS["headline"])
    headline_minimums["seeds"] = len(experiment["seeds"])
    headline_minimums["split_samples"] = {
        "train": 100_000,
        "validation": 2_000,
        "heldout_channel": 5_000,
    }
    headline_minimums["per_class_samples"] = {
        "train": 10_000,
        "validation": 200,
        "heldout_channel": 500,
    }
    runner.EVIDENCE_MINIMUMS = {
        **runner.EVIDENCE_MINIMUMS,
        "headline": headline_minimums,
    }
    runner.FACTOR_ISOLATED_SPLITS = TVT_V2_FACTOR_SPLITS
    runner.factor_isolated_split_policies = factor_isolated_split_policies_v2


def bound_scientific_gate_payload(
    *,
    gate: dict[str, Any],
    freeze_path: Path,
    freeze_digest: str,
    learning_evidence: dict[str, Any],
) -> dict[str, Any]:
    """Return the scientific gate with its exact freeze-byte provenance."""

    if not (
        _is_sha256(freeze_digest)
        and freeze_sha256(freeze_path) == freeze_digest
        and learning_evidence.get("freeze_sha256") == freeze_digest
    ):
        raise ValueError(
            "scientific gate cannot be bound to inconsistent freeze identity"
        )
    bound = {
        **gate,
        "freeze": str(freeze_path.resolve()),
        "freeze_sha256": freeze_digest,
        "learning_curve_freeze_sha256": learning_evidence[
            "freeze_sha256"
        ],
        "freeze_identity_verified": True,
    }
    return bound


def write_bound_scientific_gate(
    *,
    gate: dict[str, Any],
    output: Path,
    freeze_path: Path,
    freeze_digest: str,
    learning_evidence: dict[str, Any],
) -> dict[str, Any]:
    """Write the exact payload produced by the shared pure binding helper."""

    bound = bound_scientific_gate_payload(
        gate=gate,
        freeze_path=freeze_path,
        freeze_digest=freeze_digest,
        learning_evidence=learning_evidence,
    )
    output.write_text(
        json.dumps(
            bound,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return bound


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--learning-curve-evidence", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate the freeze and print the bound execution plan only",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        contract = load_contract(arguments.freeze)
    except FormalV2ContractError as error:
        raise ValueError(str(error)) from error
    freeze_path, freeze_digest = loaded_freeze_identity(contract)
    experiment = contract["experiment"]
    device = resolve_frozen_device(
        experiment["device"],
        arguments.device,
    )
    source_tree_start = source_tree_record(
        ROOT,
        Path(runner.__file__),
        profile=TVT_V2_SOURCE_PROFILE,
    )
    if arguments.preflight_only:
        print(
            json.dumps(
                {
                    "ok": True,
                    "execution_started": False,
                    "run_id": experiment["run_id"],
                    "models": experiment["models"],
                    "seeds": experiment["seeds"],
                    "training_sources": contract["cache"][
                        "expected_split_source_counts"
                    ]["train"],
                    "learning_curve_required": True,
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
    if arguments.cache_root is None:
        raise ValueError("--cache-root is required for execution")
    if arguments.learning_curve_evidence is None:
        raise ValueError(
            "--learning-curve-evidence is required before confirmatory execution"
        )
    validate_cache(arguments.cache_root.resolve(), contract)
    learning_evidence = validate_learning_prerequisite(
        arguments.learning_curve_evidence.resolve(),
        contract,
    )
    bind_runner_contract(contract)
    training = experiment["training"]
    model = experiment["model"]
    statistics = experiment["statistics"]
    command = [
        str(Path(runner.__file__).resolve()),
        "--cache-root",
        str(arguments.cache_root.resolve()),
        "--models",
        ",".join(experiment["models"]),
        "--seeds",
        ",".join(str(value) for value in experiment["seeds"]),
        "--reference-model",
        experiment["reference_model"],
        "--holm-candidates",
        ",".join(experiment["holm_candidates"]),
        "--device",
        device,
        "--source-binding-profile",
        TVT_V2_SOURCE_PROFILE,
        "--output",
        str((arguments.output or ROOT / experiment["output"]).resolve()),
        "--run-id",
        experiment["run_id"],
        "--cpu-threads",
        str(arguments.cpu_threads),
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
        str(statistics["bootstrap_draws"]),
        "--bootstrap-seed",
        str(statistics["bootstrap_seed"]),
    ]
    if training["use_amp"]:
        command.append("--use-amp")
    original_argv = sys.argv
    try:
        sys.argv = command
        runner.main()
    finally:
        sys.argv = original_argv
    if not source_tree_record_matches(
        source_tree_start,
        ROOT,
        Path(runner.__file__),
        profile=TVT_V2_SOURCE_PROFILE,
    ):
        raise RuntimeError(
            "TVT-v2 execution sources changed during confirmatory execution"
        )
    if freeze_sha256(freeze_path) != freeze_digest:
        raise RuntimeError(
            "formal v2 freeze changed during confirmatory execution"
        )
    from tvt_submission.validate_v2_release import derive_gate

    run_json = (
        (arguments.output or ROOT / experiment["output"]).resolve()
        / experiment["run_id"]
        / "run.json"
    )
    gate = derive_gate(
        run_json=run_json,
        learning_evidence=arguments.learning_curve_evidence.resolve(),
        freeze=freeze_path,
        write=False,
    )
    gate = write_bound_scientific_gate(
        gate=gate,
        output=run_json.parent / "v2_scientific_release_gate.json",
        freeze_path=freeze_path,
        freeze_digest=freeze_digest,
        learning_evidence=learning_evidence,
    )
    if gate["passed"] is not True:
        raise RuntimeError(
            "formal v2 execution completed but the scientific release gate "
            "remains closed: " + ",".join(gate["reasons"])
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
