"""Fail-closed V4R recovery scheduler for the interrupted TVT headline run.

This is a prospective recovery utility.  It never writes into the sealed V4
run directory.  It validates the exact recovery configuration and source run,
can seal a byte-level inventory of the source artifacts, and executes at most
one missing CSSL seed in each fresh Python/CUDA process.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable
import uuid


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "tvt_submission"
    / "configs"
    / "formal_tvt_recovery_v4r_110plus10.json"
)
DEFAULT_WORKER = ROOT / "tvt_submission" / "recovery_v4r_worker.py"
EXPECTED_CONFIG_SHA256 = (
    "e88926f3c4d0958074524665b9543734fc89455d0b463a967ae70da382cbb916"
)
CONFIG_SCHEMA = "vimd_amc.tvt.formal_recovery.v4r_110plus10"
WORKER_SCHEMA = "vimd_amc.tvt.recovery_worker.v4r"
INVENTORY_SCHEMA = "vimd_amc.tvt.recovery_source_inventory.v4r"
STATE_SCHEMA = "vimd_amc.tvt.recovery_scheduler_state.v4r"
MODEL = "cssl_amc_supervised_adaptation"
MUTEX_NAME = r"Local\VIMD_AMC_TVT_V4R_RECOVERY_110PLUS10"


class RecoveryError(RuntimeError):
    """Raised when any recovery invariant fails closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _read_json(path: Path, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RecoveryError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                RecoveryError(f"{label} contains non-finite value {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RecoveryError(f"could not read {label}: {error}") from error
    if not isinstance(value, dict):
        raise RecoveryError(f"{label} must be a JSON object")
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def _exclusive_scheduler_mutex() -> Iterable[None]:
    """Hold a process-lifetime Windows mutex; fail closed on another owner."""

    kernel32 = ctypes.windll.kernel32
    kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    handle = kernel32.CreateMutexW(None, True, MUTEX_NAME)
    if not handle:
        raise RecoveryError("could not create the V4R scheduler mutex")
    already_exists = kernel32.GetLastError() == 183
    if already_exists:
        kernel32.CloseHandle(handle)
        raise RecoveryError("another V4R recovery scheduler is already active")
    try:
        yield
    finally:
        kernel32.ReleaseMutex(handle)
        kernel32.CloseHandle(handle)


def _resolve_repo_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RecoveryError(f"{label} must be a non-empty repository path")
    raw = Path(value)
    if raw.is_absolute():
        raise RecoveryError(f"{label} must be repository-relative")
    resolved = (ROOT / raw).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as error:
        raise RecoveryError(f"{label} escapes repository root") from error
    return resolved


def _require_exact_keys(
    value: dict[str, Any], expected: Iterable[str], label: str
) -> None:
    expected_set = set(expected)
    if set(value) != expected_set:
        raise RecoveryError(
            f"{label} keys mismatch; missing={sorted(expected_set - set(value))}, "
            f"unexpected={sorted(set(value) - expected_set)}"
        )


def load_recovery_config(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.expanduser().resolve()
    if resolved != DEFAULT_CONFIG.resolve():
        raise RecoveryError(
            f"recovery config path must be canonical: {DEFAULT_CONFIG.resolve()}"
        )
    digest = _sha256(resolved)
    if digest != EXPECTED_CONFIG_SHA256:
        raise RecoveryError(
            "recovery config SHA-256 mismatch: "
            f"{digest} != {EXPECTED_CONFIG_SHA256}"
        )
    config = _read_json(resolved, "recovery config")
    _require_exact_keys(
        config,
        {
            "schema_version",
            "status",
            "created_date",
            "purpose",
            "source_run",
            "bindings",
            "missing_fit_plan",
            "scientific_invariants",
            "resource_safety",
            "composite_evidence",
        },
        "recovery config",
    )
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise RecoveryError("recovery config schema mismatch")
    return config, digest


def _validate_bound_file(path: Path, digest: Any, label: str) -> None:
    if not path.is_file() or not _is_sha256(digest):
        raise RecoveryError(f"{label} is absent or has malformed SHA-256")
    actual = _sha256(path)
    if actual != digest:
        raise RecoveryError(f"{label} SHA-256 mismatch: {actual} != {digest}")


def validate_bindings(config: dict[str, Any]) -> dict[str, str]:
    source = config["source_run"]
    bindings = config["bindings"]
    records = {
        "source_run_json": (
            _resolve_repo_path(source["run_json"], "source_run.run_json"),
            source["run_json_sha256_at_seal"],
        ),
        "v4_freeze": (
            _resolve_repo_path(bindings["v4_freeze"], "bindings.v4_freeze"),
            bindings["v4_freeze_sha256"],
        ),
        "learning_evidence": (
            _resolve_repo_path(
                bindings["learning_evidence"], "bindings.learning_evidence"
            ),
            bindings["learning_evidence_sha256"],
        ),
        "headline_cache_manifest": (
            _resolve_repo_path(
                bindings["headline_cache_manifest"],
                "bindings.headline_cache_manifest",
            ),
            bindings["headline_cache_manifest_sha256"],
        ),
    }
    for label, (path, digest) in records.items():
        _validate_bound_file(path, digest, label)
    return {label: str(path) for label, (path, _) in records.items()}


def validate_source_run(
    config: dict[str, Any], source_run_path: Path
) -> dict[str, Any]:
    run = _read_json(source_run_path, "sealed V4 run.json")
    source = config["source_run"]
    plan = config["missing_fit_plan"]
    invariants = config["scientific_invariants"]
    if run.get("run_id") != source["run_id"]:
        raise RecoveryError("sealed source run_id mismatch")
    if run.get("cache_digest") != source["cache_digest"]:
        raise RecoveryError("sealed source cache digest mismatch")
    environment = run.get("environment")
    source_tree = environment.get("source_tree") if isinstance(environment, dict) else None
    if not isinstance(source_tree, dict) or source_tree.get(
        "aggregate_digest"
    ) != source["source_tree_aggregate_digest"]:
        raise RecoveryError("sealed source-tree digest mismatch")
    models = run.get("models")
    seeds = run.get("seeds")
    results = run.get("results")
    if not isinstance(models, list) or not isinstance(seeds, list) or not isinstance(
        results, list
    ):
        raise RecoveryError("sealed source model/seed/result grid is malformed")
    expected_missing_seeds = plan.get("seeds")
    if (
        plan.get("model") != MODEL
        or seeds != expected_missing_seeds
        or MODEL not in models
        or len(set(models)) != len(models)
        or len(set(seeds)) != len(seeds)
    ):
        raise RecoveryError("sealed source grid disagrees with recovery plan")
    observed: dict[tuple[str, int], dict[str, Any]] = {}
    for result in results:
        if not isinstance(result, dict):
            raise RecoveryError("sealed source contains malformed fit result")
        key = (str(result.get("model")), int(result.get("seed", -1)))
        if key in observed:
            raise RecoveryError(f"sealed source has duplicate fit {key}")
        observed[key] = result
    full_grid = {(str(model), int(seed)) for model in models for seed in seeds}
    expected_missing = {(MODEL, int(seed)) for seed in expected_missing_seeds}
    if len(results) != int(source["completed_fit_count"]):
        raise RecoveryError("sealed source completed-fit count mismatch")
    if len(full_grid) != int(source["expected_fit_count"]):
        raise RecoveryError("sealed source expected-fit count mismatch")
    if full_grid.difference(observed) != expected_missing:
        raise RecoveryError(
            "sealed source missing grid is not exactly the ten planned CSSL seeds"
        )
    if set(observed).difference(full_grid):
        raise RecoveryError("sealed source contains unexpected fits")
    expected_regimes = set(run.get("splits", {})).difference({"train"})
    if not expected_regimes:
        raise RecoveryError("sealed source has no evaluation regime contract")
    run_root = source_run_path.parent.resolve()
    for (model, seed), result in observed.items():
        label = f"{model}/seed{seed}"
        result_path = run_root / "models" / f"{model}_seed{seed}" / "result.json"
        if _read_json(result_path, f"source result {label}") != result:
            raise RecoveryError(f"source result.json disagrees for {label}")
        checkpoint = result.get("checkpoint")
        if not isinstance(checkpoint, str):
            raise RecoveryError(f"source checkpoint path missing for {label}")
        checkpoint_path = (run_root / checkpoint).resolve()
        try:
            checkpoint_path.relative_to(run_root)
        except ValueError as error:
            raise RecoveryError(f"source checkpoint escapes run root: {label}") from error
        if not checkpoint_path.is_file() or checkpoint_path.stat().st_size <= 0:
            raise RecoveryError(f"source checkpoint missing or empty: {label}")
        regimes = result.get("regimes")
        if not isinstance(regimes, dict) or set(regimes) != expected_regimes:
            raise RecoveryError(f"source evaluation regimes incomplete: {label}")
        model_root = result_path.parent
        for regime in expected_regimes:
            prediction = model_root / f"predictions_{regime}.npz"
            if not prediction.is_file() or prediction.stat().st_size <= 0:
                raise RecoveryError(f"source prediction missing: {label}/{regime}")
    training = run.get("training_configuration")
    model_config = run.get("model_configuration")
    if not isinstance(training, dict) or not isinstance(model_config, dict):
        raise RecoveryError("sealed source training/model configuration missing")
    for key in (
        "batch_size",
        "epochs",
        "learning_rate",
        "weight_decay",
        "mask_start_epoch",
        "contrastive_start_epoch",
        "mask_ramp_epochs",
        "contrastive_ramp_epochs",
        "minimum_full_stage_epochs",
        "patience",
        "use_amp",
    ):
        if training.get(key) != invariants.get(
            "training_batch_size" if key == "batch_size" else key
        ):
            raise RecoveryError(f"sealed source training invariant drift: {key}")
    for key in (
        "n_fft",
        "hop_length",
        "spectral_channels",
        "embedding_dim",
        "environment_dim",
        "dropout",
    ):
        if model_config.get(key) != invariants.get(key):
            raise RecoveryError(f"sealed source model invariant drift: {key}")
    comparison = run.get("comparison_protocol")
    if not isinstance(comparison, dict) or comparison.get(
        "bootstrap_draws"
    ) != invariants["bootstrap_draws"] or comparison.get(
        "bootstrap_seed_base"
    ) != invariants["bootstrap_seed"]:
        raise RecoveryError("sealed source bootstrap invariant drift")
    return {
        "run": run,
        "observed_fit_count": len(observed),
        "missing": [f"{MODEL}/seed{seed}" for seed in expected_missing_seeds],
        "evaluation_regimes": sorted(expected_regimes),
    }


def build_source_inventory(
    config: dict[str, Any],
    config_digest: str,
    source_run_path: Path,
    *,
    worker: Path,
) -> dict[str, Any]:
    run_root = source_run_path.parent.resolve()
    before = _sha256(source_run_path)
    files = [path for path in run_root.rglob("*") if path.is_file()]
    records: list[dict[str, Any]] = []
    for path in sorted(files, key=lambda item: item.relative_to(run_root).as_posix()):
        stat_before = path.stat()
        digest = _sha256(path)
        stat_after = path.stat()
        if (
            stat_before.st_size != stat_after.st_size
            or stat_before.st_mtime_ns != stat_after.st_mtime_ns
        ):
            raise RecoveryError(f"source artifact changed while sealing: {path}")
        records.append(
            {
                "path": path.relative_to(run_root).as_posix(),
                "size": stat_after.st_size,
                "sha256": digest,
            }
        )
    after = _sha256(source_run_path)
    expected = config["source_run"]["run_json_sha256_at_seal"]
    if before != expected or after != expected:
        raise RecoveryError("source run.json changed while inventory was generated")
    return {
        "schema_version": INVENTORY_SCHEMA,
        "created_utc": _utc_now(),
        "recovery_config_sha256": config_digest,
        "source_run_id": config["source_run"]["run_id"],
        "source_run_root": str(run_root),
        "source_run_json_sha256": expected,
        "scheduler": {
            "path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "worker": {
            "path": worker.resolve().relative_to(ROOT.resolve()).as_posix(),
            "sha256": _sha256(worker.resolve()),
        },
        "file_count": len(records),
        "files": records,
    }


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def available_ram_gib() -> float:
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise RecoveryError("could not query Windows memory status")
    return float(status.ullAvailPhys) / (1024.0**3)


def free_gpu_mib() -> int:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RecoveryError("nvidia-smi is unavailable")
    completed = subprocess.run(
        [
            executable,
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RecoveryError(f"nvidia-smi failed: {completed.stderr.strip()}")
    try:
        values = [int(line.strip()) for line in completed.stdout.splitlines() if line.strip()]
    except ValueError as error:
        raise RecoveryError("nvidia-smi returned malformed free-memory values") from error
    if len(values) != 1:
        raise RecoveryError("V4R requires exactly one visible GPU")
    return values[0]


def _safe_relative(base: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise RecoveryError(f"{label} must be a non-empty relative path")
    candidate = (base / value).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError as error:
        raise RecoveryError(f"{label} escapes worker directory") from error
    return candidate


def validate_worker_directory(
    directory: Path,
    *,
    config: dict[str, Any],
    config_digest: str,
    seed: int,
    expected_regimes: set[str],
) -> dict[str, Any]:
    manifest_path = directory / "worker_manifest.json"
    manifest = _read_json(manifest_path, f"worker manifest seed{seed}")
    if not (
        manifest.get("schema_version") == WORKER_SCHEMA
        and manifest.get("status") == "complete"
        and manifest.get("seed") == seed
        and manifest.get("model") == MODEL
        and manifest.get("run_id")
        == config["missing_fit_plan"]["worker_run_id_template"].format(seed=seed)
        and manifest.get("recovery_config_sha256") == config_digest
        and manifest.get("source_run_json_sha256")
        == config["source_run"]["run_json_sha256_at_seal"]
        and manifest.get("cache_digest") == config["source_run"]["cache_digest"]
    ):
        raise RecoveryError(f"worker manifest identity mismatch for seed{seed}")
    for key in ("result_json", "checkpoint"):
        record = manifest.get(key)
        if not isinstance(record, dict) or set(record) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise RecoveryError(f"worker {key} binding malformed for seed{seed}")
        path = _safe_relative(directory, record["path"], f"worker {key}")
        _validate_bound_file(path, record["sha256"], f"worker {key} seed{seed}")
        if path.stat().st_size != record["size_bytes"]:
            raise RecoveryError(f"worker {key} size binding mismatch for seed{seed}")
    predictions = manifest.get("predictions")
    if not isinstance(predictions, dict) or set(predictions) != expected_regimes:
        raise RecoveryError(f"worker prediction regime set mismatch for seed{seed}")
    for regime, record in predictions.items():
        if not isinstance(record, dict) or set(record) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise RecoveryError(
                f"worker prediction binding malformed for seed{seed}/{regime}"
            )
        path = _safe_relative(
            directory, record["path"], f"worker prediction {regime}"
        )
        _validate_bound_file(
            path, record["sha256"], f"worker prediction seed{seed}/{regime}"
        )
        if path.stat().st_size != record["size_bytes"]:
            raise RecoveryError(
                f"worker prediction size mismatch for seed{seed}/{regime}"
            )
    result_path = _safe_relative(
        directory, manifest["result_json"]["path"], "worker result_json"
    )
    result = _read_json(result_path, f"worker result seed{seed}")
    if result.get("model") != MODEL or result.get("seed") != seed:
        raise RecoveryError(f"worker result identity mismatch for seed{seed}")
    if set(result.get("regimes", {})) != expected_regimes:
        raise RecoveryError(f"worker result regimes incomplete for seed{seed}")
    return manifest


def audit_existing_workers(
    *,
    config: dict[str, Any],
    config_digest: str,
    expected_regimes: set[str],
) -> dict[str, Any]:
    """Read-only validation used by status/preflight and safe skip decisions."""

    output_root = _resolve_repo_path(
        config["missing_fit_plan"]["worker_output_root"],
        "missing_fit_plan.worker_output_root",
    )
    completed: list[int] = []
    missing: list[int] = []
    for seed in config["missing_fit_plan"]["seeds"]:
        run_id = config["missing_fit_plan"]["worker_run_id_template"].format(
            seed=seed
        )
        directory = output_root / run_id
        if not directory.exists():
            missing.append(seed)
            continue
        validate_worker_directory(
            directory,
            config=config,
            config_digest=config_digest,
            seed=seed,
            expected_regimes=expected_regimes,
        )
        completed.append(seed)
    return {
        "worker_output_root": str(output_root),
        "validated_completed_seeds": completed,
        "missing_seeds": missing,
        "all_complete": not missing,
    }


def _resource_snapshot() -> dict[str, Any]:
    return {"free_gpu_mib": free_gpu_mib(), "free_ram_gib": available_ram_gib()}


def _wait_for_resources(
    *, minimum_gpu: int, minimum_ram: float, timeout_seconds: int
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        snapshot = _resource_snapshot()
        if (
            snapshot["free_gpu_mib"] >= minimum_gpu
            and snapshot["free_ram_gib"] >= minimum_ram
        ):
            return snapshot
        if time.monotonic() >= deadline:
            raise RecoveryError(
                "resource gate timed out: "
                f"actual={snapshot}, required_gpu_mib={minimum_gpu}, "
                f"required_ram_gib={minimum_ram}"
            )
        time.sleep(15)


def _state_payload(
    *,
    config_digest: str,
    status: str,
    completed: list[int],
    current_seed: int | None,
    message: str,
    worker_sha256: str,
    resource: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA,
        "updated_utc": _utc_now(),
        "recovery_config_sha256": config_digest,
        "scheduler_sha256": _sha256(Path(__file__).resolve()),
        "worker_sha256": worker_sha256,
        "status": status,
        "completed_seeds": completed,
        "current_seed": current_seed,
        "message": message,
        "resource_snapshot": resource,
    }


def execute_recovery(
    *,
    config: dict[str, Any],
    config_digest: str,
    worker: Path,
    python: Path,
    expected_worker_sha256: str,
    expected_scheduler_sha256: str,
    expected_regimes: set[str],
    resource_timeout_seconds: int,
) -> dict[str, Any]:
    worker = worker.expanduser().resolve()
    python = python.expanduser().resolve()
    if not worker.is_file() or not python.is_file():
        raise RecoveryError("worker or Python executable is absent")
    try:
        worker.relative_to(ROOT.resolve())
    except ValueError as error:
        raise RecoveryError("worker must be inside the repository") from error
    worker_digest = _sha256(worker)
    if worker_digest != expected_worker_sha256:
        raise RecoveryError("worker source changed after recovery preflight")
    if _sha256(Path(__file__).resolve()) != expected_scheduler_sha256:
        raise RecoveryError("scheduler source changed after recovery preflight")
    output_root = _resolve_repo_path(
        config["missing_fit_plan"]["worker_output_root"],
        "missing_fit_plan.worker_output_root",
    )
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "recovery_state.json"
    minimum_gpu = int(config["resource_safety"]["minimum_free_gpu_mib_before_each_worker"])
    minimum_ram = float(config["resource_safety"]["minimum_free_ram_gib_before_each_worker"])
    completed: list[int] = []
    for seed in config["missing_fit_plan"]["seeds"]:
        if _sha256(DEFAULT_CONFIG.resolve()) != config_digest:
            raise RecoveryError("recovery config changed during scheduling")
        if _sha256(Path(__file__).resolve()) != expected_scheduler_sha256:
            raise RecoveryError("scheduler source changed during scheduling")
        if _sha256(worker) != worker_digest:
            raise RecoveryError("worker source changed during scheduling")
        run_id = config["missing_fit_plan"]["worker_run_id_template"].format(seed=seed)
        final_directory = output_root / run_id
        if final_directory.exists():
            validate_worker_directory(
                final_directory,
                config=config,
                config_digest=config_digest,
                seed=seed,
                expected_regimes=expected_regimes,
            )
            completed.append(seed)
            _atomic_json(
                state_path,
                _state_payload(
                    config_digest=config_digest,
                    status="running",
                    completed=completed,
                    current_seed=None,
                    message=f"validated and skipped completed seed {seed}",
                    worker_sha256=worker_digest,
                ),
            )
            continue
        resource = _wait_for_resources(
            minimum_gpu=minimum_gpu,
            minimum_ram=minimum_ram,
            timeout_seconds=resource_timeout_seconds,
        )
        staging = output_root / f".{run_id}.staging-{uuid.uuid4().hex}"
        staging.mkdir(parents=False, exist_ok=False)
        _atomic_json(
            state_path,
            _state_payload(
                config_digest=config_digest,
                status="running",
                completed=completed,
                current_seed=seed,
                message=f"starting fresh worker for seed {seed}",
                worker_sha256=worker_digest,
                resource=resource,
            ),
        )
        log_root = output_root / "logs"
        log_root.mkdir(parents=True, exist_ok=True)
        log_token = staging.name
        stdout_partial = log_root / f".{log_token}.stdout.log.partial"
        stderr_partial = log_root / f".{log_token}.stderr.log.partial"
        command = [
            str(python),
            str(worker),
            "--config",
            str(DEFAULT_CONFIG.resolve()),
            "--seed",
            str(seed),
            "--staging-directory",
            str(staging),
        ]
        environment = os.environ.copy()
        environment["PYTORCH_CUDA_ALLOC_CONF"] = config["resource_safety"][
            "allocator_configuration"
        ]
        with stdout_partial.open("wb") as stdout, stderr_partial.open("wb") as stderr:
            completed_process = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
            stdout.flush()
            stderr.flush()
            os.fsync(stdout.fileno())
            os.fsync(stderr.fileno())
        stdout_log = staging / "stdout.log"
        stderr_log = staging / "stderr.log"
        os.replace(stdout_partial, stdout_log)
        os.replace(stderr_partial, stderr_log)
        if _sha256(worker) != worker_digest:
            raise RecoveryError(f"worker source changed while seed{seed} executed")
        if completed_process.returncode != 0:
            _atomic_json(
                state_path,
                _state_payload(
                    config_digest=config_digest,
                    status="failed",
                    completed=completed,
                    current_seed=seed,
                    message=f"worker failed with exit code {completed_process.returncode}",
                    worker_sha256=worker_digest,
                    resource=resource,
                ),
            )
            raise RecoveryError(
                f"worker seed{seed} failed with exit code {completed_process.returncode}; "
                f"staging preserved at {staging}"
            )
        validate_worker_directory(
            staging,
            config=config,
            config_digest=config_digest,
            seed=seed,
            expected_regimes=expected_regimes,
        )
        if final_directory.exists():
            raise RecoveryError(f"worker destination appeared concurrently: {final_directory}")
        os.replace(staging, final_directory)
        completed.append(seed)
        _atomic_json(
            state_path,
            _state_payload(
                config_digest=config_digest,
                status="running",
                completed=completed,
                current_seed=None,
                message=f"worker seed {seed} committed atomically",
                worker_sha256=worker_digest,
                resource=resource,
            ),
        )
    final = _state_payload(
        config_digest=config_digest,
        status="complete",
        completed=completed,
        current_seed=None,
        message="all planned recovery seeds verified complete",
        worker_sha256=worker_digest,
    )
    _atomic_json(state_path, final)
    return final


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--worker", type=Path, default=DEFAULT_WORKER)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--status-only", action="store_true")
    mode.add_argument("--seal-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--resource-timeout-seconds", type=int, default=3600)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.resource_timeout_seconds < 0:
        raise RecoveryError("--resource-timeout-seconds must be nonnegative")
    config, config_digest = load_recovery_config(arguments.config)
    bindings = validate_bindings(config)
    source_path = Path(bindings["source_run_json"])
    source_audit = validate_source_run(config, source_path)
    worker_path = arguments.worker.expanduser().resolve()
    if not worker_path.is_file():
        raise RecoveryError(f"recovery worker is absent: {worker_path}")
    try:
        worker_path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise RecoveryError("recovery worker must be inside the repository") from error
    worker_digest = _sha256(worker_path)
    worker_status = audit_existing_workers(
        config=config,
        config_digest=config_digest,
        expected_regimes=set(source_audit["evaluation_regimes"]),
    )
    scheduler_digest = _sha256(Path(__file__).resolve())
    plan = {
        "ok": True,
        "execution_started": False,
        "mode": (
            "preflight"
            if arguments.preflight_only
            else "status"
            if arguments.status_only
            else "seal"
            if arguments.seal_only
            else "execute"
        ),
        "recovery_config": str(DEFAULT_CONFIG.resolve()),
        "recovery_config_sha256": config_digest,
        "bindings": bindings,
        "source_fit_count": source_audit["observed_fit_count"],
        "missing_fits": source_audit["missing"],
        "evaluation_regimes": source_audit["evaluation_regimes"],
        "scheduler_sha256": scheduler_digest,
        "worker": str(worker_path),
        "worker_sha256": worker_digest,
        "worker_status": worker_status,
        "resource_safety": config["resource_safety"],
    }
    if arguments.preflight_only or arguments.status_only:
        print(json.dumps(plan, ensure_ascii=False, sort_keys=True))
        return 0
    output_root = _resolve_repo_path(
        config["missing_fit_plan"]["worker_output_root"],
        "missing_fit_plan.worker_output_root",
    )
    inventory = build_source_inventory(
        config,
        config_digest,
        source_path,
        worker=worker_path,
    )
    inventory_path = output_root / "sealed_source_inventory.json"
    _atomic_json(inventory_path, inventory)
    if arguments.seal_only:
        print(
            json.dumps(
                {**plan, "source_inventory": str(inventory_path), "file_count": inventory["file_count"]},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    with _exclusive_scheduler_mutex():
        if _sha256(DEFAULT_CONFIG.resolve()) != config_digest:
            raise RecoveryError("recovery config changed after sealing")
        if _sha256(worker_path) != worker_digest:
            raise RecoveryError("recovery worker changed after sealing")
        if _sha256(Path(__file__).resolve()) != scheduler_digest:
            raise RecoveryError("recovery scheduler changed after sealing")
        state = execute_recovery(
            config=config,
            config_digest=config_digest,
            worker=worker_path,
            python=arguments.python,
            expected_worker_sha256=worker_digest,
            expected_scheduler_sha256=scheduler_digest,
            expected_regimes=set(source_audit["evaluation_regimes"]),
            resource_timeout_seconds=arguments.resource_timeout_seconds,
        )
    print(json.dumps({**plan, "execution_started": True, "state": state}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RecoveryError as error:
        print(f"V4R recovery blocked: {error}", file=sys.stderr)
        raise SystemExit(2) from error
