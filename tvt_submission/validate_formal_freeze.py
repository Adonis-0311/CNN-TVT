"""Read-only, fail-closed validator for the formal TVT freeze.

The formal hand-off must be rejected before a cache command, training command,
or release command is even printed when its frozen design no longer agrees
with the executable registries and external-source lock.  This module performs
only ordinary file reads.  It deliberately parses Python source with ``ast``
instead of importing the runner or model stack, so validation cannot initialize
PyTorch, touch a GPU, start training, or create project-local import caches.
"""

from __future__ import annotations

import argparse
import ast
from datetime import date
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "tvt_submission"
    / "configs"
    / "formal_tvt_freeze_v1.json"
)
DEFAULT_RUNNER_PATH = (
    PROJECT_ROOT / "experiments" / "run_standard_experiment.py"
)
DEFAULT_CACHE_BUILDER_PATH = (
    PROJECT_ROOT / "standards" / "build_factor_cache.py"
)
DEFAULT_BASELINES_PATH = (
    PROJECT_ROOT / "src" / "vimd_amc" / "models" / "baselines.py"
)
DEFAULT_CSSL_LOCK_PATH = (
    PROJECT_ROOT
    / "tvt_submission"
    / "sources"
    / "cssl_amc_2025.lock.json"
)

VALIDATION_SCHEMA = "vimd_amc.tvt.formal_freeze.validation.v1"
FREEZE_SCHEMA = "vimd_amc.tvt.formal_freeze.v1"
FREEZE_STATUS = "preregistered_not_executed"
CSSL_LOCK_SCHEMA = "vimd_amc.external_source_lock.v1"
CSSL_REGISTRY_NAME = "cssl_amc_supervised_adaptation"
CSSL_CLASS_NAME = "CSSLAMCSupervisedAdaptation"
CSSL_IMPLEMENTATION_PATH = "src/vimd_amc/models/baselines.py"
CSSL_LOCK_LOGICAL_PATH = "tvt_submission/sources/cssl_amc_2025.lock.json"
FORMAL_CONFIG_LOGICAL_PATH = (
    "tvt_submission/configs/formal_tvt_freeze_v1.json"
)
CSSL_REQUIRED_LABEL = (
    "CSSL-AMC official-architecture supervised adaptation"
)
EXPECTED_SPLITS = (
    "train",
    "validation",
    "id_test",
    "hard_interference",
    "unseen_jammer",
    "unseen_speed",
    "heldout_channel",
    "combined_ood",
    "clean_retention",
)
PROMOTION_REQUIREMENTS = frozenset(
    {
        "run_eligible_true",
        "source_tree_unchanged",
        "all_required_models_and_seeds",
        "at_least_one_checkpoint_selection_eligible_epoch_per_fit",
        "no_fallback_checkpoint",
        "no_placeholder_result_macro",
        "human_primary_source_audit",
        "hard_all_baseline_gain_gate_artifact_derived",
        "hard_ablation_direction_gate_artifact_derived",
        "ood_two_of_three_gate_artifact_derived",
        "clean_dual_stratum_noninferiority_gate_artifact_derived",
        "mechanism_consistency_gate_artifact_derived",
        "complete_numeric_tables_artifact_derived",
    }
)
TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "created_date",
        "journal_target",
        "claim_scope",
        "cache",
        "experiment",
        "promotion_requirements",
        "prohibited_imports",
    }
)
CACHE_KEYS = frozenset(
    {
        "output",
        "preset",
        "sample_length",
        "guard_samples",
        "master_seed",
        "matlab_timeout_s",
        "expected_designation",
        "expected_split_source_counts",
    }
)
EXPERIMENT_KEYS = frozenset(
    {
        "run_id",
        "output",
        "expected_run_directory",
        "models",
        "seeds",
        "reference_model",
        "reference_selection",
        "holm_candidates",
        "recent_comparator_contract",
        "scientific_release_gates",
        "device",
        "verify_checksums",
        "validate_components",
        "training",
        "model",
        "statistics",
    }
)
SCIENTIFIC_RELEASE_GATE_KEYS = frozenset(
    {
        "method_model",
        "primary_reference_model",
        "required_nonoracle_baselines",
        "hard_macro_f1_min_gain_pp_each_baseline",
        "hard_ablation_controls",
        "hard_ablation_strictly_positive",
        "ood_regimes",
        "ood_macro_f1_min_gain_pp",
        "ood_required_pass_count",
        "clean_profile_strata",
        "clean_macro_f1_min_point_gain_pp",
        "clean_macro_f1_min_ci95_low_pp",
        "mechanism_required_finite_fields",
        "mechanism_nonnegative_fields",
        "oracle_spectral_ratio_field",
        "oracle_spectral_ratio_strictly_positive",
    }
)
RECENT_COMPARATOR_KEYS = frozenset(
    {
        "model",
        "source_lock",
        "required_label",
        "complete_published_method_reproduction",
        "structured_interference_specific",
    }
)
TRAINING_KEYS = frozenset(
    {
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "mask_start_epoch",
        "contrastive_start_epoch",
        "mask_ramp_epochs",
        "contrastive_ramp_epochs",
        "minimum_full_stage_epochs",
        "patience",
        "use_amp",
    }
)
MODEL_KEYS = frozenset(
    {
        "n_fft",
        "hop_length",
        "spectral_channels",
        "embedding_dim",
        "environment_dim",
        "dropout",
    }
)
STATISTICS_KEYS = frozenset(
    {
        "bootstrap_draws",
        "bootstrap_seed",
        "headline_inference",
        "mcnemar_role",
        "validation_excluded",
    }
)
CSSL_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "audit_date",
        "paper",
        "official_source",
        "local_adaptation",
        "prohibited_claims",
    }
)
CSSL_PAPER_KEYS = frozenset(
    {
        "title",
        "authors",
        "venue",
        "volume",
        "number",
        "pages",
        "year",
        "doi",
        "publisher_url",
    }
)
CSSL_OFFICIAL_SOURCE_KEYS = frozenset(
    {
        "repository",
        "commit",
        "commit_url",
        "license_spdx",
        "license_url",
        "license_sha256",
        "local_license_copy",
        "audited_files",
    }
)
CSSL_AUDITED_FILE_KEYS = frozenset(
    {"path", "immutable_url", "sha256", "audit_role"}
)
CSSL_LOCAL_ADAPTATION_KEYS = frozenset(
    {
        "registry_name",
        "class_name",
        "implementation",
        "input_contract",
        "retained_topology",
        "material_changes",
        "claim_label",
        "formal_training_objective",
        "formal_optimizer_budget_source",
        "formal_result_status",
        "recent_auditable_amc_comparator",
        "complete_published_method_reproduction",
        "structured_interference_specific",
    }
)
CSSL_INPUT_CONTRACT_KEYS = frozenset(
    {
        "representation",
        "sample_length",
        "side_information",
        "external_weights",
    }
)
WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
ISO_CALENDAR_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class FormalFreezeValidationError(RuntimeError):
    """The frozen design is unsafe or inconsistent; execution must stay shut."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise FormalFreezeValidationError(
            f"cannot read file for SHA-256: {path}: {error}"
        ) from error
    return digest.hexdigest()


def _load_strict_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(
                stream,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_nonstandard_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise FormalFreezeValidationError(
            f"{label} strict JSON load failed for {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise FormalFreezeValidationError(f"{label} root must be an object")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FormalFreezeValidationError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise FormalFreezeValidationError(f"{label} must be an object")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != set(expected):
        missing = sorted(set(expected).difference(actual))
        unexpected = sorted(actual.difference(expected))
        raise FormalFreezeValidationError(
            f"{label} keys differ from the frozen schema; "
            f"missing={missing}, unexpected={unexpected}"
        )


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FormalFreezeValidationError(
            f"{label} must be a nonempty string"
        )
    if value != value.strip() or "\x00" in value:
        raise FormalFreezeValidationError(
            f"{label} contains surrounding whitespace or NUL"
        )
    return value


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise FormalFreezeValidationError(
            f"{label} must be a positive integer"
        )
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise FormalFreezeValidationError(
            f"{label} must be a nonnegative integer"
        )
    return value


def _finite_number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    minimum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FormalFreezeValidationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise FormalFreezeValidationError(f"{label} must be finite")
    if minimum is not None:
        valid = (
            result >= minimum
            if minimum_inclusive
            else result > minimum
        )
        if not valid:
            operator = ">=" if minimum_inclusive else ">"
            raise FormalFreezeValidationError(
                f"{label} must be {operator} {minimum}"
            )
    return result


def _unique_strings(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise FormalFreezeValidationError(
            f"{label} must be a nonempty list"
        )
    result = [
        _nonempty_string(item, f"{label}[{index}]")
        for index, item in enumerate(value)
    ]
    if len(result) != len(set(result)):
        raise FormalFreezeValidationError(f"{label} contains duplicates")
    return result


def _strict_iso_calendar_date(value: Any, label: str) -> str:
    raw = _nonempty_string(value, label)
    if ISO_CALENDAR_DATE.fullmatch(raw) is None:
        raise FormalFreezeValidationError(
            f"{label} must use strict ISO YYYY-MM-DD"
        )
    try:
        date.fromisoformat(raw)
    except ValueError as error:
        raise FormalFreezeValidationError(
            f"{label} is not a valid calendar date"
        ) from error
    return raw


def _safe_project_relative_path(
    value: Any,
    label: str,
    *,
    project_root: Path,
) -> tuple[PurePosixPath, Path]:
    raw = _nonempty_string(value, label)
    normalized = raw.replace("\\", "/")
    if (
        normalized.startswith("/")
        or normalized.startswith("//")
        or WINDOWS_ABSOLUTE.match(raw)
    ):
        raise FormalFreezeValidationError(
            f"{label} must be project-relative: {raw!r}"
        )
    components = normalized.split("/")
    if any(
        component in {"", ".", ".."} or ":" in component
        for component in components
    ):
        raise FormalFreezeValidationError(
            f"{label} contains an unsafe path component: {raw!r}"
        )
    relative = PurePosixPath(*components)
    root = project_root.resolve()
    resolved = root.joinpath(*relative.parts).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise FormalFreezeValidationError(
            f"{label} escapes the project root: {raw!r}"
        ) from error
    return relative, resolved


def _resolved_paths_overlap(first: Path, second: Path) -> bool:
    try:
        first.relative_to(second)
        return True
    except ValueError:
        pass
    try:
        second.relative_to(first)
        return True
    except ValueError:
        return False


def _load_python_tree(path: Path, label: str) -> ast.Module:
    try:
        source = path.read_text(encoding="utf-8")
        return ast.parse(source, filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as error:
        raise FormalFreezeValidationError(
            f"cannot statically parse {label} at {path}: {error}"
        ) from error


def _target_binds_name(target: ast.AST, name: str) -> bool:
    if isinstance(target, ast.Name):
        return target.id == name
    if isinstance(target, (ast.Tuple, ast.List)):
        return any(_target_binds_name(item, name) for item in target.elts)
    if isinstance(target, ast.Starred):
        return _target_binds_name(target.value, name)
    return False


class _ModuleBindingCollector(ast.NodeVisitor):
    """Collect executable bindings without descending into nested scopes."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.bindings: list[tuple[str, ast.AST, ast.expr | None]] = []

    def _record(
        self,
        kind: str,
        node: ast.AST,
        value: ast.expr | None = None,
    ) -> None:
        self.bindings.append((kind, node, value))

    def visit_Assign(self, node: ast.Assign) -> None:
        if any(_target_binds_name(target, self.name) for target in node.targets):
            self._record("assignment", node, node.value)
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if _target_binds_name(node.target, self.name):
            self._record("annotated assignment", node, node.value)
        if node.value is not None:
            self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if _target_binds_name(node.target, self.name):
            self._record("augmented assignment", node)
        self.visit(node.value)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        if _target_binds_name(node.target, self.name):
            self._record("assignment expression", node, node.value)
        self.visit(node.value)

    def visit_For(self, node: ast.For) -> None:
        if _target_binds_name(node.target, self.name):
            self._record("for target", node)
        self.visit(node.iter)
        for statement in (*node.body, *node.orelse):
            self.visit(statement)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit_For(node)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if (
                item.optional_vars is not None
                and _target_binds_name(item.optional_vars, self.name)
            ):
                self._record("with target", node)
        for statement in node.body:
            self.visit(statement)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self.visit_With(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name == self.name:
            self._record("exception target", node)
        for statement in node.body:
            self.visit(statement)

    def visit_Delete(self, node: ast.Delete) -> None:
        if any(_target_binds_name(target, self.name) for target in node.targets):
            self._record("deletion", node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            bound_name = alias.asname or alias.name.split(".", maxsplit=1)[0]
            if bound_name == self.name:
                self._record("import", node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            bound_name = alias.asname or alias.name
            if bound_name == self.name:
                self._record("import", node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name == self.name:
            self._record("function definition", node)
        for expression in (
            *node.decorator_list,
            *node.args.defaults,
            *(
                default
                for default in node.args.kw_defaults
                if default is not None
            ),
        ):
            self.visit(expression)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if node.name == self.name:
            self._record("class definition", node)
        for expression in (*node.decorator_list, *node.bases):
            self.visit(expression)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return


def _module_bindings(
    tree: ast.Module,
    name: str,
) -> list[tuple[str, ast.AST, ast.expr | None]]:
    collector = _ModuleBindingCollector(name)
    collector.visit(tree)
    return collector.bindings


def _module_literal(tree: ast.Module, name: str, label: str) -> Any:
    bindings = _module_bindings(tree, name)
    if len(bindings) != 1:
        details = [
            f"{kind}@{getattr(node, 'lineno', '?')}"
            for kind, node, _ in bindings
        ]
        raise FormalFreezeValidationError(
            f"{label} must define exactly one module-level binding for {name}; "
            f"found={details}"
        )
    kind, node, value = bindings[0]
    if (
        kind not in {"assignment", "annotated assignment"}
        or node not in tree.body
        or value is None
    ):
        raise FormalFreezeValidationError(
            f"{label} binding {name} must be one direct static assignment"
        )
    try:
        return ast.literal_eval(value)
    except (ValueError, TypeError) as error:
        raise FormalFreezeValidationError(
            f"{label} assignment {name} is not a static literal"
        ) from error


def _require_unique_module_class(
    tree: ast.Module,
    name: str,
    label: str,
) -> None:
    bindings = _module_bindings(tree, name)
    if (
        len(bindings) != 1
        or bindings[0][0] != "class definition"
        or bindings[0][1] not in tree.body
    ):
        details = [
            f"{kind}@{getattr(node, 'lineno', '?')}"
            for kind, node, _ in bindings
        ]
        raise FormalFreezeValidationError(
            f"{label} must define exactly one unshadowed module-level class "
            f"{name}; found={details}"
        )


def _function_local_literal(
    tree: ast.Module,
    function_name: str,
    variable_name: str,
    label: str,
) -> Any:
    function_bindings = _module_bindings(tree, function_name)
    if (
        len(function_bindings) != 1
        or function_bindings[0][0] != "function definition"
        or function_bindings[0][1] not in tree.body
    ):
        raise FormalFreezeValidationError(
            f"{label} must define exactly one unshadowed module-level function "
            f"{function_name}"
        )
    function = function_bindings[0][1]
    assert isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
    function_scope = ast.Module(body=function.body, type_ignores=[])
    bindings = _module_bindings(function_scope, variable_name)
    if (
        len(bindings) != 1
        or bindings[0][0] not in {"assignment", "annotated assignment"}
        or bindings[0][1] not in function_scope.body
        or bindings[0][2] is None
    ):
        raise FormalFreezeValidationError(
            f"{label} must define exactly one direct static "
            f"{function_name}.{variable_name}"
        )
    try:
        return ast.literal_eval(bindings[0][2])
    except (ValueError, TypeError) as error:
        raise FormalFreezeValidationError(
            f"{label} {function_name}.{variable_name} is not static"
        ) from error


def _validate_training(training: Mapping[str, Any]) -> dict[str, int]:
    _exact_keys(training, TRAINING_KEYS, "experiment.training")
    epochs = _positive_int(training["epochs"], "training.epochs")
    _positive_int(training["batch_size"], "training.batch_size")
    _finite_number(
        training["learning_rate"],
        "training.learning_rate",
        minimum=0.0,
        minimum_inclusive=False,
    )
    _finite_number(
        training["weight_decay"],
        "training.weight_decay",
        minimum=0.0,
    )
    mask_start = _nonnegative_int(
        training["mask_start_epoch"], "training.mask_start_epoch"
    )
    contrastive_start = _nonnegative_int(
        training["contrastive_start_epoch"],
        "training.contrastive_start_epoch",
    )
    mask_ramp = _positive_int(
        training["mask_ramp_epochs"], "training.mask_ramp_epochs"
    )
    contrastive_ramp = _positive_int(
        training["contrastive_ramp_epochs"],
        "training.contrastive_ramp_epochs",
    )
    minimum_full = _positive_int(
        training["minimum_full_stage_epochs"],
        "training.minimum_full_stage_epochs",
    )
    _positive_int(training["patience"], "training.patience")
    _require(
        training["use_amp"] is True,
        "training.use_amp must remain true for the formal CUDA freeze",
    )
    _require(
        mask_start < epochs and contrastive_start < epochs,
        "training objective start epochs must precede the epoch budget",
    )

    full_objective_index = max(
        mask_start + mask_ramp - 1,
        contrastive_start + contrastive_ramp - 1,
    )
    selection_start_index = full_objective_index + minimum_full - 1
    eligible_epochs = max(0, epochs - selection_start_index)
    _require(
        eligible_epochs >= 1,
        "training schedule leaves no checkpoint-selection-eligible epoch",
    )
    return {
        "full_objective_epoch": full_objective_index + 1,
        "selection_start_epoch": selection_start_index + 1,
        "eligible_epoch_count": eligible_epochs,
    }


def _validate_model(
    model: Mapping[str, Any],
    *,
    sample_length: int,
) -> None:
    _exact_keys(model, MODEL_KEYS, "experiment.model")
    n_fft = _positive_int(model["n_fft"], "model.n_fft")
    hop_length = _positive_int(
        model["hop_length"], "model.hop_length"
    )
    _require(
        8 <= n_fft <= sample_length,
        "model.n_fft must be in [8, cache.sample_length]",
    )
    _require(
        hop_length <= n_fft,
        "model.hop_length must not exceed model.n_fft",
    )
    for name in (
        "spectral_channels",
        "embedding_dim",
        "environment_dim",
    ):
        _positive_int(model[name], f"model.{name}")
    dropout = _finite_number(model["dropout"], "model.dropout")
    _require(0.0 <= dropout < 1.0, "model.dropout must lie in [0, 1)")


def _validate_statistics(statistics: Mapping[str, Any]) -> None:
    _exact_keys(statistics, STATISTICS_KEYS, "experiment.statistics")
    _positive_int(
        statistics["bootstrap_draws"], "statistics.bootstrap_draws"
    )
    _nonnegative_int(
        statistics["bootstrap_seed"], "statistics.bootstrap_seed"
    )
    headline = _nonempty_string(
        statistics["headline_inference"],
        "statistics.headline_inference",
    ).casefold()
    _require(
        "hierarchical" in headline
        and "algorithm seed" in headline
        and "source cluster" in headline,
        "statistics.headline_inference must retain the two-layer hierarchy",
    )
    mcnemar = _nonempty_string(
        statistics["mcnemar_role"], "statistics.mcnemar_role"
    ).casefold()
    _require(
        "per-seed" in mcnemar and "supplemental" in mcnemar,
        "statistics.mcnemar_role must remain per-seed and supplemental",
    )
    _require(
        statistics["validation_excluded"] is True,
        "statistics.validation_excluded must be true",
    )


def _json_normalize_literal(value: Any) -> Any:
    """Convert a static Python literal to its strict-JSON representation."""

    if isinstance(value, tuple):
        return [_json_normalize_literal(item) for item in value]
    if isinstance(value, list):
        return [_json_normalize_literal(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _json_normalize_literal(item)
            for key, item in value.items()
        }
    return value


def _validate_scientific_release_gates(
    gates: Mapping[str, Any],
    *,
    runner_tree: ast.Module,
    models: Sequence[str],
    reference_model: str,
    holm_candidates: Sequence[str],
) -> dict[str, Any]:
    """Lock every release threshold to one static runner declaration."""

    _exact_keys(
        gates,
        SCIENTIFIC_RELEASE_GATE_KEYS,
        "experiment.scientific_release_gates",
    )
    method_model = _module_literal(
        runner_tree,
        "FORMAL_METHOD_MODEL",
        "standard runner",
    )
    primary_reference = _module_literal(
        runner_tree,
        "FORMAL_PRIMARY_REFERENCE_MODEL",
        "standard runner",
    )
    required_baselines = _module_literal(
        runner_tree,
        "FORMAL_REQUIRED_NONORACLE_BASELINES",
        "standard runner",
    )
    formal_holm = _module_literal(
        runner_tree,
        "FORMAL_HOLM_CANDIDATES",
        "standard runner",
    )
    clean_strata = _module_literal(
        runner_tree,
        "CLEAN_RETENTION_PROFILE_STRATA",
        "standard runner",
    )
    thresholds = _module_literal(
        runner_tree,
        "SCIENTIFIC_RELEASE_THRESHOLDS",
        "standard runner",
    )
    _require(
        isinstance(method_model, str)
        and isinstance(primary_reference, str)
        and isinstance(required_baselines, tuple)
        and isinstance(formal_holm, tuple)
        and isinstance(clean_strata, dict)
        and isinstance(thresholds, dict),
        "standard-runner scientific release declarations are malformed",
    )

    expected = {
        "method_model": method_model,
        "primary_reference_model": primary_reference,
        "required_nonoracle_baselines": _json_normalize_literal(
            required_baselines
        ),
        **_json_normalize_literal(thresholds),
        "clean_profile_strata": _json_normalize_literal(clean_strata),
    }
    _require(
        dict(gates) == expected,
        "experiment.scientific_release_gates must exactly mirror the "
        "standard-runner scientific release declarations",
    )
    _require(
        reference_model == primary_reference == CSSL_REGISTRY_NAME,
        "the frozen primary reference must be the CSSL adaptation",
    )
    _require(
        list(holm_candidates) == list(formal_holm)
        and primary_reference not in holm_candidates,
        "the frozen Holm family must exactly match the runner and exclude "
        "the primary CSSL reference",
    )

    required_models = {
        method_model,
        primary_reference,
        *required_baselines,
        *formal_holm,
        *thresholds["hard_ablation_controls"],
    }
    _require(
        required_models.issubset(models),
        "scientific release declarations reference models outside the "
        f"formal suite: {sorted(required_models.difference(models))}",
    )
    _require(
        primary_reference in required_baselines,
        "the primary CSSL reference must also be one of the four "
        "non-oracle hard-regime baselines",
    )

    normalized_strata = _json_normalize_literal(clean_strata)
    _require(
        set(normalized_strata)
        == {"clean_retention_seen_acd", "clean_retention_held_be"},
        "clean-retention strata must remain the preregistered A/C/D and B/E "
        "partition",
    )
    flattened = [
        profile_index
        for indices in normalized_strata.values()
        for profile_index in indices
    ]
    _require(
        sorted(flattened) == [0, 1, 2, 3, 4]
        and len(flattened) == len(set(flattened)),
        "clean-retention profile strata must be disjoint and cover exactly "
        "target profiles 0..4",
    )
    return expected


def _validate_recent_comparator_contract(
    contract: Mapping[str, Any],
    *,
    project_root: Path,
    lock_path: Path,
) -> dict[str, Any]:
    _exact_keys(
        contract,
        RECENT_COMPARATOR_KEYS,
        "experiment.recent_comparator_contract",
    )
    _require(
        contract.get("model") == CSSL_REGISTRY_NAME,
        "recent comparator model must be the frozen CSSL registry name",
    )
    logical_lock, resolved_lock = _safe_project_relative_path(
        contract.get("source_lock"),
        "recent_comparator_contract.source_lock",
        project_root=project_root,
    )
    _require(
        logical_lock.as_posix() == CSSL_LOCK_LOGICAL_PATH
        and resolved_lock == lock_path,
        "recent comparator source_lock disagrees with the audited lock file",
    )
    _require(
        contract.get("required_label") == CSSL_REQUIRED_LABEL,
        "recent comparator required_label drifted",
    )
    _require(
        contract.get("complete_published_method_reproduction") is False,
        "recent comparator must not claim a complete published-method reproduction",
    )
    _require(
        contract.get("structured_interference_specific") is False,
        "recent comparator must not claim structured-interference specificity",
    )
    return {
        "model": CSSL_REGISTRY_NAME,
        "source_lock": logical_lock.as_posix(),
        "required_label": CSSL_REQUIRED_LABEL,
        "complete_published_method_reproduction": False,
        "structured_interference_specific": False,
    }


def _validate_cssl_lock(
    *,
    lock_path: Path,
    lock: Mapping[str, Any],
    project_root: Path,
    config_path: Path,
    baseline_path: Path,
    baseline_tree: ast.Module,
    runner_tree: ast.Module,
    recent_contract: Mapping[str, Any],
    freeze_models: Sequence[str],
    reference_model: str,
    holm_candidates: Sequence[str],
    sample_length: int,
) -> dict[str, Any]:
    _exact_keys(lock, CSSL_TOP_LEVEL_KEYS, "CSSL source lock")
    _require(
        lock.get("schema_version") == CSSL_LOCK_SCHEMA,
        "CSSL source-lock schema mismatch",
    )
    _require(
        lock.get("status")
        == "audited_architecture_adaptation_preregistered_not_executed",
        "CSSL source-lock status is not the preregistered unexecuted state",
    )
    _strict_iso_calendar_date(
        lock.get("audit_date"), "CSSL source-lock audit_date"
    )
    paper = _mapping(lock.get("paper"), "CSSL source-lock paper")
    _exact_keys(paper, CSSL_PAPER_KEYS, "CSSL source-lock paper")
    official = _mapping(
        lock.get("official_source"), "CSSL source-lock official_source"
    )
    _exact_keys(
        official,
        CSSL_OFFICIAL_SOURCE_KEYS,
        "CSSL source-lock official_source",
    )
    adaptation = _mapping(
        lock.get("local_adaptation"), "CSSL source-lock local_adaptation"
    )
    _exact_keys(
        adaptation,
        CSSL_LOCAL_ADAPTATION_KEYS,
        "CSSL source-lock local_adaptation",
    )
    _unique_strings(paper.get("authors"), "CSSL paper.authors")
    for field in ("title", "venue", "pages", "publisher_url"):
        _nonempty_string(paper.get(field), f"CSSL paper.{field}")
    for field in ("volume", "number", "year"):
        _positive_int(paper.get(field), f"CSSL paper.{field}")

    registry_name = _nonempty_string(
        adaptation.get("registry_name"),
        "CSSL local_adaptation.registry_name",
    )
    _require(
        registry_name == CSSL_REGISTRY_NAME,
        "CSSL source-lock registry name mismatch",
    )
    _require(
        registry_name == recent_contract.get("model")
        and registry_name in freeze_models
        and registry_name == reference_model
        and registry_name not in holm_candidates,
        "CSSL registry name must remain synchronized across contract, models, "
        "and the predeclared primary reference while remaining outside the "
        "Holm candidate family",
    )
    _require(
        adaptation.get("class_name") == CSSL_CLASS_NAME,
        "CSSL source-lock class name mismatch",
    )
    _require_unique_module_class(
        baseline_tree,
        CSSL_CLASS_NAME,
        "baseline registry",
    )

    implementation, implementation_path = _safe_project_relative_path(
        adaptation.get("implementation"),
        "CSSL local_adaptation.implementation",
        project_root=project_root,
    )
    _require(
        implementation.as_posix() == CSSL_IMPLEMENTATION_PATH
        and implementation_path.resolve() == baseline_path.resolve(),
        "CSSL implementation path disagrees with the audited baseline module",
    )
    _require(
        implementation_path.is_file(),
        "CSSL implementation source is absent",
    )

    baseline_commit = _module_literal(
        baseline_tree,
        "CSSL_AMC_AUDITED_COMMIT",
        "baseline registry",
    )
    baseline_repository = _module_literal(
        baseline_tree,
        "CSSL_AMC_OFFICIAL_CODE_URL",
        "baseline registry",
    )
    baseline_license = _module_literal(
        baseline_tree,
        "CSSL_AMC_LICENSE",
        "baseline registry",
    )
    baseline_paper = _module_literal(
        baseline_tree,
        "CSSL_AMC_PAPER_URL",
        "baseline registry",
    )
    commit = _nonempty_string(
        official.get("commit"), "CSSL official_source.commit"
    )
    _require(
        COMMIT_SHA.fullmatch(commit) is not None,
        "CSSL official source commit must be a 40-character lowercase SHA",
    )
    _require(
        commit == baseline_commit,
        "CSSL source-lock commit disagrees with baseline registry",
    )
    _require(
        official.get("repository") == baseline_repository,
        "CSSL source-lock repository disagrees with baseline registry",
    )
    _require(
        official.get("license_spdx") == baseline_license,
        "CSSL source-lock license disagrees with baseline registry",
    )
    doi = _nonempty_string(paper.get("doi"), "CSSL paper.doi")
    _require(
        f"https://doi.org/{doi}" == baseline_paper,
        "CSSL source-lock DOI disagrees with baseline registry",
    )
    repository = _nonempty_string(
        official.get("repository"), "CSSL official_source.repository"
    )
    _require(
        official.get("commit_url") == f"{repository}/commit/{commit}",
        "CSSL official source commit_url is not pinned to repository and commit",
    )
    _require(
        official.get("license_url")
        == (
            "https://raw.githubusercontent.com/"
            "dumingyang20/CSSL-AMC-Pytorch/"
            f"{commit}/LICENSE"
        ),
        "CSSL official license_url is not pinned to the audited commit",
    )

    optional = _function_local_literal(
        runner_tree,
        "available_model_factories",
        "optional",
        "standard runner",
    )
    _require(
        isinstance(optional, tuple)
        and (
            CSSL_REGISTRY_NAME,
            CSSL_CLASS_NAME,
            False,
        )
        in optional,
        "standard runner CSSL factory binding disagrees with source lock",
    )

    input_contract = _mapping(
        adaptation.get("input_contract"),
        "CSSL local_adaptation.input_contract",
    )
    _exact_keys(
        input_contract,
        CSSL_INPUT_CONTRACT_KEYS,
        "CSSL local_adaptation.input_contract",
    )
    _require(
        input_contract.get("representation")
        == "received complex IQ as float tensor [batch, 2, 1024]",
        "CSSL input representation drifted",
    )
    _require(
        input_contract.get("side_information") == "none",
        "CSSL source lock must not permit side information",
    )
    _require(
        input_contract.get("sample_length") == sample_length == 1024,
        "CSSL source lock and formal cache must both use 1024 samples",
    )
    _require(
        input_contract.get("external_weights") == "prohibited",
        "CSSL source lock must prohibit external weights",
    )
    _require(
        adaptation.get("formal_training_objective")
        == "paired_view_modulation_ce",
        "CSSL formal training objective drifted",
    )
    budget_path, resolved_budget_path = _safe_project_relative_path(
        adaptation.get("formal_optimizer_budget_source"),
        "CSSL formal_optimizer_budget_source",
        project_root=project_root,
    )
    _require(
        budget_path.as_posix() == FORMAL_CONFIG_LOGICAL_PATH
        and resolved_budget_path == config_path,
        "CSSL optimizer budget source does not point to the formal freeze",
    )
    _require(
        adaptation.get("formal_result_status") == "not_executed",
        "CSSL source lock must remain not_executed before the formal run",
    )
    required_claim_fragment = CSSL_REQUIRED_LABEL.removeprefix("CSSL-AMC ")
    claim_label = _nonempty_string(
        adaptation.get("claim_label"), "CSSL local_adaptation.claim_label"
    )
    _require(
        required_claim_fragment in claim_label
        and "not a reproduction" in claim_label,
        "CSSL claim label does not preserve the required adaptation limitation",
    )
    _require(
        adaptation.get("recent_auditable_amc_comparator") is True,
        "CSSL source lock must mark the adaptation as an auditable comparator",
    )
    for key in (
        "complete_published_method_reproduction",
        "structured_interference_specific",
    ):
        _require(
            adaptation.get(key) is recent_contract.get(key) is False,
            f"CSSL source lock and recent comparator contract disagree on {key}",
        )
    _unique_strings(
        adaptation.get("retained_topology"),
        "CSSL local_adaptation.retained_topology",
    )
    _unique_strings(
        adaptation.get("material_changes"),
        "CSSL local_adaptation.material_changes",
    )
    _unique_strings(lock.get("prohibited_claims"), "CSSL prohibited_claims")

    license_path_value, license_path = _safe_project_relative_path(
        official.get("local_license_copy"),
        "CSSL official_source.local_license_copy",
        project_root=project_root,
    )
    _require(
        license_path.is_file() and license_path.stat().st_size > 0,
        "CSSL local license copy is absent or empty",
    )
    try:
        license_text = license_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise FormalFreezeValidationError(
            f"cannot read CSSL local license copy: {error}"
        ) from error
    _require(
        "Apache License" in license_text,
        "CSSL local license copy does not contain the Apache license",
    )
    actual_license_sha256 = _sha256_file(license_path)
    _require(
        official.get("license_sha256") == actual_license_sha256,
        "CSSL local license SHA-256 disagrees with the source lock",
    )

    audited_files = official.get("audited_files")
    _require(
        isinstance(audited_files, list) and bool(audited_files),
        "CSSL audited_files must be a nonempty list",
    )
    audited_paths: list[str] = []
    for index, record_value in enumerate(audited_files):
        record = _mapping(
            record_value, f"CSSL audited_files[{index}]"
        )
        _exact_keys(
            record,
            CSSL_AUDITED_FILE_KEYS,
            f"CSSL audited_files[{index}]",
        )
        source_path = _nonempty_string(
            record.get("path"), f"CSSL audited_files[{index}].path"
        )
        normalized_source = source_path.replace("\\", "/")
        _require(
            not normalized_source.startswith("/")
            and ".." not in normalized_source.split("/")
            and ":" not in normalized_source,
            f"CSSL audited_files[{index}].path is unsafe",
        )
        immutable_url = _nonempty_string(
            record.get("immutable_url"),
            f"CSSL audited_files[{index}].immutable_url",
        )
        _require(
            commit in immutable_url
            and immutable_url.endswith(normalized_source),
            f"CSSL audited_files[{index}] is not pinned to its path and commit",
        )
        _require(
            SHA256.fullmatch(
                _nonempty_string(
                    record.get("sha256"),
                    f"CSSL audited_files[{index}].sha256",
                )
            )
            is not None,
            f"CSSL audited_files[{index}].sha256 is malformed",
        )
        _nonempty_string(
            record.get("audit_role"),
            f"CSSL audited_files[{index}].audit_role",
        )
        audited_paths.append(normalized_source)
    _require(
        len(audited_paths) == len(set(audited_paths)),
        "CSSL audited_files contains duplicate source paths",
    )

    return {
        "registry_name": registry_name,
        "audited_commit": commit,
        "source_lock_sha256": _sha256_file(lock_path),
        "license_path": license_path_value.as_posix(),
        "license_sha256": actual_license_sha256,
        "audited_file_count": len(audited_paths),
    }


def validate_formal_freeze(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    project_root: str | Path = PROJECT_ROOT,
    runner_path: str | Path = DEFAULT_RUNNER_PATH,
    cache_builder_path: str | Path = DEFAULT_CACHE_BUILDER_PATH,
    baselines_path: str | Path = DEFAULT_BASELINES_PATH,
    cssl_lock_path: str | Path = DEFAULT_CSSL_LOCK_PATH,
) -> dict[str, Any]:
    """Validate the frozen hand-off and return a machine-readable summary.

    No function in this call graph creates, updates, deletes, or opens a file
    for writing.
    """

    config_file = Path(config_path).expanduser().resolve()
    root = Path(project_root).expanduser().resolve()
    runner_file = Path(runner_path).expanduser().resolve()
    cache_builder_file = Path(cache_builder_path).expanduser().resolve()
    baseline_file = Path(baselines_path).expanduser().resolve()
    lock_file = Path(cssl_lock_path).expanduser().resolve()

    config = _load_strict_json(config_file, "formal freeze")
    _exact_keys(config, TOP_LEVEL_KEYS, "formal freeze")
    _require(
        config.get("schema_version") == FREEZE_SCHEMA,
        f"formal freeze schema must be {FREEZE_SCHEMA}",
    )
    _require(
        config.get("status") == FREEZE_STATUS,
        f"formal freeze status must be {FREEZE_STATUS}",
    )
    _strict_iso_calendar_date(
        config.get("created_date"), "created_date"
    )
    _require(
        config.get("journal_target")
        == "IEEE Transactions on Vehicular Technology",
        "journal_target drifted from IEEE TVT",
    )
    _nonempty_string(config.get("claim_scope"), "claim_scope")

    runner_tree = _load_python_tree(runner_file, "standard runner")
    cache_builder_tree = _load_python_tree(
        cache_builder_file, "factor-cache builder"
    )
    baseline_tree = _load_python_tree(baseline_file, "baseline registry")
    registry = _module_literal(
        runner_tree,
        "PREREGISTERED_MODEL_SUITES",
        "standard runner",
    )
    _require(
        isinstance(registry, dict)
        and isinstance(registry.get("headline"), tuple),
        "standard runner headline registry is malformed",
    )
    headline_registry = list(registry["headline"])
    _require(
        len(headline_registry) == len(set(headline_registry)),
        "standard runner headline registry contains duplicates",
    )
    formal_designation = _module_literal(
        runner_tree,
        "FORMAL_RELEASE_DESIGNATION",
        "standard runner",
    )

    cache = _mapping(config.get("cache"), "cache")
    _exact_keys(cache, CACHE_KEYS, "cache")
    cache_output, resolved_cache_output = _safe_project_relative_path(
        cache.get("output"), "cache.output", project_root=root
    )
    _require(cache.get("preset") == "headline", "cache.preset must be headline")
    sample_length = _positive_int(
        cache.get("sample_length"), "cache.sample_length"
    )
    guard_samples = _nonnegative_int(
        cache.get("guard_samples"), "cache.guard_samples"
    )
    _require(
        guard_samples < sample_length,
        "cache.guard_samples must be smaller than sample_length",
    )
    _positive_int(cache.get("master_seed"), "cache.master_seed")
    _positive_int(
        cache.get("matlab_timeout_s"), "cache.matlab_timeout_s"
    )
    _require(
        cache.get("expected_designation") == formal_designation,
        "cache.expected_designation disagrees with the runner formal designation",
    )
    split_counts = _mapping(
        cache.get("expected_split_source_counts"),
        "cache.expected_split_source_counts",
    )
    _require(
        tuple(split_counts) == EXPECTED_SPLITS,
        "formal freeze must declare the exact ordered nine-split protocol",
    )
    normalized_split_counts: dict[str, int] = {}
    for split in EXPECTED_SPLITS:
        count = _positive_int(
            split_counts[split],
            f"expected_split_source_counts.{split}",
        )
        _require(
            count % 10 == 0,
            f"expected split {split} must remain divisible across ten classes",
        )
        normalized_split_counts[split] = count
    preset_sizes = _module_literal(
        cache_builder_tree,
        "_PRESET_SIZES",
        "factor-cache builder",
    )
    _require(
        isinstance(preset_sizes, dict)
        and isinstance(preset_sizes.get("headline"), dict),
        "factor-cache builder headline preset is malformed",
    )
    headline_sizes = preset_sizes["headline"]
    _require(
        tuple(headline_sizes) == EXPECTED_SPLITS
        and headline_sizes == normalized_split_counts,
        "factor-cache builder headline preset split sizes disagree with "
        "the formal freeze",
    )
    preset_designations = _module_literal(
        cache_builder_tree,
        "_PRESET_DESIGNATIONS",
        "factor-cache builder",
    )
    _require(
        isinstance(preset_designations, dict)
        and preset_designations.get("headline")
        == cache.get("expected_designation")
        == formal_designation,
        "factor-cache builder headline designation disagrees with "
        "the formal freeze and runner",
    )

    experiment = _mapping(config.get("experiment"), "experiment")
    _exact_keys(experiment, EXPERIMENT_KEYS, "experiment")
    run_id = _nonempty_string(experiment.get("run_id"), "experiment.run_id")
    _require(
        "/" not in run_id
        and "\\" not in run_id
        and ":" not in run_id
        and run_id not in {".", ".."},
        "experiment.run_id must be one safe path component",
    )
    output, resolved_output = _safe_project_relative_path(
        experiment.get("output"), "experiment.output", project_root=root
    )
    expected_run, _ = _safe_project_relative_path(
        experiment.get("expected_run_directory"),
        "experiment.expected_run_directory",
        project_root=root,
    )
    computed_run = output / run_id
    _require(
        expected_run == computed_run,
        "experiment.output/run_id disagrees with expected_run_directory",
    )
    _require(
        not _resolved_paths_overlap(
            resolved_cache_output,
            resolved_output,
        ),
        "cache.output and experiment.output must not overlap by ancestry",
    )
    _require(
        cache_output != expected_run,
        "cache output and expected run directory must be distinct",
    )

    models = _unique_strings(experiment.get("models"), "experiment.models")
    _require(
        models == headline_registry,
        "formal models must exactly match the runner headline registry",
    )
    seeds_value = experiment.get("seeds")
    _require(
        isinstance(seeds_value, list) and bool(seeds_value),
        "experiment.seeds must be a nonempty list",
    )
    seeds = [
        _positive_int(seed, f"experiment.seeds[{index}]")
        for index, seed in enumerate(seeds_value)
    ]
    _require(
        len(seeds) == len(set(seeds)),
        "experiment.seeds contains duplicates",
    )
    _require(
        len(seeds) >= 5,
        "formal freeze requires at least five independent seeds",
    )
    reference_model = _nonempty_string(
        experiment.get("reference_model"),
        "experiment.reference_model",
    )
    _require(
        reference_model in models,
        "experiment.reference_model must be selected in models",
    )
    formal_primary_reference = _module_literal(
        runner_tree,
        "FORMAL_PRIMARY_REFERENCE_MODEL",
        "standard runner",
    )
    _require(
        reference_model
        == formal_primary_reference
        == CSSL_REGISTRY_NAME,
        "experiment.reference_model must exactly match the runner's "
        "predeclared CSSL primary reference",
    )
    _require(
        experiment.get("reference_selection")
        == "predeclared before training; not described as strongest",
        "experiment.reference_selection drifted from the preregistration",
    )
    holm_candidates = _unique_strings(
        experiment.get("holm_candidates"),
        "experiment.holm_candidates",
    )
    _require(
        set(holm_candidates).issubset(models),
        "experiment.holm_candidates must be a subset of models",
    )
    _require(
        reference_model not in holm_candidates,
        "experiment.holm_candidates cannot contain the reference model",
    )
    formal_holm_candidates = _module_literal(
        runner_tree,
        "FORMAL_HOLM_CANDIDATES",
        "standard runner",
    )
    _require(
        isinstance(formal_holm_candidates, tuple)
        and holm_candidates == list(formal_holm_candidates),
        "experiment.holm_candidates must exactly match the runner's "
        "predeclared family",
    )
    recent_contract = _validate_recent_comparator_contract(
        _mapping(
            experiment.get("recent_comparator_contract"),
            "experiment.recent_comparator_contract",
        ),
        project_root=root,
        lock_path=lock_file,
    )
    scientific_release_gates = _validate_scientific_release_gates(
        _mapping(
            experiment.get("scientific_release_gates"),
            "experiment.scientific_release_gates",
        ),
        runner_tree=runner_tree,
        models=models,
        reference_model=reference_model,
        holm_candidates=holm_candidates,
    )
    _require(
        experiment.get("device") in {"cpu", "cuda", "auto"},
        "experiment.device must be cpu, cuda, or auto",
    )
    _require(
        experiment.get("verify_checksums") is True,
        "experiment.verify_checksums must be true",
    )
    _require(
        experiment.get("validate_components") is True,
        "experiment.validate_components must be true",
    )

    training = _mapping(experiment.get("training"), "experiment.training")
    checkpoint_window = _validate_training(training)
    model = _mapping(experiment.get("model"), "experiment.model")
    _validate_model(model, sample_length=sample_length)
    statistics = _mapping(
        experiment.get("statistics"), "experiment.statistics"
    )
    _validate_statistics(statistics)

    promotion = _mapping(
        config.get("promotion_requirements"),
        "promotion_requirements",
    )
    _require(
        set(promotion) == set(PROMOTION_REQUIREMENTS),
        "promotion_requirements keys drifted from the formal gate",
    )
    failed_promotions = sorted(
        key for key, value in promotion.items() if value is not True
    )
    _require(
        not failed_promotions,
        "all promotion_requirements must be true; false="
        + repr(failed_promotions),
    )
    _unique_strings(config.get("prohibited_imports"), "prohibited_imports")

    cssl_lock = _load_strict_json(lock_file, "CSSL source lock")
    cssl = _validate_cssl_lock(
        lock_path=lock_file,
        lock=cssl_lock,
        project_root=root,
        config_path=config_file,
        baseline_path=baseline_file,
        baseline_tree=baseline_tree,
        runner_tree=runner_tree,
        recent_contract=recent_contract,
        freeze_models=models,
        reference_model=reference_model,
        holm_candidates=holm_candidates,
        sample_length=sample_length,
    )

    return {
        "schema_version": VALIDATION_SCHEMA,
        "valid": True,
        "read_only": True,
        "config_path": str(config_file),
        "config_sha256": _sha256_file(config_file),
        "freeze_schema_version": config["schema_version"],
        "freeze_status": config["status"],
        "cache_output": cache_output.as_posix(),
        "expected_run_directory": expected_run.as_posix(),
        "split_count": len(normalized_split_counts),
        "total_declared_sources": sum(normalized_split_counts.values()),
        "model_count": len(models),
        "models": models,
        "seed_count": len(seeds),
        "seeds": seeds,
        "reference_model": reference_model,
        "holm_candidate_count": len(holm_candidates),
        "holm_candidates": holm_candidates,
        "recent_comparator_contract": recent_contract,
        "scientific_release_gates": scientific_release_gates,
        "checkpoint_window": checkpoint_window,
        "runner_sha256": _sha256_file(runner_file),
        "cache_builder_sha256": _sha256_file(cache_builder_file),
        "baseline_registry_sha256": _sha256_file(baseline_file),
        "cssl": cssl,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read and validate the formal TVT freeze without starting cache "
            "generation, training, release writes, or project-local caches."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="formal freeze JSON (default: repository formal_tvt_freeze_v1.json)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        summary = validate_formal_freeze(arguments.config)
    except FormalFreezeValidationError as error:
        failure = {
            "schema_version": VALIDATION_SCHEMA,
            "valid": False,
            "read_only": True,
            "config_path": str(arguments.config.expanduser().resolve()),
            "error": str(error),
        }
        print(
            json.dumps(
                failure,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
