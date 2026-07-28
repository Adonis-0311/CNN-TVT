"""Static reproducibility contract between the formal freeze and the paper.

This test intentionally uses only the Python standard library.  It parses
source files with :mod:`ast` instead of importing the runner, model stack, or
PyTorch, so it cannot initialize a device, build a cache, or start training.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import re
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = (
    PROJECT_ROOT
    / "tvt_submission"
    / "configs"
    / "formal_tvt_freeze_v1.json"
)
BUILDER_PATH = PROJECT_ROOT / "standards" / "build_factor_cache.py"
CACHE_PATH = PROJECT_ROOT / "src" / "vimd_amc" / "standards" / "cache.py"
RUNNER_PATH = PROJECT_ROOT / "experiments" / "run_standard_experiment.py"
TRAINING_PATH = PROJECT_ROOT / "src" / "vimd_amc" / "training.py"
LOSSES_PATH = PROJECT_ROOT / "src" / "vimd_amc" / "losses.py"
MODEL_CONFIG_PATH = (
    PROJECT_ROOT / "src" / "vimd_amc" / "models" / "common.py"
)
EVALUATION_PATH = PROJECT_ROOT / "src" / "vimd_amc" / "evaluation.py"
MAIN_PATH = PROJECT_ROOT / "paper" / "main.tex"
REFERENCES_PATH = PROJECT_ROOT / "paper" / "references.bib"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _literal_assignment(path: Path, name: str):
    for node in _tree(path).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            return ast.literal_eval(node.value)
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and node.value is not None
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{path}: literal assignment {name!r} not found")


def _class_literal_defaults(path: Path, class_name: str) -> dict[str, object]:
    for node in _tree(path).body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            defaults: dict[str, object] = {}
            for statement in node.body:
                if (
                    isinstance(statement, ast.AnnAssign)
                    and isinstance(statement.target, ast.Name)
                    and statement.value is not None
                ):
                    try:
                        defaults[statement.target.id] = ast.literal_eval(
                            statement.value
                        )
                    except (ValueError, TypeError):
                        continue
            return defaults
    raise AssertionError(f"{path}: class {class_name!r} not found")


def _function(path: Path, name: str) -> ast.FunctionDef:
    for node in _tree(path).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{path}: function {name!r} not found")


def _argparse_default(path: Path, flag: str):
    function = _function(path, "parse_arguments")
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or not isinstance(
            node.func, ast.Attribute
        ):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        try:
            first = ast.literal_eval(node.args[0])
        except (ValueError, TypeError):
            continue
        if first != flag:
            continue
        for keyword in node.keywords:
            if keyword.arg == "default":
                return ast.literal_eval(keyword.value)
        raise AssertionError(f"{path}: {flag} has no explicit default")
    raise AssertionError(f"{path}: argparse flag {flag!r} not found")


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _has_call(path: Path, function_name: str, dotted_name: str) -> bool:
    return any(
        isinstance(node, ast.Call) and _dotted_name(node.func) == dotted_name
        for node in ast.walk(_function(path, function_name))
    )


def _call_keyword_literal(
    path: Path,
    function_name: str,
    dotted_name: str,
    keyword_name: str,
):
    for node in ast.walk(_function(path, function_name)):
        if not isinstance(node, ast.Call) or _dotted_name(node.func) != dotted_name:
            continue
        for keyword in node.keywords:
            if keyword.arg == keyword_name:
                return ast.literal_eval(keyword.value)
    raise AssertionError(
        f"{path}: {dotted_name} keyword {keyword_name!r} not found"
    )


def _call_positional_literal(
    path: Path,
    function_name: str,
    dotted_name: str,
    position: int,
):
    for node in ast.walk(_function(path, function_name)):
        if not isinstance(node, ast.Call) or _dotted_name(node.func) != dotted_name:
            continue
        if len(node.args) > position:
            return ast.literal_eval(node.args[position])
    raise AssertionError(
        f"{path}: {dotted_name} positional argument {position} not found"
    )


def _constant_range_loops(path: Path, function_name: str) -> tuple[int, ...]:
    values: list[int] = []
    for node in ast.walk(_function(path, function_name)):
        if (
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call)
            and _dotted_name(node.iter.func) == "range"
            and len(node.iter.args) == 1
        ):
            try:
                value = ast.literal_eval(node.iter.args[0])
            except (ValueError, TypeError):
                continue
            if isinstance(value, int) and not isinstance(value, bool):
                values.append(value)
    return tuple(values)


def _literal_for_iterable(
    path: Path,
    function_name: str,
    target_name: str,
) -> tuple[object, ...]:
    for node in ast.walk(_function(path, function_name)):
        if (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == target_name
        ):
            try:
                value = ast.literal_eval(node.iter)
            except (ValueError, TypeError):
                continue
            if isinstance(value, tuple):
                return value
    raise AssertionError(
        f"{path}: literal loop iterable for {target_name!r} not found"
    )


def _model_feature_floor(path: Path) -> int:
    function = _function(path, "make_model_config")
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or _dotted_name(node.func) != "ModelConfig":
            continue
        for keyword in node.keywords:
            if (
                keyword.arg == "feature_channels"
                and isinstance(keyword.value, ast.Call)
                and _dotted_name(keyword.value.func) == "max"
            ):
                constants = [
                    value.value
                    for value in keyword.value.args
                    if isinstance(value, ast.Constant)
                    and isinstance(value.value, int)
                    and not isinstance(value.value, bool)
                ]
                if len(constants) == 1:
                    return constants[0]
    raise AssertionError(f"{path}: feature-channel floor not found")


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _table_with_label(text: str, label: str) -> str:
    marker = rf"\label{{{label}}}"
    if text.count(marker) != 1:
        raise AssertionError(f"{label}: expected exactly one table label")
    label_index = text.index(marker)
    start = text.rfind(r"\begin{table", 0, label_index)
    if start < 0:
        raise AssertionError(f"{label}: table start not found")
    environment = "table*" if text.startswith(r"\begin{table*}", start) else "table"
    terminator = rf"\end{{{environment}}}"
    end = text.find(terminator, label_index)
    if end < 0:
        raise AssertionError(f"{label}: table end not found")
    return text[start : end + len(terminator)]


def _table_cells(table: str) -> dict[str, str]:
    declaration = table.find(r"\begin{tabular}")
    if declaration < 0:
        raise AssertionError("table has no parseable tabular body")
    body_start = table.find("\n", declaration)
    body_end = table.find(r"\end{tabular}", body_start)
    if body_start < 0 or body_end < 0:
        raise AssertionError("table has no parseable tabular body")
    body = re.sub(
        r"\\(?:toprule|midrule|bottomrule)",
        "",
        table[body_start:body_end],
    )
    cells: dict[str, str] = {}
    for row in re.split(r"\\\\", body):
        parts = [_squash(part) for part in row.split("&")]
        if len(parts) != 4 or parts[0] == "Item":
            continue
        if parts[0] in cells or parts[2] in cells:
            raise AssertionError("duplicate frozen-protocol row label")
        cells[parts[0]] = parts[1]
        cells[parts[2]] = parts[3]
    return cells


def _compact_profiles(values: tuple[str, ...]) -> str:
    if not values or any(not value.startswith("TDL-") for value in values):
        raise AssertionError(f"unexpected TDL profile names: {values}")
    return "TDL-" + "/".join(value.removeprefix("TDL-") for value in values)


def _slash_numbers(values: tuple[float, ...], *, scale: float = 1.0) -> str:
    return "/".join(f"{value * scale:g}" for value in values)


def _prose_list(values: tuple[str, ...]) -> str:
    names = {
        "partial_band": "partial-band",
        "ofdm_like": "OFDM-like",
    }
    displayed = [names.get(value, value) for value in values]
    if len(displayed) == 1:
        return displayed[0]
    if len(displayed) == 2:
        return " and ".join(displayed)
    return ", ".join(displayed[:-1]) + ", and " + displayed[-1]


class PaperReproducibilityContractTest(unittest.TestCase):
    def test_frozen_protocol_table_and_prose_follow_source_contract(self) -> None:
        freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        main = MAIN_PATH.read_text(encoding="utf-8")
        references = REFERENCES_PATH.read_text(encoding="utf-8")
        cache_source = CACHE_PATH.read_text(encoding="utf-8")

        builder_sizes = _literal_assignment(BUILDER_PATH, "_PRESET_SIZES")
        modulations = _literal_assignment(BUILDER_PATH, "_FULL_MODULATIONS")
        runner_suites = _literal_assignment(
            RUNNER_PATH, "PREREGISTERED_MODEL_SUITES"
        )
        primary_model = _literal_assignment(
            RUNNER_PATH, "FORMAL_PRIMARY_REFERENCE_MODEL"
        )
        cache_defaults = _class_literal_defaults(
            CACHE_PATH, "TDLCacheBuildConfig"
        )
        training_defaults = _class_literal_defaults(
            TRAINING_PATH, "TrainingConfig"
        )
        loss_defaults = _class_literal_defaults(LOSSES_PATH, "VIMDLossWeights")
        model_defaults = _class_literal_defaults(
            MODEL_CONFIG_PATH, "ModelConfig"
        )

        cache = freeze["cache"]
        experiment = freeze["experiment"]
        training = experiment["training"]
        model = experiment["model"]
        statistics = experiment["statistics"]
        split_counts = cache["expected_split_source_counts"]
        models = experiment["models"]
        seeds = experiment["seeds"]
        recent = experiment["recent_comparator_contract"]

        self.assertEqual(cache["preset"], "headline")
        self.assertEqual(split_counts, builder_sizes["headline"])
        self.assertEqual(models, list(runner_suites["headline"]))
        self.assertEqual(experiment["reference_model"], primary_model)
        self.assertEqual(recent["model"], primary_model)
        self.assertEqual(
            recent["required_label"],
            "CSSL-AMC official-architecture supervised adaptation",
        )

        source_count = sum(split_counts.values())
        view_slots = _literal_for_iterable(
            CACHE_PATH, "_build_pending_views", "view"
        )
        view_count = source_count * len(view_slots)
        fit_count = len(models) * len(seeds)
        self.assertEqual((source_count, view_count, fit_count), (47_000, 94_000, 55))
        self.assertEqual(view_slots, (1, 2))

        full_objective_index = max(
            training["mask_start_epoch"] + training["mask_ramp_epochs"] - 1,
            training["contrastive_start_epoch"]
            + training["contrastive_ramp_epochs"]
            - 1,
        )
        selection_start_index = (
            full_objective_index + training["minimum_full_stage_epochs"] - 1
        )
        self.assertEqual(
            (full_objective_index + 1, selection_start_index + 1),
            (8, 10),
        )

        seen_profiles = _literal_assignment(CACHE_PATH, "_SEEN_PROFILES")
        held_profiles = _literal_assignment(CACHE_PATH, "_HELDOUT_PROFILES")
        seen_jammers = _literal_assignment(CACHE_PATH, "_SEEN_JAMMERS")
        held_jammers = _literal_assignment(CACHE_PATH, "_HELDOUT_JAMMERS")
        exclusions = _literal_assignment(
            CACHE_PATH, "_FACTOR_PROTOCOL_EXCLUSIONS"
        )
        seen_speeds = _literal_assignment(CACHE_PATH, "_SEEN_SPEEDS_KMH")
        held_speeds = _literal_assignment(CACHE_PATH, "_HELDOUT_SPEEDS_KMH")
        snr_values = _literal_assignment(CACHE_PATH, "_STANDARD_SNR_DB_VALUES")
        ordinary_sir = _literal_assignment(
            CACHE_PATH, "_STANDARD_SIR_DB_VALUES"
        )
        hard_sir = _literal_assignment(CACHE_PATH, "_HARD_SIR_DB_VALUES")
        self.assertEqual(
            modulations,
            (
                "BPSK",
                "PI2BPSK",
                "QPSK",
                "8PSK",
                "16QAM",
                "64QAM",
                "256QAM",
                "GMSK",
                "CPFSK",
                "4FSK",
            ),
        )

        self.assertTrue(_has_call(TRAINING_PATH, "train_model", "torch.optim.AdamW"))
        self.assertTrue(
            _has_call(
                TRAINING_PATH,
                "_physical_teacher_target",
                "torch.amp.autocast",
            )
        )
        self.assertIs(
            _call_keyword_literal(
                TRAINING_PATH,
                "_physical_teacher_target",
                "torch.amp.autocast",
                "enabled",
            ),
            False,
        )
        feature_floor = _model_feature_floor(RUNNER_PATH)
        latency_runs = _argparse_default(RUNNER_PATH, "--latency-runs")
        cpu_threads = _argparse_default(RUNNER_PATH, "--cpu-threads")
        interop_threads = _call_positional_literal(
            RUNNER_PATH,
            "main",
            "torch.set_num_interop_threads",
            0,
        )
        warmup_runs = _constant_range_loops(
            EVALUATION_PATH, "complexity_metrics"
        )
        self.assertEqual(
            (feature_floor, latency_runs, cpu_threads, interop_threads),
            (32, 30, 1, 1),
        )
        self.assertEqual(warmup_runs, (10,))
        self.assertEqual(training_defaults["gradient_clip"], 5.0)
        self.assertEqual(training_defaults["num_workers"], 0)

        table = _table_with_label(main, "tab:frozenprotocol")
        cells = _table_cells(table)
        expected_cells = {
            "Input/cache": (
                f"$L={cache['sample_length']}$ complex I/Q samples; "
                f"{cache['guard_samples']}-sample guard; "
                f"$f_s={cache_defaults['sample_rate_hz'] / 1e6:g}$~MHz; "
                f"$f_c={cache_defaults['carrier_frequency_hz'] / 1e9:g}$~GHz; "
                f"master seed {cache['master_seed']}"
            ),
            "Targets": (
                r"BPSK, $\pi/2$-BPSK, QPSK, 8PSK, 16/64/256QAM, "
                "GMSK, CPFSK, and 4FSK"
            ),
            "Channel/mobility": (
                f"{_compact_profiles(seen_profiles)} seen, "
                f"{_compact_profiles(held_profiles)} held; "
                f"{_slash_numbers(cache_defaults['delay_spreads_s'], scale=1e9)}"
                "-ns delay spreads; "
                f"{_slash_numbers(seen_speeds)}~km/h seen, "
                f"{_slash_numbers(held_speeds)}~km/h held"
            ),
            "Jammers": (
                f"{_prose_list(seen_jammers)} seen; "
                f"{_prose_list(held_jammers)} held; "
                f"{' and '.join(exclusions)} excluded"
            ),
            "Quality grids": (
                r"SNR $\{" + ",".join(f"{value:g}" for value in snr_values)
                + r"\}$~dB; ordinary SIR $\{"
                + ",".join(f"{value:g}" for value in ordinary_sir)
                + r"\}$~dB; hard SIR $\{"
                + ",".join(f"{value:g}" for value in hard_sir)
                + r"\}$~dB"
            ),
            "Source budget": (
                f"train/validation {split_counts['train']:,}/"
                f"{split_counts['validation']:,}; each of seven test splits "
                f"{split_counts['id_test']:,}; {source_count:,} disjoint "
                f"source IDs and {view_count:,} cached views"
            ),
            "Transform/widths": (
                f"STFT {model['n_fft']}/{model['hop_length']} (FFT/hop); "
                f"complex widths {model_defaults['first_complex_channels']}/"
                f"{model_defaults['second_complex_channels']}; "
                f"feature/spectral widths "
                f"{max(feature_floor, model['spectral_channels'])}/"
                f"{model['spectral_channels']}; embedding/environment "
                f"{model['embedding_dim']}/{model['environment_dim']}; "
                f"dropout {model['dropout']:g}"
            ),
            "Routing/objective": (
                rf"$\rho\in[{model_defaults['rho_min']:g},"
                rf"{model_defaults['rho_max']:g}]$, "
                rf"$\tau_c\in[{model_defaults['temperature_min']:g},"
                rf"{model_defaults['temperature_max']:g}]$; "
                rf"$\lambda_{{j,q,m,x,\perp}}=("
                + ",".join(
                    f"{loss_defaults[name]:.2f}"
                    for name in (
                        "jammer",
                        "quality",
                        "mask",
                        "contrastive",
                        "orthogonality",
                    )
                )
                + rf")$; smoothing {loss_defaults['label_smoothing']:.2f}; "
                rf"$\tau_x={loss_defaults['contrastive_temperature']:.2f}$"
            ),
            "Optimization": (
                rf"AdamW; {training['epochs']} epochs; "
                rf"batch {training['batch_size']}; "
                rf"$\eta=3\!\times\!10^{{-4}}$; "
                rf"weight decay $10^{{-2}}$; "
                rf"gradient clip {training_defaults['gradient_clip']:g}; "
                rf"workers {training_defaults['num_workers']}; "
                rf"CPU intra/inter-op threads {cpu_threads}/{interop_threads}; "
                "CUDA AMP (teacher in FP32)"
            ),
            "Curriculum/selection": (
                f"mask/XCC first active at epochs "
                f"{training['mask_start_epoch'] + 1}/"
                f"{training['contrastive_start_epoch'] + 1} (one-based), "
                f"each ramps for {training['mask_ramp_epochs']} epochs; "
                f"full objective at epoch {full_objective_index + 1}; "
                f"selection starts at {selection_start_index + 1}; "
                f"patience {training['patience']}"
            ),
            "Execution matrix": (
                "A0--A7 plus MCLDNN, IQFormer-inspired, and CSSL adaptation; "
                r"seeds $\{" + ",".join(str(seed) for seed in seeds)
                + rf"\}}$; {fit_count} fits; {recent['required_label']} "
                "is the fixed primary anchor"
            ),
            "Inference/statistics": (
                f"one view; {statistics['bootstrap_draws']:,}-draw seed/source "
                f"hierarchical paired bootstrap "
                f"(seed {statistics['bootstrap_seed']}); class-stratified "
                "source clusters; validation excluded"
            ),
            "Latency protocol": (
                f"batch one; {warmup_runs[0]} warm-up passes; "
                f"{latency_runs} synchronized timed passes; P50/P95 on the "
                "recorded isolated device"
            ),
            "Release integrity": (
                "checksum and component validation; unchanged source tree; "
                "eligible checkpoint, no fallback; artifact-derived result "
                "cells only"
            ),
        }
        self.assertEqual(cells, expected_cells)

        caption = _squash(table)
        self.assertIn("Configured Prospectively", caption)
        self.assertIn("Artifact-Bound Release Lock", caption)
        self.assertIn(
            "The formal headline cache and run have not been generated",
            main,
        )
        self.assertIn(
            "only screening/integration cache evidence exists",
            main,
        )
        self.assertIn(
            "If built and validated, the eligible headline manifest",
            main,
        )

        results_index = main.index(r"\section{Results and Falsification Gates}")
        ablation_index = main.index(r"\label{tab:ablations}")
        ablation_end_index = main.index(r"\end{table*}", ablation_index)
        reproducibility_index = main.index(
            r"\section{Reproducibility, Limitations, and Scope}"
        )
        # IEEEtran can place a two-column float only at a later page top.  Keep
        # the source immediately before the Results section so Table IV renders
        # at the top of that section's page instead of after the references.
        self.assertLess(ablation_index, ablation_end_index)
        self.assertLess(ablation_end_index, results_index)
        self.assertLess(results_index, reproducibility_index)
        self.assertIn(
            r"Points over the Predeclared \PrimaryReference.",
            main,
        )
        self.assertNotIn("after paired multiplicity", main)
        self.assertIn("Family-Wise Simultaneous", main)

        self.assertNotIn("Table-I", cache_source)
        self.assertEqual(main.count(r"\label{tab:splits}"), 1)
        self.assertEqual(main.count(r"Table~\ref{tab:splits}"), 1)
        self.assertIn(r"\cite{du2025contrastive}", main)
        self.assertIn(r"\cite{du2025csslcode}", main)
        self.assertIn("@misc{du2025csslcode,", references)
        self.assertIn(
            "https://github.com/dumingyang20/CSSL-AMC-Pytorch/commit/"
            "2fbc5b3e12f780b0b26eb0ee2c33d592739aa24f",
            references,
        )

        conclusion = main[
            main.index(r"\section{Conclusion}") :
            main.index(r"\section*{Internal Submission Note}")
        ]
        normalized_conclusion = _squash(conclusion)
        self.assertIn(r"\ifinternalreview", conclusion)
        self.assertIn(r"\else", conclusion)
        self.assertIn(r"\RegimeHardGain", conclusion)
        self.assertIn(r"\PrimaryReference", conclusion)
        self.assertIn(
            "quantitative conclusions will be stated only after",
            normalized_conclusion,
        )
        self.assertIn(
            "Under the eligible locked simulation run",
            normalized_conclusion,
        )
        self.assertNotIn(
            "diagnostic component probes.  The internal build",
            normalized_conclusion,
        )
        self.assertIn(
            "This internal engineering draft must not be submitted unchanged",
            _squash(main),
        )
        self.assertIn(
            "This internal note is an operational gate, not that disclosure.",
            _squash(main),
        )


if __name__ == "__main__":
    unittest.main()
