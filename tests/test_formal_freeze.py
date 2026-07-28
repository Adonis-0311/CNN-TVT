from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tvt_submission import validate_formal_freeze as validator  # noqa: E402


CONFIG_PATH = (
    ROOT
    / "tvt_submission"
    / "configs"
    / "formal_tvt_freeze_v1.json"
)
LOCK_PATH = (
    ROOT / "tvt_submission" / "sources" / "cssl_amc_2025.lock.json"
)
RUNNER_PATH = ROOT / "experiments" / "run_standard_experiment.py"
BUILDER_PATH = ROOT / "standards" / "build_factor_cache.py"
BASELINES_PATH = ROOT / "src" / "vimd_amc" / "models" / "baselines.py"
VALIDATOR_PATH = ROOT / "tvt_submission" / "validate_formal_freeze.py"
RUN_LOCAL_PATH = ROOT / "tvt_submission" / "run_local.ps1"


class FormalFreezeValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.freeze = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def _write_config(
        self,
        directory: Path,
        payload: dict[str, object],
    ) -> Path:
        path = directory / "candidate.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def _assert_rejected(
        self,
        payload: dict[str, object],
        message_pattern: str,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_path = self._write_config(Path(temporary), payload)
            with self.assertRaisesRegex(
                validator.FormalFreezeValidationError,
                message_pattern,
            ):
                validator.validate_formal_freeze(config_path)

    def _copy_minimal_project(self, destination: Path) -> dict[str, Path]:
        sources = {
            "config": CONFIG_PATH,
            "lock": LOCK_PATH,
            "license": (
                ROOT
                / "tvt_submission"
                / "sources"
                / "licenses"
                / "Apache-2.0.txt"
            ),
            "baselines": BASELINES_PATH,
            "builder": BUILDER_PATH,
        }
        relative_paths = {
            "config": Path(
                "tvt_submission/configs/formal_tvt_freeze_v1.json"
            ),
            "lock": Path(
                "tvt_submission/sources/cssl_amc_2025.lock.json"
            ),
            "license": Path(
                "tvt_submission/sources/licenses/Apache-2.0.txt"
            ),
            "baselines": Path("src/vimd_amc/models/baselines.py"),
            "builder": Path("standards/build_factor_cache.py"),
        }
        copied: dict[str, Path] = {}
        for name, source in sources.items():
            target = destination / relative_paths[name]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            copied[name] = target
        return copied

    def test_real_freeze_passes_without_opening_any_file_for_write(self) -> None:
        original_open = Path.open

        def read_only_open(
            path: Path,
            mode: str = "r",
            *args: object,
            **kwargs: object,
        ):
            self.assertFalse(
                any(marker in mode for marker in ("w", "a", "x", "+")),
                f"validator attempted a write-capable open: {path} {mode}",
            )
            return original_open(path, mode, *args, **kwargs)

        with mock.patch.object(Path, "open", new=read_only_open):
            summary = validator.validate_formal_freeze(CONFIG_PATH)

        self.assertTrue(summary["valid"])
        self.assertTrue(summary["read_only"])
        self.assertEqual(summary["split_count"], 9)
        self.assertEqual(summary["model_count"], 11)
        self.assertEqual(summary["seed_count"], 5)
        self.assertEqual(summary["holm_candidate_count"], 5)
        self.assertEqual(
            summary["checkpoint_window"],
            {
                "full_objective_epoch": 8,
                "selection_start_epoch": 10,
                "eligible_epoch_count": 21,
            },
        )
        self.assertEqual(
            summary["config_sha256"],
            hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            summary["cache_builder_sha256"],
            hashlib.sha256(BUILDER_PATH.read_bytes()).hexdigest(),
        )

    def test_schema_and_status_are_fail_closed(self) -> None:
        mutations = (
            ("schema_version", "vimd_amc.tvt.formal_freeze.v0", "schema"),
            ("status", "executed", "status"),
            ("created_date", "20260728", "strict ISO YYYY-MM-DD"),
            ("created_date", "2026-W31-2", "strict ISO YYYY-MM-DD"),
            ("created_date", "2026-02-30", "valid calendar date"),
        )
        for key, value, expected in mutations:
            with self.subTest(key=key):
                candidate = copy.deepcopy(self.freeze)
                candidate[key] = value
                self._assert_rejected(candidate, expected)

    def test_paths_and_run_directory_must_be_safe_and_exact(self) -> None:
        candidates: list[tuple[dict[str, object], str]] = []

        unsafe_cache = copy.deepcopy(self.freeze)
        unsafe_cache["cache"]["output"] = "../cache"
        candidates.append((unsafe_cache, "unsafe path component"))

        absolute_output = copy.deepcopy(self.freeze)
        absolute_output["experiment"]["output"] = "C:/outside"
        candidates.append((absolute_output, "project-relative"))

        unsafe_run_id = copy.deepcopy(self.freeze)
        unsafe_run_id["experiment"]["run_id"] = "../run"
        candidates.append((unsafe_run_id, "safe path component"))

        mismatched_run = copy.deepcopy(self.freeze)
        mismatched_run["experiment"][
            "expected_run_directory"
        ] = "artifacts/a-different-run"
        candidates.append((mismatched_run, "disagrees"))

        cache_contains_output = copy.deepcopy(self.freeze)
        cache_contains_output["cache"]["output"] = "artifacts"
        cache_contains_output["experiment"]["output"] = "artifacts/runs"
        cache_contains_output["experiment"][
            "expected_run_directory"
        ] = "artifacts/runs/tvt_headline_1024_5seed_v1"
        candidates.append((cache_contains_output, "must not overlap"))

        output_contains_cache = copy.deepcopy(self.freeze)
        output_contains_cache["cache"]["output"] = "artifacts/cache"
        output_contains_cache["experiment"]["output"] = "artifacts"
        candidates.append((output_contains_cache, "must not overlap"))

        for candidate, expected in candidates:
            with self.subTest(expected=expected):
                self._assert_rejected(candidate, expected)

    def test_exact_nine_positive_split_contract_is_required(self) -> None:
        missing = copy.deepcopy(self.freeze)
        del missing["cache"]["expected_split_source_counts"]["validation"]

        zero = copy.deepcopy(self.freeze)
        zero["cache"]["expected_split_source_counts"]["id_test"] = 0

        extra = copy.deepcopy(self.freeze)
        extra["cache"]["expected_split_source_counts"]["extra"] = 10

        for candidate, expected in (
            (missing, "exact ordered nine-split"),
            (zero, "positive integer"),
            (extra, "exact ordered nine-split"),
        ):
            with self.subTest(expected=expected):
                self._assert_rejected(candidate, expected)

    def test_models_must_be_unique_and_exactly_match_headline_registry(
        self,
    ) -> None:
        duplicate = copy.deepcopy(self.freeze)
        duplicate["experiment"]["models"][-1] = duplicate["experiment"][
            "models"
        ][0]

        reordered = copy.deepcopy(self.freeze)
        reordered["experiment"]["models"] = list(
            reversed(reordered["experiment"]["models"])
        )

        for candidate, expected in (
            (duplicate, "duplicates"),
            (reordered, "exactly match"),
        ):
            with self.subTest(expected=expected):
                self._assert_rejected(candidate, expected)

        with tempfile.TemporaryDirectory() as temporary:
            runner_copy = Path(temporary) / "runner.py"
            source = RUNNER_PATH.read_text(encoding="utf-8")
            source = source.replace(
                '        "cssl_amc_supervised_adaptation",\n',
                "",
                1,
            )
            runner_copy.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(
                validator.FormalFreezeValidationError,
                "exactly match",
            ):
                validator.validate_formal_freeze(
                    CONFIG_PATH,
                    runner_path=runner_copy,
                )

    def test_audited_python_bindings_cannot_be_redefined_or_shadowed(
        self,
    ) -> None:
        runner_source = RUNNER_PATH.read_text(encoding="utf-8")
        runner_mutations = (
            (
                runner_source
                + "\nPREREGISTERED_MODEL_SUITES = "
                + '{"headline": ("shadow",), "screening": ("shadow",)}\n',
                "exactly one module-level binding",
            ),
            (
                runner_source
                + '\nFORMAL_RELEASE_DESIGNATION: str = "shadowed"\n',
                "exactly one module-level binding",
            ),
            (
                runner_source.replace(
                    "    for public_name, attribute, uses_config in optional:\n",
                    "    optional = ()\n"
                    "    for public_name, attribute, uses_config in optional:\n",
                    1,
                ),
                "exactly one direct static",
            ),
        )
        for source, expected in runner_mutations:
            with self.subTest(expected=expected):
                with tempfile.TemporaryDirectory() as temporary:
                    runner_copy = Path(temporary) / "runner.py"
                    runner_copy.write_text(source, encoding="utf-8")
                    with self.assertRaisesRegex(
                        validator.FormalFreezeValidationError,
                        expected,
                    ):
                        validator.validate_formal_freeze(
                            CONFIG_PATH,
                            runner_path=runner_copy,
                        )

        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            copied = self._copy_minimal_project(project)
            with copied["baselines"].open("a", encoding="utf-8") as stream:
                stream.write("\nCSSLAMCSupervisedAdaptation = None\n")
            with self.assertRaisesRegex(
                validator.FormalFreezeValidationError,
                "exactly one unshadowed module-level class",
            ):
                validator.validate_formal_freeze(
                    copied["config"],
                    project_root=project,
                    runner_path=RUNNER_PATH,
                    cache_builder_path=copied["builder"],
                    baselines_path=copied["baselines"],
                    cssl_lock_path=copied["lock"],
                )

    def test_cache_builder_headline_preset_is_bound_to_the_freeze(self) -> None:
        builder_source = BUILDER_PATH.read_text(encoding="utf-8")
        mutations = (
            (
                builder_source.replace(
                    '    "headline": {\n        "train": 10_000,',
                    '    "headline": {\n        "train": 20_000,',
                    1,
                ),
                "headline preset split sizes disagree",
            ),
            (
                builder_source.replace(
                    '"headline": "headline_formal_tvt_evidence",',
                    '"headline": "drifted_designation",',
                    1,
                ),
                "headline designation disagrees",
            ),
        )
        for source, expected in mutations:
            with self.subTest(expected=expected):
                with tempfile.TemporaryDirectory() as temporary:
                    project = Path(temporary)
                    copied = self._copy_minimal_project(project)
                    copied["builder"].write_text(source, encoding="utf-8")
                    with self.assertRaisesRegex(
                        validator.FormalFreezeValidationError,
                        expected,
                    ):
                        validator.validate_formal_freeze(
                            copied["config"],
                            project_root=project,
                            runner_path=RUNNER_PATH,
                            cache_builder_path=copied["builder"],
                            baselines_path=copied["baselines"],
                            cssl_lock_path=copied["lock"],
                        )

    def test_seeds_reference_and_holm_family_are_legal(self) -> None:
        duplicate_seed = copy.deepcopy(self.freeze)
        duplicate_seed["experiment"]["seeds"][-1] = duplicate_seed[
            "experiment"
        ]["seeds"][0]

        too_few_seeds = copy.deepcopy(self.freeze)
        too_few_seeds["experiment"]["seeds"] = [17, 29, 43, 71]

        unknown_reference = copy.deepcopy(self.freeze)
        unknown_reference["experiment"]["reference_model"] = "absent"

        duplicate_holm = copy.deepcopy(self.freeze)
        duplicate_holm["experiment"]["holm_candidates"][-1] = duplicate_holm[
            "experiment"
        ]["holm_candidates"][0]

        unknown_holm = copy.deepcopy(self.freeze)
        unknown_holm["experiment"]["holm_candidates"][-1] = "absent"

        reference_in_holm = copy.deepcopy(self.freeze)
        reference_in_holm["experiment"]["holm_candidates"][
            0
        ] = reference_in_holm["experiment"]["reference_model"]

        for candidate, expected in (
            (duplicate_seed, "seeds contains duplicates"),
            (too_few_seeds, "at least five"),
            (unknown_reference, "selected in models"),
            (duplicate_holm, "duplicates"),
            (unknown_holm, "subset of models"),
            (reference_in_holm, "cannot contain the reference"),
        ):
            with self.subTest(expected=expected):
                self._assert_rejected(candidate, expected)

    def test_training_statistics_and_promotion_values_are_gated(self) -> None:
        no_checkpoint = copy.deepcopy(self.freeze)
        no_checkpoint["experiment"]["training"][
            "contrastive_start_epoch"
        ] = 29

        no_bootstrap = copy.deepcopy(self.freeze)
        no_bootstrap["experiment"]["statistics"]["bootstrap_draws"] = 0

        validation_included = copy.deepcopy(self.freeze)
        validation_included["experiment"]["statistics"][
            "validation_excluded"
        ] = False

        false_promotion = copy.deepcopy(self.freeze)
        false_promotion["promotion_requirements"][
            "source_tree_unchanged"
        ] = False

        amp_disabled = copy.deepcopy(self.freeze)
        amp_disabled["experiment"]["training"]["use_amp"] = False

        for candidate, expected in (
            (no_checkpoint, "no checkpoint-selection-eligible"),
            (no_bootstrap, "positive integer"),
            (validation_included, "must be true"),
            (false_promotion, "all promotion_requirements must be true"),
            (amp_disabled, "use_amp must remain true"),
        ):
            with self.subTest(expected=expected):
                self._assert_rejected(candidate, expected)

    def test_recent_comparator_contract_is_frozen(self) -> None:
        unsafe_lock = copy.deepcopy(self.freeze)
        unsafe_lock["experiment"]["recent_comparator_contract"][
            "source_lock"
        ] = "../source.lock.json"

        wrong_label = copy.deepcopy(self.freeze)
        wrong_label["experiment"]["recent_comparator_contract"][
            "required_label"
        ] = "CSSL-AMC"

        overclaim = copy.deepcopy(self.freeze)
        overclaim["experiment"]["recent_comparator_contract"][
            "complete_published_method_reproduction"
        ] = True

        for candidate, expected in (
            (unsafe_lock, "unsafe path component"),
            (wrong_label, "required_label drifted"),
            (overclaim, "must not claim"),
        ):
            with self.subTest(expected=expected):
                self._assert_rejected(candidate, expected)

    def test_cssl_source_lock_and_baseline_registry_must_agree(self) -> None:
        mutations = (
            (
                lambda lock: lock.__setitem__("schema_version", "wrong"),
                "schema mismatch",
            ),
            (
                lambda lock: lock["official_source"].__setitem__(
                    "commit", "0" * 40
                ),
                "disagrees with baseline registry",
            ),
            (
                lambda lock: lock["local_adaptation"].__setitem__(
                    "complete_published_method_reproduction", True
                ),
                "disagree",
            ),
            (
                lambda lock: lock.__setitem__("audit_date", "20260728"),
                "strict ISO YYYY-MM-DD",
            ),
        )
        for mutate, expected in mutations:
            with self.subTest(expected=expected):
                with tempfile.TemporaryDirectory() as temporary:
                    project = Path(temporary)
                    copied = self._copy_minimal_project(project)
                    lock = json.loads(
                        copied["lock"].read_text(encoding="utf-8")
                    )
                    mutate(lock)
                    copied["lock"].write_text(
                        json.dumps(lock, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(
                        validator.FormalFreezeValidationError,
                        expected,
                    ):
                        validator.validate_formal_freeze(
                            copied["config"],
                            project_root=project,
                            runner_path=RUNNER_PATH,
                            baselines_path=copied["baselines"],
                            cssl_lock_path=copied["lock"],
                        )

    def test_cli_success_is_machine_readable_and_preserves_inputs(
        self,
    ) -> None:
        before = {
            CONFIG_PATH: hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
            LOCK_PATH: hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest(),
        }
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(VALIDATOR_PATH),
                "--config",
                str(CONFIG_PATH),
            ],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, "")
        summary = json.loads(completed.stdout)
        self.assertTrue(summary["valid"])
        self.assertTrue(summary["read_only"])
        self.assertEqual(summary["config_sha256"], before[CONFIG_PATH])
        for path, digest in before.items():
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                digest,
            )

    def test_cli_failure_is_nonzero_machine_readable_and_writes_nothing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            candidate = copy.deepcopy(self.freeze)
            candidate["status"] = "executed"
            config_path = self._write_config(directory, candidate)
            before = {
                path.name: path.read_bytes()
                for path in directory.iterdir()
                if path.is_file()
            }
            environment = dict(os.environ)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(VALIDATOR_PATH),
                    "--config",
                    str(config_path),
                ],
                cwd=directory,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(completed.stdout, "")
            failure = json.loads(completed.stderr)
            self.assertFalse(failure["valid"])
            self.assertTrue(failure["read_only"])
            after = {
                path.name: path.read_bytes()
                for path in directory.iterdir()
                if path.is_file()
            }
            self.assertEqual(after, before)
            self.assertFalse((directory / "__pycache__").exists())

    def test_run_local_invokes_validator_before_config_or_stage_output(
        self,
    ) -> None:
        source = RUN_LOCAL_PATH.read_text(encoding="utf-8")
        invocation = "& $Python -B $ValidatorPath --config $ConfigPath"
        invocation_index = source.index(invocation)
        self.assertLess(
            invocation_index,
            source.index("$Config = Get-Content"),
        )
        self.assertLess(invocation_index, source.index("function Show-Command"))
        self.assertLess(invocation_index, source.index("Write-Host"))
        self.assertIn(
            "no stage command was printed or executed",
            source,
        )
        self.assertIn('"standards\\build_factor_cache.py"', source)
        self.assertIn('"--use-amp"', source)


if __name__ == "__main__":
    unittest.main()
