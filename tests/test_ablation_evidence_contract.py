"""Static and synthetic contract tests for the formal A0--A7 evidence path."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import sys
import unittest

from tvt_submission import generate_macro_values as generator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "experiments" / "run_standard_experiment.py"
FREEZE_PATH = (
    PROJECT_ROOT
    / "tvt_submission"
    / "configs"
    / "formal_tvt_freeze_v1.json"
)


def _load_runner():
    specification = importlib.util.spec_from_file_location(
        "vimd_ablation_contract_runner",
        RUNNER_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("could not load runner")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _json_normalize(value):
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
    )


class AblationEvidenceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = _load_runner()

    def test_freeze_runner_and_generator_share_exact_family(self) -> None:
        freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        frozen = freeze["experiment"]["scientific_release_gates"][
            "hard_ablation_family"
        ]
        runner = self.runner.SCIENTIFIC_RELEASE_THRESHOLDS[
            "hard_ablation_family"
        ]
        generated = generator.SCIENTIFIC_RELEASE_THRESHOLDS[
            "hard_ablation_family"
        ]
        self.assertEqual(_json_normalize(runner), frozen)
        self.assertEqual(_json_normalize(generated), frozen)
        self.assertEqual(len(frozen["contrasts"]), 6)
        self.assertEqual(
            frozen["multiplicity_method"],
            (
                "joint_max_absolute_centered_deviation_"
                "hierarchical_paired_bootstrap"
            ),
        )

    def test_macro_contract_is_exactly_ninety_seven_and_pure_letters(
        self,
    ) -> None:
        self.assertEqual(len(generator.PROVENANCE_MACROS), 97)
        self.assertEqual(len(set(generator.PROVENANCE_MACROS)), 97)
        new_macros = (
            *generator.ABLATION_MEAN_MACROS,
            *generator.ABLATION_CONTRAST_MACROS,
        )
        self.assertEqual(len(new_macros), 24)
        self.assertTrue(all(re.fullmatch(r"[A-Za-z]+", name) for name in new_macros))

    def test_checked_in_placeholder_and_table_three_wire_all_new_macros(
        self,
    ) -> None:
        results_text = (
            PROJECT_ROOT / "paper" / "results_auto.tex"
        ).read_text(encoding="utf-8")
        main_text = (PROJECT_ROOT / "paper" / "main.tex").read_text(
            encoding="utf-8"
        )
        placeholder_names = re.findall(
            r"\\newcommand\{\\([A-Za-z]+)\}\{--\}",
            results_text,
        )
        self.assertEqual(
            set(placeholder_names),
            set(generator.PROVENANCE_MACROS),
        )
        for name in (
            *generator.ABLATION_MEAN_MACROS,
            *generator.ABLATION_CONTRAST_MACROS,
        ):
            self.assertEqual(results_text.count(rf"\newcommand{{\{name}}}"), 1)
            self.assertGreaterEqual(main_text.count(rf"\{name}"), 1)
        table = main_text[
            main_text.index(r"\label{tab:ablations}") :
            main_text.index(r"\end{table*}", main_text.index(r"\label{tab:ablations}"))
        ]
        for identifier in ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"):
            self.assertIn(identifier, table)
        self.assertIn("Family-Wise", main_text)
        self.assertNotIn("improve both mask agreement", main_text)

    def test_runner_csv_schema_is_fixed_and_six_row_oriented(self) -> None:
        self.assertEqual(len(self.runner.ABLATION_PAIRED_COLUMNS), 33)
        self.assertEqual(
            [record["contrast_id"] for record in self.runner.FORMAL_ABLATION_CONTRASTS],
            [
                "teacher",
                "multitask",
                "exact_source_contrast",
                "full_vs_single",
                "full_vs_dual",
                "bypass",
            ],
        )
        self.assertEqual(
            self.runner.FORMAL_ABLATION_CONTRASTS[0]["reference"],
            "a2_tri_no_teacher",
        )
        self.assertEqual(
            self.runner.FORMAL_ABLATION_CONTRASTS[0]["candidate"],
            "a3_tri_teacher",
        )
        self.assertEqual(
            self.runner.FORMAL_ABLATION_CONTRASTS[-1]["reference"],
            "a7_vimd_no_residual",
        )
        self.assertEqual(
            self.runner.FORMAL_ABLATION_CONTRASTS[-1]["candidate"],
            "a5_vimd_full",
        )


if __name__ == "__main__":
    unittest.main()
