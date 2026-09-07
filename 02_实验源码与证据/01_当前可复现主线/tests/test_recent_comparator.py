from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.run_experiment import build_model  # noqa: E402
from experiments.run_standard_experiment import (  # noqa: E402
    PREREGISTERED_MODEL_SUITES,
    available_model_factories,
)
from tvt_submission import generate_macro_values  # noqa: E402
from vimd_amc.models.baselines import (  # noqa: E402
    CSSL_AMC_AUDITED_COMMIT,
    CSSLAMCSupervisedAdaptation,
)
from vimd_amc.models.common import ModelConfig  # noqa: E402


class RecentComparatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_cssl_official_architecture_adaptation_shape_and_count(self) -> None:
        model = CSSLAMCSupervisedAdaptation(10).eval()
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            8_631_948,
        )
        self.assertEqual(model.encoder.readout.in_features, 128 * 512)
        self.assertEqual(len(model.encoder.stage1), 2)
        self.assertEqual(len(model.encoder.stage2), 2)
        self.assertEqual(len(model.classifier), 2)
        self.assertEqual(
            tuple(
                model.encoder.noise_level_estimator.input_convolution[
                    0
                ].weight.shape
            ),
            (32, 2, 3),
        )
        self.assertEqual(
            tuple(model.encoder.input_convolution.weight.shape),
            (64, 4, 3),
        )
        self.assertIsNone(model.encoder.input_convolution.bias)
        self.assertEqual(
            tuple(model.encoder.stage2[0].shortcut[0].weight.shape),
            (128, 64, 1),
        )
        self.assertEqual(
            tuple(model.encoder.readout.weight.shape),
            (128, 128 * 512),
        )
        with torch.no_grad():
            output = model(torch.randn(1, 2, 1024))
        self.assertEqual(tuple(output["embedding"].shape), (1, 128))
        self.assertEqual(tuple(output["logits"].shape), (1, 10))
        self.assertTrue(torch.isfinite(output["logits"]).all())

    def test_cssl_adaptation_is_strictly_1024_sample_raw_iq(self) -> None:
        model = CSSLAMCSupervisedAdaptation(10).eval()
        for invalid in (
            torch.randn(1, 1, 1024),
            torch.randn(1, 2, 1023),
            torch.randn(1, 2, 1025),
            torch.randn(2, 1024),
        ):
            with self.assertRaises(ValueError):
                model(invalid)

    def test_cssl_provenance_prevents_full_method_claim(self) -> None:
        provenance = CSSLAMCSupervisedAdaptation.provenance
        self.assertEqual(provenance["audited_commit"], CSSL_AMC_AUDITED_COMMIT)
        self.assertEqual(provenance["license"], "Apache-2.0")
        self.assertIn("supervised adaptation", provenance["claim_level"])
        self.assertIn(
            "not a reproduction",
            provenance["claim_level"],
        )
        differences = " ".join(provenance["material_differences"])
        self.assertIn("no external checkpoint", differences)
        self.assertIn("pretraining", differences)

    def test_both_runners_register_supervised_adaptation(self) -> None:
        name = "cssl_amc_supervised_adaptation"
        factories = available_model_factories()
        self.assertIn(name, factories)
        standard = factories[name](
            10,
            9,
            ModelConfig(),
        )
        self.assertIsInstance(standard.model, CSSLAMCSupervisedAdaptation)
        self.assertEqual(
            standard.objective.name,
            "paired_view_modulation_ce",
        )
        self.assertIsNone(standard.teacher)

        proxy = build_model(
            name,
            classes=10,
            jammers=9,
            config=ModelConfig(),
        )
        self.assertIsInstance(proxy.model, CSSLAMCSupervisedAdaptation)
        self.assertEqual(proxy.objective.name, "paired_view_modulation_ce")
        self.assertIsNone(proxy.teacher)

    def test_formal_freeze_and_eligibility_suite_are_synchronized(self) -> None:
        name = "cssl_amc_supervised_adaptation"
        freeze_path = (
            ROOT
            / "tvt_submission"
            / "configs"
            / "formal_tvt_freeze_v1.json"
        )
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        self.assertEqual(freeze["cache"]["sample_length"], 1024)
        self.assertEqual(
            tuple(freeze["experiment"]["models"]),
            PREREGISTERED_MODEL_SUITES["headline"],
        )
        self.assertEqual(freeze["experiment"]["reference_model"], name)
        self.assertNotIn(name, freeze["experiment"]["holm_candidates"])
        self.assertIn(name, PREREGISTERED_MODEL_SUITES["headline"])
        self.assertIn(name, generate_macro_values.BASELINE_MODELS)
        self.assertEqual(
            generate_macro_values.MODEL_LABELS[name],
            "CSSL-AMC official-architecture supervised adaptation",
        )
        contract = freeze["experiment"]["recent_comparator_contract"]
        self.assertEqual(contract["model"], name)
        self.assertFalse(contract["complete_published_method_reproduction"])
        self.assertFalse(contract["structured_interference_specific"])

    def test_source_lock_and_local_license_are_present(self) -> None:
        lock_path = (
            ROOT
            / "tvt_submission"
            / "sources"
            / "cssl_amc_2025.lock.json"
        )
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        self.assertEqual(
            lock["official_source"]["commit"],
            CSSL_AMC_AUDITED_COMMIT,
        )
        self.assertEqual(
            lock["local_adaptation"]["registry_name"],
            "cssl_amc_supervised_adaptation",
        )
        self.assertEqual(
            lock["local_adaptation"]["formal_result_status"],
            "not_executed",
        )
        self.assertTrue(
            lock["local_adaptation"]["recent_auditable_amc_comparator"]
        )
        self.assertFalse(
            lock["local_adaptation"]["complete_published_method_reproduction"]
        )
        self.assertFalse(
            lock["local_adaptation"]["structured_interference_specific"]
        )
        for source in lock["official_source"]["audited_files"]:
            self.assertRegex(source["sha256"], r"^[0-9a-f]{64}$")
        license_path = (
            ROOT
            / lock["official_source"]["local_license_copy"]
        )
        self.assertTrue(license_path.is_file())
        self.assertIn(
            "Apache License",
            license_path.read_text(encoding="utf-8"),
        )
        self.assertEqual(
            hashlib.sha256(license_path.read_bytes()).hexdigest(),
            lock["official_source"]["license_sha256"],
        )

    def test_machine_readable_status_keeps_structured_interference_open(
        self,
    ) -> None:
        status = json.loads(
            (ROOT / "tvt_submission" / "DELIVERY_STATUS.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(
            status["completed"]["recent_auditable_amc_comparator"]
        )
        self.assertTrue(
            status["not_completed"][
                "verified_recent_structured_interference_comparator"
            ]
        )
        self.assertNotIn(
            "verified_recent_interference_comparator",
            status["not_completed"],
        )

    def test_manuscript_uses_the_bounded_adaptation_label(self) -> None:
        manuscript = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
        normalized = " ".join(manuscript.split())
        self.assertIn(
            "CSSL-AMC official-architecture supervised adaptation",
            normalized,
        )
        self.assertIn(r"\cite{du2025contrastive}", manuscript)
        self.assertIn(
            "not a structured-interference-specific method",
            normalized,
        )


if __name__ == "__main__":
    unittest.main()
