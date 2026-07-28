"""Tests for non-deployable clean-oracle and transparent classical controls."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import Dataset

from vimd_amc.data.controls import CleanOracleInputDataset
from vimd_amc.models.classical import (
    ClassicalHOCyclostationaryClassifier,
    ClassicalHOCyclostationaryFeatures,
)
from vimd_amc.models.common import ModelConfig
from vimd_amc.models.oracle_probes import PhysicalTeacherRouteProbe


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "run_learnability_controls.py"
CACHE = ROOT / "standards" / "cache_smoke"


class _SentinelDataset(Dataset):
    def __init__(self) -> None:
        self.items = []
        for index in range(3):
            view = {
                "x": torch.full((2, 32), float(index + 10)),
                "clean": torch.full((2, 32), float(index + 1)),
                "snr_db": torch.tensor(float(index)),
                "condition_seed": torch.tensor(100 + index),
            }
            self.items.append(
                {
                    "view1": {name: value.clone() for name, value in view.items()},
                    "view2": {name: value.clone() for name, value in view.items()},
                    "label": torch.tensor(index),
                    "source_id": torch.tensor(900 + index),
                }
            )
        self._manifest = {"digest": "sentinel", "size": len(self.items)}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.items[index]

    def source_ids(self) -> list[int]:
        return [900, 901, 902]

    def manifest(self) -> dict[str, object]:
        return self._manifest


class CleanOracleDatasetTest(unittest.TestCase):
    def test_only_x_changes_and_source_order_metadata_are_preserved(self) -> None:
        base = _SentinelDataset()
        oracle = CleanOracleInputDataset(base)
        self.assertTrue(oracle.oracle_clean_input)
        self.assertFalse(oracle.deployment_eligible)
        self.assertEqual(oracle.source_ids(), base.source_ids())
        self.assertIs(oracle.manifest(), base.manifest())
        self.assertEqual(len(oracle), len(base))
        for index in range(len(base)):
            original = base[index]
            controlled = oracle[index]
            self.assertEqual(controlled["label"].item(), original["label"].item())
            self.assertEqual(
                controlled["source_id"].item(),
                original["source_id"].item(),
            )
            for view_name in ("view1", "view2"):
                original_view = original[view_name]
                controlled_view = controlled[view_name]
                self.assertEqual(set(controlled_view), set(original_view))
                self.assertTrue(
                    torch.equal(controlled_view["x"], original_view["clean"])
                )
                self.assertTrue(
                    torch.equal(
                        controlled_view["condition_seed"],
                        original_view["condition_seed"],
                    )
                )
                controlled_view["x"][0, 0] = -999.0
                self.assertNotEqual(
                    controlled_view["clean"][0, 0].item(),
                    -999.0,
                )


class ClassicalFeatureTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.values = torch.randn(5, 2, 128)

    def test_shape_finiteness_and_zero_parameter_extractor(self) -> None:
        extractor = ClassicalHOCyclostationaryFeatures()
        features = extractor(self.values)
        self.assertEqual(features.shape, (5, extractor.output_dim))
        self.assertEqual(features.shape[1], len(extractor.feature_names))
        self.assertTrue(torch.isfinite(features).all())
        self.assertEqual(sum(p.numel() for p in extractor.parameters()), 0)
        zeros = extractor(torch.zeros_like(self.values))
        self.assertTrue(torch.isfinite(zeros).all())

    def test_positive_scale_and_global_phase_invariance(self) -> None:
        extractor = ClassicalHOCyclostationaryFeatures()
        reference = extractor(self.values)
        scaled = extractor(3.75 * self.values)
        theta = 0.73
        i, q = self.values[:, 0], self.values[:, 1]
        rotated = torch.stack(
            (
                np.cos(theta) * i - np.sin(theta) * q,
                np.sin(theta) * i + np.cos(theta) * q,
            ),
            dim=1,
        )
        phase_shifted = extractor(rotated)
        torch.testing.assert_close(reference, scaled, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(
            reference,
            phase_shifted,
            atol=2e-5,
            rtol=2e-5,
        )

    def test_classifier_gradients_and_fixed_parameter_budget(self) -> None:
        model = ClassicalHOCyclostationaryClassifier(
            10,
            hidden_dim=32,
            dropout=0.0,
        )
        values = self.values.clone().requires_grad_(True)
        output = model(values)
        self.assertEqual(output["logits"].shape, (5, 10))
        self.assertEqual(
            model.trainable_parameter_count,
            model.extractor.output_dim * 32 + 32 + 32 * 10 + 10,
        )
        output["logits"].square().mean().backward()
        self.assertIsNotNone(values.grad)
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertTrue(
            all(
                parameter.grad is not None
                for parameter in model.classifier.parameters()
            )
        )


class LearnabilityRunnerTest(unittest.TestCase):
    def test_one_epoch_cache_smoke_emits_labeled_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vimd_control_test_") as temporary:
            command = [
                sys.executable,
                str(RUNNER),
                "--cache-root",
                str(CACHE),
                "--output",
                temporary,
                "--run-id",
                "smoke",
                "--epochs",
                "1",
                "--batch-size",
                "4",
                "--threads",
                "1",
                "--seed",
                "13",
            ]
            completed = subprocess.run(
                command,
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            output = Path(completed.stdout.strip())
            self.assertTrue((output / "run.json").is_file())
            self.assertTrue((output / "metrics.csv").is_file())
            self.assertTrue((output / "checksums.json").is_file())
            payload = json.loads(
                (output / "run.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                payload["evidence_designation"],
                "diagnostic_upper_control_only",
            )
            self.assertFalse(payload["headline_evidence_eligible"])
            self.assertFalse(payload["paper_performance_claim_allowed"])
            self.assertTrue(
                payload["clean_oracle_provenance"]["oracle_clean_input"]
            )
            self.assertFalse(
                payload["clean_oracle_provenance"]["deployment_eligible"]
            )
            self.assertTrue(
                payload["source_dependency_hashes"]["unchanged"]
            )
            self.assertEqual(len(payload["results"]), 3)


class PhysicalTeacherRouteProbeTest(unittest.TestCase):
    def test_routes_are_finite_aligned_and_inverse_audit_is_explicit(self) -> None:
        torch.manual_seed(19)
        config = ModelConfig(n_fft=32, hop_length=8)
        probe = PhysicalTeacherRouteProbe(config)
        clean = torch.randn(3, 2, 128)
        jammer = 0.5 * torch.randn(3, 2, 128)
        unexplained = 0.1 * torch.randn(3, 2, 128)
        mixture = clean + jammer + unexplained
        result = probe(mixture, clean, jammer, unexplained)
        self.assertEqual(result["ms_only"].shape, mixture.shape)
        self.assertEqual(result["ms_plus_half_mo"].shape, mixture.shape)
        self.assertTrue(torch.isfinite(result["ms_only"]).all())
        self.assertTrue(torch.isfinite(result["ms_plus_half_mo"]).all())
        torch.testing.assert_close(
            result["masks"].sum(dim=1),
            torch.ones_like(result["masks"][:, 0]),
            atol=1e-6,
            rtol=1e-6,
        )
        self.assertFalse(bool(result["covered_samples"][0]))
        self.assertTrue(bool(result["covered_samples"][1:].all()))
        round_trip, covered = probe.inverse_for_feature_probe(
            probe.front_end(mixture),
            length=mixture.shape[-1],
        )
        torch.testing.assert_close(
            round_trip[:, :, covered],
            mixture[:, :, covered],
            atol=2e-4,
            rtol=2e-4,
        )
        metadata = probe.control_metadata()
        self.assertTrue(metadata["oracle_component_access"])
        self.assertFalse(metadata["deployment_eligible"])
        self.assertFalse(metadata["waveform_reconstruction_claimed"])


if __name__ == "__main__":
    unittest.main()
