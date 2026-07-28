from __future__ import annotations

import json
import sys
from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vimd_amc.data.dataset import PairedAMCDataset, Regime  # noqa: E402
from vimd_amc.data.synthesis import SynthesisConfig, SignalSynthesizer  # noqa: E402
from vimd_amc.evaluation import (  # noqa: E402
    auxiliary_task_metrics,
    mechanism_metrics,
)
from vimd_amc.losses import jensen_shannon_mask_loss  # noqa: E402
from vimd_amc.models.common import ModelConfig  # noqa: E402
from vimd_amc.models.vimd import PhysicalTriMaskTeacher, VIMDNet  # noqa: E402


class _AuxiliaryDataset(Dataset):
    split = "heldout_channel"

    def __init__(self, *, invalid_quality_mask: bool = False):
        self.invalid_quality_mask = invalid_quality_mask
        self.jammer_labels = torch.tensor(
            [
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        )
        self.quality = torch.tensor(
            [
                [0.50, -0.50, 0.25],
                [0.25, -0.25, 0.50],
                [0.00, 0.00, 0.75],
                [-0.25, 0.00, 1.00],
            ]
        )

    def __len__(self) -> int:
        return len(self.jammer_labels)

    def manifest(self) -> dict[str, object]:
        return {
            "quality_normalization": {
                "snr_db": {"scale": 20.0, "unit": "dB"},
                "sir_db": {"scale": 20.0, "unit": "dB"},
                "doppler_hz": {"scale": 100.0, "unit": "Hz"},
            }
        }

    def __getitem__(self, index: int) -> dict[str, object]:
        marker = torch.zeros(2, 16)
        marker[0, 0] = float(index)
        quality_mask = torch.ones(3)
        if index == 3:
            quality_mask[1] = 0.0
        if self.invalid_quality_mask and index == 0:
            quality_mask[1] = 0.25
        sir_db = (
            self.quality[index, 1] * 20.0
            if quality_mask[1] > 0.5
            else torch.tensor(float("inf"))
        )
        view = {
            "x": marker,
            "jam_labels": self.jammer_labels[index],
            "quality": self.quality[index],
            "quality_mask": quality_mask,
            "snr_db": self.quality[index, 0] * 20.0,
            "sir_db": sir_db,
            "doppler_hz": self.quality[index, 2] * 100.0,
        }
        return {
            "view1": view,
            "view2": view,
            "label": torch.tensor(0, dtype=torch.long),
            "source_id": torch.tensor(index, dtype=torch.long),
        }


class _AuxiliaryModel(nn.Module):
    def __init__(self, dataset: _AuxiliaryDataset):
        super().__init__()
        probabilities = torch.tensor(
            [
                [0.90, 0.10, 0.80],
                [0.80, 0.10, 0.20],
                [0.20, 0.10, 0.90],
                [0.10, 0.10, 0.10],
            ]
        )
        self.register_buffer(
            "jammer_logits",
            torch.logit(probabilities),
        )
        normalized_error = torch.tensor([0.10, 0.20, 0.30])
        self.register_buffer(
            "quality_predictions",
            dataset.quality + normalized_error,
        )

    def forward(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        indices = values[:, 0, 0].round().long()
        return {
            "logits": torch.zeros(len(indices), 1, device=values.device),
            "jam_logits": self.jammer_logits[indices],
            "quality": self.quality_predictions[indices],
        }


class AuxiliaryAndMechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_js_is_fp32_finite_and_differentiable_under_cpu_amp(self) -> None:
        target = torch.tensor(
            [
                [[[1.0]], [[0.0]], [[0.0]]],
                [[[0.0]], [[0.0]], [[0.0]]],
            ],
            dtype=torch.float16,
        )
        predicted = torch.tensor(
            [
                [[[0.0]], [[1.0]], [[0.0]]],
                [[[1.0e-7]], [[1.0]], [[1.0e-7]]],
            ],
            dtype=torch.float16,
            requires_grad=True,
        )
        with torch.amp.autocast(
            device_type="cpu",
            dtype=torch.bfloat16,
            enabled=True,
        ):
            loss = jensen_shannon_mask_loss(target, predicted)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertGreaterEqual(float(loss.detach()), -1e-6)
        self.assertLessEqual(
            float(loss.detach()),
            float(np.log(2.0)) + 1e-5,
        )
        loss.backward()
        self.assertIsNotNone(predicted.grad)
        self.assertTrue(bool(torch.isfinite(predicted.grad).all()))
        self.assertGreater(float(predicted.grad.abs().sum()), 0.0)

    def test_auxiliary_metrics_are_physical_json_and_support_aware(self) -> None:
        dataset = _AuxiliaryDataset()
        result = auxiliary_task_metrics(
            _AuxiliaryModel(dataset),
            dataset,
            device=torch.device("cpu"),
            batch_size=2,
            seed=23,
            jammer_names=("tone", "chirp", "pulse"),
        )
        self.assertEqual(result["split"], "heldout_channel")
        self.assertEqual(result["seed"], 23)
        jammer = result["jammer_multilabel"]
        self.assertAlmostEqual(jammer["micro_f1"]["value"], 1.0)
        self.assertAlmostEqual(jammer["macro_f1"]["value"], 1.0)
        self.assertAlmostEqual(jammer["micro_auroc"]["value"], 1.0)
        self.assertEqual(
            jammer["per_class"]["chirp"]["auroc"]["status"],
            "unavailable",
        )
        quality = result["quality"]
        self.assertEqual(quality["quality_mask_validity"]["status"], "valid")
        self.assertAlmostEqual(
            quality["physical_mae"]["snr_db"]["value"],
            2.0,
            places=5,
        )
        self.assertAlmostEqual(
            quality["physical_mae"]["sir_db"]["value"],
            4.0,
            places=5,
        )
        self.assertAlmostEqual(
            quality["physical_mae"]["doppler_hz"]["value"],
            30.0,
            places=5,
        )
        for component in ("snr_db", "sir_db", "doppler_hz"):
            self.assertAlmostEqual(
                quality["normalization_consistency"][component][
                    "max_absolute_discrepancy"
                ],
                0.0,
                places=6,
            )
        # Strict JSON encoding rejects NaN/Infinity and non-native scalars.
        json.dumps(result, allow_nan=False)

    def test_unsupported_jammer_family_is_not_scored_as_trained(self) -> None:
        dataset = _AuxiliaryDataset()
        result = auxiliary_task_metrics(
            _AuxiliaryModel(dataset),
            dataset,
            device=torch.device("cpu"),
            batch_size=2,
            seed=25,
            jammer_names=("tone", "chirp", "pulse"),
            jammer_training_support_mask=(True, True, False),
            jammer_training_support_source="unit-test train split",
        )
        support = result["jammer_training_support"]
        self.assertEqual(support["mask"], [True, True, False])
        self.assertEqual(support["supported_names"], ["tone", "chirp"])
        self.assertEqual(support["unsupported_names"], ["pulse"])
        self.assertFalse(
            support[
                "held_or_excluded_logits_interpreted_as_trained_family_recognition"
            ]
        )
        jammer = result["jammer_multilabel"]
        self.assertEqual(jammer["training_supported_class_count"], 2)
        pulse = jammer["per_class"]["pulse"]
        self.assertFalse(pulse["training_supported"])
        self.assertEqual(pulse["f1"]["status"], "unavailable")
        self.assertEqual(pulse["auroc"]["status"], "unavailable")
        self.assertIn(
            "no positive training support",
            pulse["f1"]["reason"],
        )
        self.assertEqual(
            jammer["metric_scope"],
            "training-supported taxonomy columns only",
        )
        json.dumps(result, allow_nan=False)

    def test_missing_scale_and_invalid_quality_mask_are_explicit(self) -> None:
        dataset = _AuxiliaryDataset(invalid_quality_mask=True)
        result = auxiliary_task_metrics(
            _AuxiliaryModel(dataset),
            dataset,
            device=torch.device("cpu"),
            batch_size=4,
            seed=24,
            dataset_manifest={},
        )
        self.assertEqual(
            result["quality"]["quality_mask_validity"]["status"],
            "invalid",
        )
        self.assertEqual(
            result["quality"]["physical_mae"]["snr_db"]["status"],
            "unavailable",
        )
        self.assertEqual(
            result["quality_denormalization"]["status"],
            "unavailable",
        )
        json.dumps(result, allow_nan=False)

    def test_mechanism_reports_transfer_amplification_and_constituents(self) -> None:
        synthesizer = SignalSynthesizer(
            SynthesisConfig(sample_length=128)
        )
        dataset = PairedAMCDataset(
            synthesizer=synthesizer,
            split="hard",
            size=6,
            regime=Regime.hard_interference(),
            master_seed=91,
            modulations=("BPSK", "QPSK"),
            cache_in_memory=True,
        )
        normalization = dataset.manifest()["quality_normalization"]
        self.assertEqual(normalization["snr_db"]["scale"], 20.0)
        self.assertEqual(normalization["sir_db"]["scale"], 20.0)
        self.assertGreater(normalization["doppler_hz"]["scale"], 0.0)
        config = ModelConfig(
            feature_channels=16,
            embedding_dim=24,
            environment_dim=12,
            spectral_channels=8,
            n_fft=32,
            hop_length=8,
            dropout=0.0,
        )
        model = VIMDNet(2, 9, config)
        auxiliary = auxiliary_task_metrics(
            model,
            dataset,
            device=torch.device("cpu"),
            batch_size=3,
            seed=29,
        )
        for component in ("snr_db", "sir_db", "doppler_hz"):
            self.assertEqual(
                auxiliary["quality"]["physical_mae"][component]["status"],
                "available",
            )
            self.assertLess(
                auxiliary["quality"]["normalization_consistency"][component][
                    "max_absolute_discrepancy"
                ],
                1e-3,
            )
        result = mechanism_metrics(
            model,
            PhysicalTriMaskTeacher(config),
            dataset,
            device=torch.device("cpu"),
            batch_size=3,
            maximum_samples=6,
            seed=29,
            snr_strata_edges_db=(-2.0,),
            sir_strata_edges_db=(-7.5,),
        )
        self.assertEqual(
            result["signal_retention"],
            result["target_energy_transfer_ratio_mean"],
        )
        self.assertIn("signal_retention", result["deprecated_metric_aliases"])
        self.assertGreaterEqual(
            result["target_energy_transfer_ratio_max"],
            result["target_energy_transfer_ratio_mean"],
        )
        self.assertGreaterEqual(
            result["target_energy_transfer_ratio_amplification_share"],
            0.0,
        )
        self.assertLessEqual(
            result["target_energy_transfer_ratio_amplification_share"],
            1.0,
        )
        constituents = result["overlap_teacher_constituents"]
        self.assertEqual(
            set(constituents),
            {"unexplained_fraction", "signal_jammer_ambiguity"},
        )
        for record in constituents.values():
            self.assertIn("oracle_occupancy", record)
            self.assertIn(
                "predicted_overlap_direct_weighted_correlation",
                record,
            )
            self.assertIn("predicted_overlap_direct_weighted_mae", record)
        self.assertEqual(
            result["stratified_mechanism"]["snr_db"]["status"],
            "available",
        )
        self.assertEqual(
            result["stratified_mechanism"]["sir_db"]["status"],
            "available",
        )


if __name__ == "__main__":
    unittest.main()
