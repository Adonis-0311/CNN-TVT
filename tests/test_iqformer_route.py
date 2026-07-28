from __future__ import annotations

import inspect
import sys
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from vimd_amc.ablation import PAPER_ABLATION_PROTOCOLS  # noqa: E402
from experiments.run_standard_experiment import available_model_factories  # noqa: E402
from vimd_amc.models.baselines import IQFormerInspiredClassifier  # noqa: E402
from vimd_amc.models.common import ModelConfig  # noqa: E402
from vimd_amc.models.iqformer_route import (  # noqa: E402
    ComplexSTFTOverlapAdd,
    IQFormerRawOnlyControl,
    VIMDIQFormerRouteNet,
)
from vimd_amc.models.spectral import ComplexSTFT  # noqa: E402


class IQFormerRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        cls.config = ModelConfig(
            spectral_channels=16,
            environment_dim=16,
            embedding_dim=48,
            n_fft=64,
            hop_length=16,
            dropout=0.0,
        )

    def test_overlap_add_length_phase_and_edge_audit(self) -> None:
        sample_count = 256
        positions = torch.arange(sample_count, dtype=torch.float32)
        phase = 0.37 + 2.0 * torch.pi * 0.071 * positions
        values = torch.stack((phase.cos(), phase.sin()))[None]
        spectrum = ComplexSTFT(64, 16)(values)
        result = ComplexSTFTOverlapAdd(64, 16)(
            spectrum,
            output_length=sample_count,
        )
        self.assertEqual(tuple(result["iq"].shape), (1, 2, sample_count))
        coverage = result["coverage"]
        self.assertGreater(int(coverage.sum()), 245)
        self.assertGreater(int(result["unrecoverable_sample_count"]), 0)
        error = (result["iq"][..., coverage] - values[..., coverage]).abs()
        self.assertLess(float(error.max()), 2e-5)
        reference = torch.complex(
            values[:, 0, coverage],
            values[:, 1, coverage],
        )
        reconstructed = torch.complex(
            result["iq"][:, 0, coverage],
            result["iq"][:, 1, coverage],
        )
        phase_error = torch.angle(reconstructed * reference.conj()).abs()
        self.assertLess(float(phase_error.max()), 2e-5)

    def test_overlap_add_reports_uncovered_tail_and_has_mask_gradient(self) -> None:
        values = torch.randn(2, 2, 250)
        spectrum = ComplexSTFT(64, 16)(values)
        mask_logits = torch.randn_like(spectrum.real, requires_grad=True)
        mask = torch.sigmoid(mask_logits)
        result = ComplexSTFTOverlapAdd(64, 16)(
            mask * spectrum,
            output_length=250,
        )
        self.assertEqual(result["iq"].shape[-1], 250)
        self.assertGreater(int(result["unrecoverable_sample_count"]), 1)
        result["iq"].square().mean().backward()
        self.assertIsNotNone(mask_logits.grad)
        self.assertGreater(float(mask_logits.grad.abs().sum()), 0.0)

    def test_candidate_uses_one_shared_encoder_twice(self) -> None:
        model = VIMDIQFormerRouteNet(10, 9, self.config).eval()
        encoders = [
            module
            for module in model.modules()
            if isinstance(module, IQFormerInspiredClassifier)
        ]
        self.assertEqual(len(encoders), 1)
        calls = []
        handle = model.shared_encoder.iq_stem.register_forward_hook(
            lambda *_: calls.append(1)
        )
        try:
            output = model(torch.randn(2, 2, 256))
        finally:
            handle.remove()
        self.assertEqual(len(calls), 2)
        self.assertEqual(tuple(output["logits"].shape), (2, 10))
        self.assertTrue(torch.all(output["route_gate"] >= 0.10))
        self.assertTrue(torch.all(output["route_gate"] <= 0.90))

    def test_route_gradient_reaches_student_mask(self) -> None:
        model = VIMDIQFormerRouteNet(10, 9, self.config)
        output = model(torch.randn(3, 2, 256))
        output["logits"].square().mean().backward()
        gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.tri_mask.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient, 0.0)

    def test_teacher_free_strict_reload_and_raw_control_encoder_match(self) -> None:
        candidate = VIMDIQFormerRouteNet(10, 9, self.config).eval()
        reloaded = VIMDIQFormerRouteNet(10, 9, self.config).eval()
        reloaded.load_state_dict(candidate.state_dict(), strict=True)
        values = torch.randn(2, 2, 256)
        with torch.no_grad():
            first = candidate(values)["logits"]
            second = reloaded(values)["logits"]
        torch.testing.assert_close(first, second)
        self.assertFalse(any("teacher" in key for key in candidate.state_dict()))
        self.assertEqual(
            tuple(inspect.signature(VIMDIQFormerRouteNet.forward).parameters),
            ("self", "values"),
        )
        raw = IQFormerRawOnlyControl(10)
        self.assertEqual(
            sum(
                parameter.numel()
                for parameter in raw.shared_encoder.parameters()
            ),
            sum(
                parameter.numel()
                for parameter in candidate.shared_encoder.parameters()
            ),
        )

    def test_a0_a7_registry_is_untouched(self) -> None:
        self.assertEqual(len(PAPER_ABLATION_PROTOCOLS), 8)
        self.assertNotIn("vimd_iqformer_route", PAPER_ABLATION_PROTOCOLS)
        self.assertNotIn("iqformer_raw_only_control", PAPER_ABLATION_PROTOCOLS)
        factories = available_model_factories()
        self.assertIn("diagnostic_iqformer_raw_only", factories)
        self.assertIn("diagnostic_vimd_v3_iqformer_route", factories)


if __name__ == "__main__":
    unittest.main()
