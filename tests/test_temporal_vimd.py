from __future__ import annotations

import inspect
import sys
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.run_standard_experiment import available_model_factories  # noqa: E402
from vimd_amc.ablation import PAPER_ABLATION_PROTOCOLS  # noqa: E402
from vimd_amc.models.common import ModelConfig  # noqa: E402
from vimd_amc.models.temporal_vimd import (  # noqa: E402
    DescriptorAssistedVIMDTemporalNet,
    VIMDTemporalCurriculumNet,
    VIMDTemporalNet,
)


class TemporalVIMDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        cls.config = ModelConfig(
            spectral_channels=16,
            environment_dim=16,
            embedding_dim=48,
            n_fft=32,
            hop_length=8,
            dropout=0.0,
        )

    def test_shapes_masks_and_phase_temporal_embedding(self) -> None:
        model = VIMDTemporalNet(5, 7, self.config)
        output = model(torch.randn(3, 2, 128))
        self.assertEqual(tuple(output["logits"].shape), (3, 5))
        self.assertEqual(tuple(output["embedding"].shape), (3, 48))
        self.assertEqual(tuple(output["temporal_embedding"].shape), (3, 48))
        self.assertEqual(tuple(output["masks"].shape), (3, 3, 32, 13))
        torch.testing.assert_close(
            output["masks"].sum(dim=1),
            torch.ones_like(output["masks"][:, 0]),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_gradient_reaches_temporal_and_mask_paths(self) -> None:
        model = VIMDTemporalNet(5, 7, self.config)
        output = model(torch.randn(4, 2, 128))
        loss = output["logits"].square().mean()
        loss.backward()
        temporal_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.modulation_temporal_branch.parameters()
            if parameter.grad is not None
        )
        mask_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.tri_mask.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(temporal_gradient, 0.0)
        self.assertGreater(mask_gradient, 0.0)

    def test_inference_contract_has_no_teacher_or_component_arguments(self) -> None:
        signature = inspect.signature(VIMDTemporalNet.forward)
        self.assertEqual(tuple(signature.parameters), ("self", "values"))
        model = VIMDTemporalNet(5, 7, self.config)
        self.assertFalse(model.requires_teacher_at_inference)
        self.assertFalse(model.supports_external_routing)
        output = model.eval()(torch.randn(2, 2, 128))
        self.assertTrue(torch.isfinite(output["logits"]).all())

    def test_registration_is_diagnostic_and_does_not_change_a0_a7(self) -> None:
        before = tuple(PAPER_ABLATION_PROTOCOLS)
        factories = available_model_factories()
        self.assertIn("diagnostic_vimd_v2_temporal", factories)
        self.assertNotIn("diagnostic_vimd_v2_temporal", PAPER_ABLATION_PROTOCOLS)
        self.assertEqual(before, tuple(PAPER_ABLATION_PROTOCOLS))
        built = factories["diagnostic_vimd_v2_temporal"](5, 7, self.config)
        self.assertIsInstance(built.model, VIMDTemporalNet)
        self.assertTrue(built.objective.use_mask_supervision)
        self.assertIsNotNone(built.teacher)

    def test_parameter_count_is_stable_for_audited_small_configuration(self) -> None:
        model = VIMDTemporalNet(5, 7, self.config)
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            67_941,
        )

    def test_curriculum_routing_is_temporary_and_state_dict_is_teacher_free(self) -> None:
        model = VIMDTemporalCurriculumNet(5, 7, self.config)
        values = torch.randn(2, 2, 128)
        model.train()
        student = model(values)
        teacher_masks = torch.zeros_like(student["masks"])
        teacher_masks[:, 0] = 1.0
        with model.teacher_route_context(teacher_masks, 1.0):
            routed = model(values)
            torch.testing.assert_close(
                routed["routing_masks"],
                teacher_masks,
                atol=1e-6,
                rtol=1e-6,
            )
        restored = model(values)
        torch.testing.assert_close(
            restored["routing_masks"],
            restored["masks"],
            atol=1e-6,
            rtol=1e-6,
        )
        self.assertIsNone(model._teacher_route_masks)
        self.assertFalse(
            any("teacher_route" in name for name in model.state_dict())
        )

    def test_curriculum_checkpoint_loads_for_teacher_free_inference(self) -> None:
        first = VIMDTemporalCurriculumNet(5, 7, self.config).eval()
        second = VIMDTemporalCurriculumNet(5, 7, self.config).eval()
        second.load_state_dict(first.state_dict(), strict=True)
        values = torch.randn(2, 2, 128)
        with torch.no_grad():
            first_output = first(values)
            second_output = second(values)
        torch.testing.assert_close(first_output["logits"], second_output["logits"])
        self.assertEqual(
            tuple(inspect.signature(VIMDTemporalCurriculumNet.forward).parameters),
            ("self", "values"),
        )

    def test_eval_ignores_accidentally_active_training_route_context(self) -> None:
        model = VIMDTemporalCurriculumNet(5, 7, self.config).eval()
        values = torch.randn(2, 2, 128)
        baseline = model(values)
        teacher_masks = torch.zeros_like(baseline["masks"])
        teacher_masks[:, 2] = 1.0
        with model.teacher_route_context(teacher_masks, 1.0):
            actual = model(values)
        torch.testing.assert_close(
            actual["routing_masks"],
            actual["masks"],
            atol=1e-6,
            rtol=1e-6,
        )

    def test_descriptor_assisted_candidate_is_fixed_and_mixture_only(self) -> None:
        model = DescriptorAssistedVIMDTemporalNet(5, 7, self.config)
        values = torch.randn(3, 2, 128)
        output = model(values)
        self.assertEqual(tuple(output["logits"].shape), (3, 5))
        self.assertEqual(tuple(output["descriptor_features"].shape), (3, 61))
        self.assertEqual(tuple(output["descriptor_embedding"].shape), (3, 48))
        self.assertEqual(
            sum(
                parameter.numel()
                for parameter in model.fixed_descriptor.parameters()
            ),
            0,
        )
        self.assertEqual(
            tuple(
                inspect.signature(
                    DescriptorAssistedVIMDTemporalNet.forward
                ).parameters
            ),
            ("self", "values"),
        )
        loss = output["logits"].square().mean()
        loss.backward()
        descriptor_head_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.descriptor_projector.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(descriptor_head_gradient, 0.0)


if __name__ == "__main__":
    unittest.main()
