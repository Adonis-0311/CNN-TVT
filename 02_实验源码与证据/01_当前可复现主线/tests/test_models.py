from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vimd_amc.losses import (  # noqa: E402
    VIMDLossWeights,
    masked_jammer_binary_cross_entropy,
    orthogonality_loss,
    paired_cross_condition_contrastive_loss,
    vimd_two_view_loss,
)
from vimd_amc.models.common import ModelConfig  # noqa: E402
from vimd_amc.models.vimd import PhysicalTriMaskTeacher, VIMDNet  # noqa: E402


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(4)
        self.model = VIMDNet(
            num_classes=4,
            num_jammers=9,
            config=ModelConfig(
                feature_channels=32,
                embedding_dim=48,
                spectral_channels=16,
                n_fft=32,
                hop_length=8,
            ),
        )

    def test_tri_mask_is_normalized(self) -> None:
        output = self.model(torch.randn(3, 2, 128))
        self.assertEqual(tuple(output["masks"].shape[:2]), (3, 3))
        torch.testing.assert_close(
            output["masks"].sum(dim=1),
            torch.ones_like(output["masks"][:, 0]),
            atol=1e-6,
            rtol=1e-6,
        )
        self.assertTrue(torch.all(output["rho"] >= self.model.config.rho_min))
        self.assertTrue(torch.all(output["rho"] <= self.model.config.rho_max))

    def test_teacher_mask_is_normalized_and_handles_no_jammer(self) -> None:
        teacher = PhysicalTriMaskTeacher(self.model.config)
        clean = torch.randn(2, 2, 128)
        jammer = torch.zeros_like(clean)
        target = teacher(clean, jammer, torch.zeros_like(clean))
        torch.testing.assert_close(
            target.sum(dim=1),
            torch.ones_like(target[:, 0]),
            atol=1e-6,
            rtol=1e-6,
        )
        self.assertGreater(float(target[:, 0].mean()), 0.999)

    def test_teacher_is_scale_aware_and_swap_symmetric(self) -> None:
        teacher = PhysicalTriMaskTeacher(self.model.config)
        clean = torch.randn(2, 2, 128)
        jammer = torch.randn_like(clean)
        unexplained = 0.05 * torch.randn_like(clean)
        weak = teacher(clean, 0.1 * jammer, unexplained)
        strong = teacher(clean, 10.0 * jammer, unexplained)
        self.assertGreater(float(weak[:, 0].mean()), float(strong[:, 0].mean()))
        self.assertLess(float(weak[:, 1].mean()), float(strong[:, 1].mean()))
        original = teacher(clean, jammer, unexplained)
        swapped = teacher(jammer, clean, unexplained)
        torch.testing.assert_close(original[:, 0], swapped[:, 1], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(original[:, 2], swapped[:, 2], atol=1e-6, rtol=1e-6)
        for scale in (1e-4, 1e4):
            scaled = teacher(scale * clean, scale * jammer, scale * unexplained)
            torch.testing.assert_close(original, scaled, atol=1e-5, rtol=1e-5)

    def test_teacher_assigns_unexplained_energy_to_overlap_route(self) -> None:
        teacher = PhysicalTriMaskTeacher(self.model.config)
        zero = torch.zeros(2, 2, 128)
        noise = torch.randn_like(zero)
        target = teacher(zero, zero, noise)
        self.assertLess(float(target[:, :2].abs().max()), 1e-6)
        self.assertGreater(float(target[:, 2].mean()), 0.999)

    def test_full_loss_is_finite(self) -> None:
        batch_size = 4
        first_x = torch.randn(batch_size, 2, 128)
        second_x = torch.randn(batch_size, 2, 128)
        first_output = self.model(first_x)
        second_output = self.model(second_x)
        teacher = PhysicalTriMaskTeacher(self.model.config)
        first_batch = {
            "jam_labels": torch.zeros(batch_size, 9),
            "quality": torch.zeros(batch_size, 3),
            "quality_mask": torch.ones(batch_size, 3),
        }
        second_batch = {
            "jam_labels": torch.zeros(batch_size, 9),
            "quality": torch.zeros(batch_size, 3),
            "quality_mask": torch.ones(batch_size, 3),
        }
        losses = vimd_two_view_loss(
            first_output=first_output,
            second_output=second_output,
            first_batch=first_batch,
            second_batch=second_batch,
            labels=torch.tensor([0, 1, 2, 3]),
            first_teacher_mask=teacher(
                first_x,
                torch.zeros_like(first_x),
                torch.zeros_like(first_x),
            ),
            second_teacher_mask=teacher(
                second_x,
                torch.zeros_like(second_x),
                torch.zeros_like(second_x),
            ),
            weights=VIMDLossWeights(),
            enable_mask=True,
            enable_contrastive=True,
        )
        self.assertTrue(torch.isfinite(losses["total"]))

    def test_paired_contrastive_is_finite_for_single_class(self) -> None:
        first = torch.randn(5, 12, requires_grad=True)
        second = torch.randn(5, 12, requires_grad=True)
        loss = paired_cross_condition_contrastive_loss(
            first,
            second,
            torch.zeros(5, dtype=torch.long),
            0.1,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(loss.detach()), 0.0)

    def test_jammer_support_mask_removes_unsupported_bce_gradient(self) -> None:
        logits = torch.tensor(
            [[0.2, -8.0, 7.0], [-0.4, 9.0, -6.0]],
            requires_grad=True,
        )
        targets = torch.tensor(
            [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )
        support = torch.tensor([1.0, 0.0, 0.0])
        loss = masked_jammer_binary_cross_entropy(
            logits,
            targets,
            support,
        )
        changed_unsupported = logits.detach().clone()
        changed_unsupported[:, 1:] += 100.0
        changed_loss = masked_jammer_binary_cross_entropy(
            changed_unsupported,
            targets,
            support,
        )
        torch.testing.assert_close(loss.detach(), changed_loss)
        loss.backward()
        self.assertGreater(float(logits.grad[:, 0].abs().sum()), 0.0)
        torch.testing.assert_close(
            logits.grad[:, 1:],
            torch.zeros_like(logits.grad[:, 1:]),
            atol=0.0,
            rtol=0.0,
        )

    def test_jammer_support_mask_rejects_no_supervised_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            masked_jammer_binary_cross_entropy(
                torch.zeros(2, 3),
                torch.zeros(2, 3),
                torch.zeros(3),
            )

    def test_orthogonality_has_dimension_stable_scale(self) -> None:
        base = torch.randn(32, 16)
        identical = orthogonality_loss(base, base)
        orthogonal = orthogonality_loss(
            torch.cat((base[:, :8], torch.zeros_like(base[:, 8:])), dim=1),
            torch.cat((torch.zeros_like(base[:, :8]), base[:, 8:]), dim=1),
        )
        self.assertGreater(float(identical), 0.9)
        self.assertLess(float(orthogonal), 1e-6)


if __name__ == "__main__":
    unittest.main()
