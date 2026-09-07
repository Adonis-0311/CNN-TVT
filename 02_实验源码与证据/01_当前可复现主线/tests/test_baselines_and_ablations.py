from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np
from scipy.signal import stft as scipy_stft
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.run_experiment import build_model  # noqa: E402
from experiments.run_standard_experiment import available_model_factories  # noqa: E402
from vimd_amc.ablation import PAPER_ABLATION_PROTOCOLS  # noqa: E402
from vimd_amc.evaluation import _operation_counter, complexity_metrics  # noqa: E402
from vimd_amc.losses import VIMDLossWeights, vimd_two_view_loss  # noqa: E402
from vimd_amc.models.baselines import (  # noqa: E402
    DiagnosticDualMaskCEClassifier,
    IQFormerInspiredClassifier,
    MCLDNNReimplementation,
    _IQFormerSTFT,
)
from vimd_amc.models.common import ModelConfig  # noqa: E402
from vimd_amc.models.vimd import PhysicalTriMaskTeacher, VIMDNet  # noqa: E402
from vimd_amc.models.vimd import (  # noqa: E402
    DualMaskVIMDNet,
    PhysicalDualMaskTeacher,
)
from vimd_amc.reproducibility import source_tree_record  # noqa: E402
from vimd_amc.training import (  # noqa: E402
    _physical_teacher_target,
    _staged_cosine_learning_rate_factor,
)


class StrongBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_mcldnn_audited_topology_and_batch_shape(self) -> None:
        model = MCLDNNReimplementation(11).eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 406_199)
        for batch_size in (1, 3):
            output = model(torch.randn(batch_size, 2, 128))
            self.assertEqual(tuple(output["logits"].shape), (batch_size, 11))
            self.assertEqual(tuple(output["embedding"].shape), (batch_size, 128))
        self.assertIn("reimplementation", model.provenance["claim_level"])

    def test_iqformer_inspired_audited_shape_and_label(self) -> None:
        model = IQFormerInspiredClassifier(11).eval()
        # Matches the parameter count of the audited public RML2016
        # configuration for 11 classes, despite the explicitly documented
        # preprocessing/runtime differences.
        self.assertEqual(sum(p.numel() for p in model.parameters()), 355_049)
        for batch_size in (1, 2):
            output = model(torch.randn(batch_size, 2, 128))
            self.assertEqual(tuple(output["logits"].shape), (batch_size, 11))
            self.assertEqual(tuple(output["embedding"].shape), (batch_size, 64))
        self.assertIn("not an exact reproduction", model.provenance["claim_level"])

    def test_internal_iqformer_stft_matches_public_scipy_preprocessing(self) -> None:
        rng = np.random.default_rng(81)
        in_phase = rng.normal(size=128).astype(np.float32)
        _, _, reference = scipy_stft(
            in_phase,
            1.0,
            "blackman",
            31,
            30,
            128,
        )
        values = torch.from_numpy(np.stack((in_phase, in_phase))[None])
        actual = _IQFormerSTFT()(values).squeeze(0).squeeze(0).numpy()
        np.testing.assert_allclose(
            actual,
            reference[:32].real,
            atol=2e-6,
            rtol=2e-5,
        )

    def test_recurrent_and_frontend_complexity_are_reported(self) -> None:
        mcldnn = complexity_metrics(
            MCLDNNReimplementation(4),
            sample_length=128,
            device=torch.device("cpu"),
            latency_runs=1,
        )
        iqformer = complexity_metrics(
            IQFormerInspiredClassifier(4),
            sample_length=128,
            device=torch.device("cpu"),
            latency_runs=1,
        )
        self.assertGreater(mcldnn["recurrent_macs_excluding_stft"], 0.0)
        self.assertGreater(iqformer["recurrent_macs_excluding_stft"], 0.0)
        self.assertGreater(iqformer["stft_estimated_real_operations"], 0.0)

    def test_linear_mac_counter_includes_all_token_positions(self) -> None:
        layer = torch.nn.Linear(3, 2)
        with _operation_counter(layer) as counter:
            layer(torch.randn(1, 5, 3))
        self.assertEqual(counter["convolution_linear_macs"], 1 * 5 * 3 * 2)

    def test_source_tree_fingerprint_is_stable_and_includes_entrypoint(self) -> None:
        first = source_tree_record(ROOT, ROOT / "experiments" / "run_experiment.py")
        second = source_tree_record(ROOT, ROOT / "experiments" / "run_experiment.py")
        self.assertEqual(first, second)
        self.assertEqual(len(first["aggregate_digest"]), 64)
        self.assertIn("experiments/run_experiment.py", first["files"])
        self.assertIn("src/vimd_amc/models/baselines.py", first["files"])


class AblationProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ModelConfig(
            spectral_channels=16,
            environment_dim=16,
            embedding_dim=32,
            n_fft=32,
            hop_length=8,
            dropout=0.0,
        )

    def test_dual_mask_is_normalized(self) -> None:
        model = DiagnosticDualMaskCEClassifier(4, self.config)
        output = model(torch.randn(3, 2, 128))
        self.assertEqual(tuple(output["masks"].shape[:2]), (3, 2))
        torch.testing.assert_close(
            output["masks"].sum(dim=1),
            torch.ones_like(output["masks"][:, 0]),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_manuscript_ablation_table_is_the_single_fixed_mapping(self) -> None:
        observed = [
            (
                protocol["ablation_id"],
                protocol["mask_routes"],
                protocol["teacher"],
                protocol["mtl"],
                protocol["xcc"],
                protocol["residual_path"],
            )
            for protocol in PAPER_ABLATION_PROTOCOLS.values()
        ]
        self.assertEqual(
            observed,
            [
                ("A0", 0, "none", False, False, False),
                ("A1", 1, "none", False, False, False),
                ("A2", 3, "none", False, False, True),
                ("A3", 3, "fixed_physical_tri", False, False, True),
                ("A4", 3, "fixed_physical_tri", True, False, True),
                ("A5", 3, "fixed_physical_tri", True, True, True),
                (
                    "A6",
                    2,
                    "fixed_physical_dual_collapsed_from_tri",
                    True,
                    True,
                    True,
                ),
                ("A7", 3, "fixed_physical_tri", True, True, False),
            ],
        )

    def test_a2_to_a5_and_a7_share_tri_architecture(self) -> None:
        names = (
            "a2_tri_no_teacher",
            "a3_tri_teacher",
            "a4_tri_teacher_mtl",
            "a5_vimd_full",
            "a7_vimd_no_residual",
        )
        built = [
            build_model(name, classes=4, jammers=9, config=self.config)
            for name in names
        ]
        self.assertTrue(all(isinstance(item.model, VIMDNet) for item in built))
        parameter_counts = {
            sum(parameter.numel() for parameter in item.model.parameters())
            for item in built
        }
        self.assertEqual(len(parameter_counts), 1)
        self.assertIsNone(built[0].teacher)
        self.assertFalse(built[0].objective.use_mask_supervision)
        self.assertTrue(built[1].objective.use_mask_supervision)
        self.assertTrue(built[2].objective.use_jammer_auxiliary)
        self.assertFalse(built[2].objective.use_cross_condition_contrastive)
        self.assertTrue(built[3].objective.use_jammer_auxiliary)
        self.assertTrue(built[3].objective.use_cross_condition_contrastive)
        self.assertTrue(built[4].objective.use_jammer_auxiliary)
        self.assertTrue(built[4].objective.use_cross_condition_contrastive)
        self.assertTrue(built[3].model.use_residual)
        self.assertFalse(built[4].model.use_residual)

    def test_a6_is_dual_branch_full_objective_with_dual_teacher(self) -> None:
        built = build_model(
            "a6_dual_full",
            classes=4,
            jammers=9,
            config=self.config,
        )
        self.assertIsInstance(built.model, DualMaskVIMDNet)
        self.assertIsInstance(built.teacher, PhysicalDualMaskTeacher)
        self.assertTrue(built.objective.use_mask_supervision)
        self.assertTrue(built.objective.use_jammer_auxiliary)
        self.assertTrue(built.objective.use_quality_auxiliary)
        self.assertTrue(built.objective.use_cross_condition_contrastive)
        output = built.model(torch.randn(3, 2, 128))
        self.assertEqual(tuple(output["masks"].shape[:2]), (3, 2))
        self.assertIn("jammer_embedding", output)
        self.assertTrue(torch.all(output["rho"] > 0))
        target = built.teacher(
            torch.randn(3, 2, 128),
            torch.randn(3, 2, 128),
            torch.randn(3, 2, 128),
        )
        self.assertEqual(tuple(target.shape[:2]), (3, 2))
        torch.testing.assert_close(
            target.sum(dim=1),
            torch.ones_like(target[:, 0]),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_a7_removes_only_applied_residual_from_full_tri_model(self) -> None:
        with_residual = build_model(
            "a5_vimd_full",
            classes=4,
            jammers=9,
            config=self.config,
        ).model.eval()
        without_residual = build_model(
            "a7_vimd_no_residual",
            classes=4,
            jammers=9,
            config=self.config,
        ).model.eval()
        without_residual.load_state_dict(with_residual.state_dict())
        values = torch.randn(3, 2, 128)
        first = with_residual(values)
        second = without_residual(values)
        torch.testing.assert_close(first["masks"], second["masks"])
        torch.testing.assert_close(
            first["rho_predicted"],
            second["rho_predicted"],
        )
        self.assertTrue(torch.all(first["rho"] > 0))
        self.assertEqual(float(second["rho"].abs().max()), 0.0)
        expected_difference = first["rho"][:, :, None].expand_as(
            first["modulation_weight"]
        )
        torch.testing.assert_close(
            first["modulation_weight"] - second["modulation_weight"],
            expected_difference,
        )

    def test_standard_cache_runner_registers_complete_ablation_ladder(self) -> None:
        factories = available_model_factories()
        names = tuple(f"a{index}_" for index in range(8))
        for prefix in names:
            self.assertTrue(
                any(name.startswith(prefix) for name in factories),
                msg=f"missing standard-runner ablation prefix {prefix}",
            )
        tri_names = (
            "a2_tri_no_teacher",
            "a3_tri_teacher",
            "a4_tri_teacher_mtl",
            "a5_vimd_full",
            "a7_vimd_no_residual",
        )
        built = [factories[name](4, 9, self.config) for name in tri_names]
        self.assertEqual(
            len(
                {
                    sum(parameter.numel() for parameter in item.model.parameters())
                    for item in built
                }
            ),
            1,
        )
        self.assertEqual(
            [item.objective.name for item in built],
            [
                "tri_mask_no_teacher_no_mtl_no_xcc_with_residual",
                "tri_mask_fixed_teacher",
                "tri_mask_teacher_mtl_no_xcc",
                "full_vimd",
                "full_vimd",
            ],
        )
        standard_dual = factories["a6_dual_full"](4, 9, self.config)
        self.assertIsInstance(standard_dual.model, DualMaskVIMDNet)
        self.assertIsInstance(standard_dual.teacher, PhysicalDualMaskTeacher)
        standard_no_residual = factories["a7_vimd_no_residual"](
            4,
            9,
            self.config,
        )
        self.assertFalse(standard_no_residual.model.use_residual)

    def test_tri_ce_loss_needs_no_teacher_or_auxiliary_targets(self) -> None:
        model = VIMDNet(4, 9, self.config)
        first_output = model(torch.randn(4, 2, 128))
        second_output = model(torch.randn(4, 2, 128))
        empty_batch: dict[str, torch.Tensor] = {}
        losses = vimd_two_view_loss(
            first_output=first_output,
            second_output=second_output,
            first_batch=empty_batch,
            second_batch=empty_batch,
            labels=torch.tensor([0, 1, 2, 3]),
            first_teacher_mask=None,
            second_teacher_mask=None,
            weights=VIMDLossWeights(),
            enable_mask=False,
            enable_contrastive=False,
            enable_jammer=False,
            enable_quality=False,
            enable_orthogonality=False,
        )
        for name in ("jammer", "quality", "mask", "contrastive", "orthogonality"):
            self.assertEqual(float(losses[name]), 0.0)

    def test_teacher_helper_forces_float32_then_returns_student_dtype(self) -> None:
        teacher = PhysicalTriMaskTeacher(self.config)
        view = {
            "clean": torch.randn(2, 2, 128).half(),
            "jammer": torch.randn(2, 2, 128).half(),
            "unexplained": torch.randn(2, 2, 128).half(),
        }
        target = _physical_teacher_target(
            teacher,
            view,
            output_dtype=torch.float16,
            device_type="cpu",
        )
        self.assertEqual(target.dtype, torch.float16)
        self.assertTrue(torch.isfinite(target).all())
        torch.testing.assert_close(
            target.float().sum(dim=1),
            torch.ones_like(target[:, 0].float()),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_lr_is_held_through_minimum_full_objective_stage(self) -> None:
        factors = [
            _staged_cosine_learning_rate_factor(
                completed,
                total_epochs=10,
                selection_start_epoch=4,
            )
            for completed in range(0, 11)
        ]
        self.assertEqual(factors[:5], [1.0] * 5)
        self.assertLess(factors[5], 1.0)
        self.assertTrue(
            all(
                left >= right
                for left, right in zip(factors[4:], factors[5:])
            )
        )


if __name__ == "__main__":
    unittest.main()
