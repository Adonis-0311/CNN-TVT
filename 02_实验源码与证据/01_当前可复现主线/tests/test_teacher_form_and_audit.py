from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.run_experiment import build_model  # noqa: E402
from experiments.run_standard_experiment import (  # noqa: E402
    available_model_factories,
)
from vimd_amc.ablation import (  # noqa: E402
    LOCAL_NORMALIZATION_TEACHER_POLICY,
    PAPER_ABLATION_PROTOCOLS,
    TEACHER_FORM_ABLATION_PROTOCOLS,
    canonical_model_name,
    paper_ablation_protocol,
)
from vimd_amc.models.common import ModelConfig  # noqa: E402
from vimd_amc.models.vimd import (  # noqa: E402
    LocalWhitenedTriMaskTeacher,
    PhysicalTriMaskTeacher,
    ProportionalTriMaskTeacher,
    TriMaskTeacherSpec,
    VIMDNet,
    build_tri_mask_teacher,
)
from vimd_amc.teacher_audit import (  # noqa: E402
    FORMAL_TVT_EVIDENCE_DESIGNATION,
    FORMAL_TVT_V2_EVIDENCE_DESIGNATION,
    hard_split_teacher_cell_statistics,
    occupancy_gain_mechanism_test,
)


class _TinyHardDataset(Dataset):
    split = "hard_interference"

    def __init__(
        self,
        evidence_designation: str = "screening_not_formal_tvt_evidence",
    ) -> None:
        self.evidence_designation = evidence_designation
        generator = torch.Generator().manual_seed(20260728)
        self.items = []
        self.records = []
        families = (("tone", "chirp"), ("tone", "tone"))
        for source_index, source_families in enumerate(families):
            views = {}
            view_records = []
            for view_index, (view_name, family) in enumerate(
                zip(("view1", "view2"), source_families)
            ):
                clean = torch.randn(2, 64, generator=generator)
                jammer = (1.0 + source_index + view_index) * torch.randn(
                    2,
                    64,
                    generator=generator,
                )
                unexplained = 0.2 * torch.randn(
                    2,
                    64,
                    generator=generator,
                )
                views[view_name] = {
                    "clean": clean,
                    "jammer": jammer,
                    "unexplained": unexplained,
                    "snr_db": torch.tensor(-10.0),
                    "sir_db": torch.tensor(-15.0),
                }
                view_records.append(
                    {
                        "jammer_name": family,
                        "snr_db": -10.0,
                        "sir_db": -15.0,
                    }
                )
            self.items.append(views)
            self.records.append(
                {
                    "index": source_index,
                    "views": view_records,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, object]:
        views = self.items[index]
        return {
            **views,
            "label": torch.tensor(0),
            "source_id": torch.tensor(index),
        }

    def manifest(self) -> dict[str, object]:
        return {
            "configuration": {
                "evidence_designation": self.evidence_designation
            },
            "cache_digest": "unit-test-digest",
            "records": {"hard_interference": self.records},
        }


class TeacherFormAndAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def setUp(self) -> None:
        self.config = ModelConfig(
            n_fft=16,
            hop_length=4,
            spectral_channels=8,
            environment_dim=8,
            embedding_dim=16,
            dropout=0.0,
        )

    def test_margin_teacher_implements_direct_closed_form(self) -> None:
        teacher = PhysicalTriMaskTeacher(self.config)
        clean = torch.randn(3, 2, 64)
        jammer = 2.5 * torch.randn_like(clean)
        unexplained = 0.4 * torch.randn_like(clean)
        decomposition = teacher.decompose(clean, jammer, unexplained)
        signal_power = teacher.front_end(clean).abs().square()
        jammer_power = teacher.front_end(jammer).abs().square()
        unexplained_power = teacher.front_end(unexplained).abs().square()
        total = signal_power + jammer_power + unexplained_power
        epsilon = (1e-8 * total.mean((1, 2), keepdim=True)).clamp_min(
            torch.finfo(total.dtype).tiny
        )
        denominator = total.clamp_min(epsilon)
        q_signal = signal_power / denominator
        q_jammer = jammer_power / denominator
        q_unexplained = unexplained_power / denominator
        expected = torch.stack(
            (
                torch.relu(q_signal - q_jammer),
                torch.relu(q_jammer - q_signal),
                q_unexplained + 2.0 * torch.minimum(q_signal, q_jammer),
            ),
            dim=1,
        )
        expected = expected / expected.sum(dim=1, keepdim=True)
        torch.testing.assert_close(decomposition["masks"], expected)

    def test_proportional_teacher_is_qs_qj_qu(self) -> None:
        teacher = ProportionalTriMaskTeacher(self.config)
        clean = torch.randn(2, 2, 64)
        jammer = 4.0 * torch.randn_like(clean)
        unexplained = 0.3 * torch.randn_like(clean)
        masks = teacher(clean, jammer, unexplained)
        powers = torch.stack(
            (
                teacher.front_end(clean).abs().square(),
                teacher.front_end(jammer).abs().square(),
                teacher.front_end(unexplained).abs().square(),
            ),
            dim=1,
        )
        expected = powers / powers.sum(dim=1, keepdim=True)
        torch.testing.assert_close(masks, expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            masks.sum(dim=1),
            torch.ones_like(masks[:, 0]),
        )

    def test_local_teacher_has_frozen_serializable_spec(self) -> None:
        spec = TriMaskTeacherSpec(
            mode="local_whitened_margin",
            local_frequency_bins=7,
            local_time_frames=3,
            local_whitening_exponent=0.5,
        )
        with self.assertRaises(FrozenInstanceError):
            spec.local_frequency_bins = 9  # type: ignore[misc]
        teacher = build_tri_mask_teacher(self.config, spec)
        self.assertIsInstance(teacher, LocalWhitenedTriMaskTeacher)
        self.assertEqual(teacher.spec.to_dict(), spec.to_dict())
        masks = teacher(
            torch.randn(2, 2, 64),
            torch.randn(2, 2, 64),
            torch.randn(2, 2, 64),
        )
        self.assertTrue(bool(torch.isfinite(masks).all()))
        torch.testing.assert_close(
            masks.sum(dim=1),
            torch.ones_like(masks[:, 0]),
            atol=1e-6,
            rtol=1e-6,
        )
        self.assertFalse(
            LOCAL_NORMALIZATION_TEACHER_POLICY[
                "primary_a3_to_a7_enabled"
            ]
        )

    def test_a3_prime_is_inserted_without_renumbering_a0_a7(self) -> None:
        self.assertEqual(len(PAPER_ABLATION_PROTOCOLS), 8)
        self.assertEqual(
            tuple(
                protocol["ablation_id"]
                for protocol in PAPER_ABLATION_PROTOCOLS.values()
            ),
            tuple(f"A{index}" for index in range(8)),
        )
        self.assertEqual(
            TEACHER_FORM_ABLATION_PROTOCOLS[
                "a3p_tri_proportional_teacher"
            ]["ablation_id"],
            "A3\u2032",
        )
        self.assertEqual(
            canonical_model_name("a3_ratio_teacher"),
            "a3p_tri_proportional_teacher",
        )
        self.assertEqual(
            paper_ablation_protocol("a3_ratio_teacher")["teacher"],
            "proportional_component_power_tri",
        )

    def test_both_runners_build_a3_prime_with_same_architecture(self) -> None:
        primary = build_model(
            "a3p_tri_proportional_teacher",
            classes=4,
            jammers=9,
            config=self.config,
        )
        alias = build_model(
            "a3_ratio_teacher",
            classes=4,
            jammers=9,
            config=self.config,
        )
        self.assertIsInstance(primary.model, VIMDNet)
        self.assertIsInstance(primary.teacher, ProportionalTriMaskTeacher)
        self.assertIsInstance(alias.teacher, ProportionalTriMaskTeacher)
        self.assertTrue(primary.objective.use_mask_supervision)
        factories = available_model_factories()
        standard = factories["a3p_tri_proportional_teacher"](
            4,
            9,
            self.config,
        )
        diagnostic = factories["diagnostic_a3_local_whitened_teacher"](
            4,
            9,
            self.config,
        )
        self.assertIsInstance(standard.teacher, ProportionalTriMaskTeacher)
        self.assertIsInstance(
            diagnostic.teacher,
            LocalWhitenedTriMaskTeacher,
        )

    def test_hard_split_audit_is_exact_grouped_and_development_only(self) -> None:
        dataset = _TinyHardDataset()
        teacher = PhysicalTriMaskTeacher(self.config)
        result = hard_split_teacher_cell_statistics(
            {"closed_form_margin": teacher},
            dataset,
            batch_size=1,
        )
        self.assertFalse(result["formal_tvt_evidence_eligible"])
        self.assertIn("development-only", result["claim_scope"])
        self.assertEqual(result["occupancy_gain_mechanism"]["status"], "pending")
        self.assertEqual(
            set(result["jammer_spectral_occupancy"]["by_jammer_family"]),
            {"tone", "chirp"},
        )
        record = result["teachers"]["closed_form_margin"]
        self.assertEqual(record["overall"]["source_view_count"], 4)
        self.assertEqual(
            record["by_jammer_family"]["tone"]["source_view_count"],
            3,
        )
        self.assertEqual(
            record["by_jammer_family"]["chirp"]["source_view_count"],
            1,
        )
        manual = []
        for item in dataset.items:
            for view_name in ("view1", "view2"):
                view = item[view_name]
                manual.append(
                    teacher(
                        view["clean"].unsqueeze(0),
                        view["jammer"].unsqueeze(0),
                        view["unexplained"].unsqueeze(0),
                    )[0, 0].numpy().reshape(-1)
                )
        pooled = np.concatenate(manual)
        self.assertAlmostEqual(
            record["overall"]["mean"],
            float(np.mean(pooled, dtype=np.float64)),
        )
        self.assertAlmostEqual(
            record["overall"]["p90"],
            float(np.quantile(pooled, 0.90, method="linear")),
        )
        self.assertAlmostEqual(
            record["overall"]["nonzero_cell_fraction"],
            float(np.mean(pooled > 0.0)),
        )
        self.assertEqual(
            record["operating_points_db"]["snr=-10,sir=-15"]["status"],
            "available",
        )
        json.dumps(result, allow_nan=False)

    def test_hard_split_audit_accepts_frozen_v1_and_v2_designations(
        self,
    ) -> None:
        teacher = PhysicalTriMaskTeacher(self.config)
        for designation in (
            FORMAL_TVT_EVIDENCE_DESIGNATION,
            FORMAL_TVT_V2_EVIDENCE_DESIGNATION,
        ):
            with self.subTest(designation=designation):
                result = hard_split_teacher_cell_statistics(
                    {"closed_form_margin": teacher},
                    _TinyHardDataset(designation),
                    batch_size=2,
                )
                self.assertEqual(
                    result["cache_evidence_designation"],
                    designation,
                )
                self.assertTrue(result["formal_tvt_evidence_eligible"])
                self.assertEqual(
                    result["claim_scope"],
                    "formal-cache descriptive teacher evidence",
                )

    def test_occupancy_gain_hypothesis_uses_exact_negative_spearman(self) -> None:
        families = ("tone", "multitone", "chirp", "sweep", "partial_band", "comb")
        rows = [
            {
                "family": family,
                "occupancy": 0.1 * (index + 1),
                "gain_pp": float(6 - index),
            }
            for index, family in enumerate(families)
        ]
        result = occupancy_gain_mechanism_test(rows)
        self.assertEqual(result["permutation_count"], 720)
        self.assertAlmostEqual(result["spearman_rho"], -1.0)
        self.assertAlmostEqual(
            result["one_sided_negative_exact_permutation_p"],
            1.0 / 720.0,
        )
        self.assertTrue(result["negative_rank_test_passed"])
        self.assertTrue(result["nonincreasing_order_holds"])
        self.assertTrue(result["confirmatory_mechanism_passed"])

        tied_rows = [dict(row) for row in rows]
        tied_rows[1]["occupancy"] = tied_rows[0]["occupancy"]
        tied = occupancy_gain_mechanism_test(tied_rows)
        self.assertLess(tied["spearman_rho"], 0.0)
        self.assertEqual(tied["protocol"]["tie_policy"], "average ranks for both occupancy and gain")


if __name__ == "__main__":
    unittest.main()
