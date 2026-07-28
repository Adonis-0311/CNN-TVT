from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vimd_amc.data.dataset import PairedAMCDataset, Regime  # noqa: E402
from vimd_amc.data.split import assert_disjoint_source_ids  # noqa: E402
from vimd_amc.data.synthesis import SignalSynthesizer, SynthesisConfig  # noqa: E402


class DatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        synthesizer = SignalSynthesizer(SynthesisConfig(sample_length=128))
        self.train = PairedAMCDataset(
            synthesizer=synthesizer,
            split="train",
            size=12,
            regime=Regime.train(),
            master_seed=7,
            modulations=("BPSK", "QPSK", "16QAM"),
        )
        self.validation = PairedAMCDataset(
            synthesizer=synthesizer,
            split="validation",
            size=8,
            regime=Regime.validation(),
            master_seed=7,
            modulations=("BPSK", "QPSK", "16QAM"),
        )

    def test_source_sequences_are_disjoint(self) -> None:
        assert_disjoint_source_ids(self.train.source_ids(), self.validation.source_ids())

    def test_views_share_source_but_not_condition(self) -> None:
        item = self.train[3]
        self.assertNotEqual(
            int(item["view1"]["condition_seed"]),
            int(item["view2"]["condition_seed"]),
        )
        self.assertEqual(int(item["source_id"]), self.train.source_ids()[3])
        self.assertFalse(torch.equal(item["view1"]["x"], item["view2"]["x"]))

    def test_manifest_is_stable(self) -> None:
        self.assertEqual(self.train.manifest()["digest"], self.train.manifest()["digest"])

    def test_no_jammer_quality_mask_disables_sir(self) -> None:
        synthesizer = self.train.synthesizer
        dataset = PairedAMCDataset(
            synthesizer=synthesizer,
            split="clean",
            size=2,
            regime=Regime.clean_high_snr(),
            master_seed=9,
            modulations=("BPSK",),
        )
        item = dataset[0]
        self.assertEqual(float(item["view1"]["quality_mask"][1]), 0.0)
        self.assertTrue(torch.isinf(item["view1"]["sir_db"]))


if __name__ == "__main__":
    unittest.main()
