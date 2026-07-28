from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vimd_amc.data.synthesis import (  # noqa: E402
    SignalSynthesizer,
    SynthesisConfig,
    measured_sir_db,
    measured_snr_db,
    signal_power,
)


class SynthesisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.synthesizer = SignalSynthesizer(SynthesisConfig(sample_length=256))

    def _sample(self, **overrides):
        arguments = {
            "modulation": "QPSK",
            "source_seed": 1234,
            "condition_seed": 9876,
            "snr_db": -3.0,
            "sir_db": -6.0,
            "jammer_name": "chirp",
            "channel_scenario": "proxy_highway_nlos",
            "speed_kmh": 120.0,
            "overlap_profile": "high",
        }
        arguments.update(overrides)
        return self.synthesizer.synthesize(**arguments)

    def test_power_control_is_realized_not_only_expected(self) -> None:
        sample = self._sample()
        self.assertAlmostEqual(measured_snr_db(sample.clean, sample.noise), -3.0, places=5)
        self.assertAlmostEqual(measured_sir_db(sample.clean, sample.jammer), -6.0, places=5)
        self.assertAlmostEqual(signal_power(sample.mixture), 1.0, places=5)
        reconstructed = (
            sample.clean + sample.jammer + sample.noise + sample.receiver_artifact
        )
        self.assertLess(float(np.max(np.abs(reconstructed - sample.mixture))), 2e-6)

    def test_no_jammer_uses_validity_mask_semantics(self) -> None:
        sample = self._sample(jammer_name="none")
        self.assertIsNone(sample.metadata["sir_db"])
        self.assertFalse(sample.metadata["sir_valid"])
        self.assertTrue(np.isinf(sample.metadata["measured_sir_db"]))
        self.assertEqual(signal_power(sample.jammer), 0.0)

    def test_same_seed_is_elementwise_deterministic(self) -> None:
        first = self._sample()
        second = self._sample()
        np.testing.assert_array_equal(first.mixture, second.mixture)
        np.testing.assert_array_equal(first.clean, second.clean)
        np.testing.assert_array_equal(first.jammer, second.jammer)
        self.assertEqual(first.metadata, second.metadata)

    def test_overlap_profiles_change_inband_jammer_energy(self) -> None:
        low, high = [], []
        for condition_seed in range(30, 42):
            low.append(
                self._sample(
                    condition_seed=condition_seed,
                    jammer_name="tone",
                    overlap_profile="low",
                ).metadata["jammer_to_signal_overlap"]
            )
            high.append(
                self._sample(
                    condition_seed=condition_seed,
                    jammer_name="tone",
                    overlap_profile="high",
                ).metadata["jammer_to_signal_overlap"]
            )
        self.assertLess(float(np.median(low)), 0.25)
        self.assertGreater(float(np.median(high)), 0.75)


if __name__ == "__main__":
    unittest.main()
