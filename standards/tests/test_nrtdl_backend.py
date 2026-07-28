"""Integration test for the optional MATLAB 5G Toolbox backend."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
import unittest

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from vimd_amc.standards import (  # noqa: E402
    NRTDLConfiguration,
    apply_nrtdl_batch,
)


@unittest.skipUnless(shutil.which("matlab"), "MATLAB is not available on PATH")
class MATLABNRTDLBackendTest(unittest.TestCase):
    def test_batch_shape_determinism_seed_and_doppler(self) -> None:
        rng = np.random.default_rng(907)
        waveform = (
            rng.standard_normal(2048) + 1j * rng.standard_normal(2048)
        ) / np.sqrt(2.0)
        batch = np.repeat(waveform[None, :], 7, axis=0)
        configs = [
            NRTDLConfiguration("TDL-A", 100e-9, 0.0, 5.9e9, 11),
            NRTDLConfiguration("TDL-A", 100e-9, 0.0, 5.9e9, 11),
            NRTDLConfiguration("TDL-A", 100e-9, 0.0, 5.9e9, 12),
            NRTDLConfiguration("TDL-B", 300e-9, 20.0, 5.9e9, 16),
            NRTDLConfiguration("TDL-C", 300e-9, 35.0, 5.9e9, 21),
            NRTDLConfiguration("TDL-D", 100e-9, 50.0, 5.9e9, 31),
            NRTDLConfiguration("TDL-E", 100e-9, 45.0, 5.9e9, 41),
        ]

        result = apply_nrtdl_batch(batch, configs)

        self.assertEqual(result.waveforms.shape, batch.shape)
        self.assertTrue(np.all(np.isfinite(result.waveforms)))
        self.assertTrue(np.all(np.linalg.norm(result.waveforms, axis=1) > 0))
        self.assertTrue(np.array_equal(result.waveforms[0], result.waveforms[1]))
        self.assertFalse(np.allclose(result.waveforms[0], result.waveforms[2]))
        self.assertEqual(result.metadata[0]["maximum_doppler_hz"], 0.0)
        self.assertGreater(result.metadata[3]["maximum_doppler_hz"], 0.0)
        self.assertEqual(
            tuple(item["profile"] for item in result.metadata),
            (
                "TDL-A",
                "TDL-A",
                "TDL-A",
                "TDL-B",
                "TDL-C",
                "TDL-D",
                "TDL-E",
            ),
        )
        self.assertTrue(
            all(item["sample_rate_hz"] == 1_000_000.0 for item in result.metadata)
        )


if __name__ == "__main__":
    unittest.main()
