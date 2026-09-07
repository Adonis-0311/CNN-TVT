"""Unit tests for the standards cache CLI configuration surface."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from vimd_amc.data.synthesis import SynthesisConfig, SignalSynthesizer  # noqa: E402
from vimd_amc.standards.cache import (  # noqa: E402
    TDLCacheBuildConfig,
    _build_pending_views,
)

_SPEC = importlib.util.spec_from_file_location(
    "vimd_cache_builder_cli",
    REPOSITORY_ROOT / "standards" / "cache_builder.py",
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("could not load standards/cache_builder.py")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class CacheBuilderCLITest(unittest.TestCase):
    def test_defaults_preserve_smoke_configuration(self) -> None:
        args = _MODULE.parse_args([])
        config = _MODULE.config_from_args(args)
        self.assertEqual(config.sample_length, 128)
        self.assertEqual(config.guard_samples, 48)
        self.assertEqual(
            config.split_sizes,
            (("train", 2), ("validation", 1), ("heldout_channel", 2)),
        )
        self.assertIsNone(config.snr_db_values)
        self.assertIsNone(config.sir_db_values)
        self.assertEqual(config.evidence_designation, "integration_smoke_only")

    def test_explicit_lists_are_translated_without_loss(self) -> None:
        args = _MODULE.parse_args(
            [
                "--modulations",
                "BPSK,QPSK,256QAM",
                "--jammers",
                "tone,multitone,cochannel",
                "--train-profiles",
                "TDL-A,TDL-C,TDL-D",
                "--heldout-profiles",
                "TDL-B,TDL-E",
                "--speeds-kmh",
                "0,60,120,150",
                "--delay-spreads-ns",
                "30,100,300",
                "--snr-db=-8,-4,0,8,16",
                "--sir-db-values=-12,-6,0,8",
                "--evidence-designation",
                "screening_not_formal_tvt_evidence",
            ]
        )
        config = _MODULE.config_from_args(args)
        self.assertEqual(config.modulations, ("BPSK", "QPSK", "256QAM"))
        self.assertEqual(
            config.jammer_choices, ("tone", "multitone", "cochannel")
        )
        self.assertEqual(config.speeds_kmh, (0.0, 60.0, 120.0, 150.0))
        self.assertEqual(len(config.delay_spreads_s), 3)
        for actual, expected in zip(
            config.delay_spreads_s, (30e-9, 100e-9, 300e-9), strict=True
        ):
            self.assertAlmostEqual(actual, expected, places=18)
        self.assertEqual(config.snr_db_values, (-8.0, -4.0, 0.0, 8.0, 16.0))
        self.assertEqual(config.sir_db_values, (-12.0, -6.0, 0.0, 8.0))

    def test_rejects_invalid_explicit_values(self) -> None:
        cases = (
            (["--speeds-kmh=-1,60"], "speeds"),
            (["--delay-spreads-ns=0,30"], "delay"),
            (["--snr-db-values=0,nan"], "snr_db_values"),
            (["--sir-db-values=-4,-4"], "sir_db_values"),
            (["--train-profiles=TDL-A,TDL-Z"], "unknown TDL"),
            (
                [
                    "--train-profiles=TDL-A,TDL-C",
                    "--heldout-profiles=TDL-C,TDL-E",
                ],
                "overlap",
            ),
            (["--modulations=BPSK,BPSK"], "modulations"),
            (["--modulations=BPSK,UNKNOWN"], "unsupported modulations"),
            (["--jammers=tone,none"], "active jammer"),
            (["--jammers=tone,unknown"], "unsupported jammers"),
            (["--evidence-designation="], "evidence_designation"),
        )
        for argv, expected in cases:
            with self.subTest(argv=argv):
                args = _MODULE.parse_args(argv)
                with self.assertRaisesRegex(ValueError, expected):
                    _MODULE.config_from_args(args)


class ActiveJammerCropPolicyTest(unittest.TestCase):
    def test_sparse_pulse_retry_uses_only_prechannel_crop_power(self) -> None:
        config = TDLCacheBuildConfig(
            split_sizes=(
                ("train", 415),
                ("validation", 1),
                ("heldout_channel", 1),
            ),
            sample_length=256,
            guard_samples=64,
            master_seed=20260727,
            modulations=(
                "BPSK",
                "PI2BPSK",
                "QPSK",
                "8PSK",
                "16QAM",
                "64QAM",
                "256QAM",
                "GMSK",
                "CPFSK",
                "4FSK",
            ),
            jammer_choices=(
                "tone",
                "chirp",
                "pulse",
                "partial_band",
                "cochannel",
                "multitone",
            ),
            snr_db_values=(-10.0, -6.0, -2.0, 2.0, 6.0, 10.0, 14.0, 18.0),
            sir_db_values=(-15.0, -10.0, -5.0, 0.0, 5.0, 10.0),
        )
        synthesizer = SignalSynthesizer(
            SynthesisConfig(
                sample_length=256,
                sample_rate_hz=1e6,
                carrier_hz=5.9e9,
            )
        )
        pending, _ = _build_pending_views(synthesizer, config)
        regression_view = pending[828]
        self.assertEqual(
            (regression_view.split, regression_view.index, regression_view.view),
            ("train", 414, 1),
        )
        self.assertEqual(regression_view.jammer_name, "pulse")
        self.assertEqual(regression_view.jammer_generation_attempt, 1)
        self.assertGreater(
            regression_view.jammer_center_power_before_channel, 1e-12
        )
        self.assertEqual(len(regression_view.jammer_generation_rejections), 1)
        rejection = regression_view.jammer_generation_rejections[0]
        self.assertEqual(rejection["attempt"], 0)
        self.assertEqual(rejection["observed_power"], 0.0)
        self.assertEqual(rejection["threshold"], 1e-12)
        self.assertIn("prechannel", rejection["reason"])


if __name__ == "__main__":
    unittest.main()
