"""Tests for the pre-registered nine-split factor-isolation cache."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from vimd_amc.data.split import assert_disjoint_source_ids  # noqa: E402
from vimd_amc.data.synthesis import SynthesisConfig, SignalSynthesizer  # noqa: E402
from vimd_amc.standards import (  # noqa: E402
    CachedPairedAMCDataset,
    FACTOR_ISOLATED_SPLITS,
    TDLCacheBuildConfig,
    build_tdl_paired_cache,
    factor_isolated_split_policies,
    validate_cached_components,
)
from vimd_amc.standards.cache import (  # noqa: E402
    _build_pending_views,
    _stratified_clean_slots,
)


_SPEC = importlib.util.spec_from_file_location(
    "vimd_factor_cache_cli",
    REPOSITORY_ROOT / "standards" / "build_factor_cache.py",
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("could not load standards/build_factor_cache.py")
_CLI = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CLI)


def _micro_config() -> TDLCacheBuildConfig:
    args = _CLI.parse_args(
        [
            "--output",
            str(REPOSITORY_ROOT / "standards" / "_never_written"),
            "--preset",
            "micro",
            "--sample-length",
            "64",
            "--guard-samples",
            "48",
        ]
    )
    return _CLI.config_from_args(args)


class FactorPolicyUnitTest(unittest.TestCase):
    def test_policy_matches_locked_table_and_source_ids_are_global_disjoint(
        self,
    ) -> None:
        config = _micro_config()
        self.assertEqual(
            tuple(policy.split for policy in config.split_policies or ()),
            FACTOR_ISOLATED_SPLITS,
        )
        self.assertTrue(
            all(policy.size >= 1 for policy in config.split_policies or ())
        )
        policies = {
            policy.split: policy for policy in config.split_policies or ()
        }
        self.assertFalse(
            set(policies["train"].jammer_choices)
            & set(policies["unseen_jammer"].jammer_choices)
        )
        self.assertEqual(
            set(policies["train"].jammer_choices),
            {
                "tone",
                "multitone",
                "chirp",
                "sweep",
                "partial_band",
                "comb",
            },
        )
        self.assertEqual(
            set(policies["unseen_jammer"].jammer_choices),
            {"pulse", "ofdm_like"},
        )
        self.assertFalse(
            set(policies["train"].speeds_kmh)
            & set(policies["unseen_speed"].speeds_kmh)
        )
        self.assertFalse(
            set(policies["train"].profiles)
            & set(policies["heldout_channel"].profiles)
        )
        self.assertLessEqual(
            max(policies["hard_interference"].sir_db_values or (float("inf"),)),
            0.0,
        )
        self.assertEqual(
            policies["clean_retention"].jammer_choices, ("none",)
        )
        self.assertEqual(policies["train"].clean_fraction, 0.20)
        self.assertEqual(policies["validation"].clean_fraction, 0.20)
        self.assertEqual(policies["id_test"].clean_fraction, 0.20)
        self.assertIsNone(policies["clean_retention"].sir_db_values)
        self.assertEqual(
            set(policies["clean_retention"].profiles),
            {"TDL-A", "TDL-B", "TDL-C", "TDL-D", "TDL-E"},
        )

        synthesizer = SignalSynthesizer(
            SynthesisConfig(
                sample_length=config.sample_length,
                sample_rate_hz=config.sample_rate_hz,
                carrier_hz=config.carrier_frequency_hz,
            )
        )
        pending, source_groups = _build_pending_views(synthesizer, config)
        self.assertEqual(len(pending), 2 * len(FACTOR_ISOLATED_SPLITS))
        assert_disjoint_source_ids(*source_groups.values())
        grouped = {
            split: [item for item in pending if item.split == split]
            for split in FACTOR_ISOLATED_SPLITS
        }
        self.assertTrue(
            all(item.sir_db <= 0 for item in grouped["hard_interference"])
        )
        for split in ("train", "validation", "id_test"):
            self.assertEqual(
                sum(item.jammer_name == "none" for item in grouped[split]),
                1,
            )
            self.assertEqual(
                sum(item.jammer_name != "none" for item in grouped[split]),
                1,
            )
        self.assertTrue(
            all(
                item.jammer_name == "none"
                and np.count_nonzero(item.jammer_with_guard) == 0
                for item in grouped["clean_retention"]
            )
        )
        self.assertTrue(
            {
                item.speed_kmh for item in grouped["unseen_speed"]
            }.issubset({180.0, 250.0})
        )

    def test_policy_rejects_seen_held_jammer_overlap(self) -> None:
        config = _micro_config()
        policies = list(config.split_policies or ())
        train = next(policy for policy in policies if policy.split == "train")
        unseen_index = next(
            index
            for index, policy in enumerate(policies)
            if policy.split == "unseen_jammer"
        )
        policies[unseen_index] = replace(
            policies[unseen_index],
            jammer_choices=(train.jammer_choices[0],),
        )
        invalid = replace(config, split_policies=tuple(policies))
        with self.assertRaisesRegex(ValueError, "jammer families.*disjoint"):
            invalid.validate()

    def test_split_size_override_does_not_change_factor_policy(self) -> None:
        args = _CLI.parse_args(
            [
                "--output",
                str(REPOSITORY_ROOT / "standards" / "_never_written"),
                "--preset",
                "micro",
                "--split-size",
                "train=11",
            ]
        )
        config = _CLI.config_from_args(args)
        policies = {
            policy.split: policy for policy in config.split_policies or ()
        }
        self.assertEqual(policies["train"].size, 11)
        self.assertEqual(policies["validation"].size, 1)
        self.assertEqual(
            policies["unseen_speed"].speeds_kmh, (180.0, 250.0)
        )
        self.assertEqual(
            _stratified_clean_slots(10, 0.20),
            _stratified_clean_slots(10, 0.20),
        )
        self.assertEqual(len(_stratified_clean_slots(10, 0.20)), 4)
        self.assertEqual(len(_stratified_clean_slots(10, 0.0)), 0)
        self.assertEqual(len(_stratified_clean_slots(10, 1.0)), 20)


@unittest.skipUnless(shutil.which("matlab"), "MATLAB is not available on PATH")
class FactorCacheIntegrationTest(unittest.TestCase):
    def test_micro_cache_materializes_every_split_and_clean_validity(
        self,
    ) -> None:
        config = _micro_config()
        with tempfile.TemporaryDirectory(
            prefix="vimd_factor_cache_"
        ) as temporary:
            result = build_tdl_paired_cache(
                Path(temporary) / "factor_micro",
                config=config,
                matlab_timeout_s=300.0,
            )
            manifest = result.manifest
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(
                tuple(manifest["files"]), FACTOR_ISOLATED_SPLITS
            )
            self.assertEqual(
                tuple(manifest["preregistered_split_policy"]),
                FACTOR_ISOLATED_SPLITS,
            )
            self.assertTrue(
                all(
                    audit["all_actual_values_within_policy"]
                    for audit in manifest["factor_coverage"].values()
                )
            )
            self.assertEqual(
                set(manifest["quality_normalization"]),
                {"snr_db", "sir_db", "doppler_hz"},
            )
            self.assertEqual(
                manifest["quality_normalization"]["snr_db"]["scale"],
                20.0,
            )
            self.assertEqual(len(manifest["jammer_taxonomy"]), 9)
            self.assertEqual(len(set(manifest["jammer_taxonomy"])), 9)
            self.assertEqual(
                set(manifest["protocol_exclusions"]),
                {"cochannel", "mixed"},
            )
            datasets: list[CachedPairedAMCDataset] = []
            try:
                for split in FACTOR_ISOLATED_SPLITS:
                    dataset = CachedPairedAMCDataset(
                        result.root, split, verify_checksums=True
                    )
                    datasets.append(dataset)
                    audit = validate_cached_components(dataset)
                    self.assertLessEqual(audit["max_component_error"], 2e-6)
                    self.assertLessEqual(audit["max_snr_error_db"], 2e-3)
                    self.assertLessEqual(audit["max_sir_error_db"], 2e-3)
                clean = datasets[-1]
                self.assertTrue(
                    np.all(clean._arrays["quality_mask"][:, :, 1] == 0)
                )
                self.assertTrue(np.all(clean._arrays["jam_labels"] == 0))
                self.assertEqual(
                    np.count_nonzero(clean._arrays["jammer"]), 0
                )
                assert_disjoint_source_ids(
                    *(dataset.source_ids() for dataset in datasets)
                )
            finally:
                for dataset in datasets:
                    dataset.close()


if __name__ == "__main__":
    unittest.main()
