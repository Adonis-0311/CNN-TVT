"""Integration tests for the deterministic offline MATLAB-TDL cache."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from vimd_amc.data.split import assert_disjoint_source_ids  # noqa: E402
from vimd_amc.standards import (  # noqa: E402
    CachedPairedAMCDataset,
    TDLCacheBuildConfig,
    build_tdl_paired_cache,
    validate_cached_components,
)


@unittest.skipUnless(shutil.which("matlab"), "MATLAB is not available on PATH")
class TDLCacheIntegrationTest(unittest.TestCase):
    def test_cache_evidence_and_determinism(self) -> None:
        config = TDLCacheBuildConfig(
            split_sizes=(
                ("train", 2),
                ("validation", 1),
                ("heldout_channel", 2),
            ),
            sample_length=96,
            guard_samples=48,
            master_seed=8701,
        )
        with tempfile.TemporaryDirectory(prefix="vimd_tdl_cache_") as temporary:
            root = Path(temporary)
            first = build_tdl_paired_cache(root / "first", config=config)
            second = build_tdl_paired_cache(root / "second", config=config)

            self.assertEqual(
                first.manifest["cache_digest"],
                second.manifest["cache_digest"],
            )
            self.assertEqual(
                first.manifest["source_ids"],
                second.manifest["source_ids"],
            )
            for split, _ in config.split_sizes:
                first_dataset = CachedPairedAMCDataset(
                    first.root, split, verify_checksums=True
                )
                second_dataset = CachedPairedAMCDataset(
                    second.root, split, verify_checksums=True
                )
                evidence = validate_cached_components(first_dataset)
                self.assertLessEqual(evidence["max_component_error"], 2e-6)
                self.assertLessEqual(evidence["max_snr_error_db"], 2e-3)
                self.assertLessEqual(evidence["max_sir_error_db"], 2e-3)
                for name, specification in first.manifest["files"][split].items():
                    self.assertEqual(
                        specification["sha256"],
                        second.manifest["files"][split][name]["sha256"],
                    )
                item = first_dataset[0]
                self.assertEqual(tuple(item["view1"]["x"].shape), (2, 96))
                self.assertEqual(tuple(item["view2"]["x"].shape), (2, 96))
                self.assertEqual(int(item["source_id"]), first_dataset.source_ids()[0])
                self.assertNotEqual(
                    int(item["view1"]["target_channel_seed"]),
                    int(item["view1"]["jammer_channel_seed"]),
                )
                self.assertTrue(
                    np.array_equal(
                        first_dataset._arrays["x"],
                        second_dataset._arrays["x"],
                    )
                )
                first_dataset.close()
                second_dataset.close()

            source_groups = first.manifest["source_ids"]
            assert_disjoint_source_ids(
                source_groups["train"],
                source_groups["validation"],
                source_groups["heldout_channel"],
            )
            for split in ("train", "validation"):
                profiles = {
                    view["tdl_target_profile"]
                    for record in first.manifest["records"][split]
                    for view in record["views"]
                } | {
                    view["tdl_jammer_profile"]
                    for record in first.manifest["records"][split]
                    for view in record["views"]
                }
                self.assertTrue(profiles.issubset({"TDL-A", "TDL-C", "TDL-D"}))
            heldout_profiles = {
                view["tdl_target_profile"]
                for record in first.manifest["records"]["heldout_channel"]
                for view in record["views"]
            } | {
                view["tdl_jammer_profile"]
                for record in first.manifest["records"]["heldout_channel"]
                for view in record["views"]
            }
            self.assertTrue(heldout_profiles.issubset({"TDL-B", "TDL-E"}))
            self.assertGreater(len(heldout_profiles), 0)
            for split in ("train", "validation", "heldout_channel"):
                for record in first.manifest["records"][split]:
                    for view in record["views"]:
                        self.assertNotEqual(
                            view["tdl_target_seed"], view["tdl_jammer_seed"]
                        )
                        self.assertEqual(view["tdl_guard_samples"], 48)
                        self.assertGreaterEqual(
                            view["tdl_guard_margin_samples"], 0
                        )
                        self.assertEqual(
                            view["tdl_crop_stop_sample"]
                            - view["tdl_crop_start_sample"],
                            96,
                        )


if __name__ == "__main__":
    unittest.main()
