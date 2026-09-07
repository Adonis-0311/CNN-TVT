from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest


from tvt_submission import run_v4r_recovery as recovery


class V4RRecoverySchedulerTests(unittest.TestCase):
    def test_canonical_config_is_hash_bound(self) -> None:
        config, digest = recovery.load_recovery_config(recovery.DEFAULT_CONFIG)
        self.assertEqual(digest, recovery.EXPECTED_CONFIG_SHA256)
        self.assertEqual(config["schema_version"], recovery.CONFIG_SCHEMA)

    def test_atomic_json_replaces_complete_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            recovery._atomic_json(path, {"status": "running", "value": 1})
            recovery._atomic_json(path, {"status": "complete", "value": 2})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"status": "complete", "value": 2},
            )
            self.assertEqual(list(path.parent.glob(".state.json.*.tmp")), [])

    def test_safe_relative_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(recovery.RecoveryError):
                recovery._safe_relative(Path(directory), "../escape", "test")

    def test_scheduler_mutex_rejects_second_owner(self) -> None:
        with recovery._exclusive_scheduler_mutex():
            with self.assertRaises(recovery.RecoveryError):
                with recovery._exclusive_scheduler_mutex():
                    self.fail("second scheduler unexpectedly acquired mutex")

    def test_source_run_has_exact_110_plus_10_grid(self) -> None:
        config, _ = recovery.load_recovery_config(recovery.DEFAULT_CONFIG)
        source = recovery._resolve_repo_path(
            config["source_run"]["run_json"], "source run"
        )
        audit = recovery.validate_source_run(config, source)
        self.assertEqual(audit["observed_fit_count"], 110)
        self.assertEqual(len(audit["missing"]), 10)
        self.assertEqual(len(audit["evaluation_regimes"]), 11)
        self.assertTrue(
            all(item.startswith(f"{recovery.MODEL}/seed") for item in audit["missing"])
        )


if __name__ == "__main__":
    unittest.main()
