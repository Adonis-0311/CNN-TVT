"""Safety tests for the explicitly non-upload-ready handoff package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import unittest
import zipfile

from tvt_submission import build_pre_submission_package as packager


class PreSubmissionPackageTest(unittest.TestCase):
    def test_collection_is_whitelisted_and_excludes_evidence_outputs(self) -> None:
        files = packager._collect()
        relative = [packager._safe_relative(path) for path in files]
        self.assertIn("paper/main.tex", relative)
        self.assertIn("paper/IEEEtran.cls", relative)
        self.assertIn("tvt_submission/run_local.ps1", relative)
        self.assertNotIn("paper/release_lock.json", relative)
        for token in relative:
            parts = PurePosixPath(token).parts
            self.assertNotEqual(parts[0], "artifacts")
            self.assertNotEqual(parts[0], "diagnostics")
            self.assertNotEqual(parts[0], "tmp")
            self.assertNotIn("cache_factor_screening_1024_v1", parts)
            self.assertNotIn("cache_factor_headline_1024_v1", parts)

    def test_manifest_is_explicitly_not_upload_ready(self) -> None:
        manifest = packager.build_manifest(packager._collect())
        self.assertEqual(manifest["schema_version"], packager.SCHEMA)
        self.assertFalse(manifest["upload_ready"])
        self.assertTrue(manifest["reason_not_upload_ready"])
        self.assertFalse(manifest["long_running_work_started_by_packager"])
        records = manifest["files"]
        self.assertEqual(
            [record["path"] for record in records],
            sorted(record["path"] for record in records),
        )
        self.assertEqual(
            len(records),
            len({record["path"] for record in records}),
        )
        for record in records:
            self.assertRegex(record["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(record["bytes"], 0)

    def test_archive_is_repeatable_and_paths_are_safe(self) -> None:
        files = packager._collect()
        manifest = packager.build_manifest(files)
        with tempfile.TemporaryDirectory(
            prefix="vimd_pre_submission_package_"
        ) as temporary:
            output = Path(temporary) / "handoff.zip"
            first = packager.write_archive(
                output,
                files,
                manifest,
                replace=False,
            )
            first_bytes = output.read_bytes()
            second = packager.write_archive(
                output,
                files,
                manifest,
                replace=True,
            )
            self.assertEqual(first["archive_sha256"], second["archive_sha256"])
            self.assertEqual(first_bytes, output.read_bytes())
            self.assertEqual(
                first["archive_sha256"],
                hashlib.sha256(first_bytes).hexdigest(),
            )
            with zipfile.ZipFile(output) as archive:
                names = archive.namelist()
                self.assertEqual(len(names), len(set(names)))
                for name in names:
                    token = PurePosixPath(name)
                    self.assertFalse(token.is_absolute())
                    self.assertNotIn("..", token.parts)
                payload = json.loads(
                    archive.read(
                        "TVT_VIMD_Net/PRE_SUBMISSION_MANIFEST.json"
                    )
                )
            self.assertFalse(payload["upload_ready"])


if __name__ == "__main__":
    unittest.main()
