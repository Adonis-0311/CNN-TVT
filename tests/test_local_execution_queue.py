from __future__ import annotations

import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "tvt_submission" / "run_all_local_after_gpu_free.ps1"
FORMAL = ROOT / "tvt_submission" / "run_local.ps1"
CANDIDATE = ROOT / "tvt_submission" / "run_candidate_local.ps1"
QUEUE_DOC = ROOT / "tvt_submission" / "LOCAL_EXECUTION_QUEUE.md"
FORMAL_DOC = ROOT / "tvt_submission" / "LOCAL_FORMAL_RUN_HANDOFF.md"
SCRIPTS = (QUEUE, FORMAL, CANDIDATE)
DELIVERABLES = (*SCRIPTS, QUEUE_DOC, FORMAL_DOC)
FORMAL_OUTPUT_SENTINELS = (
    ROOT
    / "standards"
    / "cache_factor_headline_1024_v1"
    / "manifest.json",
    ROOT / "artifacts" / "tvt_headline_1024_5seed_v1" / "run.json",
    ROOT / "tvt_submission" / "formal_macro_values.json",
    ROOT / "paper" / "release_lock.json",
)


def _powershell() -> str | None:
    return shutil.which("pwsh") or shutil.which("powershell")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sentinel_state() -> dict[Path, tuple[bool, int | None, int | None]]:
    state: dict[Path, tuple[bool, int | None, int | None]] = {}
    for path in FORMAL_OUTPUT_SENTINELS:
        if path.is_file():
            stat = path.stat()
            state[path] = (True, stat.st_size, stat.st_mtime_ns)
        else:
            state[path] = (False, None, None)
    return state


class LocalExecutionQueueTests(unittest.TestCase):
    def test_all_powershell_entry_points_parse(self) -> None:
        shell = _powershell()
        if shell is None:
            self.skipTest("PowerShell is unavailable")
        for path in SCRIPTS:
            escaped = str(path).replace("'", "''")
            command = (
                "$tokens=$null;$errors=$null;"
                "[void][System.Management.Automation.Language.Parser]::"
                f"ParseFile('{escaped}',[ref]$tokens,[ref]$errors);"
                "if(@($errors).Count -ne 0){"
                "$errors|ForEach-Object{$_.Message}|Write-Error;exit 1}"
            )
            completed = subprocess.run(
                [
                    shell,
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    command,
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
            )
            self.assertEqual(
                completed.returncode,
                0,
                completed.stderr.decode(errors="replace"),
            )

    def test_default_preflight_is_read_only(self) -> None:
        shell = _powershell()
        if shell is None:
            self.skipTest("PowerShell is unavailable")
        before_hashes = {path: _digest(path) for path in DELIVERABLES}
        before_outputs = _sentinel_state()
        completed = subprocess.run(
            [
                shell,
                "-NoProfile",
                "-NonInteractive",
                "-File",
                str(QUEUE),
                "-Plan",
                "preflight",
                "-Python",
                sys.executable,
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
        )
        output = (completed.stdout + completed.stderr).decode(
            errors="replace"
        )
        self.assertEqual(completed.returncode, 0, output)
        self.assertIn(
            "Queue dry-run complete. No cache, simulation, training, "
            "candidate, macro, or release write was started.",
            output,
        )
        self.assertEqual(
            {path: _digest(path) for path in DELIVERABLES},
            before_hashes,
        )
        self.assertEqual(_sentinel_state(), before_outputs)

    def test_every_execute_entry_has_the_same_fail_closed_gates(self) -> None:
        for path in SCRIPTS:
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertIn(
                    "START_TVT_ONLY_WHEN_MACHINE_IS_IDLE",
                    source,
                )
                self.assertIn(
                    "Global\\VIMD_AMC_TVT_LOCAL_EXECUTION_V1",
                    source,
                )
                self.assertIn("Assert-ExecutionAcknowledged", source)
                self.assertIn("Enter-ExecutionMutex", source)
                self.assertIn("Exit-ExecutionMutex", source)
                self.assertIn("Assert-MachineIdle", source)
                self.assertIn("MinimumFreeGpuMiB", source)
                self.assertIn("MinimumFreeDiskGiB", source)
                self.assertIn("Get-ActiveComputeProcesses", source)
                self.assertIn("soffice", source)
                self.assertIn(r"\.bin", source)
                self.assertNotIn("$ProjectToken", source)
                guard = source.rindex("\nAssert-ExecutionAcknowledged\n")
                lock = source.rindex("\nEnter-ExecutionMutex\n")
                self.assertLess(guard, lock)

    def test_reuse_is_explicit_digest_pinned_and_macro_path_is_unique(
        self,
    ) -> None:
        queue = QUEUE.read_text(encoding="utf-8")
        formal = FORMAL.read_text(encoding="utf-8")
        self.assertIn("[switch]$AllowValidatedReuse", queue)
        self.assertIn("[switch]$AllowValidatedReuse", formal)
        self.assertIn("ExpectedCacheManifestSha256", queue)
        self.assertIn("ExpectedRunJsonSha256", queue)
        self.assertIn("Assert-PinnedFileDigest", formal)
        self.assertIn("Assert-RunRecordEligible", formal)
        self.assertIn(
            '$MacroValues = Join-Path $PackageRoot "formal_macro_values.json"',
            queue,
        )
        self.assertNotIn(
            '$MacroValues = Join-Path $RunRoot "formal_macro_values.json"',
            queue,
        )

    def test_documented_execute_blocks_include_safety_parameters(self) -> None:
        for path in (QUEUE_DOC, FORMAL_DOC):
            source = path.read_text(encoding="utf-8")
            blocks = re.findall(
                r"```powershell\s*(.*?)```",
                source,
                flags=re.DOTALL,
            )
            execute_blocks = [block for block in blocks if "-Execute" in block]
            self.assertGreater(len(execute_blocks), 0, path.name)
            for block in execute_blocks:
                with self.subTest(path=path.name, block=block[:80]):
                    self.assertIn("-Acknowledgement", block)
                    self.assertIn("-MinimumFreeGpuMiB", block)
                    self.assertIn("-MinimumFreeDiskGiB", block)

    def test_handoff_matches_current_reference_and_macro_contract(self) -> None:
        source = FORMAL_DOC.read_text(encoding="utf-8")
        self.assertIn(
            "固定主参考是 `cssl_amc_supervised_adaptation`",
            source,
        )
        self.assertIn("97 个 provenance 宏", source)
        self.assertIn("共 98 个", source)
        self.assertIn("共 99 个", source)
        self.assertIn("`PFifty`", source)
        self.assertIn("`PNinetyFive`", source)
        self.assertIn("分别高于 A0、MCLDNN", source)
        self.assertIn("`scientific_release_gate`", source)
        self.assertIn("ablation_paired_statistics.csv", source)
        self.assertIn("A3-A2", source)
        self.assertIn("A4-A3", source)
        self.assertIn("A5-A4", source)
        self.assertIn("A5-A1", source)
        self.assertIn("A5-A6", source)
        self.assertIn("A5-A7", source)
        self.assertIn(
            "joint_max_absolute_centered_deviation_"
            "hierarchical_paired_bootstrap",
            source,
        )
        self.assertIn("正式总数仍为 55", source)

    def test_write_release_without_execute_is_blocked_and_read_only(
        self,
    ) -> None:
        shell = _powershell()
        if shell is None:
            self.skipTest("PowerShell is unavailable")
        before_outputs = _sentinel_state()
        completed = subprocess.run(
            [
                shell,
                "-NoProfile",
                "-NonInteractive",
                "-File",
                str(QUEUE),
                "-Plan",
                "formal",
                "-WriteRelease",
                "-Python",
                sys.executable,
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
        )
        output = (completed.stdout + completed.stderr).decode(
            errors="replace"
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("WriteRelease is invalid without Execute", output)
        self.assertEqual(_sentinel_state(), before_outputs)


if __name__ == "__main__":
    unittest.main()
