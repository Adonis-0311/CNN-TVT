"""Offline tests for the read-only TVT paper-build gate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from tvt_submission import validate_paper_build as gate


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class PaperBuildGateTest(unittest.TestCase):
    def _results_text(
        self,
        *,
        placeholders: bool,
        release_sentinel: bool,
    ) -> str:
        values = {name: "--" for name in gate.RESULT_MACROS}
        if not placeholders:
            for name in gate.RESULT_MACROS:
                if name in gate.TEXT_RESULT_MACROS:
                    values[name] = "fixture text"
                elif name.endswith(("Accuracy", "MacroFOne", "WorstRecall")):
                    values[name] = "70.00"
                elif name.endswith("NLL"):
                    values[name] = "0.50"
                elif name.endswith("ECE"):
                    values[name] = "0.05"
                elif name.startswith("Regime") and name.endswith("Reference"):
                    values[name] = "70.00"
                elif name.startswith("Regime") and name.endswith("AFive"):
                    values[name] = "72.00"
                elif name.startswith("Regime") and name.endswith("Gain"):
                    values[name] = "2.00"
                elif name.startswith("Regime") and name.endswith("CILow"):
                    values[name] = "1.00"
                elif name.startswith("Regime") and name.endswith("CIHigh"):
                    values[name] = "3.00"
                else:
                    values[name] = "0.50"
        values["ResultSource"] = (
            "No eligible locked run"
            if placeholders
            else r"locked formal run \texttt{fixture}"
        )
        if "PrimaryReference" in values:
            values["PrimaryReference"] = (
                "--" if placeholders else gate.PRIMARY_REFERENCE_VALUE
            )
        if "VIMDLatencyDevice" in values:
            values["VIMDLatencyDevice"] = (
                "--" if placeholders else "CPU fixture device"
            )
        if "VIMDParameters" in values and not placeholders:
            values["VIMDParameters"] = "39500"
        if "VIMDLatencyP50" in values and not placeholders:
            values["VIMDLatencyP50"] = "2.00"
        if "VIMDLatencyP95" in values and not placeholders:
            values["VIMDLatencyP95"] = "3.00"
        lines = ["% Synthetic result-macro fixture."]
        lines.extend(
            rf"\newcommand{{\{name}}}{{{values[name]}}}"
            for name in gate.RESULT_MACROS
        )
        if release_sentinel:
            lines.append(
                rf"\newcommand{{\{gate.RELEASE_SENTINEL}}}"
                rf"{{{gate.RELEASE_SENTINEL_VALUE}}}"
            )
        return "\n".join(lines) + "\n"

    def _write_lock(
        self,
        paper_root: Path,
        *,
        results_path: Path | None = None,
    ) -> None:
        results = (
            paper_root / "results_auto.tex"
            if results_path is None
            else results_path
        )
        payload = {
            "schema_version": gate.RELEASE_LOCK_SCHEMA,
            "submission_unlocked": True,
            "generated_utc": "2026-07-28T00:00:00+00:00",
            "run_id": "formal_fixture",
            "cache_digest": "a" * 64,
            "formal_cache_designation": (
                gate.release_contract.FORMAL_DESIGNATION
            ),
            "release_sentinel_name": gate.RELEASE_SENTINEL,
            "release_sentinel_value": gate.RELEASE_SENTINEL_VALUE,
            "results_auto_sha256": hashlib.sha256(
                results.read_bytes()
            ).hexdigest(),
            "run_json_sha256": "b" * 64,
            "macro_value_manifest_sha256": "c" * 64,
            "source_gate_sha256": "d" * 64,
            "artifact_audit": {"passed": True},
            "macro_provenance": {
                name: {
                    "source_artifact": "metrics.csv",
                    "source_sha256": "e" * 64,
                    "derivation": f"Deterministic fixture derivation for {name}",
                }
                for name in gate.release_contract.RESULT_MACROS
            },
        }
        (paper_root / "release_lock.json").write_text(
            json.dumps(payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _write_log(
        self,
        path: Path,
        *,
        pdf_bytes: int,
        pages: int,
        extra: str = "",
        output_record: bool = True,
    ) -> None:
        lines = [
            "This is pdfTeX, Version fixture",
            extra,
        ]
        if output_record:
            # Deliberately reproduce MiKTeX's wrapped "pages" token.
            lines.extend(
                [
                    (
                        "Output written on C:\\fixture\\main.pdf "
                        f"({pages} pag"
                    ),
                    f"es, {pdf_bytes} bytes).",
                    "PDF statistics:",
                ]
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _pdf_bytes(self, *, pages: int, text: str = "Fixture") -> bytes:
        self.assertGreater(pages, 0)
        font_id = 3
        page_ids = [4 + 2 * index for index in range(pages)]
        content_ids = [page_id + 1 for page_id in page_ids]
        objects: dict[int, bytes] = {
            1: b"<< /Type /Catalog /Pages 2 0 R >>",
            2: (
                b"<< /Type /Pages /Count "
                + str(pages).encode("ascii")
                + b" /Kids ["
                + b" ".join(
                    f"{page_id} 0 R".encode("ascii") for page_id in page_ids
                )
                + b"] >>"
            ),
            font_id: (
                b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
            ),
        }
        escaped = (
            text.replace("\\", "\\\\")
            .replace("(", "\\(")
            .replace(")", "\\)")
        )
        for page_id, content_id in zip(page_ids, content_ids):
            stream = (
                f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET\n"
            ).encode("ascii")
            objects[page_id] = (
                b"<< /Type /Page /Parent 2 0 R "
                b"/MediaBox [0 0 612 792] "
                b"/Resources << /Font << /F1 3 0 R >> >> "
                + f"/Contents {content_id} 0 R >>".encode("ascii")
            )
            objects[content_id] = (
                f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
                + stream
                + b"endstream"
            )
        output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for object_id in range(1, max(objects) + 1):
            offsets.append(len(output))
            output.extend(f"{object_id} 0 obj\n".encode("ascii"))
            output.extend(objects[object_id])
            output.extend(b"\nendobj\n")
        xref = len(output)
        output.extend(
            f"xref\n0 {len(offsets)}\n".encode("ascii")
        )
        output.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
        output.extend(
            (
                f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
                f"startxref\n{xref}\n%%EOF\n"
            ).encode("ascii")
        )
        return bytes(output)

    def _write_fls(
        self,
        paper_root: Path,
        *,
        inputs: list[Path] | None = None,
    ) -> None:
        build_root = paper_root / "build"
        selected_inputs = (
            [paper_root / "main.tex", paper_root / "results_auto.tex"]
            if inputs is None
            else inputs
        )
        lines = [f"PWD {paper_root.resolve()}"]
        lines.extend(f"INPUT {path.resolve()}" for path in selected_inputs)
        lines.extend(
            [
                f"OUTPUT {(build_root / 'main.log').resolve()}",
                f"OUTPUT {(build_root / 'main.pdf').resolve()}",
            ]
        )
        (build_root / "main.fls").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

    def _set_fixture_times(
        self,
        paper_root: Path,
        *,
        source_time: float,
        build_time: float,
    ) -> None:
        for path in paper_root.rglob("*"):
            if not path.is_file() or path.is_relative_to(paper_root / "build"):
                continue
            os.utime(
                path,
                (source_time, source_time),
            )
        for name in ("main.log", "main.pdf", "main.fls"):
            os.utime(
                paper_root / "build" / name,
                (build_time, build_time),
            )

    def _fixture(
        self,
        root: Path,
        *,
        internal_review: bool = True,
        placeholders: bool = True,
        release_sentinel: bool = False,
        release_lock: bool = False,
        pages: int = 7,
        log_extra: str = "",
        output_record: bool = True,
        pdf_text: str = "Fixture",
    ) -> Path:
        paper_root = root / "paper"
        build_root = paper_root / "build"
        build_root.mkdir(parents=True)
        review_token = "true" if internal_review else "false"
        public_macro_references = " ".join(
            rf"\{name}{{}}" for name in gate.RESULT_MACROS
        )
        (paper_root / "main.tex").write_text(
            "\n".join(
                [
                    r"\documentclass{article}",
                    r"\newif\ifinternalreview",
                    rf"\internalreview{review_token}",
                    r"\input{results_auto.tex}",
                    r"\begin{document}",
                    "Fixture",
                    r"\ifinternalreview\else",
                    public_macro_references,
                    r"\fi",
                    r"\end{document}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (paper_root / "results_auto.tex").write_text(
            self._results_text(
                placeholders=placeholders,
                release_sentinel=release_sentinel,
            ),
            encoding="utf-8",
        )
        if release_lock:
            self._write_lock(paper_root)
        pdf = build_root / "main.pdf"
        pdf.write_bytes(self._pdf_bytes(pages=pages, text=pdf_text))
        self._write_log(
            build_root / "main.log",
            pdf_bytes=pdf.stat().st_size,
            pages=pages,
            extra=log_extra,
            output_record=output_record,
        )
        self._write_fls(paper_root)
        base = time.time() - 1_000
        self._set_fixture_times(
            paper_root,
            source_time=base,
            build_time=base + 10,
        )
        return paper_root

    def _checks(self, report: dict) -> dict[str, dict]:
        return {record["id"]: record for record in report["checks"]}

    def test_internal_build_accepts_safe_placeholders_without_writing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="vimd_paper_internal_"
        ) as temporary:
            paper_root = self._fixture(Path(temporary))
            before = {
                path.relative_to(paper_root): (
                    path.stat().st_mtime_ns,
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
                for path in paper_root.rglob("*")
                if path.is_file()
            }
            report = gate.audit_paper_build(
                paper_root=paper_root,
                mode="internal",
            )
            after = {
                path.relative_to(paper_root): (
                    path.stat().st_mtime_ns,
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
                for path in paper_root.rglob("*")
                if path.is_file()
            }
            self.assertTrue(report["ok"], report["issues"])
            self.assertTrue(report["internal_build_validated"])
            self.assertFalse(report["release_eligible"])
            self.assertEqual(report["page_count"], 7)
            self.assertEqual(
                set(report["placeholder_macros"]),
                set(gate.RESULT_MACROS),
            )
            self.assertEqual(before, after)

    def test_release_build_accepts_locked_nonplaceholder_fourteen_pages(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="vimd_paper_release_"
        ) as temporary:
            paper_root = self._fixture(
                Path(temporary),
                internal_review=False,
                placeholders=False,
                release_sentinel=True,
                release_lock=True,
                pages=14,
            )
            report = gate.audit_paper_build(
                paper_root=paper_root,
                mode="release",
            )
            self.assertTrue(report["ok"], report["issues"])
            self.assertTrue(report["release_eligible"])
            self.assertFalse(report["internal_review"])
            self.assertEqual(report["placeholder_macros"], [])
            self.assertTrue(report["release_sentinel_defined"])
            self.assertTrue(
                self._checks(report)["release_lock_results_digest"]["passed"]
            )

    def test_log_scientific_and_layout_failures_are_rejected(self) -> None:
        cases = {
            "fatal": (
                "! Undefined control sequence.",
                "log_no_fatal_errors",
            ),
            "citation": (
                "LaTeX Warning: Citation `missing' on page 1 undefined.",
                "log_no_undefined_citations",
            ),
            "reference": (
                "LaTeX Warning: There were undefined references.",
                "log_no_undefined_references",
            ),
            "overfull": (
                r"Overfull \hbox (10.0pt too wide) in paragraph.",
                "log_no_overfull_boxes",
            ),
            "rerun": (
                "LaTeX Warning: Label(s) may have changed. Rerun to get "
                "cross-references right.",
                "log_no_rerun_required",
            ),
        }
        for name, (log_extra, check_id) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory(
                prefix=f"vimd_paper_log_{name}_"
            ) as temporary:
                paper_root = self._fixture(
                    Path(temporary),
                    log_extra=log_extra,
                )
                report = gate.audit_paper_build(
                    paper_root=paper_root,
                    mode="internal",
                )
                self.assertFalse(report["ok"])
                self.assertFalse(self._checks(report)[check_id]["passed"])

    def test_missing_unparseable_and_mismatched_outputs_are_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="vimd_paper_missing_log_"
        ) as temporary:
            paper_root = self._fixture(Path(temporary))
            (paper_root / "build" / "main.log").unlink()
            report = gate.audit_paper_build(
                paper_root=paper_root,
                mode="internal",
            )
            self.assertFalse(report["ok"])
            self.assertFalse(
                self._checks(report)["artifact_latex_log"]["passed"]
            )

        with tempfile.TemporaryDirectory(
            prefix="vimd_paper_unparseable_log_"
        ) as temporary:
            paper_root = self._fixture(
                Path(temporary),
                output_record=False,
            )
            report = gate.audit_paper_build(
                paper_root=paper_root,
                mode="internal",
            )
            self.assertFalse(report["ok"])
            self.assertFalse(
                self._checks(report)["log_output_record"]["passed"]
            )

        with tempfile.TemporaryDirectory(
            prefix="vimd_paper_mismatch_"
        ) as temporary:
            paper_root = self._fixture(Path(temporary))
            pdf = paper_root / "build" / "main.pdf"
            original_time = pdf.stat().st_mtime
            pdf.write_bytes(b"not-a-pdf-and-a-different-size")
            os.utime(pdf, (original_time, original_time))
            report = gate.audit_paper_build(
                paper_root=paper_root,
                mode="internal",
            )
            self.assertFalse(report["ok"])
            checks = self._checks(report)
            self.assertFalse(checks["log_pdf_byte_match"]["passed"])
            self.assertFalse(checks["pdf_signature"]["passed"])

    def test_source_newer_than_build_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="vimd_paper_stale_"
        ) as temporary:
            paper_root = self._fixture(Path(temporary))
            build_time = (paper_root / "build" / "main.log").stat().st_mtime
            os.utime(
                paper_root / "main.tex",
                (build_time + 30, build_time + 30),
            )
            report = gate.audit_paper_build(
                paper_root=paper_root,
                mode="internal",
            )
            self.assertFalse(report["ok"])
            self.assertFalse(
                self._checks(report)["freshness_main_tex"]["passed"]
            )

    def test_release_rejects_wrong_mode_placeholders_missing_lock_and_pages(
        self,
    ) -> None:
        scenarios = (
            {
                "name": "wrong_review_mode",
                "fixture": {
                    "internal_review": True,
                    "placeholders": False,
                    "release_sentinel": True,
                    "release_lock": True,
                },
                "check": "review_mode",
            },
            {
                "name": "placeholders",
                "fixture": {
                    "internal_review": False,
                    "placeholders": True,
                    "release_sentinel": True,
                    "release_lock": True,
                },
                "check": "release_results_nonplaceholder",
            },
            {
                "name": "missing_lock",
                "fixture": {
                    "internal_review": False,
                    "placeholders": False,
                    "release_sentinel": True,
                    "release_lock": False,
                },
                "check": "artifact_release_lock",
            },
            {
                "name": "missing_sentinel",
                "fixture": {
                    "internal_review": False,
                    "placeholders": False,
                    "release_sentinel": False,
                    "release_lock": True,
                },
                "check": "release_sentinel",
            },
            {
                "name": "page_limit",
                "fixture": {
                    "internal_review": False,
                    "placeholders": False,
                    "release_sentinel": True,
                    "release_lock": True,
                    "pages": 15,
                },
                "check": "release_page_limit",
            },
        )
        for scenario in scenarios:
            with self.subTest(
                name=scenario["name"]
            ), tempfile.TemporaryDirectory(
                prefix=f"vimd_paper_release_{scenario['name']}_"
            ) as temporary:
                paper_root = self._fixture(
                    Path(temporary),
                    **scenario["fixture"],
                )
                report = gate.audit_paper_build(
                    paper_root=paper_root,
                    mode="release",
                )
                self.assertFalse(report["ok"])
                self.assertFalse(
                    self._checks(report)[scenario["check"]]["passed"]
                )

    def test_release_lock_must_bind_exact_result_bytes(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="vimd_paper_lock_digest_"
        ) as temporary:
            paper_root = self._fixture(
                Path(temporary),
                internal_review=False,
                placeholders=False,
                release_sentinel=True,
                release_lock=True,
            )
            lock_path = paper_root / "release_lock.json"
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            lock["results_auto_sha256"] = "0" * 64
            lock_path.write_text(
                json.dumps(lock, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            build_time = (
                paper_root / "build" / "main.log"
            ).stat().st_mtime
            os.utime(lock_path, (build_time - 5, build_time - 5))
            report = gate.audit_paper_build(
                paper_root=paper_root,
                mode="release",
            )
            self.assertFalse(report["ok"])
            self.assertFalse(
                self._checks(report)["release_lock_results_digest"]["passed"]
            )

    def test_cli_emits_failure_json_on_stdout_and_nonzero(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="vimd_paper_cli_"
        ) as temporary:
            paper_root = self._fixture(
                Path(temporary),
                output_record=False,
            )
            environment = os.environ.copy()
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(gate.__file__).resolve()),
                    "--paper-root",
                    str(paper_root),
                    "--mode",
                    "internal",
                ],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                env=environment,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(completed.stderr, "")
            payload = json.loads(completed.stdout)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["schema_version"], gate.REPORT_SCHEMA)
            self.assertTrue(payload["read_only"])

    def test_current_internal_build_has_a_machine_readable_audit(self) -> None:
        """Exercise the checked-in build, whether fresh or correctly stale."""

        report = gate.audit_paper_build(
            paper_root=REPOSITORY_ROOT / "paper",
            mode="internal",
        )
        self.assertEqual(report["schema_version"], gate.REPORT_SCHEMA)
        self.assertEqual(report["mode"], "internal")
        self.assertTrue(report["read_only"])
        self.assertFalse(report["compiled_by_validator"])
        self.assertIsInstance(report["page_count"], int)
        self.assertGreater(report["page_count"], 0)
        checks = self._checks(report)
        for check_id in (
            "artifact_main_tex",
            "artifact_results_auto",
            "artifact_latex_log",
            "artifact_pdf",
            "log_output_record",
            "log_pdf_byte_match",
            "pdf_signature",
        ):
            self.assertTrue(checks[check_id]["passed"], report["issues"])
        self.assertEqual(report["ok"], not report["issues"])


if __name__ == "__main__":
    unittest.main()
