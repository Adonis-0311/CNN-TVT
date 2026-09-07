from __future__ import annotations

from pathlib import Path

import pytest

from tvt_submission import recovery_v4r_worker as worker


def test_invalid_recovery_config_sha_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    altered = tmp_path / "formal_tvt_recovery_v4r_110plus10.json"
    altered.write_bytes(worker.CANONICAL_RECOVERY_CONFIG.read_bytes() + b"\n")
    monkeypatch.setattr(worker, "CANONICAL_RECOVERY_CONFIG", altered.resolve())

    with pytest.raises(
        ValueError,
        match="canonical V4R recovery config SHA-256 mismatch",
    ):
        worker._validate_bindings(altered.resolve(), 17)


def test_invalid_seed_fails_before_staging_creation(tmp_path: Path) -> None:
    staging = tmp_path / "must_not_be_created"

    with pytest.raises(ValueError, match="not in the frozen recovery plan"):
        worker.main(
            [
                "--config",
                str(worker.CANONICAL_RECOVERY_CONFIG),
                "--seed",
                "999",
                "--staging-directory",
                str(staging),
                "--preflight-only",
            ]
        )

    assert not staging.exists()
