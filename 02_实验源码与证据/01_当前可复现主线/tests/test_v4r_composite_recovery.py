from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tvt_submission import build_v4r_composite as composite


def test_not_ready_build_does_not_create_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "must_not_exist"
    state = {
        "ready": False,
        "status": "not_ready",
        "output_root": output,
        "missing_workers": [17],
        "invalid_workers": [],
        "config": {},
        "source_run": {},
        "workers": {},
        "input_records": [],
        "grid": [],
    }
    monkeypatch.setattr(composite, "preflight", lambda _path: state)

    result = composite.build(tmp_path / "config.json")

    assert result["status"] == "not_ready"
    assert result["missing_workers"] == [17]
    assert not output.exists()


def test_descriptor_requires_exact_path_and_hash(tmp_path: Path) -> None:
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"sealed checkpoint")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()

    path, observed = composite._descriptor_file(
        tmp_path,
        {
            "path": "model.pt",
            "sha256": digest,
            "size_bytes": artifact.stat().st_size,
        },
        label="checkpoint",
    )

    assert path == artifact
    assert observed == digest
    with pytest.raises(composite.CompositeError, match="SHA-256 drift"):
        composite._descriptor_file(
            tmp_path,
            {
                "path": "model.pt",
                "sha256": "0" * 64,
                "size_bytes": artifact.stat().st_size,
            },
            label="checkpoint",
        )


def test_expected_grid_preserves_frozen_model_seed_order() -> None:
    run = {"models": ["a0", "cssl"], "seeds": [17, 29]}

    assert composite._expected_grid(run) == [
        ("a0", 17),
        ("a0", 29),
        ("cssl", 17),
        ("cssl", 29),
    ]


def test_formal_ablation_uses_freeze_bound_seed_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        composite.runner,
        "FORMAL_ABLATION_ALGORITHM_SEEDS",
        (17, 29, 43, 71, 101, 131, 173, 211, 257, 307),
    )
    headline_seeds = [17, 29, 43, 71, 101, 131, 173, 211, 257, 307]

    assert composite._formal_ablation_seeds(headline_seeds) == [
        17,
        29,
        43,
        71,
        101,
        131,
        173,
        211,
        257,
        307,
    ]
    with pytest.raises(composite.CompositeError, match="missing preregistered"):
        composite._formal_ablation_seeds(headline_seeds[:-1])


def test_materialization_does_not_claim_derived_result_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    model_root = source_root / "models" / "a0_seed17"
    model_root.mkdir(parents=True)
    (model_root / "model.pt").write_bytes(b"checkpoint")
    (model_root / "predictions_validation.npz").write_bytes(b"prediction")
    (model_root / "result.json").write_text("{}", encoding="utf-8")
    staging = tmp_path / "staging"
    staging.mkdir()
    state = {
        "source_run": {
            "results": [{"model": "a0", "seed": 17}],
        },
        "source_root": source_root,
        "workers": {},
        "config": {"missing_fit_plan": {"model": "cssl"}},
        "grid": [("a0", 17)],
        "splits": ["validation"],
    }
    monkeypatch.setattr(composite, "_relative", lambda path: path.as_posix())

    records = composite._materialize_inputs(state, staging)

    assert all("composite_path" in record for record in records)
    assert not any(
        record.get("materialization") == "derived_result_written_later"
        for record in records
    )
