"""Smoke validation for the MATLAB nrTDLChannel batch bridge."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from vimd_amc.standards import (  # noqa: E402
    NRTDLConfiguration,
    apply_nrtdl_batch,
)


def main() -> None:
    rng = np.random.default_rng(20260727)
    sample_count = 4096
    source = (
        rng.standard_normal(sample_count)
        + 1j * rng.standard_normal(sample_count)
    ) / np.sqrt(2.0)
    waveforms = np.repeat(source[None, :], 8, axis=0)
    configurations = [
        NRTDLConfiguration("TDL-A", 100e-9, 0.0, 5.9e9, 101),
        NRTDLConfiguration("TDL-A", 100e-9, 0.0, 5.9e9, 101),
        NRTDLConfiguration("TDL-A", 100e-9, 0.0, 5.9e9, 102),
        NRTDLConfiguration("TDL-A", 100e-9, 50.0, 5.9e9, 101),
        NRTDLConfiguration("TDL-B", 300e-9, 25.0, 5.9e9, 151),
        NRTDLConfiguration("TDL-C", 300e-9, 20.0, 5.9e9, 201),
        NRTDLConfiguration("TDL-D", 100e-9, 30.0, 5.9e9, 301),
        NRTDLConfiguration("TDL-E", 100e-9, 40.0, 5.9e9, 401),
    ]

    result = apply_nrtdl_batch(
        waveforms,
        configurations,
        sample_rate_hz=1_000_000.0,
    )

    assert result.waveforms.shape == waveforms.shape
    assert np.all(np.isfinite(result.waveforms))
    assert np.all(np.linalg.norm(result.waveforms, axis=1) > 0)
    assert np.array_equal(result.waveforms[0], result.waveforms[1]), (
        "identical profile/speed/seed inputs must be bitwise deterministic"
    )
    assert not np.allclose(result.waveforms[0], result.waveforms[2]), (
        "different seeds must produce different fading realizations"
    )
    assert not np.allclose(result.waveforms[0], result.waveforms[3]), (
        "zero and positive Doppler configurations must differ"
    )
    assert result.metadata[0]["maximum_doppler_hz"] == 0.0
    assert result.metadata[3]["maximum_doppler_hz"] > 0.0
    assert tuple(item["profile"] for item in result.metadata) == (
        "TDL-A",
        "TDL-A",
        "TDL-A",
        "TDL-A",
        "TDL-B",
        "TDL-C",
        "TDL-D",
        "TDL-E",
    )
    assert all(item["sample_rate_hz"] == 1_000_000.0 for item in result.metadata)
    assert all(item["num_transmit_antennas"] == 1 for item in result.metadata)
    assert all(item["num_receive_antennas"] == 1 for item in result.metadata)

    summary = {
        "status": "passed",
        "batch_shape": list(result.waveforms.shape),
        "profiles": [item["profile"] for item in result.metadata],
        "seeds": [item["seed"] for item in result.metadata],
        "doppler_hz": [
            round(float(item["maximum_doppler_hz"]), 6)
            for item in result.metadata
        ],
        "num_paths": [item["num_paths"] for item in result.metadata],
        "output_rms": [
            round(float(item["output_rms"]), 6) for item in result.metadata
        ],
        "matlab_release": result.metadata[0]["matlab_release"],
        "five_g_toolbox_version": result.metadata[0][
            "five_g_toolbox_version"
        ],
        "standard_reference": result.metadata[0]["standard_reference"],
        "deterministic_duplicate_bitwise_equal": bool(
            np.array_equal(result.waveforms[0], result.waveforms[1])
        ),
        "different_seed_distinct": bool(
            not np.allclose(result.waveforms[0], result.waveforms[2])
        ),
        "positive_doppler_distinct": bool(
            not np.allclose(result.waveforms[0], result.waveforms[3])
        ),
    }
    output_path = REPOSITORY_ROOT / "standards" / "smoke_result.json"
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
