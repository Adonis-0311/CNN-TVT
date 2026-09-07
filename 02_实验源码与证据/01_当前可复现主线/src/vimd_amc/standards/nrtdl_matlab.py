"""Offline Python bridge to MATLAB 5G Toolbox ``nrTDLChannel``.

The bridge deliberately batches waveforms into a single MATLAB process.  It is
intended for deterministic, auditable dataset-cache generation rather than
online use in a PyTorch data-loader worker.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Sequence

import numpy as np
from scipy.io import loadmat, savemat


_PROFILE_CODES = {
    "TDL-A": 0,
    "TDL-B": 1,
    "TDL-C": 2,
    "TDL-D": 3,
    "TDL-E": 4,
}
_SPEED_OF_LIGHT_MPS = 299_792_458.0


@dataclass(frozen=True)
class NRTDLConfiguration:
    """Per-waveform 3GPP TDL configuration."""

    profile: str
    delay_spread_s: float
    speed_mps: float
    carrier_frequency_hz: float
    seed: int

    @property
    def maximum_doppler_hz(self) -> float:
        return (
            abs(float(self.speed_mps))
            * float(self.carrier_frequency_hz)
            / _SPEED_OF_LIGHT_MPS
        )

    def validate(self) -> None:
        if self.profile not in _PROFILE_CODES:
            raise ValueError(
                f"Unsupported profile {self.profile!r}; "
                f"choose one of {tuple(_PROFILE_CODES)}."
            )
        if not np.isfinite(self.delay_spread_s) or self.delay_spread_s <= 0:
            raise ValueError("delay_spread_s must be positive and finite.")
        if not np.isfinite(self.speed_mps) or self.speed_mps < 0:
            raise ValueError("speed_mps must be nonnegative and finite.")
        if (
            not np.isfinite(self.carrier_frequency_hz)
            or self.carrier_frequency_hz <= 0
        ):
            raise ValueError("carrier_frequency_hz must be positive and finite.")
        if not isinstance(self.seed, (int, np.integer)):
            raise TypeError("seed must be an integer.")
        if not 0 <= int(self.seed) <= np.iinfo(np.uint32).max:
            raise ValueError("seed must lie in the uint32 range.")


@dataclass(frozen=True)
class NRTDLBatchResult:
    """Channelized waveforms and MATLAB-returned audit metadata."""

    waveforms: np.ndarray
    metadata: tuple[dict[str, object], ...]
    matlab_stdout: str


def _as_matlab_literal(path: Path) -> str:
    """Return a path as a safely quoted MATLAB character vector literal."""
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def _find_matlab(explicit: str | os.PathLike[str] | None) -> str:
    if explicit is not None:
        candidate = Path(explicit)
        if candidate.is_file():
            return str(candidate.resolve())
        resolved = shutil.which(str(explicit))
        if resolved is not None:
            return resolved
        raise FileNotFoundError(f"MATLAB executable was not found: {explicit}")
    resolved = shutil.which("matlab")
    if resolved is None:
        raise FileNotFoundError(
            "MATLAB is not on PATH. Pass matlab_executable or set up PATH."
        )
    return resolved


def _matlab_backend_dir() -> Path:
    """Locate the repository-owned MATLAB function from an editable checkout."""
    package_file = Path(__file__).resolve()
    repository_root = package_file.parents[3]
    candidate = repository_root / "standards" / "matlab"
    if not (candidate / "vimd_apply_nrtdl_batch.m").is_file():
        raise FileNotFoundError(
            "MATLAB backend not found at expected path: "
            f"{candidate / 'vimd_apply_nrtdl_batch.m'}"
        )
    return candidate


def _column(raw: np.ndarray, count: int, name: str) -> np.ndarray:
    values = np.asarray(raw).reshape(-1)
    if values.size != count:
        raise RuntimeError(
            f"MATLAB output {name!r} has {values.size} entries; expected {count}."
        )
    return values


def _matlab_text(raw: object) -> str:
    """Normalize scipy's MATLAB char/cell representations to one string."""
    value = raw
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"U", "S"}:
            return "".join(str(item) for item in value.reshape(-1)).strip()
        return str(value.tolist())
    return str(value).strip()


def _cell_numeric(raw: np.ndarray, count: int, name: str) -> tuple[np.ndarray, ...]:
    cells = np.asarray(raw, dtype=object).reshape(-1)
    if cells.size != count:
        raise RuntimeError(
            f"MATLAB output {name!r} has {cells.size} entries; expected {count}."
        )
    return tuple(np.asarray(cell, dtype=np.float64).reshape(-1) for cell in cells)


def apply_nrtdl_batch(
    waveforms: np.ndarray,
    configurations: Sequence[NRTDLConfiguration],
    *,
    sample_rate_hz: float = 1_000_000.0,
    matlab_executable: str | os.PathLike[str] | None = None,
    timeout_s: float = 300.0,
    work_directory: str | os.PathLike[str] | None = None,
) -> NRTDLBatchResult:
    """Apply independent SISO ``nrTDLChannel`` objects to a waveform batch.

    Parameters
    ----------
    waveforms:
        Complex array shaped ``[batch, samples]``.
    configurations:
        Exactly one channel configuration per waveform.
    sample_rate_hz:
        Common sample rate.  The current evidence prototype uses 1 MHz.
    matlab_executable:
        Optional path/name for ``matlab``.  PATH lookup is the default.
    timeout_s:
        End-to-end timeout for the single MATLAB batch process.
    work_directory:
        Optional directory in which temporary input/output MAT files are
        created.  When omitted, a system temporary directory is used and
        removed automatically.
    """
    samples = np.asarray(waveforms)
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] < 2:
        raise ValueError("waveforms must have shape [batch, samples>=2].")
    if not np.iscomplexobj(samples):
        raise TypeError("waveforms must be a complex-valued array.")
    if not np.all(np.isfinite(samples.real)) or not np.all(
        np.isfinite(samples.imag)
    ):
        raise ValueError("waveforms contain nonfinite samples.")
    if not np.any(np.abs(samples) > 0):
        raise ValueError("the complete waveform batch is zero.")
    if len(configurations) != samples.shape[0]:
        raise ValueError(
            "configurations must contain exactly one entry per waveform."
        )
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive and finite.")
    for configuration in configurations:
        configuration.validate()

    matlab = _find_matlab(matlab_executable)
    backend_dir = _matlab_backend_dir()
    owns_tempdir = work_directory is None
    if owns_tempdir:
        temporary_context = tempfile.TemporaryDirectory(
            prefix="vimd_nrtdl_"
        )
        transfer_dir = Path(temporary_context.name)
    else:
        temporary_context = None
        transfer_dir = Path(work_directory).resolve()
        transfer_dir.mkdir(parents=True, exist_ok=True)

    try:
        input_path = transfer_dir / "nrtdl_input.mat"
        output_path = transfer_dir / "nrtdl_output.mat"
        savemat(
            input_path,
            {
                # MATLAB/system objects use [time, channels/batch].
                "waveforms": np.asarray(samples.T, dtype=np.complex128),
                "profile_codes": np.array(
                    [_PROFILE_CODES[c.profile] for c in configurations],
                    dtype=np.int32,
                ),
                "delay_spread_s": np.array(
                    [c.delay_spread_s for c in configurations],
                    dtype=np.float64,
                ),
                "speed_mps": np.array(
                    [c.speed_mps for c in configurations],
                    dtype=np.float64,
                ),
                "carrier_frequency_hz": np.array(
                    [c.carrier_frequency_hz for c in configurations],
                    dtype=np.float64,
                ),
                "seeds": np.array(
                    [c.seed for c in configurations], dtype=np.uint32
                ),
                "sample_rate_hz": np.array(
                    [[sample_rate_hz]], dtype=np.float64
                ),
            },
            do_compression=True,
            oned_as="column",
        )
        if output_path.exists():
            output_path.unlink()

        expression = (
            f"addpath({_as_matlab_literal(backend_dir)});"
            f"vimd_apply_nrtdl_batch({_as_matlab_literal(input_path)},"
            f"{_as_matlab_literal(output_path)});"
        )
        completed = subprocess.run(
            [matlab, "-batch", expression],
            cwd=str(transfer_dir),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        combined_output = "\n".join(
            item for item in (completed.stdout, completed.stderr) if item
        ).strip()
        if completed.returncode != 0:
            raise RuntimeError(
                "MATLAB nrTDLChannel batch failed with exit code "
                f"{completed.returncode}:\n{combined_output}"
            )
        if not output_path.is_file():
            raise RuntimeError(
                "MATLAB exited successfully but did not create the output MAT "
                f"file. MATLAB output:\n{combined_output}"
            )

        returned = loadmat(output_path, squeeze_me=False, struct_as_record=False)
        expected_shape = (samples.shape[1], samples.shape[0])
        matlab_waveforms = np.asarray(returned["channelizedWaveforms"])
        if matlab_waveforms.shape != expected_shape:
            raise RuntimeError(
                "MATLAB returned channelizedWaveforms with shape "
                f"{matlab_waveforms.shape}; expected {expected_shape}."
            )
        output_waveforms = np.asarray(
            matlab_waveforms.T, dtype=np.complex128
        )
        if not np.all(np.isfinite(output_waveforms)):
            raise RuntimeError("MATLAB returned nonfinite waveform samples.")

        count = samples.shape[0]
        profile_codes = _column(
            returned["metadataProfileCodes"], count, "metadataProfileCodes"
        ).astype(int)
        profile_cells = np.asarray(
            returned["metadataProfileNames"], dtype=object
        ).reshape(-1)
        if profile_cells.size != count:
            raise RuntimeError("metadataProfileNames has an invalid length.")
        profile_names = tuple(_matlab_text(item) for item in profile_cells)
        delay_spreads = _column(
            returned["metadataDelaySpreadS"], count, "metadataDelaySpreadS"
        )
        speeds = _column(returned["metadataSpeedMps"], count, "metadataSpeedMps")
        carriers = _column(
            returned["metadataCarrierFrequencyHz"],
            count,
            "metadataCarrierFrequencyHz",
        )
        dopplers = _column(
            returned["metadataMaximumDopplerHz"],
            count,
            "metadataMaximumDopplerHz",
        )
        seeds = _column(returned["metadataSeeds"], count, "metadataSeeds").astype(
            np.uint64
        )
        path_delays = _cell_numeric(
            returned["metadataPathDelaysS"], count, "metadataPathDelaysS"
        )
        path_gains = _cell_numeric(
            returned["metadataAveragePathGainsDb"],
            count,
            "metadataAveragePathGainsDb",
        )
        num_paths = _column(
            returned["metadataNumPaths"], count, "metadataNumPaths"
        ).astype(int)
        filter_delays = _column(
            returned["metadataChannelFilterDelaySamples"],
            count,
            "metadataChannelFilterDelaySamples",
        ).astype(int)
        maximum_delays = _column(
            returned["metadataMaximumChannelDelaySamples"],
            count,
            "metadataMaximumChannelDelaySamples",
        ).astype(int)
        input_rms = _column(
            returned["metadataInputRms"], count, "metadataInputRms"
        )
        output_rms = _column(
            returned["metadataOutputRms"], count, "metadataOutputRms"
        )

        common = {
            "sample_rate_hz": float(
                np.asarray(returned["metadataSampleRateHz"]).squeeze()
            ),
            "num_transmit_antennas": int(
                np.asarray(returned["metadataNumTransmitAntennas"]).squeeze()
            ),
            "num_receive_antennas": int(
                np.asarray(returned["metadataNumReceiveAntennas"]).squeeze()
            ),
            "normalize_path_gains": bool(
                np.asarray(returned["metadataNormalizePathGains"]).squeeze()
            ),
            "normalize_channel_outputs": bool(
                np.asarray(
                    returned["metadataNormalizeChannelOutputs"]
                ).squeeze()
            ),
            "channel_filtering": bool(
                np.asarray(returned["metadataChannelFiltering"]).squeeze()
            ),
            "initial_time_s": float(
                np.asarray(returned["metadataInitialTimeS"]).squeeze()
            ),
            "random_stream": _matlab_text(returned["metadataRandomStream"]),
            "standard_reference": _matlab_text(
                returned["metadataStandardReference"]
            ),
            "channel_class": _matlab_text(returned["metadataChannelClass"]),
            "matlab_release": _matlab_text(returned["metadataMatlabRelease"]),
            "five_g_toolbox_version": _matlab_text(
                returned["metadataFiveGToolboxVersion"]
            ),
            "generated_utc": _matlab_text(returned["metadataGeneratedUtc"]),
        }
        metadata: list[dict[str, object]] = []
        for index in range(count):
            metadata.append(
                {
                    **common,
                    "profile_code": int(profile_codes[index]),
                    "profile": profile_names[index],
                    "delay_spread_s": float(delay_spreads[index]),
                    "speed_mps": float(speeds[index]),
                    "carrier_frequency_hz": float(carriers[index]),
                    "maximum_doppler_hz": float(dopplers[index]),
                    "seed": int(seeds[index]),
                    "path_delays_s": path_delays[index],
                    "average_path_gains_db": path_gains[index],
                    "num_paths": int(num_paths[index]),
                    "channel_filter_delay_samples": int(filter_delays[index]),
                    "maximum_channel_delay_samples": int(maximum_delays[index]),
                    "input_rms": float(input_rms[index]),
                    "output_rms": float(output_rms[index]),
                }
            )

        return NRTDLBatchResult(
            waveforms=output_waveforms,
            metadata=tuple(metadata),
            matlab_stdout=combined_output,
        )
    finally:
        if temporary_context is not None:
            temporary_context.cleanup()
