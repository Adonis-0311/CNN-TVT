"""Regression tests for bounded deterministic MATLAB cache streaming."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from vimd_amc.standards import (  # noqa: E402
    DEFAULT_MATLAB_BATCH_SIZE,
    FACTOR_ISOLATED_SPLITS,
    MAT_V7_MAX_VARIABLE_BYTES,
    NRTDLBatchResult,
    TDLCacheBuildConfig,
    build_tdl_paired_cache,
    cache_build_execution_preflight,
    factor_isolated_split_policies,
)


_SPEC = importlib.util.spec_from_file_location(
    "vimd_streaming_cache_cli",
    REPOSITORY_ROOT / "standards" / "build_factor_cache.py",
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("could not load standards/build_factor_cache.py")
_CLI = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CLI)


def _identity_nrtdl(
    waveforms: np.ndarray,
    configurations: list[object],
    **_: object,
) -> NRTDLBatchResult:
    samples = np.asarray(waveforms, dtype=np.complex128)
    metadata = []
    for waveform, configuration in zip(
        samples,
        configurations,
        strict=True,
    ):
        metadata.append(
            {
                "profile": configuration.profile,
                "seed": int(configuration.seed),
                "delay_spread_s": float(configuration.delay_spread_s),
                "speed_mps": float(configuration.speed_mps),
                "carrier_frequency_hz": float(
                    configuration.carrier_frequency_hz
                ),
                "maximum_doppler_hz": float(
                    configuration.maximum_doppler_hz
                ),
                "num_paths": 1,
                "channel_filter_delay_samples": 0,
                "maximum_channel_delay_samples": 0,
                "path_delays_s": np.asarray([0.0]),
                "average_path_gains_db": np.asarray([0.0]),
                "sample_rate_hz": 1_000_000.0,
                "num_transmit_antennas": 1,
                "num_receive_antennas": 1,
                "normalize_path_gains": True,
                "normalize_channel_outputs": True,
                "channel_filtering": True,
                "initial_time_s": 0.0,
                "random_stream": "mt19937ar with seed",
                "standard_reference": "unit-test identity TDL",
                "channel_class": "identity_nrTDLChannel",
                "matlab_release": "unit-test",
                "five_g_toolbox_version": "unit-test",
                "generated_utc": "excluded-from-manifest",
                "input_rms": float(np.sqrt(np.mean(np.abs(waveform) ** 2))),
                "output_rms": float(np.sqrt(np.mean(np.abs(waveform) ** 2))),
            }
        )
    return NRTDLBatchResult(
        waveforms=samples.copy(),
        metadata=tuple(metadata),
        matlab_stdout="unit-test",
    )


def _factor_micro_config() -> TDLCacheBuildConfig:
    sizes = {split: 1 for split in FACTOR_ISOLATED_SPLITS}
    policies = factor_isolated_split_policies(sizes)
    return TDLCacheBuildConfig(
        split_sizes=tuple(
            (policy.split, policy.size) for policy in policies
        ),
        sample_length=64,
        guard_samples=48,
        master_seed=20260727,
        split_policies=policies,
    )


def _headline_v2_config() -> TDLCacheBuildConfig:
    arguments = _CLI.parse_args(
        [
            "--output",
            str(REPOSITORY_ROOT / "standards" / "_never_written"),
            "--preset",
            "headline_v2",
            "--sample-length",
            "1024",
            "--guard-samples",
            "96",
        ]
    )
    return _CLI.config_from_args(arguments)


class CacheExecutionPreflightTest(unittest.TestCase):
    def test_headline_v2_is_safely_chunked_below_mat_v7_limit(self) -> None:
        config = _headline_v2_config()
        plan = cache_build_execution_preflight(
            config,
            matlab_batch_size=DEFAULT_MATLAB_BATCH_SIZE,
        )
        self.assertEqual(plan["total_source_count"], 152_000)
        self.assertEqual(plan["total_view_count"], 304_000)
        self.assertEqual(plan["extended_sample_length"], 1216)
        self.assertTrue(
            plan["unbatched_transfer_exceeds_mat_v7_limit"]
        )
        self.assertTrue(plan["chunk_transfer_within_mat_v7_limit"])
        self.assertLessEqual(
            plan["maximum_single_transfer_bytes"],
            MAT_V7_MAX_VARIABLE_BYTES,
        )
        self.assertEqual(plan["matlab_chunk_count"], 38)
        self.assertFalse(plan["batch_size_affects_manifest_digest"])

    def test_preflight_rejects_a_batch_above_mat_v7_limit(self) -> None:
        config = _headline_v2_config()
        bytes_per_waveform = (
            (config.sample_length + 2 * config.guard_samples)
            * np.dtype(np.complex128).itemsize
        )
        unsafe = MAT_V7_MAX_VARIABLE_BYTES // bytes_per_waveform
        with self.assertRaisesRegex(
            ValueError,
            "MATLAB v7 single-variable limit",
        ):
            cache_build_execution_preflight(
                config,
                matlab_batch_size=unsafe,
            )


class StreamedCacheBuilderTest(unittest.TestCase):
    def test_chunk_boundaries_do_not_change_cache_digest_or_order(self) -> None:
        config = _factor_micro_config()
        chunked_call_sizes: list[int] = []

        def chunked_backend(
            waveforms: np.ndarray,
            configurations: list[object],
            **kwargs: object,
        ) -> NRTDLBatchResult:
            del kwargs
            chunked_call_sizes.append(len(waveforms))
            self.assertTrue(np.any(np.abs(waveforms) > 0))
            return _identity_nrtdl(waveforms, configurations)

        with tempfile.TemporaryDirectory(
            prefix="vimd_streamed_cache_"
        ) as temporary:
            root = Path(temporary)
            with patch(
                "vimd_amc.standards.cache.apply_nrtdl_batch",
                side_effect=chunked_backend,
            ):
                chunked = build_tdl_paired_cache(
                    root / "chunked",
                    config=config,
                    matlab_batch_size=4,
                )
            with patch(
                "vimd_amc.standards.cache.apply_nrtdl_batch",
                side_effect=_identity_nrtdl,
            ):
                single = build_tdl_paired_cache(
                    root / "single",
                    config=config,
                    matlab_batch_size=100,
                )

        self.assertEqual(chunked.manifest, single.manifest)
        self.assertNotIn("execution_preflight", chunked.manifest)
        self.assertNotIn(
            "matlab_batch_size",
            chunked.manifest["configuration"],
        )
        self.assertEqual(
            chunked.manifest["source_ids"],
            single.manifest["source_ids"],
        )
        self.assertEqual(
            chunked.manifest["records"],
            single.manifest["records"],
        )
        self.assertEqual(
            chunked.manifest["files"],
            single.manifest["files"],
        )
        self.assertEqual(
            chunked_call_sizes,
            [4, 4, 4, 4, 4, 4, 4, 4, 2, 3],
        )


if __name__ == "__main__":
    unittest.main()
