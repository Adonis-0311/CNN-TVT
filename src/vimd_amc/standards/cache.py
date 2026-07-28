"""Deterministic offline paired-view caches using MATLAB ``nrTDLChannel``.

The builder keeps the standards-backed channel outside the online data-loader:
target and jammer waveforms are generated with a fixed guard, independently
channelized in two batched MATLAB calls, audibly cropped, and then passed
through :meth:`SignalSynthesizer.finalize_received_components` for the shared
receiver/noise/SNR/SIR bookkeeping used by the proxy experiments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from vimd_amc.data.split import (
    assert_disjoint_source_ids,
    manifest_digest,
    source_sequence_id,
    stable_seed,
)
from vimd_amc.data.synthesis import (
    SynthesisConfig,
    SignalSynthesizer,
    measured_sir_db,
    measured_snr_db,
)

from .nrtdl_matlab import NRTDLConfiguration, apply_nrtdl_batch


_PROFILE_INDEX = {
    "TDL-A": 0,
    "TDL-B": 1,
    "TDL-C": 2,
    "TDL-D": 3,
    "TDL-E": 4,
}
_SOURCE_SPLIT_KEY = {
    "train": "train",
    "validation": "validation",
    "heldout_channel": "unseen_channel",
}
FACTOR_ISOLATED_SPLITS = (
    "train",
    "validation",
    "id_test",
    "hard_interference",
    "unseen_jammer",
    "unseen_speed",
    "heldout_channel",
    "combined_ood",
    "clean_retention",
)
_SEEN_PROFILES = ("TDL-A", "TDL-C", "TDL-D")
_HELDOUT_PROFILES = ("TDL-B", "TDL-E")
_SEEN_JAMMERS = (
    "tone",
    "multitone",
    "chirp",
    "sweep",
    "partial_band",
    "comb",
)
_HELDOUT_JAMMERS = ("pulse", "ofdm_like")
_SEEN_SPEEDS_KMH = (0.0, 60.0, 120.0, 150.0)
_HELDOUT_SPEEDS_KMH = (180.0, 250.0)
_STANDARD_SNR_DB_VALUES = (-10.0, -6.0, -2.0, 2.0, 6.0, 10.0, 14.0, 18.0)
_STANDARD_SIR_DB_VALUES = (-10.0, -5.0, 0.0, 5.0, 10.0)
_HARD_SIR_DB_VALUES = (-15.0, -10.0, -5.0, 0.0)
_QUALITY_DB_SCALE = 20.0
_QUALITY_CLIP = (-1.0, 1.5)
_FACTOR_PROTOCOL_EXCLUSIONS = {
    "cochannel": {
        "status": "excluded_from_primary_modulation_classification",
        "reason": (
            "A single-antenna mixture of two in-taxonomy modulated emitters "
            "does not identify which waveform owns a preassigned target label "
            "under source exchange, especially near equal power."
        ),
        "required_resolution": (
            "physical emitter anchor, out-of-taxonomy collision label, "
            "dominant-emitter label, or separate ambiguity-stress protocol"
        ),
    },
    "mixed": {
        "status": "excluded_from_locked_factor_protocol",
        "reason": (
            "The generator can compose two jammer primitives, but no frozen "
            "combination policy or independently identified mixture label is "
            "pre-registered for the nine-split headline experiment."
        ),
        "required_resolution": (
            "separate pre-registered compositional generalization experiment"
        ),
    },
}
_ACTIVE_JAMMER_PRECHANNEL_POWER_THRESHOLD = 1e-12
_ARRAY_DTYPES: dict[str, np.dtype[Any]] = {
    "x": np.dtype(np.float32),
    "clean": np.dtype(np.float32),
    "jammer": np.dtype(np.float32),
    "noise": np.dtype(np.float32),
    "receiver_artifact": np.dtype(np.float32),
    "unexplained": np.dtype(np.float32),
    "jam_labels": np.dtype(np.float32),
    "quality": np.dtype(np.float32),
    "quality_mask": np.dtype(np.float32),
    "snr_db": np.dtype(np.float32),
    "sir_db": np.dtype(np.float32),
    "overlap": np.dtype(np.float32),
    "speed_kmh": np.dtype(np.float32),
    "doppler_hz": np.dtype(np.float32),
    "target_profile_index": np.dtype(np.int64),
    "jammer_profile_index": np.dtype(np.int64),
    "condition_seed": np.dtype(np.int64),
    "target_channel_seed": np.dtype(np.uint32),
    "jammer_channel_seed": np.dtype(np.uint32),
    "label": np.dtype(np.int64),
    "source_id": np.dtype(np.int64),
}


@dataclass(frozen=True)
class FactorSplitPolicy:
    """Pre-registered factor support for one source-disjoint cache split."""

    split: str
    role: str
    size: int
    source_key: str
    profiles: tuple[str, ...]
    jammer_choices: tuple[str, ...]
    speeds_kmh: tuple[float, ...]
    snr_db_values: tuple[float, ...]
    sir_db_values: tuple[float, ...] | None
    clean_fraction: float = 0.0
    held_factors: tuple[str, ...] = ()
    isolation_factors: tuple[str, ...] = ()


def factor_isolated_split_policies(
    split_sizes: Mapping[str, int] | None = None,
) -> tuple[FactorSplitPolicy, ...]:
    """Return the locked factor-isolation split policy.

    Sizes are deliberately supplied separately from factor support so a micro
    pipeline sentinel and a powered experiment can share exactly the same
    isolation contract.  The defaults are small administrative placeholders,
    not a statistical power claim.
    """

    sizes = {split: 10 for split in FACTOR_ISOLATED_SPLITS}
    if split_sizes is not None:
        unknown = set(split_sizes).difference(FACTOR_ISOLATED_SPLITS)
        if unknown:
            raise ValueError(
                f"unknown factor-isolated split sizes: {sorted(unknown)}"
            )
        sizes.update({str(key): int(value) for key, value in split_sizes.items()})

    def policy(
        split: str,
        role: str,
        *,
        profiles: tuple[str, ...] = _SEEN_PROFILES,
        jammers: tuple[str, ...] = _SEEN_JAMMERS,
        speeds: tuple[float, ...] = _SEEN_SPEEDS_KMH,
        sir_values: tuple[float, ...] | None = _STANDARD_SIR_DB_VALUES,
        clean_fraction: float = 0.0,
        held_factors: tuple[str, ...] = (),
        isolation_factors: tuple[str, ...] = (),
    ) -> FactorSplitPolicy:
        return FactorSplitPolicy(
            split=split,
            role=role,
            size=sizes[split],
            source_key=f"factor_isolated::{split}",
            profiles=profiles,
            jammer_choices=jammers,
            speeds_kmh=speeds,
            snr_db_values=_STANDARD_SNR_DB_VALUES,
            sir_db_values=sir_values,
            clean_fraction=clean_fraction,
            held_factors=held_factors,
            isolation_factors=isolation_factors,
        )

    return (
        policy("train", "model_fitting", clean_fraction=0.20),
        policy("validation", "checkpoint_selection", clean_fraction=0.20),
        policy("id_test", "in_distribution_test", clean_fraction=0.20),
        policy(
            "hard_interference",
            "seen_factor_hard_interference_test",
            sir_values=_HARD_SIR_DB_VALUES,
            isolation_factors=("sir_severity",),
        ),
        policy(
            "unseen_jammer",
            "jammer_family_held_out_test",
            jammers=_HELDOUT_JAMMERS,
            held_factors=("jammer_family",),
            isolation_factors=("jammer_family",),
        ),
        policy(
            "unseen_speed",
            "speed_held_out_test",
            speeds=_HELDOUT_SPEEDS_KMH,
            held_factors=("speed",),
            isolation_factors=("speed",),
        ),
        policy(
            "heldout_channel",
            "tdl_profile_held_out_test",
            profiles=_HELDOUT_PROFILES,
            held_factors=("tdl_profile",),
            isolation_factors=("tdl_profile",),
        ),
        policy(
            "combined_ood",
            "combined_jammer_speed_channel_held_out_test",
            profiles=_HELDOUT_PROFILES,
            jammers=_HELDOUT_JAMMERS,
            speeds=_HELDOUT_SPEEDS_KMH,
            held_factors=("jammer_family", "speed", "tdl_profile"),
            isolation_factors=("jammer_family", "speed", "tdl_profile"),
        ),
        policy(
            "clean_retention",
            "no_interference_retention_test",
            profiles=(*_SEEN_PROFILES, *_HELDOUT_PROFILES),
            jammers=("none",),
            sir_values=None,
            clean_fraction=1.0,
            isolation_factors=("interference_presence",),
        ),
    )


@dataclass(frozen=True)
class TDLCacheBuildConfig:
    """Configuration for one source-disjoint train/validation/held-out cache."""

    split_sizes: tuple[tuple[str, int], ...] = (
        ("train", 4),
        ("validation", 2),
        ("heldout_channel", 4),
    )
    sample_length: int = 256
    guard_samples: int = 64
    sample_rate_hz: float = 1_000_000.0
    carrier_frequency_hz: float = 5_900_000_000.0
    master_seed: int = 20260727
    modulations: tuple[str, ...] = ("BPSK", "QPSK", "8PSK", "16QAM")
    jammer_choices: tuple[str, ...] = (
        "tone",
        "chirp",
        "pulse",
        "partial_band",
        "cochannel",
    )
    train_profiles: tuple[str, ...] = ("TDL-A", "TDL-C", "TDL-D")
    heldout_profiles: tuple[str, ...] = ("TDL-B", "TDL-E")
    delay_spreads_s: tuple[float, ...] = (30e-9, 100e-9, 300e-9)
    speeds_kmh: tuple[float, ...] = (0.0, 60.0, 120.0)
    snr_db_range: tuple[float, float] = (-8.0, 16.0)
    sir_db_range: tuple[float, float] = (-12.0, 8.0)
    snr_db_values: tuple[float, ...] | None = None
    sir_db_values: tuple[float, ...] | None = None
    evidence_designation: str = "integration_smoke_only"
    split_policies: tuple[FactorSplitPolicy, ...] | None = None

    def validate(self) -> None:
        split_sizes = dict(self.split_sizes)
        if len(split_sizes) != len(self.split_sizes):
            raise ValueError("split_sizes contains duplicate split names.")
        if any(size <= 0 for size in split_sizes.values()):
            raise ValueError("every split size must be positive.")
        if self.split_policies is None:
            if set(split_sizes) != set(_SOURCE_SPLIT_KEY):
                raise ValueError(
                    "split_sizes must define train, validation, and "
                    "heldout_channel."
                )
        else:
            self._validate_split_policies(split_sizes)
        if self.sample_length < 16:
            raise ValueError("sample_length must be at least 16.")
        if self.guard_samples <= 0:
            raise ValueError("guard_samples must be positive.")
        if self.sample_rate_hz <= 0 or self.carrier_frequency_hz <= 0:
            raise ValueError("sample and carrier frequencies must be positive.")
        if not self.modulations:
            raise ValueError("at least one modulation is required.")
        if len(set(self.modulations)) != len(self.modulations):
            raise ValueError("modulations contains duplicate values.")
        synthesis_defaults = SynthesisConfig()
        unknown_modulations = set(self.modulations).difference(
            synthesis_defaults.modulations
        )
        if unknown_modulations:
            raise ValueError(
                f"unsupported modulations: {sorted(unknown_modulations)}"
            )
        if not self.jammer_choices or "none" in self.jammer_choices:
            raise ValueError(
                "the standards cache requires at least one active jammer choice."
            )
        if len(set(self.jammer_choices)) != len(self.jammer_choices):
            raise ValueError("jammer_choices contains duplicate values.")
        unknown_jammers = set(self.jammer_choices).difference(
            synthesis_defaults.jammer_types
        )
        if unknown_jammers:
            raise ValueError(
                f"unsupported jammers: {sorted(unknown_jammers)}"
            )
        if not self.train_profiles or not self.heldout_profiles:
            raise ValueError("training and held-out profile pools cannot be empty.")
        if len(set(self.train_profiles)) != len(self.train_profiles):
            raise ValueError("train_profiles contains duplicate values.")
        if len(set(self.heldout_profiles)) != len(self.heldout_profiles):
            raise ValueError("heldout_profiles contains duplicate values.")
        unknown = (
            set(self.train_profiles) | set(self.heldout_profiles)
        ).difference(_PROFILE_INDEX)
        if unknown:
            raise ValueError(f"unknown TDL profiles: {sorted(unknown)}")
        overlap = set(self.train_profiles).intersection(self.heldout_profiles)
        if overlap:
            raise ValueError(
                f"train and held-out profile pools overlap: {sorted(overlap)}"
            )
        if not self.delay_spreads_s:
            raise ValueError("at least one delay spread is required.")
        if any(
            not np.isfinite(delay) or delay <= 0
            for delay in self.delay_spreads_s
        ):
            raise ValueError("delay spreads must be positive and finite.")
        if len(set(self.delay_spreads_s)) != len(self.delay_spreads_s):
            raise ValueError("delay_spreads_s contains duplicate values.")
        if not self.speeds_kmh:
            raise ValueError("at least one speed is required.")
        if any(
            not np.isfinite(speed) or speed < 0 for speed in self.speeds_kmh
        ):
            raise ValueError("speeds must be nonnegative and finite.")
        if len(set(self.speeds_kmh)) != len(self.speeds_kmh):
            raise ValueError("speeds_kmh contains duplicate values.")
        if (
            len(self.snr_db_range) != 2
            or not all(np.isfinite(value) for value in self.snr_db_range)
        ):
            raise ValueError("snr_db_range must contain two finite values.")
        if (
            len(self.sir_db_range) != 2
            or not all(np.isfinite(value) for value in self.sir_db_range)
        ):
            raise ValueError("sir_db_range must contain two finite values.")
        if self.snr_db_range[0] > self.snr_db_range[1]:
            raise ValueError("snr_db_range is reversed.")
        if self.sir_db_range[0] > self.sir_db_range[1]:
            raise ValueError("sir_db_range is reversed.")
        for name, values in (
            ("snr_db_values", self.snr_db_values),
            ("sir_db_values", self.sir_db_values),
        ):
            if values is None:
                continue
            if not values:
                raise ValueError(f"{name} cannot be empty when provided.")
            if any(not np.isfinite(value) for value in values):
                raise ValueError(f"{name} must contain only finite values.")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} contains duplicate values.")
        if not self.evidence_designation.strip():
            raise ValueError("evidence_designation cannot be empty.")

    def _validate_split_policies(self, split_sizes: Mapping[str, int]) -> None:
        assert self.split_policies is not None
        policies = {policy.split: policy for policy in self.split_policies}
        if len(policies) != len(self.split_policies):
            raise ValueError("split_policies contains duplicate split names.")
        if set(policies) != set(split_sizes):
            raise ValueError(
                "split_policies and split_sizes must contain identical splits."
            )
        if not {"train", "validation"}.issubset(policies):
            raise ValueError("split_policies must include train and validation.")
        if len({policy.source_key for policy in policies.values()}) != len(
            policies
        ):
            raise ValueError("every split policy requires a unique source_key.")
        synthesis_defaults = SynthesisConfig()
        for split, policy in policies.items():
            if policy.size != split_sizes[split] or policy.size <= 0:
                raise ValueError(
                    f"{split} policy size disagrees with split_sizes."
                )
            if not policy.role.strip() or not policy.source_key.strip():
                raise ValueError(f"{split} role/source_key cannot be blank.")
            if not policy.profiles:
                raise ValueError(f"{split} requires at least one TDL profile.")
            unknown_profiles = set(policy.profiles).difference(_PROFILE_INDEX)
            if unknown_profiles:
                raise ValueError(
                    f"{split} has unknown TDL profiles: "
                    f"{sorted(unknown_profiles)}"
                )
            if len(set(policy.profiles)) != len(policy.profiles):
                raise ValueError(f"{split} profiles contain duplicates.")
            if not policy.jammer_choices:
                raise ValueError(f"{split} requires a jammer policy.")
            unknown_jammers = set(policy.jammer_choices).difference(
                {*synthesis_defaults.jammer_types, "none"}
            )
            if unknown_jammers:
                raise ValueError(
                    f"{split} has unsupported jammers: "
                    f"{sorted(unknown_jammers)}"
                )
            if len(set(policy.jammer_choices)) != len(policy.jammer_choices):
                raise ValueError(f"{split} jammer choices contain duplicates.")
            if "none" in policy.jammer_choices and policy.jammer_choices != (
                "none",
            ):
                raise ValueError(
                    f"{split} cannot mix clean and active jammer records."
                )
            if (
                not np.isfinite(policy.clean_fraction)
                or not 0.0 <= policy.clean_fraction <= 1.0
            ):
                raise ValueError(
                    f"{split} clean_fraction must lie in [0, 1]."
                )
            if policy.jammer_choices == ("none",):
                if policy.clean_fraction != 1.0:
                    raise ValueError(
                        f"{split} all-clean policy requires clean_fraction=1."
                    )
            elif policy.clean_fraction >= 1.0:
                raise ValueError(
                    f"{split} active jammer pool requires clean_fraction < 1."
                )
            if not policy.speeds_kmh or any(
                not np.isfinite(speed) or speed < 0
                for speed in policy.speeds_kmh
            ):
                raise ValueError(f"{split} speeds must be nonnegative and finite.")
            if len(set(policy.speeds_kmh)) != len(policy.speeds_kmh):
                raise ValueError(f"{split} speeds contain duplicates.")
            if not policy.snr_db_values or any(
                not np.isfinite(value) for value in policy.snr_db_values
            ):
                raise ValueError(f"{split} SNR values must be finite and nonempty.")
            if policy.jammer_choices == ("none",):
                if policy.sir_db_values not in (None, ()):
                    raise ValueError(
                        f"{split} clean policy must not declare SIR values."
                    )
            elif not policy.sir_db_values or any(
                not np.isfinite(value) for value in policy.sir_db_values
            ):
                raise ValueError(
                    f"{split} active-jammer SIR values must be finite and nonempty."
                )
        if set(FACTOR_ISOLATED_SPLITS).issubset(policies):
            train = policies["train"]
            for split in ("validation", "id_test"):
                candidate = policies[split]
                if (
                    candidate.profiles != train.profiles
                    or candidate.jammer_choices != train.jammer_choices
                    or candidate.speeds_kmh != train.speeds_kmh
                    or candidate.clean_fraction != train.clean_fraction
                ):
                    raise ValueError(
                        f"{split} must share train factor support."
                    )
            held_jammers = set(
                policies["unseen_jammer"].jammer_choices
            ) | set(policies["combined_ood"].jammer_choices)
            if set(train.jammer_choices).intersection(held_jammers):
                raise ValueError("seen and held jammer families must be disjoint.")
            held_speeds = set(policies["unseen_speed"].speeds_kmh) | set(
                policies["combined_ood"].speeds_kmh
            )
            if set(train.speeds_kmh).intersection(held_speeds):
                raise ValueError("seen and held speeds must be disjoint.")
            held_profiles = set(
                policies["heldout_channel"].profiles
            ) | set(policies["combined_ood"].profiles)
            if set(train.profiles).intersection(held_profiles):
                raise ValueError("seen and held TDL profiles must be disjoint.")
            hard_sir = policies["hard_interference"].sir_db_values or ()
            if not hard_sir or max(hard_sir) > 0:
                raise ValueError("hard_interference must enforce SIR <= 0 dB.")
            clean = policies["clean_retention"]
            if clean.jammer_choices != ("none",) or clean.sir_db_values is not None:
                raise ValueError(
                    "clean_retention requires jammer='none' and invalid SIR."
                )

    def policy_for_split(self, split: str) -> FactorSplitPolicy | None:
        if self.split_policies is None:
            return None
        for policy in self.split_policies:
            if policy.split == split:
                return policy
        raise KeyError(split)


@dataclass(frozen=True)
class TDLCacheBuildResult:
    root: Path
    manifest: dict[str, Any]


@dataclass
class _PendingView:
    split: str
    index: int
    view: int
    source_id: int
    modulation: str
    condition_seed: int
    snr_db: float
    sir_db: float
    jammer_name: str
    overlap_profile: str
    speed_kmh: float
    target_configuration: NRTDLConfiguration
    jammer_configuration: NRTDLConfiguration
    source_with_guard: np.ndarray
    jammer_with_guard: np.ndarray
    jammer_labels: np.ndarray
    jammer_components: tuple[str, ...]
    jammer_generation_attempt: int
    jammer_waveform_seed: int
    jammer_center_power_before_channel: float
    jammer_generation_rejections: tuple[dict[str, object], ...]


def _complex_to_iq(values: np.ndarray) -> np.ndarray:
    return np.stack((values.real, values.imag), axis=0).astype(
        np.float32, copy=False
    )


def _iq_to_complex(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    return array[0].astype(np.float64) + 1j * array[1].astype(np.float64)


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _actual_factor_coverage(
    records: Mapping[str, Sequence[Mapping[str, object]]],
    config: TDLCacheBuildConfig,
) -> dict[str, dict[str, object]]:
    """Summarize what was actually materialized, independently of policy."""

    coverage: dict[str, dict[str, object]] = {}
    for split, split_records in records.items():
        views = [
            view
            for record in split_records
            for view in record["views"]  # type: ignore[index]
        ]
        policy = config.policy_for_split(split)
        requested: dict[str, object] | None = (
            _jsonable(asdict(policy)) if policy is not None else None
        )
        active_sir_values = sorted(
            {
                float(view["sir_db"])
                for view in views
                if bool(view["sir_valid"])
            }
        )
        actual = {
            "modulations": sorted(
                {str(record["modulation"]) for record in split_records}
            ),
            "jammer_choices": sorted(
                {str(view["jammer_name"]) for view in views}
            ),
            "speeds_kmh": sorted(
                {float(view["vehicle_speed_kmh"]) for view in views}
            ),
            "target_profiles": sorted(
                {str(view["tdl_target_profile"]) for view in views}
            ),
            "jammer_profiles": sorted(
                {str(view["tdl_jammer_profile"]) for view in views}
            ),
            "snr_db_values": sorted(
                {float(view["snr_db"]) for view in views}
            ),
            "sir_db_values": active_sir_values,
        }
        policy_coverage: dict[str, bool] | None = None
        if policy is not None:
            allowed_jammers = set(policy.jammer_choices)
            if policy.clean_fraction > 0.0:
                allowed_jammers.add("none")
            policy_coverage = {
                "profiles": set(actual["target_profiles"]).issubset(
                    policy.profiles
                )
                and set(actual["jammer_profiles"]).issubset(policy.profiles),
                "jammers": set(actual["jammer_choices"]).issubset(
                    allowed_jammers
                ),
                "speeds": set(actual["speeds_kmh"]).issubset(
                    policy.speeds_kmh
                ),
                "snr": set(actual["snr_db_values"]).issubset(
                    policy.snr_db_values
                ),
                "sir": (
                    not active_sir_values
                    if policy.sir_db_values is None
                    else set(active_sir_values).issubset(policy.sir_db_values)
                ),
            }
        coverage[split] = {
            "source_count": len(split_records),
            "view_count": len(views),
            "sir_valid_view_count": sum(
                bool(view["sir_valid"]) for view in views
            ),
            "sir_invalid_view_count": sum(
                not bool(view["sir_valid"]) for view in views
            ),
            "requested_clean_fraction": (
                policy.clean_fraction if policy is not None else None
            ),
            "requested_clean_view_count": (
                len(_stratified_clean_slots(len(split_records), policy.clean_fraction))
                if policy is not None
                else None
            ),
            "actual_clean_fraction": (
                sum(not bool(view["sir_valid"]) for view in views)
                / max(len(views), 1)
            ),
            "requested_policy": requested,
            "actual": actual,
            "all_actual_values_within_policy": (
                all(policy_coverage.values())
                if policy_coverage is not None
                else None
            ),
            "factor_checks": policy_coverage,
        }
    return coverage


def _split_profiles(split: str, config: TDLCacheBuildConfig) -> tuple[str, ...]:
    policy = config.policy_for_split(split)
    if policy is not None:
        return policy.profiles
    return (
        config.heldout_profiles
        if split == "heldout_channel"
        else config.train_profiles
    )


def _stratified_clean_slots(
    size: int,
    clean_fraction: float,
) -> frozenset[int]:
    """Place an exact, deterministic clean quota across paired-view slots."""

    view_count = 2 * size
    if clean_fraction <= 0.0:
        return frozenset()
    if clean_fraction >= 1.0:
        return frozenset(range(view_count))
    clean_count = max(1, int(round(view_count * clean_fraction)))
    clean_count = min(clean_count, view_count - 1)
    slots = {
        min(
            view_count - 1,
            int((index + 0.5) * view_count / clean_count),
        )
        for index in range(clean_count)
    }
    if len(slots) != clean_count:
        raise AssertionError("clean stratification produced duplicate slots")
    return frozenset(slots)


def _build_pending_views(
    synthesizer: SignalSynthesizer,
    config: TDLCacheBuildConfig,
) -> tuple[list[_PendingView], dict[str, list[int]]]:
    extended_length = config.sample_length + 2 * config.guard_samples
    pending: list[_PendingView] = []
    source_groups: dict[str, list[int]] = {}
    for split, size in config.split_sizes:
        policy = config.policy_for_split(split)
        source_key = (
            policy.source_key if policy is not None else _SOURCE_SPLIT_KEY[split]
        )
        source_ids: list[int] = []
        profile_pool = _split_profiles(split, config)
        clean_slots = (
            _stratified_clean_slots(size, policy.clean_fraction)
            if policy is not None
            else frozenset()
        )
        for index in range(size):
            source_id = (
                source_sequence_id(source_key, index, config.master_seed)
                if policy is None
                else stable_seed(
                    "factor_cache_source",
                    config.master_seed,
                    source_key,
                    index,
                    bits=63,
                )
            )
            source_ids.append(source_id)
            modulation = config.modulations[index % len(config.modulations)]
            source = synthesizer.generate_source(
                modulation, source_id, extended_length
            )
            for view in (1, 2):
                view_slot = 2 * index + view - 1
                scheduled_clean = view_slot in clean_slots
                condition_seed = stable_seed(
                    config.master_seed, split, index, view, "condition"
                )
                condition_rng = np.random.default_rng(condition_seed)
                snr_db = float(
                    condition_rng.choice(policy.snr_db_values)
                    if policy is not None
                    else condition_rng.choice(config.snr_db_values)
                    if config.snr_db_values is not None
                    else condition_rng.uniform(*config.snr_db_range)
                )
                if scheduled_clean or (
                    policy is not None and policy.sir_db_values is None
                ):
                    # Finite storage sentinel.  Its quality-mask entry is zero
                    # and the per-view manifest records sir_valid=false.
                    sir_db = 0.0
                else:
                    sir_db = float(
                        condition_rng.choice(policy.sir_db_values)
                        if policy is not None
                        else condition_rng.choice(config.sir_db_values)
                        if config.sir_db_values is not None
                        else condition_rng.uniform(*config.sir_db_range)
                    )
                jammer_name = (
                    "none"
                    if scheduled_clean
                    else str(
                        condition_rng.choice(
                            policy.jammer_choices
                            if policy is not None
                            else config.jammer_choices
                        )
                    )
                )
                overlap_profile = (
                    None
                    if jammer_name == "none"
                    else
                    "high"
                    if jammer_name == "cochannel"
                    else str(condition_rng.choice(("low", "mid", "high")))
                )
                speed_kmh = float(
                    condition_rng.choice(
                        policy.speeds_kmh
                        if policy is not None
                        else config.speeds_kmh
                    )
                )
                target_profile_rng = np.random.default_rng(
                    stable_seed(condition_seed, "target_profile")
                )
                jammer_profile_rng = np.random.default_rng(
                    stable_seed(condition_seed, "jammer_profile")
                )
                target_profile = str(target_profile_rng.choice(profile_pool))
                jammer_profile = str(jammer_profile_rng.choice(profile_pool))
                target_delay_rng = np.random.default_rng(
                    stable_seed(condition_seed, "target_delay")
                )
                jammer_delay_rng = np.random.default_rng(
                    stable_seed(condition_seed, "jammer_delay")
                )
                target_delay = float(
                    target_delay_rng.choice(config.delay_spreads_s)
                )
                jammer_delay = float(
                    jammer_delay_rng.choice(config.delay_spreads_s)
                )
                target_channel_seed = stable_seed(
                    config.master_seed,
                    split,
                    index,
                    view,
                    "target_nrtdl",
                    bits=32,
                )
                jammer_channel_seed = stable_seed(
                    config.master_seed,
                    split,
                    index,
                    view,
                    "jammer_nrtdl",
                    bits=32,
                )
                if target_channel_seed == jammer_channel_seed:
                    raise AssertionError(
                        "target and jammer channel seeds unexpectedly collide"
                    )
                jammer_generation_rejections: list[dict[str, object]] = []
                maximum_attempts = 1 if jammer_name == "none" else 32
                for jammer_generation_attempt in range(maximum_attempts):
                    jammer_seed_parts: tuple[object, ...] = (
                        (condition_seed, "jammer_waveform")
                        if jammer_generation_attempt == 0
                        else (
                            condition_seed,
                            "jammer_waveform_retry",
                            jammer_generation_attempt,
                        )
                    )
                    jammer_waveform_seed = stable_seed(*jammer_seed_parts)
                    jammer_rng = np.random.default_rng(jammer_waveform_seed)
                    raw_jammer, jammer_labels, jammer_components = (
                        synthesizer.generate_jammer(
                            jammer_name,
                            extended_length,
                            jammer_rng,
                            modulation,
                            overlap_profile=overlap_profile,
                        )
                    )
                    retained = raw_jammer[
                        config.guard_samples : (
                            config.guard_samples + config.sample_length
                        )
                    ]
                    jammer_center_power = float(
                        np.mean(np.abs(retained) ** 2)
                    )
                    if (
                        jammer_name == "none"
                        or
                        jammer_center_power
                        > _ACTIVE_JAMMER_PRECHANNEL_POWER_THRESHOLD
                    ):
                        break
                    jammer_generation_rejections.append(
                        {
                            "attempt": jammer_generation_attempt,
                            "named_seed": jammer_waveform_seed,
                            "reason": (
                                "prechannel_retained_window_power_not_above_"
                                "threshold"
                            ),
                            "observed_power": jammer_center_power,
                            "threshold": (
                                _ACTIVE_JAMMER_PRECHANNEL_POWER_THRESHOLD
                            ),
                        }
                    )
                else:
                    raise RuntimeError(
                        "could not generate an active jammer in the retained "
                        f"window after {maximum_attempts} deterministic attempts: "
                        f"split={split}, "
                        f"index={index}, view={view}, jammer={jammer_name}"
                    )
                speed_mps = speed_kmh / 3.6
                pending.append(
                    _PendingView(
                        split=split,
                        index=index,
                        view=view,
                        source_id=source_id,
                        modulation=modulation,
                        condition_seed=condition_seed,
                        snr_db=snr_db,
                        sir_db=sir_db,
                        jammer_name=jammer_name,
                        overlap_profile=overlap_profile,
                        speed_kmh=speed_kmh,
                        target_configuration=NRTDLConfiguration(
                            target_profile,
                            target_delay,
                            speed_mps,
                            config.carrier_frequency_hz,
                            target_channel_seed,
                        ),
                        jammer_configuration=NRTDLConfiguration(
                            jammer_profile,
                            jammer_delay,
                            speed_mps,
                            config.carrier_frequency_hz,
                            jammer_channel_seed,
                        ),
                        source_with_guard=source,
                        jammer_with_guard=raw_jammer,
                        jammer_labels=jammer_labels,
                        jammer_components=jammer_components,
                        jammer_generation_attempt=jammer_generation_attempt,
                        jammer_waveform_seed=jammer_waveform_seed,
                        jammer_center_power_before_channel=jammer_center_power,
                        jammer_generation_rejections=tuple(
                            jammer_generation_rejections
                        ),
                    )
                )
        source_groups[split] = source_ids
    assert_disjoint_source_ids(*source_groups.values())
    return pending, source_groups


def _array_shapes(
    size: int,
    sample_length: int,
    jammer_classes: int,
) -> dict[str, tuple[int, ...]]:
    return {
        "x": (size, 2, 2, sample_length),
        "clean": (size, 2, 2, sample_length),
        "jammer": (size, 2, 2, sample_length),
        "noise": (size, 2, 2, sample_length),
        "receiver_artifact": (size, 2, 2, sample_length),
        "unexplained": (size, 2, 2, sample_length),
        "jam_labels": (size, 2, jammer_classes),
        "quality": (size, 2, 3),
        "quality_mask": (size, 2, 3),
        "snr_db": (size, 2),
        "sir_db": (size, 2),
        "overlap": (size, 2),
        "speed_kmh": (size, 2),
        "doppler_hz": (size, 2),
        "target_profile_index": (size, 2),
        "jammer_profile_index": (size, 2),
        "condition_seed": (size, 2),
        "target_channel_seed": (size, 2),
        "jammer_channel_seed": (size, 2),
        "label": (size,),
        "source_id": (size,),
    }


def _open_split_arrays(
    split_directory: Path,
    *,
    size: int,
    sample_length: int,
    jammer_classes: int,
) -> dict[str, np.memmap]:
    split_directory.mkdir(parents=True, exist_ok=False)
    arrays: dict[str, np.memmap] = {}
    for name, shape in _array_shapes(size, sample_length, jammer_classes).items():
        arrays[name] = np.lib.format.open_memmap(
            split_directory / f"{name}.npy",
            mode="w+",
            dtype=_ARRAY_DTYPES[name],
            shape=shape,
        )
    return arrays


def _channel_audit(
    *,
    prefix: str,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    return {
        f"{prefix}_profile": str(metadata["profile"]),
        f"{prefix}_seed": int(metadata["seed"]),
        f"{prefix}_delay_spread_s": float(metadata["delay_spread_s"]),
        f"{prefix}_speed_mps": float(metadata["speed_mps"]),
        f"{prefix}_carrier_frequency_hz": float(
            metadata["carrier_frequency_hz"]
        ),
        f"{prefix}_maximum_doppler_hz": float(
            metadata["maximum_doppler_hz"]
        ),
        f"{prefix}_num_paths": int(metadata["num_paths"]),
        f"{prefix}_channel_filter_delay_samples": int(
            metadata["channel_filter_delay_samples"]
        ),
        f"{prefix}_maximum_channel_delay_samples": int(
            metadata["maximum_channel_delay_samples"]
        ),
        f"{prefix}_path_delays_s": np.asarray(
            metadata["path_delays_s"], dtype=np.float64
        ),
        f"{prefix}_average_path_gains_db": np.asarray(
            metadata["average_path_gains_db"], dtype=np.float64
        ),
    }


def build_tdl_paired_cache(
    output_root: str | Path,
    *,
    config: TDLCacheBuildConfig | None = None,
    matlab_executable: str | Path | None = None,
    matlab_timeout_s: float = 300.0,
) -> TDLCacheBuildResult:
    """Build one immutable source-disjoint paired cache.

    The destination must not already exist.  This prevents an interrupted or
    differently configured run from silently overwriting auditable evidence.
    """
    build_config = config or TDLCacheBuildConfig()
    build_config.validate()
    root = Path(output_root).resolve()
    if root.exists():
        raise FileExistsError(
            f"cache destination already exists; choose a new path: {root}"
        )
    root.mkdir(parents=True, exist_ok=False)

    synthesis_config = SynthesisConfig(
        sample_length=build_config.sample_length,
        sample_rate_hz=build_config.sample_rate_hz,
        carrier_hz=build_config.carrier_frequency_hz,
    )
    synthesizer = SignalSynthesizer(synthesis_config)
    unknown_modulations = set(build_config.modulations).difference(
        synthesis_config.modulations
    )
    if unknown_modulations:
        raise ValueError(
            f"unsupported modulations: {sorted(unknown_modulations)}"
        )
    unknown_jammers = set(build_config.jammer_choices).difference(
        synthesis_config.jammer_types
    )
    if unknown_jammers:
        raise ValueError(f"unsupported jammers: {sorted(unknown_jammers)}")

    pending, source_groups = _build_pending_views(
        synthesizer, build_config
    )
    target_batch = np.stack(
        [item.source_with_guard for item in pending], axis=0
    )
    jammer_batch = np.stack(
        [item.jammer_with_guard for item in pending], axis=0
    )
    if not np.any(np.abs(jammer_batch) > 0):
        raise ValueError(
            "a cache containing only clean records cannot provide the batched "
            "jammer-channel audit; include at least one active-jammer split"
        )
    target_result = apply_nrtdl_batch(
        target_batch,
        [item.target_configuration for item in pending],
        sample_rate_hz=build_config.sample_rate_hz,
        matlab_executable=matlab_executable,
        timeout_s=matlab_timeout_s,
    )
    jammer_result = apply_nrtdl_batch(
        jammer_batch,
        [item.jammer_configuration for item in pending],
        sample_rate_hz=build_config.sample_rate_hz,
        matlab_executable=matlab_executable,
        timeout_s=matlab_timeout_s,
    )

    split_arrays = {
        split: _open_split_arrays(
            root / split,
            size=size,
            sample_length=build_config.sample_length,
            jammer_classes=len(synthesis_config.jammer_types),
        )
        for split, size in build_config.split_sizes
    }
    records: dict[str, list[dict[str, object]]] = {
        split: [
            {
                "index": index,
                "source_sequence_id": int(source_groups[split][index]),
                "modulation": build_config.modulations[
                    index % len(build_config.modulations)
                ],
                "views": [None, None],
            }
            for index in range(size)
        ]
        for split, size in build_config.split_sizes
    }
    max_doppler_hz = max(
        item.target_configuration.maximum_doppler_hz for item in pending
    )
    component_audit: dict[str, dict[str, float | int | None]] = {
        split: {
            "view_count": 0,
            "active_jammer_view_count": 0,
            "clean_view_count": 0,
            "max_component_error": 0.0,
            "max_snr_error_db": 0.0,
            "max_sir_error_db": 0.0,
            "min_active_jammer_power": None,
            "max_clean_jammer_power": 0.0,
            "min_guard_margin_samples": None,
        }
        for split, _ in build_config.split_sizes
    }
    for flat_index, item in enumerate(pending):
        target_metadata = target_result.metadata[flat_index]
        jammer_metadata = jammer_result.metadata[flat_index]
        target_required_guard = int(
            target_metadata["channel_filter_delay_samples"]
        ) + int(target_metadata["maximum_channel_delay_samples"])
        jammer_required_guard = int(
            jammer_metadata["channel_filter_delay_samples"]
        ) + int(jammer_metadata["maximum_channel_delay_samples"])
        required_guard = max(target_required_guard, jammer_required_guard)
        if build_config.guard_samples < required_guard:
            raise RuntimeError(
                "fixed guard is insufficient for returned nrTDLChannel delay: "
                f"guard={build_config.guard_samples}, required={required_guard}, "
                f"split={item.split}, index={item.index}, view={item.view}"
            )
        crop_start = build_config.guard_samples
        crop_stop = crop_start + build_config.sample_length
        channel_metadata: dict[str, object] = {
            "tdl_standard_reference": target_metadata["standard_reference"],
            "tdl_matlab_release": target_metadata["matlab_release"],
            "tdl_five_g_toolbox_version": target_metadata[
                "five_g_toolbox_version"
            ],
            "tdl_guard_samples": build_config.guard_samples,
            "tdl_crop_start_sample": crop_start,
            "tdl_crop_stop_sample": crop_stop,
            "tdl_required_guard_samples": required_guard,
            "tdl_guard_margin_samples": (
                build_config.guard_samples - required_guard
            ),
            "jammer_generation_attempt": item.jammer_generation_attempt,
            "jammer_waveform_named_seed": item.jammer_waveform_seed,
            "jammer_center_power_before_channel": (
                item.jammer_center_power_before_channel
            ),
            "jammer_generation_rejections": (
                item.jammer_generation_rejections
            ),
            "jammer_generation_selection_stage": (
                "prechannel_retained_window_power_only"
            ),
            "jammer_generation_power_threshold": (
                _ACTIVE_JAMMER_PRECHANNEL_POWER_THRESHOLD
            ),
            "jammer_generation_uses_postchannel_or_model_outcome": False,
            **_channel_audit(
                prefix="tdl_target", metadata=target_metadata
            ),
            **_channel_audit(
                prefix="tdl_jammer", metadata=jammer_metadata
            ),
        }
        sample = synthesizer.finalize_received_components(
            clean=target_result.waveforms[flat_index, crop_start:crop_stop],
            raw_jammer=jammer_result.waveforms[
                flat_index, crop_start:crop_stop
            ],
            modulation=item.modulation,
            source_seed=item.source_id,
            condition_seed=item.condition_seed,
            snr_db=item.snr_db,
            sir_db=item.sir_db,
            jammer_name=item.jammer_name,
            jammer_labels=item.jammer_labels,
            jammer_components=item.jammer_components,
            channel_scenario=(
                f"{item.target_configuration.profile}/"
                f"{item.jammer_configuration.profile}"
            ),
            channel_model="matlab_nrTDLChannel_3GPP_TR_38_901",
            speed_kmh=item.speed_kmh,
            doppler_norm=(
                item.target_configuration.maximum_doppler_hz
                / build_config.sample_rate_hz
            ),
            overlap_profile=item.overlap_profile,
            channel_metadata=channel_metadata,
        )
        unexplained = sample.noise + sample.receiver_artifact
        residual = sample.mixture - (
            sample.clean + sample.jammer + unexplained
        )
        maximum_component_residual = float(np.max(np.abs(residual)))
        if maximum_component_residual > 2e-6:
            raise AssertionError(
                "component sum failed before cache write: "
                f"{maximum_component_residual}"
            )
        snr_error = abs(
            measured_snr_db(sample.clean, sample.noise) - item.snr_db
        )
        active_jammer = item.jammer_name != "none"
        jammer_power = float(np.mean(np.abs(sample.jammer) ** 2))
        sir_error = (
            abs(measured_sir_db(sample.clean, sample.jammer) - item.sir_db)
            if active_jammer
            else 0.0
        )
        if not active_jammer and jammer_power > 1e-12:
            raise AssertionError(
                f"clean split contains jammer energy: {jammer_power}"
            )
        if snr_error > 2e-4 or sir_error > 2e-4:
            raise AssertionError(
                f"power target mismatch: SNR error={snr_error}, "
                f"SIR error={sir_error}"
            )
        split_audit = component_audit[item.split]
        split_audit["view_count"] = int(split_audit["view_count"] or 0) + 1
        if active_jammer:
            split_audit["active_jammer_view_count"] = (
                int(split_audit["active_jammer_view_count"] or 0) + 1
            )
            prior_minimum = split_audit["min_active_jammer_power"]
            split_audit["min_active_jammer_power"] = (
                jammer_power
                if prior_minimum is None
                else min(float(prior_minimum), jammer_power)
            )
        else:
            split_audit["clean_view_count"] = (
                int(split_audit["clean_view_count"] or 0) + 1
            )
            split_audit["max_clean_jammer_power"] = max(
                float(split_audit["max_clean_jammer_power"] or 0.0),
                jammer_power,
            )
        split_audit["max_component_error"] = max(
            float(split_audit["max_component_error"] or 0.0),
            maximum_component_residual,
        )
        split_audit["max_snr_error_db"] = max(
            float(split_audit["max_snr_error_db"] or 0.0),
            snr_error,
        )
        split_audit["max_sir_error_db"] = max(
            float(split_audit["max_sir_error_db"] or 0.0),
            sir_error,
        )
        guard_margin = int(
            channel_metadata["tdl_guard_margin_samples"]
        )
        prior_guard = split_audit["min_guard_margin_samples"]
        split_audit["min_guard_margin_samples"] = (
            guard_margin
            if prior_guard is None
            else min(int(prior_guard), guard_margin)
        )

        arrays = split_arrays[item.split]
        row = item.index
        view_index = item.view - 1
        arrays["x"][row, view_index] = _complex_to_iq(sample.mixture)
        arrays["clean"][row, view_index] = _complex_to_iq(sample.clean)
        arrays["jammer"][row, view_index] = _complex_to_iq(sample.jammer)
        arrays["noise"][row, view_index] = _complex_to_iq(sample.noise)
        arrays["receiver_artifact"][row, view_index] = _complex_to_iq(
            sample.receiver_artifact
        )
        arrays["unexplained"][row, view_index] = _complex_to_iq(unexplained)
        arrays["jam_labels"][row, view_index] = sample.jammer_multihot
        arrays["quality"][row, view_index] = np.array(
            [
                np.clip(
                    item.snr_db / _QUALITY_DB_SCALE,
                    *_QUALITY_CLIP,
                ),
                np.clip(
                    item.sir_db / _QUALITY_DB_SCALE,
                    *_QUALITY_CLIP,
                ),
                np.clip(
                    item.target_configuration.maximum_doppler_hz
                    / max(max_doppler_hz, 1e-12),
                    0.0,
                    _QUALITY_CLIP[1],
                ),
            ],
            dtype=np.float32,
        )
        arrays["quality_mask"][row, view_index] = np.array(
            [1.0, float(active_jammer), 1.0],
            dtype=np.float32,
        )
        arrays["snr_db"][row, view_index] = item.snr_db
        arrays["sir_db"][row, view_index] = item.sir_db
        arrays["overlap"][row, view_index] = float(
            sample.metadata["jammer_to_signal_overlap"]
        )
        arrays["speed_kmh"][row, view_index] = item.speed_kmh
        arrays["doppler_hz"][row, view_index] = (
            item.target_configuration.maximum_doppler_hz
        )
        arrays["target_profile_index"][row, view_index] = _PROFILE_INDEX[
            item.target_configuration.profile
        ]
        arrays["jammer_profile_index"][row, view_index] = _PROFILE_INDEX[
            item.jammer_configuration.profile
        ]
        arrays["condition_seed"][row, view_index] = item.condition_seed
        arrays["target_channel_seed"][row, view_index] = (
            item.target_configuration.seed
        )
        arrays["jammer_channel_seed"][row, view_index] = (
            item.jammer_configuration.seed
        )
        arrays["label"][row] = build_config.modulations.index(
            item.modulation
        )
        arrays["source_id"][row] = item.source_id

        record_metadata = {
            key: value
            for key, value in sample.metadata.items()
            if key != "generated_utc"
        }
        records[item.split][row]["views"][view_index] = _jsonable(
            {
                **record_metadata,
                "component_sum_max_abs_error": maximum_component_residual,
                "snr_target_abs_error_db": snr_error,
                "sir_target_abs_error_db": sir_error,
            }
        )

    for arrays in split_arrays.values():
        for array in arrays.values():
            array.flush()
    # Release memmap handles before checksumming and consumer loading.
    del array
    del arrays
    split_arrays.clear()

    files: dict[str, dict[str, dict[str, object]]] = {}
    for split, size in build_config.split_sizes:
        files[split] = {}
        shapes = _array_shapes(
            size,
            build_config.sample_length,
            len(synthesis_config.jammer_types),
        )
        for name, shape in shapes.items():
            path = root / split / f"{name}.npy"
            files[split][name] = {
                "path": f"{split}/{name}.npy",
                "dtype": str(_ARRAY_DTYPES[name]),
                "shape": list(shape),
                "sha256": _sha256_file(path),
            }

    profile_policy: dict[str, object]
    if build_config.split_policies is None:
        profile_policy = {
            "train_and_validation": list(build_config.train_profiles),
            "heldout_channel": list(build_config.heldout_profiles),
        }
    else:
        profile_policy = {
            policy.split: list(policy.profiles)
            for policy in build_config.split_policies
        }
    factor_coverage = _actual_factor_coverage(records, build_config)
    deterministic_payload = {
        "schema_version": (
            2 if build_config.split_policies is not None else 1
        ),
        "builder": "vimd_amc.standards.cache.build_tdl_paired_cache",
        "configuration": asdict(build_config),
        "jammer_taxonomy": list(synthesis_config.jammer_types),
        "protocol_exclusions": (
            _FACTOR_PROTOCOL_EXCLUSIONS
            if build_config.split_policies is not None
            else None
        ),
        "profile_policy": profile_policy,
        "split_roles": (
            {
                policy.split: policy.role
                for policy in build_config.split_policies
            }
            if build_config.split_policies is not None
            else {
                "train": "model_fitting",
                "validation": "checkpoint_selection",
                "heldout_channel": "tdl_profile_held_out_test",
            }
        ),
        "preregistered_split_policy": (
            {
                policy.split: asdict(policy)
                for policy in build_config.split_policies
            }
            if build_config.split_policies is not None
            else None
        ),
        "quality_normalization": {
            "snr_db": {
                "scale": _QUALITY_DB_SCALE,
                "unit": "dB",
                "formula": "clip(snr_db / scale, -1.0, 1.5)",
                "quality_index": 0,
                "validity": "always",
            },
            "sir_db": {
                "scale": _QUALITY_DB_SCALE,
                "unit": "dB",
                "formula": "clip(sir_db / scale, -1.0, 1.5)",
                "quality_index": 1,
                "validity": "quality_mask[1] == 1",
            },
            "doppler_hz": {
                "scale": max(max_doppler_hz, 1e-12),
                "unit": "Hz",
                "formula": "clip(maximum_doppler_hz / scale, 0.0, 1.5)",
                "quality_index": 2,
                "validity": "always",
                "speed_to_doppler_formula": (
                    "speed_kmh / 3.6 * carrier_frequency_hz / 299792458"
                ),
            },
        },
        "factor_coverage": factor_coverage,
        "component_audit": component_audit,
        "jammer_generation_policy": {
            "selection_stage": "prechannel_retained_window_power_only",
            "retained_window_power_must_exceed": (
                _ACTIVE_JAMMER_PRECHANNEL_POWER_THRESHOLD
            ),
            "maximum_deterministic_attempts": 32,
            "attempt_zero_named_seed": (
                "stable_seed(condition_seed, 'jammer_waveform')"
            ),
            "retry_named_seed": (
                "stable_seed(condition_seed, 'jammer_waveform_retry', attempt)"
            ),
            "uses_postchannel_or_model_outcome": False,
            "rejection_reason_recorded_per_view": True,
        },
        "source_ids": source_groups,
        "records": records,
        "files": files,
    }
    cache_digest = manifest_digest([_jsonable(deterministic_payload)])
    manifest: dict[str, Any] = {
        **_jsonable(deterministic_payload),
        "cache_digest": cache_digest,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return TDLCacheBuildResult(root=root, manifest=manifest)


class CachedPairedAMCDataset(Dataset):
    """Read-only memory-mapped paired AMC dataset produced by the TDL builder."""

    def __init__(
        self,
        cache_root: str | Path,
        split: str,
        *,
        verify_checksums: bool = False,
    ):
        self.cache_root = Path(cache_root).resolve()
        manifest_path = self.cache_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"cache manifest not found: {manifest_path}")
        self._manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        digest_payload = {
            key: value
            for key, value in self._manifest.items()
            if key != "cache_digest"
        }
        actual_manifest_digest = manifest_digest([digest_payload])
        if actual_manifest_digest != self._manifest.get("cache_digest"):
            raise RuntimeError(
                "cache manifest digest mismatch: "
                f"{actual_manifest_digest} != "
                f"{self._manifest.get('cache_digest')}"
            )
        if split not in self._manifest["files"]:
            raise ValueError(f"split is absent from cache manifest: {split}")
        if split not in self._manifest["source_ids"]:
            raise ValueError(
                f"split source IDs are absent from cache manifest: {split}"
            )
        self.split = split
        self._arrays: dict[str, np.ndarray] = {}
        for name, specification in self._manifest["files"][split].items():
            path = self.cache_root / specification["path"]
            if verify_checksums:
                actual = _sha256_file(path)
                if actual != specification["sha256"]:
                    raise RuntimeError(
                        f"checksum mismatch for {path}: "
                        f"{actual} != {specification['sha256']}"
                    )
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(array.shape) != specification["shape"]:
                raise RuntimeError(f"shape mismatch for cache array: {path}")
            self._arrays[name] = array
        self.size = int(self._arrays["source_id"].shape[0])
        source_sets = list(self._manifest["source_ids"].values())
        assert_disjoint_source_ids(*source_sets)

    def __len__(self) -> int:
        return self.size

    def source_ids(self) -> list[int]:
        return [int(value) for value in self._arrays["source_id"]]

    def manifest(self) -> dict[str, Any]:
        return self._manifest

    def close(self) -> None:
        """Close Windows file mappings so cache directories can be removed."""
        for array in self._arrays.values():
            mapping = getattr(array, "_mmap", None)
            if mapping is not None:
                mapping.close()
        self._arrays.clear()

    def __enter__(self) -> "CachedPairedAMCDataset":
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        arrays = getattr(self, "_arrays", None)
        if arrays:
            self.close()

    @staticmethod
    def _tensor(values: np.ndarray) -> torch.Tensor:
        # Copy one sample out of the read-only memmap to avoid writable-tensor
        # warnings while retaining memory mapping for the full cache.
        return torch.from_numpy(np.array(values, copy=True))

    def _view(self, index: int, view: int) -> dict[str, torch.Tensor]:
        result = {
            "x": self._tensor(self._arrays["x"][index, view]),
            "clean": self._tensor(self._arrays["clean"][index, view]),
            "jammer": self._tensor(self._arrays["jammer"][index, view]),
            "noise": self._tensor(self._arrays["noise"][index, view]),
            "receiver_artifact": self._tensor(
                self._arrays["receiver_artifact"][index, view]
            ),
            "unexplained": self._tensor(
                self._arrays["unexplained"][index, view]
            ),
            "jam_labels": self._tensor(
                self._arrays["jam_labels"][index, view]
            ),
            "quality": self._tensor(self._arrays["quality"][index, view]),
            "quality_mask": self._tensor(
                self._arrays["quality_mask"][index, view]
            ),
            "snr_db": self._tensor(self._arrays["snr_db"][index, view]),
            "sir_db": self._tensor(self._arrays["sir_db"][index, view]),
            "overlap": self._tensor(self._arrays["overlap"][index, view]),
            "speed_kmh": self._tensor(
                self._arrays["speed_kmh"][index, view]
            ),
            "scenario_index": self._tensor(
                self._arrays["target_profile_index"][index, view]
            ),
            "target_profile_index": self._tensor(
                self._arrays["target_profile_index"][index, view]
            ),
            "jammer_profile_index": self._tensor(
                self._arrays["jammer_profile_index"][index, view]
            ),
            "condition_seed": self._tensor(
                self._arrays["condition_seed"][index, view]
            ),
            "target_channel_seed": self._tensor(
                self._arrays["target_channel_seed"][index, view]
            ),
            "jammer_channel_seed": self._tensor(
                self._arrays["jammer_channel_seed"][index, view]
            ),
        }
        if "doppler_hz" in self._arrays:
            result["doppler_hz"] = self._tensor(
                self._arrays["doppler_hz"][index, view]
            )
        return result

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0:
            index += self.size
        if index < 0 or index >= self.size:
            raise IndexError(index)
        return {
            "view1": self._view(index, 0),
            "view2": self._view(index, 1),
            "label": self._tensor(self._arrays["label"][index]),
            "source_id": self._tensor(self._arrays["source_id"][index]),
        }


def validate_cached_components(
    dataset: CachedPairedAMCDataset,
    *,
    component_tolerance: float = 2e-6,
    power_tolerance_db: float = 2e-3,
) -> dict[str, float | int]:
    """Recompute component identities and SNR/SIR from cached IQ arrays."""
    max_component_error = 0.0
    max_snr_error_db = 0.0
    max_sir_error_db = 0.0
    min_active_jammer_power = float("inf")
    max_clean_jammer_power = 0.0
    max_quality_normalization_error = 0.0
    active_jammer_view_count = 0
    clean_view_count = 0
    for index in range(len(dataset)):
        item = dataset[index]
        for view_name in ("view1", "view2"):
            view = item[view_name]
            x = _iq_to_complex(view["x"].numpy())
            clean = _iq_to_complex(view["clean"].numpy())
            jammer = _iq_to_complex(view["jammer"].numpy())
            noise = _iq_to_complex(view["noise"].numpy())
            unexplained = _iq_to_complex(view["unexplained"].numpy())
            jammer_power = float(np.mean(np.abs(jammer) ** 2))
            sir_valid = bool(float(view["quality_mask"][1]) > 0.5)
            if sir_valid:
                if not np.isfinite(jammer_power) or jammer_power <= 0.0:
                    raise AssertionError(
                        "cached active jammer has nonpositive/nonfinite power: "
                        f"index={index}, view={view_name}, power={jammer_power}"
                    )
                min_active_jammer_power = min(
                    min_active_jammer_power, jammer_power
                )
                active_jammer_view_count += 1
            else:
                if not np.isfinite(jammer_power) or jammer_power > 1e-12:
                    raise AssertionError(
                        "cached clean record has unexpected jammer power: "
                        f"index={index}, view={view_name}, power={jammer_power}"
                    )
                if bool(torch.any(view["jam_labels"] != 0)):
                    raise AssertionError(
                        "cached clean record has a positive jammer label: "
                        f"index={index}, view={view_name}"
                    )
                max_clean_jammer_power = max(
                    max_clean_jammer_power, jammer_power
                )
                clean_view_count += 1
            stored_snr_db = float(view["snr_db"])
            stored_sir_db = float(view["sir_db"])
            quality = view["quality"].numpy()
            if not np.isfinite(stored_snr_db):
                raise AssertionError(
                    f"cached SNR is nonfinite: index={index}, view={view_name}"
                )
            if not np.isfinite(stored_sir_db):
                raise AssertionError(
                    f"cached SIR is nonfinite: index={index}, view={view_name}"
                )
            normalization = dataset.manifest().get("quality_normalization")
            if isinstance(normalization, Mapping):
                expected_quality = np.array(
                    [
                        np.clip(
                            stored_snr_db
                            / float(normalization["snr_db"]["scale"]),
                            *_QUALITY_CLIP,
                        ),
                        np.clip(
                            stored_sir_db
                            / float(normalization["sir_db"]["scale"]),
                            *_QUALITY_CLIP,
                        ),
                        np.clip(
                            float(view["doppler_hz"])
                            / float(normalization["doppler_hz"]["scale"]),
                            0.0,
                            _QUALITY_CLIP[1],
                        ),
                    ],
                    dtype=np.float32,
                )
                max_quality_normalization_error = max(
                    max_quality_normalization_error,
                    float(np.max(np.abs(quality - expected_quality))),
                )
            max_component_error = max(
                max_component_error,
                float(np.max(np.abs(x - clean - jammer - unexplained))),
            )
            max_snr_error_db = max(
                max_snr_error_db,
                abs(measured_snr_db(clean, noise) - stored_snr_db),
            )
            if sir_valid:
                max_sir_error_db = max(
                    max_sir_error_db,
                    abs(
                        measured_sir_db(clean, jammer)
                        - stored_sir_db
                    ),
                )
    if max_component_error > component_tolerance:
        raise AssertionError(
            f"cached component residual {max_component_error} exceeds "
            f"{component_tolerance}"
        )
    if max_snr_error_db > power_tolerance_db:
        raise AssertionError(
            f"cached SNR error {max_snr_error_db} dB exceeds "
            f"{power_tolerance_db} dB"
        )
    if max_sir_error_db > power_tolerance_db:
        raise AssertionError(
            f"cached SIR error {max_sir_error_db} dB exceeds "
            f"{power_tolerance_db} dB"
        )
    if max_quality_normalization_error > 1e-6:
        raise AssertionError(
            "cached physical-quality normalization error "
            f"{max_quality_normalization_error} exceeds 1e-6"
        )
    return {
        "max_component_error": max_component_error,
        "max_snr_error_db": max_snr_error_db,
        "max_sir_error_db": max_sir_error_db,
        "min_active_jammer_power": (
            min_active_jammer_power
            if active_jammer_view_count
            else 0.0
        ),
        "max_clean_jammer_power": max_clean_jammer_power,
        "max_quality_normalization_error": (
            max_quality_normalization_error
        ),
        "active_jammer_view_count": active_jammer_view_count,
        "clean_view_count": clean_view_count,
    }
