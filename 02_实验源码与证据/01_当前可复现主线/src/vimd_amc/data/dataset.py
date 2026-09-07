"""Paired-view datasets for controlled cross-condition AMC experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .split import manifest_digest, source_sequence_id, stable_seed
from .synthesis import SampleComponents, SignalSynthesizer


@dataclass(frozen=True)
class Regime:
    name: str
    snr_db: tuple[float, float]
    sir_db: tuple[float, float]
    jammer_choices: tuple[str, ...]
    channel_scenarios: tuple[str, ...]
    speed_kmh: tuple[float, ...]
    mixture_pool: tuple[str, ...] | None = None
    overlap_profiles: tuple[str, ...] | None = ("low", "mid", "high")

    @classmethod
    def train(cls) -> "Regime":
        return cls(
            name="train",
            snr_db=(-10.0, 20.0),
            sir_db=(-12.0, 20.0),
            jammer_choices=(
                "none",
                "tone",
                "multitone",
                "chirp",
                "pulse",
                "partial_band",
                "cochannel",
                "mixed",
            ),
            channel_scenarios=(
                "proxy_highway_los",
                "proxy_highway_nlos",
                "proxy_urban_los",
            ),
            speed_kmh=(0.0, 30.0, 60.0, 90.0, 120.0, 150.0),
            mixture_pool=("tone", "chirp", "pulse", "partial_band", "cochannel"),
        )

    @classmethod
    def validation(cls) -> "Regime":
        base = cls.train()
        return cls(
            name="validation",
            snr_db=base.snr_db,
            sir_db=base.sir_db,
            jammer_choices=base.jammer_choices,
            channel_scenarios=base.channel_scenarios,
            speed_kmh=base.speed_kmh,
            mixture_pool=base.mixture_pool,
            overlap_profiles=base.overlap_profiles,
        )

    @classmethod
    def in_distribution_test(cls) -> "Regime":
        base = cls.train()
        return cls(
            name="test",
            snr_db=base.snr_db,
            sir_db=base.sir_db,
            jammer_choices=base.jammer_choices,
            channel_scenarios=base.channel_scenarios,
            speed_kmh=base.speed_kmh,
            mixture_pool=base.mixture_pool,
            overlap_profiles=base.overlap_profiles,
        )

    @classmethod
    def hard_interference(cls) -> "Regime":
        return cls(
            name="hard",
            snr_db=(-8.0, 4.0),
            sir_db=(-15.0, 0.0),
            jammer_choices=("tone", "chirp", "pulse", "partial_band", "cochannel", "mixed"),
            channel_scenarios=("proxy_highway_nlos", "proxy_urban_los"),
            speed_kmh=(60.0, 120.0, 150.0),
            mixture_pool=("tone", "chirp", "pulse", "partial_band", "cochannel"),
            overlap_profiles=("mid", "high"),
        )

    @classmethod
    def unseen_jammer(cls) -> "Regime":
        return cls(
            name="unseen_jammer",
            snr_db=(-10.0, 16.0),
            sir_db=(-15.0, 5.0),
            jammer_choices=("sweep", "comb", "ofdm_like", "mixed"),
            channel_scenarios=(
                "proxy_highway_los",
                "proxy_highway_nlos",
                "proxy_urban_los",
            ),
            speed_kmh=(30.0, 90.0, 150.0),
            mixture_pool=("sweep", "comb", "ofdm_like"),
            overlap_profiles=("low", "mid", "high"),
        )

    @classmethod
    def unseen_speed(cls) -> "Regime":
        return cls(
            name="unseen_speed",
            snr_db=(-8.0, 16.0),
            sir_db=(-10.0, 10.0),
            jammer_choices=("tone", "chirp", "pulse", "partial_band", "cochannel", "mixed"),
            channel_scenarios=(
                "proxy_highway_los",
                "proxy_highway_nlos",
                "proxy_urban_los",
            ),
            speed_kmh=(180.0, 250.0),
            mixture_pool=("tone", "chirp", "pulse", "partial_band", "cochannel"),
            overlap_profiles=("low", "mid", "high"),
        )

    @classmethod
    def unseen_channel(cls) -> "Regime":
        return cls(
            name="unseen_channel",
            snr_db=(-8.0, 16.0),
            sir_db=(-10.0, 10.0),
            jammer_choices=("tone", "chirp", "pulse", "partial_band", "cochannel", "mixed"),
            channel_scenarios=("proxy_urban_nlos",),
            speed_kmh=(0.0, 30.0, 60.0, 90.0, 120.0, 150.0),
            mixture_pool=("tone", "chirp", "pulse", "partial_band", "cochannel"),
            overlap_profiles=("low", "mid", "high"),
        )

    @classmethod
    def unseen_speed_and_channel(cls) -> "Regime":
        return cls(
            name="unseen_speed_and_channel",
            snr_db=(-8.0, 16.0),
            sir_db=(-10.0, 10.0),
            jammer_choices=("tone", "chirp", "pulse", "partial_band", "cochannel", "mixed"),
            channel_scenarios=("proxy_urban_nlos",),
            speed_kmh=(180.0, 250.0),
            mixture_pool=("tone", "chirp", "pulse", "partial_band", "cochannel"),
            overlap_profiles=("low", "mid", "high"),
        )

    @classmethod
    def unseen_mobility(cls) -> "Regime":
        """Backward-compatible alias for the deliberately combined stress test."""

        return cls.unseen_speed_and_channel()

    @classmethod
    def clean_high_snr(cls) -> "Regime":
        return cls(
            name="clean",
            snr_db=(12.0, 20.0),
            sir_db=(20.0, 20.0),
            jammer_choices=("none",),
            channel_scenarios=("proxy_highway_los", "proxy_urban_los"),
            speed_kmh=(0.0, 60.0, 120.0),
            overlap_profiles=None,
        )


def _to_iq_tensor(values: np.ndarray) -> torch.Tensor:
    stacked = np.stack((values.real, values.imag), axis=0).astype(np.float32, copy=False)
    return torch.from_numpy(stacked)


class PairedAMCDataset(Dataset):
    """Two independently impaired views of one immutable source sequence."""

    def __init__(
        self,
        *,
        synthesizer: SignalSynthesizer,
        split: str,
        size: int,
        regime: Regime,
        master_seed: int,
        modulations: Sequence[str] | None = None,
        cache_in_memory: bool = False,
    ):
        if size <= 0:
            raise ValueError("size must be positive")
        self.synthesizer = synthesizer
        self.split = split
        self.size = int(size)
        self.regime = regime
        self.master_seed = int(master_seed)
        self.cache_in_memory = bool(cache_in_memory)
        self._cache: dict[int, dict[str, object]] = {}
        self.modulations = tuple(modulations or synthesizer.config.modulations)
        unknown = set(self.modulations).difference(synthesizer.config.modulations)
        if unknown:
            raise ValueError(f"Unknown modulations: {sorted(unknown)}")
        c = synthesizer.config
        self.max_doppler_norm = (
            (c.max_speed_kmh / 3.6) * c.carrier_hz / 299_792_458.0 / c.sample_rate_hz
        )
        self.quality_normalization = {
            "snr_db": {"scale": 20.0, "offset": 0.0, "unit": "dB"},
            "sir_db": {"scale": 20.0, "offset": 0.0, "unit": "dB"},
            "doppler_hz": {
                "scale": self.max_doppler_norm * c.sample_rate_hz,
                "offset": 0.0,
                "unit": "Hz",
            },
        }

    def __len__(self) -> int:
        return self.size

    def source_ids(self) -> list[int]:
        return [
            source_sequence_id(self.split, index, self.master_seed)
            for index in range(self.size)
        ]

    def manifest(self) -> dict[str, object]:
        records = [
            {
                "index": index,
                "source_sequence_id": source_sequence_id(self.split, index, self.master_seed),
                "modulation": self.modulations[index % len(self.modulations)],
                "split": self.split,
                "views": [
                    self._draw_condition(index, 1),
                    self._draw_condition(index, 2),
                ],
            }
            for index in range(self.size)
        ]
        regime_configuration = asdict(self.regime)
        synthesis_configuration = asdict(self.synthesizer.config)
        quality_normalization = self.quality_normalization
        return {
            "split": self.split,
            "regime": self.regime.name,
            "regime_configuration": regime_configuration,
            "synthesis_configuration": synthesis_configuration,
            "quality_normalization": quality_normalization,
            "size": self.size,
            "master_seed": self.master_seed,
            "digest": manifest_digest(
                [
                    {
                        "records": records,
                        "regime_configuration": regime_configuration,
                        "synthesis_configuration": synthesis_configuration,
                        "quality_normalization": quality_normalization,
                    }
                ]
            ),
            "records": records,
        }

    def _draw_condition(self, index: int, view: int) -> dict[str, object]:
        condition_seed = stable_seed(self.master_seed, self.split, self.regime.name, index, view)
        rng = np.random.default_rng(condition_seed)
        snr = float(rng.uniform(*self.regime.snr_db))
        sir = float(rng.uniform(*self.regime.sir_db))
        jammer = str(rng.choice(self.regime.jammer_choices))
        scenario = str(rng.choice(self.regime.channel_scenarios))
        speed = float(rng.choice(self.regime.speed_kmh))
        if jammer == "none":
            overlap_profile = None
        elif jammer == "cochannel":
            overlap_profile = "high"
        elif self.regime.overlap_profiles:
            overlap_profile = str(rng.choice(self.regime.overlap_profiles))
        else:
            overlap_profile = None
        return {
            "condition_seed": condition_seed,
            "snr_db": snr,
            "sir_db": sir,
            "jammer_name": jammer,
            "channel_scenario": scenario,
            "speed_kmh": speed,
            "overlap_profile": overlap_profile,
        }

    def _materialize_view(
        self,
        *,
        source_id: int,
        modulation: str,
        condition: dict[str, object],
    ) -> dict[str, torch.Tensor]:
        sample: SampleComponents = self.synthesizer.synthesize(
            modulation=modulation,
            source_seed=source_id,
            condition_seed=int(condition["condition_seed"]),
            snr_db=float(condition["snr_db"]),
            sir_db=float(condition["sir_db"]),
            jammer_name=str(condition["jammer_name"]),
            channel_scenario=str(condition["channel_scenario"]),
            speed_kmh=float(condition["speed_kmh"]),
            mixture_pool=self.regime.mixture_pool,
            overlap_profile=(
                None
                if condition["overlap_profile"] is None
                else str(condition["overlap_profile"])
            ),
        )
        metadata = sample.metadata
        sir_valid = bool(metadata["sir_valid"])
        sir_value = float(metadata["sir_db"]) if sir_valid else 0.0
        quality = torch.tensor(
            [
                np.clip(
                    float(metadata["snr_db"])
                    / float(self.quality_normalization["snr_db"]["scale"]),
                    -1.0,
                    1.5,
                ),
                np.clip(
                    sir_value
                    / float(self.quality_normalization["sir_db"]["scale"]),
                    -1.0,
                    1.5,
                ),
                np.clip(float(metadata["doppler_norm"]) / self.max_doppler_norm, 0.0, 1.5),
            ],
            dtype=torch.float32,
        )
        scenario_index = self.synthesizer.config.channel_scenarios.index(
            str(metadata["channel_scenario"])
        )
        return {
            "x": _to_iq_tensor(sample.mixture),
            "clean": _to_iq_tensor(sample.clean),
            "jammer": _to_iq_tensor(sample.jammer),
            "noise": _to_iq_tensor(sample.noise),
            "receiver_artifact": _to_iq_tensor(sample.receiver_artifact),
            "unexplained": _to_iq_tensor(sample.noise + sample.receiver_artifact),
            "jam_labels": torch.from_numpy(sample.jammer_multihot.copy()),
            "quality": quality,
            "quality_mask": torch.tensor([1.0, float(sir_valid), 1.0], dtype=torch.float32),
            "snr_db": torch.tensor(float(metadata["snr_db"]), dtype=torch.float32),
            "sir_db": torch.tensor(
                float(metadata["sir_db"]) if sir_valid else float("inf"),
                dtype=torch.float32,
            ),
            "doppler_hz": torch.tensor(
                float(metadata["doppler_norm"])
                * self.synthesizer.config.sample_rate_hz,
                dtype=torch.float32,
            ),
            "overlap": torch.tensor(
                float(metadata["jammer_to_signal_overlap"]),
                dtype=torch.float32,
            ),
            "speed_kmh": torch.tensor(float(metadata["vehicle_speed_kmh"]), dtype=torch.float32),
            "scenario_index": torch.tensor(scenario_index, dtype=torch.long),
            "condition_seed": torch.tensor(int(metadata["condition_seed"]), dtype=torch.long),
        }

    def _build_item(self, index: int) -> dict[str, object]:
        if index < 0 or index >= self.size:
            raise IndexError(index)
        modulation = self.modulations[index % len(self.modulations)]
        source_id = source_sequence_id(self.split, index, self.master_seed)
        first = self._materialize_view(
            source_id=source_id,
            modulation=modulation,
            condition=self._draw_condition(index, 1),
        )
        second = self._materialize_view(
            source_id=source_id,
            modulation=modulation,
            condition=self._draw_condition(index, 2),
        )
        return {
            "view1": first,
            "view2": second,
            "label": torch.tensor(
                self.modulations.index(modulation),
                dtype=torch.long,
            ),
            "source_id": torch.tensor(source_id, dtype=torch.long),
        }

    def __getitem__(self, index: int) -> dict[str, object]:
        if not self.cache_in_memory:
            return self._build_item(index)
        if index not in self._cache:
            self._cache[index] = self._build_item(index)
        return self._cache[index]
