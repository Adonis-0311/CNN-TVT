"""Deterministic complex-baseband synthesis with explicit power bookkeeping.

The generator is deliberately transparent.  Every sample returns the
channel-distorted target component, the scaled jammer component, and AWGN
separately so that SNR/SIR and latent-mask supervision can be audited.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from typing import Any, Iterable

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class SynthesisConfig:
    sample_length: int = 512
    samples_per_symbol: int = 4
    sample_rate_hz: float = 1_000_000.0
    carrier_hz: float = 5_900_000_000.0
    rrc_rolloff: float = 0.35
    rrc_span_symbols: int = 8
    modulations: tuple[str, ...] = (
        "BPSK",
        "PI2BPSK",
        "QPSK",
        "8PSK",
        "16QAM",
        "64QAM",
        "256QAM",
        "GMSK",
        "CPFSK",
        "4FSK",
    )
    jammer_types: tuple[str, ...] = (
        "tone",
        "multitone",
        "chirp",
        "sweep",
        "pulse",
        "partial_band",
        "comb",
        "cochannel",
        "ofdm_like",
    )
    channel_scenarios: tuple[str, ...] = (
        "proxy_highway_los",
        "proxy_highway_nlos",
        "proxy_urban_los",
        "proxy_urban_nlos",
    )
    max_speed_kmh: float = 250.0


@dataclass
class SampleComponents:
    mixture: np.ndarray
    clean: np.ndarray
    jammer: np.ndarray
    noise: np.ndarray
    receiver_artifact: np.ndarray
    modulation_index: int
    jammer_multihot: np.ndarray
    metadata: dict[str, Any]


def signal_power(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x) ** 2))


def measured_snr_db(clean: np.ndarray, noise: np.ndarray) -> float:
    return float(10.0 * np.log10((signal_power(clean) + EPS) / (signal_power(noise) + EPS)))


def measured_sir_db(clean: np.ndarray, jammer: np.ndarray) -> float:
    if signal_power(jammer) <= EPS:
        return float("inf")
    return float(10.0 * np.log10((signal_power(clean) + EPS) / (signal_power(jammer) + EPS)))


def _unit_power(x: np.ndarray) -> np.ndarray:
    return x / np.sqrt(signal_power(x) + EPS)


def _named_rng(parent_seed: int, name: str, attempt: int = 0) -> np.random.Generator:
    payload = f"{parent_seed}|{name}|{attempt}".encode("utf-8")
    child_seed = int.from_bytes(blake2b(payload, digest_size=8).digest(), "little")
    return np.random.default_rng(child_seed)


def _rrc_taps(beta: float, samples_per_symbol: int, span_symbols: int) -> np.ndarray:
    """Return energy-normalized root-raised-cosine taps."""
    half = span_symbols * samples_per_symbol // 2
    t = np.arange(-half, half + 1, dtype=np.float64) / samples_per_symbol
    h = np.zeros_like(t)
    for idx, value in enumerate(t):
        if abs(value) < 1e-12:
            h[idx] = 1.0 + beta * (4.0 / np.pi - 1.0)
        elif beta > 0 and abs(abs(4.0 * beta * value) - 1.0) < 1e-10:
            h[idx] = (
                beta
                / np.sqrt(2.0)
                * (
                    (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                    + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
                )
            )
        else:
            numerator = (
                np.sin(np.pi * value * (1.0 - beta))
                + 4.0 * beta * value * np.cos(np.pi * value * (1.0 + beta))
            )
            denominator = np.pi * value * (1.0 - (4.0 * beta * value) ** 2)
            h[idx] = numerator / denominator
    return h / np.sqrt(np.sum(h**2) + EPS)


class SignalSynthesizer:
    """Generate traceable AMC samples from a source seed and condition seed."""

    # Development-only heuristic profiles.  They are intentionally prefixed
    # ``proxy_`` and must not be described as 3GPP-compliant.
    _CHANNEL_PROFILES: dict[str, tuple[list[int], list[float], float]] = {
        "proxy_highway_los": ([0, 2, 5], [0.0, -9.0, -15.0], 10.0),
        "proxy_highway_nlos": ([0, 2, 6, 11], [0.0, -3.0, -8.0, -14.0], 0.0),
        "proxy_urban_los": ([0, 1, 4, 9], [0.0, -5.0, -10.0, -16.0], 6.0),
        "proxy_urban_nlos": ([0, 1, 3, 7, 13], [0.0, -2.0, -6.0, -11.0, -17.0], 0.0),
    }

    def __init__(self, config: SynthesisConfig | None = None):
        self.config = config or SynthesisConfig()
        self._rrc = _rrc_taps(
            self.config.rrc_rolloff,
            self.config.samples_per_symbol,
            self.config.rrc_span_symbols,
        )

    def modulation_index(self, name: str) -> int:
        return self.config.modulations.index(name)

    def _linear_symbols(self, name: str, count: int, rng: np.random.Generator) -> np.ndarray:
        if name == "BPSK":
            return (2 * rng.integers(0, 2, count) - 1).astype(np.complex128)
        if name == "PI2BPSK":
            base = 2 * rng.integers(0, 2, count) - 1
            return base * np.exp(1j * np.pi * np.arange(count) / 2.0)
        if name in {"QPSK", "8PSK"}:
            order = 4 if name == "QPSK" else 8
            index = rng.integers(0, order, count)
            offset = np.pi / order
            return np.exp(1j * (2.0 * np.pi * index / order + offset))
        if name.endswith("QAM"):
            order = int(name[:-3])
            side = int(np.sqrt(order))
            levels = np.arange(-(side - 1), side, 2, dtype=np.float64)
            symbols = rng.choice(levels, count) + 1j * rng.choice(levels, count)
            return _unit_power(symbols)
        raise ValueError(f"Unsupported linear modulation: {name}")

    def _pulse_shape(self, symbols: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        sps = self.config.samples_per_symbol
        upsampled = np.zeros(len(symbols) * sps, dtype=np.complex128)
        upsampled[::sps] = symbols
        shaped = np.convolve(upsampled, self._rrc, mode="same")
        timing = int(rng.integers(0, sps))
        if timing:
            shaped = np.roll(shaped, timing)
        return shaped

    def _continuous_phase(self, name: str, count: int, rng: np.random.Generator) -> np.ndarray:
        sps = self.config.samples_per_symbol
        if name == "GMSK":
            symbol_count = int(np.ceil(count / sps)) + 16
            bits = (2 * rng.integers(0, 2, symbol_count) - 1).astype(np.float64)
            frequency = np.repeat(bits, sps)
            span = 4 * sps
            axis = np.arange(-span, span + 1) / sps
            gaussian = np.exp(-0.5 * (axis / 0.55) ** 2)
            gaussian /= gaussian.sum()
            frequency = np.convolve(frequency, gaussian, mode="same")
            phase = np.cumsum((np.pi * 0.5 / sps) * frequency)
        elif name == "CPFSK":
            symbol_count = int(np.ceil(count / sps)) + 8
            levels = 2 * rng.integers(0, 2, symbol_count) - 1
            frequency = np.repeat(levels, sps) * (0.25 / sps)
            phase = np.cumsum(2.0 * np.pi * frequency)
        elif name == "4FSK":
            symbol_count = int(np.ceil(count / sps)) + 8
            levels = rng.choice(np.array([-3.0, -1.0, 1.0, 3.0]), symbol_count)
            frequency = np.repeat(levels, sps) * (0.12 / sps)
            phase = np.cumsum(2.0 * np.pi * frequency)
        else:
            raise ValueError(f"Unsupported continuous-phase modulation: {name}")
        return np.exp(1j * (phase + rng.uniform(-np.pi, np.pi)))

    def generate_source(self, modulation: str, source_seed: int, length: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(source_seed)
        target_length = length or self.config.sample_length
        margin = len(self._rrc) + 4 * self.config.samples_per_symbol
        required = target_length + margin
        if modulation in {"GMSK", "CPFSK", "4FSK"}:
            waveform = self._continuous_phase(modulation, required, rng)
        else:
            symbol_count = int(np.ceil(required / self.config.samples_per_symbol)) + 8
            waveform = self._pulse_shape(self._linear_symbols(modulation, symbol_count, rng), rng)
        if len(waveform) < target_length:
            waveform = np.resize(waveform, target_length)
        start_max = max(1, len(waveform) - target_length + 1)
        start = int(rng.integers(0, start_max))
        return _unit_power(waveform[start : start + target_length])

    def _apply_channel(
        self,
        source: np.ndarray,
        scenario: str,
        speed_kmh: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, float]:
        delays, powers_db, rician_k = self._CHANNEL_PROFILES[scenario]
        powers = 10.0 ** (np.asarray(powers_db, dtype=np.float64) / 10.0)
        powers /= powers.sum()
        speed_mps = speed_kmh / 3.6
        doppler_hz = speed_mps * self.config.carrier_hz / 299_792_458.0
        doppler_norm = doppler_hz / self.config.sample_rate_hz
        time = np.arange(len(source), dtype=np.float64)
        output = np.zeros_like(source, dtype=np.complex128)
        for tap_index, (delay, tap_power) in enumerate(zip(delays, powers)):
            scatter = (rng.normal() + 1j * rng.normal()) / np.sqrt(2.0)
            if tap_index == 0 and rician_k > 0:
                base = (
                    np.sqrt(rician_k / (rician_k + 1.0))
                    + np.sqrt(1.0 / (rician_k + 1.0)) * scatter
                )
            else:
                base = scatter
            direction = rng.uniform(-1.0, 1.0)
            evolution = np.exp(
                1j * (rng.uniform(-np.pi, np.pi) + 2.0 * np.pi * doppler_norm * direction * time)
            )
            # The caller supplies guard samples and crops the stable center,
            # preventing the leading-zero scenario fingerprint that would be
            # created by window-local convolution.
            shifted = np.zeros_like(source)
            if delay == 0:
                shifted[:] = source
            elif delay < len(source):
                shifted[delay:] = source[:-delay]
            output += np.sqrt(tap_power) * base * evolution * shifted
        return _unit_power(output), float(doppler_norm)

    @staticmethod
    def _apply_linear_receiver_impairments(
        signal: np.ndarray,
        cfo_norm: float,
        phase_offset: float,
        iq_gain_db: float,
        iq_phase_deg: float,
    ) -> np.ndarray:
        time = np.arange(len(signal), dtype=np.float64)
        rotated = signal * np.exp(1j * (2.0 * np.pi * cfo_norm * time + phase_offset))
        gain = 10.0 ** (iq_gain_db / 20.0)
        phase = np.deg2rad(iq_phase_deg)
        in_phase = gain * rotated.real
        quadrature = (rotated.imag * np.cos(phase) + rotated.real * np.sin(phase)) / gain
        return in_phase + 1j * quadrature

    @staticmethod
    def _stft_power(signal: np.ndarray, fft_size: int = 128, hop: int = 32) -> np.ndarray:
        if len(signal) < fft_size:
            signal = np.pad(signal, (0, fft_size - len(signal)))
        window = np.hanning(fft_size)
        frames = []
        for start in range(0, len(signal) - fft_size + 1, hop):
            spectrum = np.fft.fftshift(np.fft.fft(signal[start : start + fft_size] * window))
            frames.append(np.abs(spectrum) ** 2)
        return np.stack(frames, axis=0)

    @classmethod
    def spectral_overlap_metrics(
        cls,
        clean: np.ndarray,
        jammer: np.ndarray,
    ) -> dict[str, float]:
        if signal_power(jammer) <= EPS:
            return {
                "jammer_to_signal_overlap": 0.0,
                "histogram_intersection": 0.0,
                "overlap_sir_db": float("inf"),
            }
        target_power = cls._stft_power(clean)
        jammer_power = cls._stft_power(jammer)
        flat = target_power.ravel()
        order = np.argsort(flat)[::-1]
        cumulative = np.cumsum(flat[order])
        cutoff_index = int(np.searchsorted(cumulative, 0.99 * cumulative[-1], side="left"))
        support = np.zeros_like(flat, dtype=bool)
        support[order[: cutoff_index + 1]] = True
        support = support.reshape(target_power.shape)
        jammer_in_support = float(jammer_power[support].sum())
        jammer_total = float(jammer_power.sum())
        target_total = float(target_power.sum())
        target_distribution = target_power / (target_total + EPS)
        jammer_distribution = jammer_power / (jammer_total + EPS)
        return {
            "jammer_to_signal_overlap": jammer_in_support / (jammer_total + EPS),
            "histogram_intersection": float(
                np.minimum(target_distribution, jammer_distribution).sum()
            ),
            "overlap_sir_db": float(
                10.0 * np.log10((target_total + EPS) / (jammer_in_support + EPS))
            ),
        }

    def _frequency_by_overlap(
        self,
        rng: np.random.Generator,
        overlap_profile: str | None,
    ) -> float:
        nominal_edge = 0.5 * (1.0 + self.config.rrc_rolloff) / self.config.samples_per_symbol
        sign = float(rng.choice((-1.0, 1.0)))
        if overlap_profile == "high":
            return float(rng.uniform(-0.75 * nominal_edge, 0.75 * nominal_edge))
        if overlap_profile == "mid":
            return sign * float(rng.uniform(0.85 * nominal_edge, 1.25 * nominal_edge))
        if overlap_profile == "low":
            return sign * float(rng.uniform(1.65 * nominal_edge, 0.45))
        return float(rng.uniform(-0.42, 0.42))

    def _partial_band_noise(
        self,
        length: int,
        rng: np.random.Generator,
        overlap_profile: str | None,
    ) -> np.ndarray:
        spectrum = rng.normal(size=length) + 1j * rng.normal(size=length)
        spectrum = np.fft.fftshift(spectrum)
        mask = np.zeros(length, dtype=np.float64)
        width = int(rng.uniform(0.12, 0.38) * length)
        normalized_center = self._frequency_by_overlap(rng, overlap_profile)
        center = int(np.clip((normalized_center + 0.5) * length, width // 2, length - width // 2 - 1))
        low = max(0, center - width // 2)
        high = min(length, low + width)
        mask[low:high] = 1.0
        return np.fft.ifft(np.fft.ifftshift(spectrum * mask))

    def _ofdm_like(self, length: int, rng: np.random.Generator) -> np.ndarray:
        fft_size, cp = 64, 16
        blocks: list[np.ndarray] = []
        while sum(len(block) for block in blocks) < length:
            carriers = np.zeros(fft_size, dtype=np.complex128)
            active = np.r_[np.arange(6, 31), np.arange(34, 59)]
            qpsk = np.exp(1j * (np.pi / 4.0 + np.pi / 2.0 * rng.integers(0, 4, len(active))))
            carriers[active] = qpsk
            symbol = np.fft.ifft(carriers)
            blocks.append(np.r_[symbol[-cp:], symbol])
        return np.concatenate(blocks)[:length]

    def _single_jammer(
        self,
        name: str,
        length: int,
        rng: np.random.Generator,
        target_modulation: str,
        overlap_profile: str | None,
    ) -> np.ndarray:
        time = np.arange(length, dtype=np.float64)
        if name == "tone":
            frequency = self._frequency_by_overlap(rng, overlap_profile)
            jammer = np.exp(1j * (2.0 * np.pi * frequency * time + rng.uniform(-np.pi, np.pi)))
        elif name == "multitone":
            jammer = np.zeros(length, dtype=np.complex128)
            for _ in range(int(rng.integers(2, 6))):
                frequency = self._frequency_by_overlap(rng, overlap_profile)
                jammer += np.exp(1j * (2.0 * np.pi * frequency * time + rng.uniform(-np.pi, np.pi)))
        elif name == "chirp":
            edge = 0.5 * (1.0 + self.config.rrc_rolloff) / self.config.samples_per_symbol
            if overlap_profile == "high":
                start, stop = -0.9 * edge, 0.9 * edge
            elif overlap_profile == "mid":
                sign = float(rng.choice((-1.0, 1.0)))
                start, stop = sign * 0.7 * edge, sign * 1.6 * edge
            elif overlap_profile == "low":
                sign = float(rng.choice((-1.0, 1.0)))
                start, stop = sign * 1.7 * edge, sign * 2.5 * edge
            else:
                start = rng.uniform(-0.38, -0.08)
                stop = rng.uniform(0.08, 0.38)
            slope = (stop - start) / max(1, length - 1)
            jammer = np.exp(1j * 2.0 * np.pi * (start * time + 0.5 * slope * time**2))
        elif name == "sweep":
            period = int(rng.integers(max(16, length // 8), max(17, length // 2)))
            phase_position = (time % period) / period
            instantaneous = -0.4 + 0.8 * phase_position
            jammer = np.exp(1j * 2.0 * np.pi * np.cumsum(instantaneous))
        elif name == "pulse":
            carrier = np.exp(
                1j * 2.0 * np.pi * self._frequency_by_overlap(rng, overlap_profile) * time
            )
            gate = np.zeros(length, dtype=np.float64)
            for _ in range(int(rng.integers(2, 7))):
                width = int(rng.integers(max(3, length // 80), max(4, length // 12)))
                start = int(rng.integers(0, max(1, length - width)))
                gate[start : start + width] = 1.0
            jammer = carrier * gate
        elif name == "partial_band":
            jammer = self._partial_band_noise(length, rng, overlap_profile)
        elif name == "comb":
            jammer = np.zeros(length, dtype=np.complex128)
            spacing = rng.uniform(0.04, 0.1)
            center = self._frequency_by_overlap(rng, overlap_profile)
            for harmonic in range(-3, 4):
                jammer += np.exp(
                    1j
                    * (
                        2.0 * np.pi * (center + harmonic * spacing) * time
                        + rng.uniform(-np.pi, np.pi)
                    )
                )
        elif name == "cochannel":
            # Independent sampling intentionally allows same-modulation
            # collisions, preventing label-exclusion leakage.
            modulation = self.config.modulations[
                int(rng.integers(0, len(self.config.modulations)))
            ]
            jammer = self.generate_source(modulation, int(rng.integers(0, 2**31 - 1)), length)
            delay = int(rng.integers(0, max(1, self.config.samples_per_symbol * 2)))
            if delay:
                guarded = self.generate_source(
                    modulation,
                    int(rng.integers(0, 2**31 - 1)),
                    length + delay,
                )
                jammer = guarded[delay : delay + length]
        elif name == "ofdm_like":
            jammer = self._ofdm_like(length, rng)
        else:
            raise ValueError(f"Unknown jammer: {name}")
        return _unit_power(jammer)

    def generate_jammer(
        self,
        jammer_name: str,
        length: int,
        rng: np.random.Generator,
        target_modulation: str,
        mixture_pool: Iterable[str] | None = None,
        overlap_profile: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        labels = np.zeros(len(self.config.jammer_types), dtype=np.float32)
        if jammer_name == "none":
            return np.zeros(length, dtype=np.complex128), labels, ()
        if jammer_name == "mixed":
            pool = tuple(mixture_pool or self.config.jammer_types)
            chosen = tuple(rng.choice(pool, size=2, replace=False).tolist())
            jammer = np.zeros(length, dtype=np.complex128)
            for name in chosen:
                weight = 10.0 ** (rng.uniform(-3.0, 3.0) / 20.0)
                jammer += weight * self._single_jammer(
                    name,
                    length,
                    rng,
                    target_modulation,
                    overlap_profile,
                )
                labels[self.config.jammer_types.index(name)] = 1.0
            return _unit_power(jammer), labels, chosen
        jammer = self._single_jammer(
            jammer_name,
            length,
            rng,
            target_modulation,
            overlap_profile,
        )
        labels[self.config.jammer_types.index(jammer_name)] = 1.0
        return jammer, labels, (jammer_name,)

    def finalize_received_components(
        self,
        *,
        clean: np.ndarray,
        raw_jammer: np.ndarray,
        modulation: str,
        source_seed: int,
        condition_seed: int,
        snr_db: float,
        sir_db: float,
        jammer_name: str,
        jammer_labels: np.ndarray,
        jammer_components: tuple[str, ...],
        channel_scenario: str,
        channel_model: str,
        speed_kmh: float,
        doppler_norm: float,
        overlap_profile: str | None,
        channel_metadata: dict[str, Any] | None = None,
    ) -> SampleComponents:
        """Apply shared receiver effects, exact SNR/SIR scaling, and AGC.

        This public boundary lets an offline standards backend provide
        independently channelized target and jammer waveforms while retaining
        exactly the same receiver/noise bookkeeping as the proxy generator.
        """
        clean = np.asarray(clean, dtype=np.complex128).reshape(-1)
        raw_jammer = np.asarray(raw_jammer, dtype=np.complex128).reshape(-1)
        if len(clean) != self.config.sample_length or len(raw_jammer) != len(clean):
            raise ValueError("channelized clean/jammer components have an invalid length")
        if not np.isfinite(clean).all() or not np.isfinite(raw_jammer).all():
            raise ValueError("channelized components contain non-finite samples")
        if signal_power(clean) <= EPS:
            raise ValueError("channelized target component has zero power")
        receiver_rng = _named_rng(condition_seed, "receiver")
        noise_rng = _named_rng(condition_seed, "noise")
        raw_noise = _unit_power(
            noise_rng.normal(size=len(clean)) + 1j * noise_rng.normal(size=len(clean))
        )

        # Shared linear receiver effects are applied consistently to every RF
        # component.  Additive DC is tracked separately and is never presented
        # to the mask teacher as target signal.
        cfo_norm = float(receiver_rng.uniform(-0.012, 0.012))
        phase_offset = float(receiver_rng.uniform(-np.pi, np.pi))
        iq_gain_db = float(receiver_rng.uniform(-0.8, 0.8))
        iq_phase_deg = float(receiver_rng.uniform(-4.0, 4.0))
        clean = self._apply_linear_receiver_impairments(
            clean,
            cfo_norm,
            phase_offset,
            iq_gain_db,
            iq_phase_deg,
        )
        raw_jammer = self._apply_linear_receiver_impairments(
            raw_jammer,
            cfo_norm,
            phase_offset,
            iq_gain_db,
            iq_phase_deg,
        )
        raw_noise = self._apply_linear_receiver_impairments(
            raw_noise,
            cfo_norm,
            phase_offset,
            iq_gain_db,
            iq_phase_deg,
        )
        clean = _unit_power(clean)
        if jammer_name == "none":
            jammer = np.zeros_like(clean)
            reported_sir: float | None = None
        else:
            if signal_power(raw_jammer) <= EPS:
                raise ValueError("active channelized jammer component has zero power")
            desired_jammer_power = signal_power(clean) / (10.0 ** (sir_db / 10.0))
            jammer = raw_jammer * np.sqrt(
                desired_jammer_power / (signal_power(raw_jammer) + EPS)
            )
            reported_sir = float(sir_db)

        desired_noise_power = signal_power(clean) / (10.0 ** (snr_db / 10.0))
        noise = raw_noise * np.sqrt(
            desired_noise_power / (signal_power(raw_noise) + EPS)
        )

        dc_offset = complex(
            receiver_rng.normal(scale=0.015),
            receiver_rng.normal(scale=0.015),
        )
        receiver_artifact = np.full(len(clean), dc_offset, dtype=np.complex128)

        overlap = self.spectral_overlap_metrics(clean, jammer)
        if jammer_name == "none":
            activity_fraction = 0.0
            active_sir_db = float("inf")
        else:
            activity = np.abs(jammer) > 0.1 * np.max(np.abs(jammer))
            activity_fraction = float(activity.mean())
            active_sir_db = (
                float(
                    10.0
                    * np.log10(
                        (np.mean(np.abs(clean[activity]) ** 2) + EPS)
                        / (np.mean(np.abs(jammer[activity]) ** 2) + EPS)
                    )
                )
                if activity.any()
                else float("inf")
            )

        mixture = clean + jammer + noise + receiver_artifact
        agc = np.sqrt(signal_power(mixture) + EPS)
        mixture = mixture / agc
        clean = clean / agc
        jammer = jammer / agc
        noise = noise / agc
        receiver_artifact = receiver_artifact / agc

        metadata: dict[str, Any] = {
            "source_sequence_id": int(source_seed),
            "condition_seed": int(condition_seed),
            "modulation": modulation,
            "snr_db": float(snr_db),
            "sir_db": reported_sir,
            "measured_snr_db": measured_snr_db(clean, noise),
            "measured_sir_db": measured_sir_db(clean, jammer),
            "sir_valid": jammer_name != "none",
            "interference_present": jammer_name != "none",
            "jammer_name": jammer_name,
            "jammer_components": jammer_components,
            "channel_scenario": channel_scenario,
            "channel_model": channel_model,
            "vehicle_speed_kmh": float(speed_kmh),
            "doppler_norm": float(doppler_norm),
            "cfo_norm": cfo_norm,
            "iq_gain_db": iq_gain_db,
            "iq_phase_deg": iq_phase_deg,
            "window_rms_normalization_gain": float(1.0 / agc),
            "receiver_dc_real": float(dc_offset.real),
            "receiver_dc_imag": float(dc_offset.imag),
            "overlap_profile_requested": overlap_profile,
            "jammer_activity_fraction": activity_fraction,
            "active_sir_db": active_sir_db,
            **(channel_metadata or {}),
            **overlap,
        }
        return SampleComponents(
            mixture=mixture.astype(np.complex64),
            clean=clean.astype(np.complex64),
            jammer=jammer.astype(np.complex64),
            noise=noise.astype(np.complex64),
            receiver_artifact=receiver_artifact.astype(np.complex64),
            modulation_index=self.modulation_index(modulation),
            jammer_multihot=np.asarray(jammer_labels, dtype=np.float32),
            metadata=metadata,
        )

    def synthesize(
        self,
        *,
        modulation: str,
        source_seed: int,
        condition_seed: int,
        snr_db: float,
        sir_db: float,
        jammer_name: str,
        channel_scenario: str,
        speed_kmh: float,
        mixture_pool: Iterable[str] | None = None,
        overlap_profile: str | None = None,
    ) -> SampleComponents:
        if modulation not in self.config.modulations:
            raise ValueError(f"Unknown modulation: {modulation}")
        if channel_scenario not in self._CHANNEL_PROFILES:
            raise ValueError(f"Unknown channel scenario: {channel_scenario}")
        channel_rng = _named_rng(condition_seed, "channel")
        jammer_rng = _named_rng(condition_seed, "jammer")

        maximum_delay = max(self._CHANNEL_PROFILES[channel_scenario][0])
        guard = maximum_delay + self.config.rrc_span_symbols * self.config.samples_per_symbol
        source = self.generate_source(
            modulation,
            source_seed,
            self.config.sample_length + 2 * guard,
        )
        channel_output, doppler_norm = self._apply_channel(
            source,
            channel_scenario,
            speed_kmh,
            channel_rng,
        )
        clean = channel_output[guard : guard + self.config.sample_length]

        raw_jammer, jammer_labels, jammer_components = self.generate_jammer(
            jammer_name,
            len(clean),
            jammer_rng,
            modulation,
            mixture_pool=mixture_pool,
            overlap_profile=overlap_profile,
        )
        return self.finalize_received_components(
            clean=clean,
            raw_jammer=raw_jammer,
            modulation=modulation,
            source_seed=source_seed,
            condition_seed=condition_seed,
            snr_db=snr_db,
            sir_db=sir_db,
            jammer_name=jammer_name,
            jammer_labels=jammer_labels,
            jammer_components=jammer_components,
            channel_scenario=channel_scenario,
            channel_model="controlled_v2x_motivated_heuristic_proxy_v1",
            speed_kmh=speed_kmh,
            doppler_norm=doppler_norm,
            overlap_profile=overlap_profile,
        )
