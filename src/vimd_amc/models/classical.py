"""Transparent higher-order/cyclostationary feature controls for AMC.

The implementation is local and deliberately modest: fixed, named complex
statistics are followed by a small classifier.  It is not presented as a
reproduction of any published HOC or cyclostationary detector.
"""

from __future__ import annotations

import math
from typing import Final

import torch
from torch import nn
import torch.nn.functional as F


_MOMENT_ORDERS: Final[tuple[int, ...]] = (2, 3, 4, 5, 6, 8)
_PHASE_HARMONICS: Final[tuple[int, ...]] = (1, 2, 4)
_AUTOCORRELATION_LAGS: Final[tuple[int, ...]] = (1, 2, 4, 8, 16)
_CYCLIC_LAGS: Final[tuple[int, ...]] = (0, 1, 2, 4)
_CYCLIC_COUNTS: Final[tuple[int, ...]] = (1, 2, 4, 8)


def _feature_names() -> tuple[str, ...]:
    names: list[str] = [
        *(f"abs_normalized_complex_moment_order_{order}" for order in _MOMENT_ORDERS),
        "abs_normalized_cumulant_c40",
        "normalized_cumulant_c42_signed",
        "abs_normalized_cumulant_c60",
        "normalized_amplitude_mean",
        "normalized_amplitude_std",
        "normalized_amplitude_moment_order_3",
        "normalized_amplitude_moment_order_4",
        "normalized_amplitude_moment_order_6",
    ]
    for harmonic in _PHASE_HARMONICS:
        names.extend(
            (
                f"phase_difference_harmonic_{harmonic}_real",
                f"phase_difference_harmonic_{harmonic}_imag",
                f"phase_difference_harmonic_{harmonic}_resultant",
            )
        )
    for lag in _AUTOCORRELATION_LAGS:
        names.extend(
            (
                f"normalized_autocorrelation_lag_{lag}_real",
                f"normalized_autocorrelation_lag_{lag}_imag",
                f"normalized_autocorrelation_lag_{lag}_magnitude",
            )
        )
    for lag in _CYCLIC_LAGS:
        for count in _CYCLIC_COUNTS:
            names.append(
                f"cyclic_autocorrelation_magnitude_lag_{lag}_cycles_{count}"
            )
    names.extend(
        (
            "spectral_entropy_normalized",
            "spectral_flatness",
            "spectral_concentration",
            "spectral_centroid_normalized",
            "spectral_spread_normalized",
            "spectral_lower_quartile_energy",
            "spectral_upper_quartile_energy",
        )
    )
    return tuple(names)


class ClassicalHOCyclostationaryFeatures(nn.Module):
    """Fixed scale/phase-invariant complex statistics.

    Input is a real-valued I/Q tensor ``[batch, 2, time]``.  Statistics are
    evaluated in float64/complex128 for stability and returned in the input
    floating dtype (float32 for half/bfloat16 input).  The extractor has no
    learned parameters.

    "Cyclostationary" here has a precise local definition: for lag ``l`` and
    cycle count ``k``, the feature is the magnitude of
    ``mean(z[n+l] conj(z[n]) exp(-j 2 pi k n/N))`` after unit-power
    normalization.  It is a compact control, not a full spectral-correlation
    estimator.
    """

    feature_names: Final[tuple[str, ...]] = _feature_names()
    minimum_length: Final[int] = max(_AUTOCORRELATION_LAGS) + 1

    def __init__(self, *, stability_epsilon: float = 1e-12):
        super().__init__()
        if not math.isfinite(stability_epsilon) or stability_epsilon <= 0:
            raise ValueError("stability_epsilon must be positive and finite")
        self.stability_epsilon = float(stability_epsilon)

    @property
    def output_dim(self) -> int:
        return len(self.feature_names)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1] != 2:
            raise ValueError(
                "classical feature extractor expects [batch, 2, time] I/Q"
            )
        if values.shape[-1] < self.minimum_length:
            raise ValueError(
                f"classical features require at least {self.minimum_length} samples"
            )
        if not values.is_floating_point():
            raise TypeError("classical feature extractor requires floating input")

        output_dtype = (
            values.dtype
            if values.dtype in (torch.float32, torch.float64)
            else torch.float32
        )
        real = values[:, 0].to(torch.float64)
        imag = values[:, 1].to(torch.float64)
        z = torch.complex(real, imag)
        z = z - z.mean(dim=-1, keepdim=True)
        power = z.abs().square().mean(dim=-1, keepdim=True)
        scale = power.clamp_min(self.stability_epsilon).sqrt()
        z = z / scale
        amplitude = z.abs()
        unit_power = z.abs().square().mean(dim=-1)

        features: list[torch.Tensor] = []
        moments: dict[int, torch.Tensor] = {}
        for order in _MOMENT_ORDERS:
            moment = (z**order).mean(dim=-1)
            moments[order] = moment
            features.append(moment.abs())

        c40 = moments[4] - 3.0 * moments[2].square()
        absolute_fourth = amplitude.pow(4).mean(dim=-1)
        c42 = absolute_fourth - moments[2].abs().square() - 2.0 * unit_power.square()
        c60 = (
            moments[6]
            - 15.0 * moments[4] * moments[2]
            - 10.0 * moments[3].square()
            + 30.0 * moments[2].pow(3)
        )
        features.extend((c40.abs(), c42, c60.abs()))

        features.extend(
            (
                amplitude.mean(dim=-1),
                amplitude.std(dim=-1, unbiased=False),
                amplitude.pow(3).mean(dim=-1),
                absolute_fourth,
                amplitude.pow(6).mean(dim=-1),
            )
        )

        adjacent = z[:, 1:] * z[:, :-1].conj()
        unit_adjacent = adjacent / adjacent.abs().clamp_min(
            self.stability_epsilon
        )
        for harmonic in _PHASE_HARMONICS:
            circular = unit_adjacent.pow(harmonic).mean(dim=-1)
            features.extend((circular.real, circular.imag, circular.abs()))

        for lag in _AUTOCORRELATION_LAGS:
            autocorrelation = (z[:, lag:] * z[:, :-lag].conj()).mean(dim=-1)
            features.extend(
                (
                    autocorrelation.real,
                    autocorrelation.imag,
                    autocorrelation.abs(),
                )
            )

        sample_count = z.shape[-1]
        for lag in _CYCLIC_LAGS:
            left = z[:, lag:] if lag else z
            right = z[:, : sample_count - lag].conj() if lag else z.conj()
            lag_product = left * right
            positions = torch.arange(
                lag_product.shape[-1],
                dtype=torch.float64,
                device=values.device,
            )
            for cycle_count in _CYCLIC_COUNTS:
                phase = torch.exp(
                    torch.complex(
                        torch.zeros_like(positions),
                        -2.0
                        * math.pi
                        * float(cycle_count)
                        * positions
                        / float(sample_count),
                    )
                )
                features.append((lag_product * phase).mean(dim=-1).abs())

        spectral_power = torch.fft.fftshift(
            torch.fft.fft(z, dim=-1).abs().square(),
            dim=-1,
        )
        spectral_mass = spectral_power.sum(dim=-1, keepdim=True)
        spectral_probability = spectral_power / spectral_mass.clamp_min(
            self.stability_epsilon
        )
        spectral_entropy = -(
            spectral_probability
            * spectral_probability.clamp_min(self.stability_epsilon).log()
        ).sum(dim=-1) / math.log(float(sample_count))
        spectral_flatness = torch.exp(
            spectral_power.clamp_min(self.stability_epsilon).log().mean(dim=-1)
        ) / spectral_power.mean(dim=-1).clamp_min(self.stability_epsilon)
        spectral_concentration = spectral_probability.square().sum(dim=-1)
        frequencies = torch.linspace(
            -1.0,
            1.0,
            sample_count,
            dtype=torch.float64,
            device=values.device,
        )
        spectral_centroid = (spectral_probability * frequencies).sum(dim=-1)
        spectral_spread = (
            spectral_probability
            * (frequencies - spectral_centroid[:, None]).square()
        ).sum(dim=-1).clamp_min(0.0).sqrt()
        quartile = max(1, sample_count // 4)
        lower_energy = spectral_probability[:, :quartile].sum(dim=-1)
        upper_energy = spectral_probability[:, -quartile:].sum(dim=-1)
        features.extend(
            (
                spectral_entropy,
                spectral_flatness,
                spectral_concentration,
                spectral_centroid,
                spectral_spread,
                lower_energy,
                upper_energy,
            )
        )

        result = torch.stack(features, dim=-1)
        if result.shape[-1] != self.output_dim:
            raise RuntimeError("internal classical feature schema mismatch")
        result = torch.nan_to_num(
            result,
            nan=0.0,
            posinf=1.0e6,
            neginf=-1.0e6,
        ).clamp(-1.0e6, 1.0e6)
        return result.to(output_dtype)


class ClassicalHOCyclostationaryClassifier(nn.Module):
    """Fixed classical features plus a fixed-budget linear/MLP classifier."""

    provenance: Final[dict[str, object]] = {
        "display_name": "Local HOC/cyclostationary-feature control",
        "implementation_origin": "local",
        "claim_level": (
            "transparent local classical-feature control; not a literature "
            "reimplementation"
        ),
        "deployment_input": "received mixture x",
        "fixed_feature_families": (
            "normalized complex moments/cumulants",
            "normalized amplitude moments",
            "phase-difference circular statistics",
            "lag autocorrelation",
            "finite-frame cyclic autocorrelation magnitudes",
            "spectral entropy/shape",
        ),
    }

    def __init__(
        self,
        num_classes: int,
        *,
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes must exceed one")
        if hidden_dim < 0:
            raise ValueError("hidden_dim cannot be negative")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        self.extractor = ClassicalHOCyclostationaryFeatures()
        self.normalizer = nn.LayerNorm(
            self.extractor.output_dim,
            elementwise_affine=False,
        )
        if hidden_dim == 0:
            self.classifier = nn.Linear(self.extractor.output_dim, num_classes)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.extractor.output_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def forward(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.extractor(values)
        parameter = next(self.classifier.parameters())
        embedding = self.normalizer(features.to(parameter.dtype))
        return {
            "logits": self.classifier(embedding),
            "embedding": embedding,
            "classical_features": features,
        }


__all__ = [
    "ClassicalHOCyclostationaryFeatures",
    "ClassicalHOCyclostationaryClassifier",
]
