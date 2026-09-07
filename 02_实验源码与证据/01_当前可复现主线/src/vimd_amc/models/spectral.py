"""Physically aligned complex-STFT building blocks.

The mask lattice is the complex STFT lattice itself.  This avoids assigning a
time-frequency oracle to arbitrary learned feature channels, and it preserves
phase in every masked branch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F

from .common import ModelConfig, _group_count


class ComplexSTFT(nn.Module):
    """Deterministic full-spectrum STFT for complex I/Q tensors."""

    def __init__(self, n_fft: int, hop_length: int):
        super().__init__()
        if n_fft < 8:
            raise ValueError("n_fft must be at least 8")
        if hop_length <= 0 or hop_length > n_fft:
            raise ValueError("hop_length must be in [1, n_fft]")
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.register_buffer(
            "window",
            torch.hann_window(self.n_fft, periodic=True),
            persistent=True,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1] != 2:
            raise ValueError("ComplexSTFT expects [batch, 2, time] I/Q tensors")
        if values.shape[-1] < self.n_fft:
            raise ValueError(
                f"input length {values.shape[-1]} is shorter than n_fft={self.n_fft}"
            )
        complex_values = torch.complex(values[:, 0], values[:, 1])
        return torch.stft(
            complex_values,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(dtype=values.dtype, device=values.device),
            center=False,
            normalized=True,
            onesided=False,
            return_complex=True,
        )


def complex_spectral_features(spectrum: torch.Tensor) -> torch.Tensor:
    """Return phase-preserving real/imaginary/log-magnitude channels."""

    if not torch.is_complex(spectrum) or spectrum.ndim != 3:
        raise ValueError("spectrum must be complex with shape [batch, frequency, frame]")
    magnitude = spectrum.abs()
    return torch.stack(
        (
            spectrum.real,
            spectrum.imag,
            torch.log1p(magnitude),
        ),
        dim=1,
    )


class SpectralResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        stride: tuple[int, int] = (1, 1),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.dropout = nn.Dropout2d(dropout)
        self.skip = (
            nn.Identity()
            if stride == (1, 1)
            else nn.Conv2d(channels, channels, kernel_size=1, stride=stride, bias=False)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.skip(values)
        output = self.depthwise(values)
        output = self.pointwise(output)
        output = self.dropout(F.silu(self.norm(output)))
        return F.silu(output + residual)


def spectral_statistical_pool(values: torch.Tensor) -> torch.Tensor:
    flattened = values.flatten(start_dim=2)
    mean = flattened.mean(dim=-1)
    std = flattened.var(dim=-1, unbiased=False).add(1e-6).sqrt()
    maximum = flattened.amax(dim=-1)
    return torch.cat((mean, std, maximum), dim=1)


class SpectralContextEncoder(nn.Module):
    """Mixture encoder that preserves the mask lattice resolution."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        channels = config.spectral_channels
        self.network = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(),
            SpectralResidualBlock(channels, dropout=config.dropout),
            SpectralResidualBlock(channels, dropout=config.dropout),
        )

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        return self.network(complex_spectral_features(spectrum))


class SpectralEnvironmentEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        channels = config.spectral_channels
        hidden = max(64, config.environment_dim * 2)
        self.network = nn.Sequential(
            nn.Linear(channels * 3, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.environment_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(spectral_statistical_pool(features))


class SpectralBranchEncoder(nn.Module):
    """Encode a masked complex spectrum without discarding phase."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        channels = config.spectral_channels
        self.network = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(),
            SpectralResidualBlock(
                channels,
                stride=(2, 1),
                dropout=config.dropout,
            ),
            SpectralResidualBlock(
                channels,
                stride=(2, 2),
                dropout=config.dropout,
            ),
            SpectralResidualBlock(channels, dropout=config.dropout),
        )
        self.projector = nn.Sequential(
            nn.Linear(channels * 3, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.SiLU(),
        )

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        features = self.network(complex_spectral_features(spectrum))
        return self.projector(spectral_statistical_pool(features))


class ConditionedTriMask(nn.Module):
    """Environment-conditioned three-way mask on the physical STFT lattice."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        channels = config.spectral_channels
        self.config = config
        self.film = nn.Linear(config.environment_dim, channels * 2)
        self.mask_network = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, 3, kernel_size=1),
        )
        self.temperature_head = nn.Linear(config.environment_dim, 1)
        self.overlap_head = nn.Linear(config.environment_dim, 1)
        self.residual_head = nn.Linear(config.environment_dim, 1)

    def forward(
        self,
        features: torch.Tensor,
        condition: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        gamma, beta = self.film(condition).chunk(2, dim=1)
        conditioned = (
            (1.0 + 0.5 * torch.tanh(gamma))[:, :, None, None] * features
            + beta[:, :, None, None]
        )
        temperature = self.config.temperature_min + (
            self.config.temperature_max - self.config.temperature_min
        ) * torch.sigmoid(self.temperature_head(condition))
        logits = self.mask_network(conditioned)
        masks = torch.softmax(logits / temperature[:, :, None, None], dim=1)
        lambda_overlap = torch.sigmoid(self.overlap_head(condition))
        rho = self.config.rho_min + (
            self.config.rho_max - self.config.rho_min
        ) * torch.sigmoid(self.residual_head(condition))
        return {
            "masks": masks,
            "lambda_overlap": lambda_overlap,
            "rho": rho,
            "temperature": temperature,
        }


class ConditionedDualMask(nn.Module):
    """Environment-conditioned target/non-target mask with residual control."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        channels = config.spectral_channels
        self.config = config
        self.film = nn.Linear(config.environment_dim, channels * 2)
        self.mask_network = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, 2, kernel_size=1),
        )
        self.temperature_head = nn.Linear(config.environment_dim, 1)
        self.residual_head = nn.Linear(config.environment_dim, 1)

    def forward(
        self,
        features: torch.Tensor,
        condition: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        gamma, beta = self.film(condition).chunk(2, dim=1)
        conditioned = (
            (1.0 + 0.5 * torch.tanh(gamma))[:, :, None, None] * features
            + beta[:, :, None, None]
        )
        temperature = self.config.temperature_min + (
            self.config.temperature_max - self.config.temperature_min
        ) * torch.sigmoid(self.temperature_head(condition))
        masks = torch.softmax(
            self.mask_network(conditioned) / temperature[:, :, None, None],
            dim=1,
        )
        rho = self.config.rho_min + (
            self.config.rho_max - self.config.rho_min
        ) * torch.sigmoid(self.residual_head(condition))
        return {
            "masks": masks,
            "rho": rho,
            "temperature": temperature,
        }


@dataclass(frozen=True)
class TriMaskTeacherSpec:
    """Serializable, immutable definition of one three-route teacher."""

    mode: Literal[
        "closed_form_margin",
        "proportional_power",
        "local_whitened_margin",
    ] = "closed_form_margin"
    local_frequency_bins: int = 9
    local_time_frames: int = 5
    local_whitening_exponent: float = 1.0

    def __post_init__(self) -> None:
        if self.local_frequency_bins <= 0 or self.local_frequency_bins % 2 == 0:
            raise ValueError("local_frequency_bins must be a positive odd integer")
        if self.local_time_frames <= 0 or self.local_time_frames % 2 == 0:
            raise ValueError("local_time_frames must be a positive odd integer")
        if not 0.0 <= self.local_whitening_exponent <= 1.0:
            raise ValueError("local_whitening_exponent must lie in [0, 1]")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _component_power_shares(
    front_end: ComplexSTFT,
    clean: torch.Tensor,
    jammer: torch.Tensor,
    unexplained: torch.Tensor,
) -> dict[str, torch.Tensor]:
    signal_power = front_end(clean).abs().square()
    jammer_power = front_end(jammer).abs().square()
    unexplained_power = front_end(unexplained).abs().square()
    total = signal_power + jammer_power + unexplained_power
    power_epsilon = (
        1e-8 * total.mean(dim=(1, 2), keepdim=True)
    ).clamp_min(torch.finfo(total.dtype).tiny)
    denominator = total.clamp_min(power_epsilon)
    return {
        "signal_power": signal_power,
        "jammer_power": jammer_power,
        "unexplained_power": unexplained_power,
        "component_power": total,
        "power_epsilon": power_epsilon,
        "q_signal": signal_power / denominator,
        "q_jammer": jammer_power / denominator,
        "q_unexplained": unexplained_power / denominator,
        "empty": total <= power_epsilon,
    }


def _normalize_teacher_masks(
    masks: torch.Tensor,
    empty: torch.Tensor,
) -> torch.Tensor:
    if empty.any():
        masks[:, 0] = masks[:, 0].masked_fill(empty, 0.0)
        masks[:, 1] = masks[:, 1].masked_fill(empty, 0.0)
        masks[:, 2] = masks[:, 2].masked_fill(empty, 1.0)
    return masks / masks.sum(dim=1, keepdim=True).clamp_min(1e-8)


class PhysicalTriMaskTeacher(nn.Module):
    """Fixed dominant-component/uncertainty oracle in direct closed form.

    For every complex-STFT cell, component powers are normalized into target
    (q_s), jammer (q_j), and unexplained (q_u) shares. The masks are:

        M_s = [q_s-q_j]_+
        M_j = [q_j-q_s]_+
        M_o = q_u + 2 min(q_s,q_j)

    This is algebraically identical to the earlier relative-dominance
    construction, but makes the low-SIR behavior explicit. Equal-power
    target/jammer cells go to the overlap/uncertainty route, while noise and
    receiver artifacts are always unexplained. The masks are non-negative and
    sum to one per cell.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.front_end = ComplexSTFT(config.n_fft, config.hop_length)
        self.spec = TriMaskTeacherSpec(mode="closed_form_margin")

    @torch.no_grad()
    def decompose(
        self,
        clean: torch.Tensor,
        jammer: torch.Tensor,
        unexplained: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        components = _component_power_shares(
            self.front_end,
            clean,
            jammer,
            unexplained,
        )
        q_signal = components["q_signal"]
        q_jammer = components["q_jammer"]
        q_unexplained = components["q_unexplained"]
        signal_mask = F.relu(q_signal - q_jammer)
        jammer_mask = F.relu(q_jammer - q_signal)
        signal_jammer_ambiguity = 2.0 * torch.minimum(q_signal, q_jammer)
        overlap_mask = q_unexplained + signal_jammer_ambiguity
        masks = torch.stack((signal_mask, jammer_mask, overlap_mask), dim=1)
        masks = _normalize_teacher_masks(masks, components["empty"])
        return {
            "masks": masks,
            "unexplained_fraction": q_unexplained,
            "signal_jammer_ambiguity": signal_jammer_ambiguity,
            "component_power": components["component_power"],
        }

    @torch.no_grad()
    def forward(
        self,
        clean: torch.Tensor,
        jammer: torch.Tensor,
        unexplained: torch.Tensor,
    ) -> torch.Tensor:
        return self.decompose(clean, jammer, unexplained)["masks"]


class ProportionalTriMaskTeacher(nn.Module):
    """A3-prime oracle using the direct component-power proportions.

    For every cell the target is simply ``(q_s, q_j, q_u)``. This is the
    natural IRM-style control for testing whether the margin/ambiguity rule is
    useful beyond access to the simulator component ledger.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.front_end = ComplexSTFT(config.n_fft, config.hop_length)
        self.spec = TriMaskTeacherSpec(mode="proportional_power")

    @torch.no_grad()
    def decompose(
        self,
        clean: torch.Tensor,
        jammer: torch.Tensor,
        unexplained: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        components = _component_power_shares(
            self.front_end,
            clean,
            jammer,
            unexplained,
        )
        masks = torch.stack(
            (
                components["q_signal"],
                components["q_jammer"],
                components["q_unexplained"],
            ),
            dim=1,
        )
        masks = _normalize_teacher_masks(masks, components["empty"])
        return {
            "masks": masks,
            "unexplained_fraction": components["q_unexplained"],
            "signal_jammer_ambiguity": torch.zeros_like(
                components["q_unexplained"]
            ),
            "component_power": components["component_power"],
        }

    @torch.no_grad()
    def forward(
        self,
        clean: torch.Tensor,
        jammer: torch.Tensor,
        unexplained: torch.Tensor,
    ) -> torch.Tensor:
        return self.decompose(clean, jammer, unexplained)["masks"]


class LocalWhitenedTriMaskTeacher(nn.Module):
    """Diagnostic margin teacher after per-component local power whitening.

    Target and jammer power maps are divided by their own local mean envelopes
    before relative dominance is computed. The raw unexplained share remains
    unchanged, so whitening cannot erase the noise/receiver-artifact burden.
    Full whitening deliberately removes window-level SIR from the target versus
    jammer comparison; this is why the variant is diagnostic and disabled by
    the locked primary-teacher policy until a development-only comparison is
    completed.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        frequency_bins: int = 9,
        time_frames: int = 5,
        whitening_exponent: float = 1.0,
    ):
        super().__init__()
        self.front_end = ComplexSTFT(config.n_fft, config.hop_length)
        self.spec = TriMaskTeacherSpec(
            mode="local_whitened_margin",
            local_frequency_bins=int(frequency_bins),
            local_time_frames=int(time_frames),
            local_whitening_exponent=float(whitening_exponent),
        )

    def _whiten(self, power: torch.Tensor) -> torch.Tensor:
        kernel = (
            self.spec.local_frequency_bins,
            self.spec.local_time_frames,
        )
        envelope = F.avg_pool2d(
            power.unsqueeze(1),
            kernel_size=kernel,
            stride=1,
            padding=(kernel[0] // 2, kernel[1] // 2),
            count_include_pad=False,
        ).squeeze(1)
        epsilon = (
            1e-8 * power.mean(dim=(1, 2), keepdim=True)
        ).clamp_min(torch.finfo(power.dtype).tiny)
        denominator = envelope.clamp_min(epsilon).pow(
            self.spec.local_whitening_exponent
        )
        return power / denominator

    @torch.no_grad()
    def decompose(
        self,
        clean: torch.Tensor,
        jammer: torch.Tensor,
        unexplained: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        components = _component_power_shares(
            self.front_end,
            clean,
            jammer,
            unexplained,
        )
        whitened_signal = self._whiten(components["signal_power"])
        whitened_jammer = self._whiten(components["jammer_power"])
        whitened_total = whitened_signal + whitened_jammer
        relative_signal = whitened_signal / whitened_total.clamp_min(1e-8)
        relative_jammer = whitened_jammer / whitened_total.clamp_min(1e-8)
        explained = components["q_signal"] + components["q_jammer"]
        dominance = relative_signal - relative_jammer
        signal_mask = explained * F.relu(dominance)
        jammer_mask = explained * F.relu(-dominance)
        signal_jammer_ambiguity = explained * (1.0 - dominance.abs())
        overlap_mask = (
            components["q_unexplained"] + signal_jammer_ambiguity
        )
        masks = torch.stack((signal_mask, jammer_mask, overlap_mask), dim=1)
        masks = _normalize_teacher_masks(masks, components["empty"])
        return {
            "masks": masks,
            "unexplained_fraction": components["q_unexplained"],
            "signal_jammer_ambiguity": signal_jammer_ambiguity,
            "component_power": components["component_power"],
            "whitened_signal_power": whitened_signal,
            "whitened_jammer_power": whitened_jammer,
        }

    @torch.no_grad()
    def forward(
        self,
        clean: torch.Tensor,
        jammer: torch.Tensor,
        unexplained: torch.Tensor,
    ) -> torch.Tensor:
        return self.decompose(clean, jammer, unexplained)["masks"]


def build_tri_mask_teacher(
    config: ModelConfig,
    spec: TriMaskTeacherSpec,
) -> nn.Module:
    """Build a teacher from an immutable, artifact-ready specification."""

    if spec.mode == "closed_form_margin":
        return PhysicalTriMaskTeacher(config)
    if spec.mode == "proportional_power":
        return ProportionalTriMaskTeacher(config)
    if spec.mode == "local_whitened_margin":
        return LocalWhitenedTriMaskTeacher(
            config,
            frequency_bins=spec.local_frequency_bins,
            time_frames=spec.local_time_frames,
            whitening_exponent=spec.local_whitening_exponent,
        )
    raise ValueError(f"unsupported teacher mode: {spec.mode}")


class PhysicalDualMaskTeacher(nn.Module):
    """Two-route projection of the fixed three-way physical teacher.

    The modulation-dominant route is retained.  Jammer-dominant and
    overlap/unexplained mass are collapsed into a single non-target route.
    This is the unique A6 teacher definition and keeps the route-count
    comparison anchored to the same component-power oracle as A3--A5/A7.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.tri_teacher = PhysicalTriMaskTeacher(config)

    @torch.no_grad()
    def decompose(
        self,
        clean: torch.Tensor,
        jammer: torch.Tensor,
        unexplained: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        tri = self.tri_teacher.decompose(clean, jammer, unexplained)
        tri_masks = tri["masks"]
        masks = torch.stack(
            (
                tri_masks[:, 0],
                tri_masks[:, 1] + tri_masks[:, 2],
            ),
            dim=1,
        )
        return {
            **tri,
            "tri_masks": tri_masks,
            "masks": masks,
        }

    @torch.no_grad()
    def forward(
        self,
        clean: torch.Tensor,
        jammer: torch.Tensor,
        unexplained: torch.Tensor,
    ) -> torch.Tensor:
        return self.decompose(clean, jammer, unexplained)["masks"]
