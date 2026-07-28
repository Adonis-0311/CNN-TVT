"""Shared neural building blocks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


def _group_count(channels: int, preferred: int = 8) -> int:
    for groups in range(min(preferred, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


@dataclass(frozen=True)
class ModelConfig:
    first_complex_channels: int = 8
    second_complex_channels: int = 16
    feature_channels: int = 64
    environment_dim: int = 32
    embedding_dim: int = 96
    dropout: float = 0.1
    rho_min: float = 0.05
    rho_max: float = 0.35
    temperature_min: float = 0.5
    temperature_max: float = 2.0
    n_fft: int = 64
    hop_length: int = 16
    spectral_channels: int = 32


class ComplexConv1d(nn.Module):
    """Complex convolution implemented with two shared real-valued kernels."""

    def __init__(
        self,
        in_complex_channels: int,
        out_complex_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | None = None,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        kwargs = dict(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.real_kernel = nn.Conv1d(in_complex_channels, out_complex_channels, **kwargs)
        self.imag_kernel = nn.Conv1d(in_complex_channels, out_complex_channels, **kwargs)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1] % 2:
            raise ValueError("ComplexConv1d expects [batch, 2*C, time]")
        half = values.shape[1] // 2
        real, imag = values[:, :half], values[:, half:]
        out_real = self.real_kernel(real) - self.imag_kernel(imag)
        out_imag = self.real_kernel(imag) + self.imag_kernel(real)
        return torch.cat((out_real, out_imag), dim=1)


class ComplexStem(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        c1 = config.first_complex_channels
        c2 = config.second_complex_channels
        self.layers = nn.Sequential(
            ComplexConv1d(1, c1, kernel_size=7),
            nn.GroupNorm(_group_count(2 * c1), 2 * c1),
            nn.SiLU(),
            ComplexConv1d(c1, c2, kernel_size=5, stride=2),
            nn.GroupNorm(_group_count(2 * c2), 2 * c2),
            nn.SiLU(),
        )
        self.out_channels = 2 * c2

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class SeparableResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, dropout: float):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=5,
            stride=stride,
            padding=2,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.skip(values)
        output = self.depthwise(values)
        output = self.pointwise(output)
        output = self.dropout(F.silu(self.norm(output)))
        return F.silu(output + residual)


class CausalConv1d(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.left_padding,
            groups=groups,
            bias=bias,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output = self.conv(values)
        return output[..., : values.shape[-1]]


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.depthwise = CausalConv1d(
            channels,
            kernel_size=3,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output = self.depthwise(values)
        output = self.pointwise(output)
        output = self.dropout(F.silu(self.norm(output)))
        return F.silu(values + output)


class FeatureEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.stem = ComplexStem(config)
        feature_channels = config.feature_channels
        middle = max(48, feature_channels)
        self.local = nn.Sequential(
            SeparableResidualBlock(
                self.stem.out_channels,
                middle,
                stride=2,
                dropout=config.dropout,
            ),
            SeparableResidualBlock(
                middle,
                feature_channels,
                stride=2,
                dropout=config.dropout,
            ),
        )
        self.temporal = nn.Sequential(
            *[
                TemporalResidualBlock(feature_channels, dilation=dilation, dropout=config.dropout)
                for dilation in (1, 2, 4, 8)
            ]
        )
        self.out_channels = feature_channels

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.temporal(self.local(self.stem(values)))


def statistical_pool(values: torch.Tensor) -> torch.Tensor:
    mean = values.mean(dim=-1)
    std = values.var(dim=-1, unbiased=False).add(1e-6).sqrt()
    maximum = values.amax(dim=-1)
    return torch.cat((mean, std, maximum), dim=1)


class EnvironmentEncoder(nn.Module):
    def __init__(self, feature_channels: int, environment_dim: int, dropout: float):
        super().__init__()
        hidden = max(64, environment_dim * 2)
        self.network = nn.Sequential(
            nn.Linear(feature_channels * 3, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, environment_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(statistical_pool(features))


class BranchEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        channels = config.feature_channels
        self.temporal = nn.Sequential(
            TemporalResidualBlock(channels, dilation=1, dropout=config.dropout),
            TemporalResidualBlock(channels, dilation=2, dropout=config.dropout),
        )
        self.projector = nn.Sequential(
            nn.Linear(channels * 3, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.SiLU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projector(statistical_pool(self.temporal(features)))
