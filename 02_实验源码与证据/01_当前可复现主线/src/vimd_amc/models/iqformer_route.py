"""Diagnostic VIMD-v3 using one shared IQFormer-inspired encoder.

The received mixture and a differentiable student-mask route are encoded by
the exact same module instance.  The raw path always retains a non-zero
coefficient, providing an inference-time safety path without any teacher or
component input.
"""

from __future__ import annotations

import torch
from torch import nn

from .baselines import IQFormerInspiredClassifier
from .common import ModelConfig
from .spectral import (
    ComplexSTFT,
    ConditionedTriMask,
    SpectralContextEncoder,
    SpectralEnvironmentEncoder,
)


class ComplexSTFTOverlapAdd(nn.Module):
    """Differentiable inverse for the project's center-free complex STFT.

    Hann-window endpoints and a non-covering final hop can leave samples with
    zero overlap weight. Those samples are explicitly zero-filled and exposed
    through ``coverage`` and ``unrecoverable_sample_count`` rather than hidden
    by an epsilon division.
    """

    def __init__(self, n_fft: int, hop_length: int, *, epsilon: float = 1e-4):
        super().__init__()
        if n_fft < 8:
            raise ValueError("n_fft must be at least 8")
        if hop_length <= 0 or hop_length > n_fft:
            raise ValueError("hop_length must be in [1, n_fft]")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.epsilon = float(epsilon)
        self.register_buffer(
            "window",
            torch.hann_window(self.n_fft, periodic=True),
            persistent=True,
        )

    def forward(
        self,
        spectrum: torch.Tensor,
        *,
        output_length: int,
    ) -> dict[str, torch.Tensor]:
        if not torch.is_complex(spectrum) or spectrum.ndim != 3:
            raise ValueError("spectrum must be complex [batch, frequency, frame]")
        if spectrum.shape[1] != self.n_fft:
            raise ValueError(
                f"expected {self.n_fft} frequency bins, got {spectrum.shape[1]}"
            )
        if output_length <= 0:
            raise ValueError("output_length must be positive")
        batch, _, frame_count = spectrum.shape
        covered_length = (frame_count - 1) * self.hop_length + self.n_fft
        work_length = max(int(output_length), covered_length)
        dtype = spectrum.real.dtype
        device = spectrum.device
        window = self.window.to(device=device, dtype=dtype)
        # ComplexSTFT(normalized=True) uses the orthonormal FFT convention.
        frames = torch.fft.ifft(
            spectrum.transpose(1, 2),
            n=self.n_fft,
            dim=-1,
            norm="ortho",
        )
        numerator = torch.zeros(
            batch,
            work_length,
            dtype=frames.dtype,
            device=device,
        )
        denominator = torch.zeros(work_length, dtype=dtype, device=device)
        window_square = window.square()
        for frame_index in range(frame_count):
            start = frame_index * self.hop_length
            stop = start + self.n_fft
            numerator[:, start:stop] = (
                numerator[:, start:stop]
                + frames[:, frame_index] * window
            )
            denominator[start:stop] = (
                denominator[start:stop] + window_square
            )
        numerator = numerator[:, :output_length]
        denominator = denominator[:output_length]
        coverage = denominator > self.epsilon
        safe_denominator = denominator.masked_fill(~coverage, 1.0)
        reconstructed = numerator / safe_denominator[None]
        reconstructed = reconstructed.masked_fill(~coverage[None], 0.0)
        iq = torch.stack(
            (reconstructed.real, reconstructed.imag),
            dim=1,
        )
        return {
            "iq": iq,
            "coverage": coverage,
            "overlap_denominator": denominator,
            "unrecoverable_sample_count": (~coverage).sum(),
            "covered_fraction": coverage.float().mean(),
        }


class IQFormerRawOnlyControl(nn.Module):
    """Raw-only control with the exact encoder/head used by VIMD-v3."""

    provenance = {
        "display_name": "IQFormer shared-encoder raw-only diagnostic control",
        "claim_level": "internal diagnostic control",
        "inference_inputs": ["received_iq_mixture"],
    }

    def __init__(self, num_classes: int):
        super().__init__()
        self.shared_encoder = IQFormerInspiredClassifier(num_classes)

    def forward(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.shared_encoder(values)


class VIMDIQFormerRouteNet(nn.Module):
    """Tri-mask route plus raw safety path through one shared IQFormer encoder."""

    # Uses a dedicated deployment-only non-collapse audit; the manuscript's
    # VIMD mechanism probe assumes a separate jammer branch that is absent here.
    supports_tri_mechanism = False
    requires_teacher_at_inference = False
    provenance = {
        "display_name": "VIMD-v3 shared-IQFormer route diagnostic",
        "claim_level": "internal diagnostic candidate; outside paper A0--A7",
        "inference_inputs": ["received_iq_mixture"],
        "teacher_or_component_inputs_at_inference": False,
        "shared_encoder": True,
        "replaces_a5": False,
    }

    def __init__(
        self,
        num_classes: int,
        num_jammers: int,
        config: ModelConfig | None = None,
    ):
        super().__init__()
        del num_jammers  # No auxiliary jammer head in this bounded diagnostic.
        self.config = config or ModelConfig()
        self.mask_routes = 3
        self.use_residual = True
        self.front_end = ComplexSTFT(
            self.config.n_fft,
            self.config.hop_length,
        )
        self.overlap_add = ComplexSTFTOverlapAdd(
            self.config.n_fft,
            self.config.hop_length,
        )
        self.context_encoder = SpectralContextEncoder(self.config)
        self.environment_encoder = SpectralEnvironmentEncoder(self.config)
        self.tri_mask = ConditionedTriMask(self.config)
        self.shared_encoder = IQFormerInspiredClassifier(num_classes)
        self.condition_gate = nn.Linear(self.config.environment_dim, 1)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return self.front_end(values)

    def forward(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        spectrum = self.front_end(values)
        features = self.context_encoder(spectrum)
        condition = self.environment_encoder(features)
        decomposition = self.tri_mask(features, condition)
        modulation_mask, _, overlap_mask = decomposition["masks"].unbind(dim=1)
        lambda_overlap = decomposition["lambda_overlap"][:, :, None]
        rho = decomposition["rho"][:, :, None]
        modulation_weight = modulation_mask + lambda_overlap * overlap_mask + rho
        modulation_spectrum = modulation_weight * spectrum
        route = self.overlap_add(
            modulation_spectrum,
            output_length=values.shape[-1],
        )
        route_values = route["iq"].to(values.dtype)

        # Calling the same object twice is intentional and test-audited.
        raw_embedding = self.shared_encoder.encode(values)
        route_embedding = self.shared_encoder.encode(route_values)
        gate = 0.10 + 0.80 * torch.sigmoid(self.condition_gate(condition))
        embedding = raw_embedding + gate * (route_embedding - raw_embedding)
        logits = self.shared_encoder.classifier(embedding)
        return {
            "logits": logits,
            "embedding": embedding,
            "raw_embedding": raw_embedding,
            "route_embedding": route_embedding,
            "route_iq": route_values,
            "route_gate": gate,
            "spectrum": spectrum,
            "features": features,
            "mod_spectrum": modulation_spectrum,
            "modulation_weight": modulation_weight,
            "masks": decomposition["masks"],
            "lambda_overlap": decomposition["lambda_overlap"],
            "rho": decomposition["rho"],
            "temperature": decomposition["temperature"],
            "ola_coverage": route["coverage"],
            "ola_covered_fraction": route["covered_fraction"],
            "ola_unrecoverable_sample_count": route[
                "unrecoverable_sample_count"
            ],
        }


__all__ = [
    "ComplexSTFTOverlapAdd",
    "IQFormerRawOnlyControl",
    "VIMDIQFormerRouteNet",
]
