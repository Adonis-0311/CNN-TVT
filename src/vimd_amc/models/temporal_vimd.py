"""Diagnostic VIMD-v2 candidate with phase-aware spectral time modelling.

This module is deliberately separate from :class:`VIMDNet`.  It does not
replace manuscript ablation A5 and it does not consume clean, jammer, or
teacher tensors at inference.  The only input is the received I/Q mixture.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn
import torch.nn.functional as F

from .common import ModelConfig, TemporalResidualBlock, _group_count
from .classical import ClassicalHOCyclostationaryFeatures
from .spectral import (
    ComplexSTFT,
    ConditionedTriMask,
    SpectralBranchEncoder,
    SpectralContextEncoder,
    SpectralEnvironmentEncoder,
    complex_spectral_features,
)


class _MultiScaleTemporalBlock(nn.Module):
    """Parallel local time kernels followed by a residual channel mixer."""

    def __init__(self, channels: int, dropout: float):
        super().__init__()
        branches = 3
        if channels % branches:
            raise ValueError("temporal channels must be divisible by three")
        branch_channels = channels // branches
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    branch_channels,
                    kernel_size=kernel,
                    padding=kernel // 2,
                    bias=False,
                )
                for kernel in (3, 5, 7)
            ]
        )
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.dropout = nn.Dropout(dropout)
        self.mixer = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        local = torch.cat([branch(values) for branch in self.branches], dim=1)
        local = self.mixer(self.dropout(F.silu(self.norm(local))))
        return F.silu(values + local)


class PhaseAwareMaskedSpectralTemporalEncoder(nn.Module):
    """Retain complex STFT-frame order until after a dilated temporal stack.

    Frequency-bin identity is retained by flattening the real, imaginary, and
    log-magnitude planes into channels.  A pointwise projection is followed by
    parallel short-context filters and causal dilated residual blocks.  Only
    then is the time axis summarized.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        temporal_channels = max(48, config.embedding_dim)
        temporal_channels += (-temporal_channels) % 3
        self.expected_frequency_bins = int(config.n_fft)
        self.temporal_channels = temporal_channels
        self.input_projection = nn.Sequential(
            nn.Conv1d(
                3 * self.expected_frequency_bins,
                temporal_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(temporal_channels), temporal_channels),
            nn.SiLU(),
        )
        self.local = _MultiScaleTemporalBlock(
            temporal_channels,
            dropout=config.dropout,
        )
        self.dilated = nn.Sequential(
            *[
                TemporalResidualBlock(
                    temporal_channels,
                    dilation=dilation,
                    dropout=config.dropout,
                )
                for dilation in (1, 2, 4, 8)
            ]
        )
        self.projector = nn.Sequential(
            nn.Linear(temporal_channels * 4, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.SiLU(),
        )

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        features = complex_spectral_features(spectrum)
        batch, planes, frequency, frames = features.shape
        if frequency != self.expected_frequency_bins:
            raise ValueError(
                "spectral temporal encoder received "
                f"{frequency} bins, expected {self.expected_frequency_bins}"
            )
        sequence = features.reshape(batch, planes * frequency, frames)
        sequence = self.dilated(self.local(self.input_projection(sequence)))
        mean = sequence.mean(dim=-1)
        std = sequence.var(dim=-1, unbiased=False).add(1e-6).sqrt()
        maximum = sequence.amax(dim=-1)
        terminal = sequence[..., -1]
        return self.projector(torch.cat((mean, std, maximum, terminal), dim=1))


class VIMDTemporalNet(nn.Module):
    """VIMD-v2 diagnostic candidate with a mixture-only inference contract."""

    supports_tri_mechanism = True
    requires_teacher_at_inference = False
    supports_external_routing = False
    provenance = {
        "display_name": "VIMD-v2 phase-aware temporal diagnostic candidate",
        "claim_level": "internal diagnostic candidate; outside paper A0--A7",
        "inference_inputs": ["received_iq_mixture"],
        "teacher_or_component_inputs_at_inference": False,
        "replaces_a5": False,
    }

    def __init__(
        self,
        num_classes: int,
        num_jammers: int,
        config: ModelConfig | None = None,
        *,
        use_residual: bool = True,
    ):
        super().__init__()
        self.config = config or ModelConfig()
        self.use_residual = bool(use_residual)
        self.mask_routes = 3
        self.front_end = ComplexSTFT(self.config.n_fft, self.config.hop_length)
        self.context_encoder = SpectralContextEncoder(self.config)
        self.environment_encoder = SpectralEnvironmentEncoder(self.config)
        self.tri_mask = ConditionedTriMask(self.config)
        self.modulation_spectral_branch = SpectralBranchEncoder(self.config)
        self.modulation_temporal_branch = (
            PhaseAwareMaskedSpectralTemporalEncoder(self.config)
        )
        fusion_input = (
            2 * self.config.embedding_dim + self.config.environment_dim
        )
        self.modulation_fusion = nn.Sequential(
            nn.Linear(fusion_input, self.config.embedding_dim),
            nn.LayerNorm(self.config.embedding_dim),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
        )
        self.jammer_branch = SpectralBranchEncoder(self.config)
        self.modulation_head = nn.Linear(self.config.embedding_dim, num_classes)
        joint_dim = self.config.environment_dim + self.config.embedding_dim
        self.jammer_head = nn.Sequential(
            nn.Linear(self.config.embedding_dim, self.config.embedding_dim),
            nn.SiLU(),
            nn.Linear(self.config.embedding_dim, num_jammers),
        )
        self.quality_head = nn.Sequential(
            nn.Linear(joint_dim, self.config.embedding_dim),
            nn.SiLU(),
            nn.Linear(self.config.embedding_dim, 3),
        )

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return self.front_end(values)

    def _routing_masks(self, student_masks: torch.Tensor) -> torch.Tensor:
        """Return masks used to route spectra; base candidate is student-only."""

        return student_masks

    def forward(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        spectrum = self.front_end(values)
        context_features = self.context_encoder(spectrum)
        condition = self.environment_encoder(context_features)
        decomposition = self.tri_mask(context_features, condition)
        routing_masks = self._routing_masks(decomposition["masks"])
        modulation_mask, jammer_mask, overlap_mask = routing_masks.unbind(dim=1)
        lambda_overlap = decomposition["lambda_overlap"][:, :, None]
        predicted_rho = decomposition["rho"]
        applied_rho = (
            predicted_rho
            if self.use_residual
            else torch.zeros_like(predicted_rho)
        )
        rho = applied_rho[:, :, None]
        modulation_weight = modulation_mask + lambda_overlap * overlap_mask + rho
        jammer_weight = jammer_mask + (1.0 - lambda_overlap) * overlap_mask
        modulation_spectrum = modulation_weight * spectrum
        jammer_spectrum = jammer_weight * spectrum

        spectral_embedding = self.modulation_spectral_branch(
            modulation_spectrum
        )
        temporal_embedding = self.modulation_temporal_branch(
            modulation_spectrum
        )
        modulation_embedding = self.modulation_fusion(
            torch.cat(
                (spectral_embedding, temporal_embedding, condition),
                dim=1,
            )
        )
        jammer_embedding = self.jammer_branch(jammer_spectrum)
        joint = torch.cat((condition, jammer_embedding), dim=1)
        return {
            "logits": self.modulation_head(modulation_embedding),
            "jam_logits": self.jammer_head(jammer_embedding),
            "quality": self.quality_head(joint),
            "embedding": modulation_embedding,
            "spectral_embedding": spectral_embedding,
            "temporal_embedding": temporal_embedding,
            "jammer_embedding": jammer_embedding,
            "spectrum": spectrum,
            "features": context_features,
            "mod_spectrum": modulation_spectrum,
            "jammer_spectrum": jammer_spectrum,
            "modulation_weight": modulation_weight,
            "jammer_weight": jammer_weight,
            "routing_masks": routing_masks,
            **decomposition,
            "rho_predicted": predicted_rho,
            "rho": applied_rho,
        }


class VIMDTemporalCurriculumNet(VIMDTemporalNet):
    """Training-only teacher-routing candidate with teacher-free state.

    Teacher masks are installed only inside :meth:`teacher_route_context`.
    They are ordinary temporary attributes, never parameters or persistent
    buffers, and are ignored whenever the module is in evaluation mode.  The
    public ``forward`` signature remains the inherited mixture-only
    ``forward(values)``.
    """

    requires_teacher_at_inference = False
    supports_external_routing = True
    provenance = {
        "display_name": "VIMD-v2 fully annealed teacher-routing diagnostic",
        "claim_level": "internal diagnostic candidate; outside paper A0--A7",
        "inference_inputs": ["received_iq_mixture"],
        "teacher_or_component_inputs_at_inference": False,
        "replaces_a5": False,
        "training_only_teacher_routing": True,
    }

    def __init__(
        self,
        num_classes: int,
        num_jammers: int,
        config: ModelConfig | None = None,
        *,
        use_residual: bool = True,
    ):
        super().__init__(
            num_classes,
            num_jammers,
            config,
            use_residual=use_residual,
        )
        self._teacher_route_masks: torch.Tensor | None = None
        self._teacher_route_coefficient = 0.0

    @contextmanager
    def teacher_route_context(
        self,
        teacher_masks: torch.Tensor,
        coefficient: float,
    ) -> Iterator[None]:
        """Temporarily blend physical-teacher routes during a training call."""

        numeric = float(coefficient)
        if not 0.0 <= numeric <= 1.0:
            raise ValueError("teacher route coefficient must be in [0, 1]")
        if teacher_masks.ndim != 4 or teacher_masks.shape[1] != 3:
            raise ValueError("teacher masks must have shape [batch, 3, freq, frame]")
        if self._teacher_route_masks is not None:
            raise RuntimeError("teacher routing contexts cannot be nested")
        self._teacher_route_masks = teacher_masks.detach()
        self._teacher_route_coefficient = numeric
        try:
            yield
        finally:
            self._teacher_route_masks = None
            self._teacher_route_coefficient = 0.0

    def _routing_masks(self, student_masks: torch.Tensor) -> torch.Tensor:
        teacher_masks = self._teacher_route_masks
        coefficient = self._teacher_route_coefficient
        if not self.training or teacher_masks is None or coefficient <= 0.0:
            return student_masks
        if teacher_masks.shape != student_masks.shape:
            raise ValueError(
                "teacher/student routing mask shape mismatch: "
                f"{tuple(teacher_masks.shape)} != {tuple(student_masks.shape)}"
            )
        teacher_masks = teacher_masks.to(
            device=student_masks.device,
            dtype=student_masks.dtype,
        )
        blended = (
            (1.0 - coefficient) * student_masks
            + coefficient * teacher_masks
        )
        return blended / blended.sum(dim=1, keepdim=True).clamp_min(1e-8)


class DescriptorAssistedVIMDTemporalNet(VIMDTemporalNet):
    """Late-fuse fixed 61-D HOC/cyclostationary mixture descriptors.

    This is a low-cost diagnostic that tests whether the neural temporal path
    is missing transparent sufficient statistics.  The descriptors consume
    the same received I/Q input as the neural path and have no learned
    parameters.  The candidate is not a replacement for A5 or a main-method
    claim.
    """

    requires_teacher_at_inference = False
    supports_external_routing = False
    provenance = {
        "display_name": "descriptor-assisted VIMD-v2 diagnostic candidate",
        "claim_level": "internal diagnostic candidate; outside paper A0--A7",
        "inference_inputs": ["received_iq_mixture"],
        "teacher_or_component_inputs_at_inference": False,
        "fixed_descriptor_dim": 61,
        "descriptor_origin": "local transparent HOC/cyclostationary control",
        "replaces_a5": False,
    }

    def __init__(
        self,
        num_classes: int,
        num_jammers: int,
        config: ModelConfig | None = None,
        *,
        use_residual: bool = True,
    ):
        super().__init__(
            num_classes,
            num_jammers,
            config,
            use_residual=use_residual,
        )
        self.fixed_descriptor = ClassicalHOCyclostationaryFeatures()
        self.descriptor_normalizer = nn.LayerNorm(
            self.fixed_descriptor.output_dim,
            elementwise_affine=False,
        )
        self.descriptor_projector = nn.Sequential(
            nn.Linear(
                self.fixed_descriptor.output_dim,
                self.config.embedding_dim,
            ),
            nn.LayerNorm(self.config.embedding_dim),
            nn.SiLU(),
        )
        self.late_fusion_head = nn.Linear(
            2 * self.config.embedding_dim,
            num_classes,
        )

    def forward(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        output = super().forward(values)
        with torch.no_grad():
            descriptors = self.fixed_descriptor(values)
        descriptor_parameter = next(self.descriptor_projector.parameters())
        normalized = self.descriptor_normalizer(
            descriptors.to(descriptor_parameter.dtype)
        )
        descriptor_embedding = self.descriptor_projector(normalized)
        fusion_logits = self.late_fusion_head(
            torch.cat((output["embedding"], descriptor_embedding), dim=1)
        )
        return {
            **output,
            # Residual logit fusion retains the original neural classifier as
            # an auditable path rather than leaving unused trainable weights.
            "logits": output["logits"] + fusion_logits,
            "descriptor_features": descriptors,
            "descriptor_embedding": descriptor_embedding,
        }


__all__ = [
    "DescriptorAssistedVIMDTemporalNet",
    "PhaseAwareMaskedSpectralTemporalEncoder",
    "VIMDTemporalNet",
    "VIMDTemporalCurriculumNet",
]
