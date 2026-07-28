"""Overlap-aware VIMD-Net on a physically aligned complex-STFT lattice."""

from __future__ import annotations

import torch
from torch import nn

from .common import ModelConfig
from .spectral import (
    ComplexSTFT,
    ConditionedDualMask,
    ConditionedTriMask,
    PhysicalDualMaskTeacher,
    PhysicalTriMaskTeacher,
    SpectralBranchEncoder,
    SpectralContextEncoder,
    SpectralEnvironmentEncoder,
)


class VIMDNet(nn.Module):
    """Vehicular interference–modulation disentanglement network.

    The predicted masks and fixed physical teacher share exactly the same STFT
    bins.  Masks are real-valued, so applying them to a complex spectrum
    preserves the phase of every retained time-frequency coefficient.
    """

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
        self.supports_tri_mechanism = True
        self.front_end = ComplexSTFT(self.config.n_fft, self.config.hop_length)
        self.context_encoder = SpectralContextEncoder(self.config)
        self.environment_encoder = SpectralEnvironmentEncoder(self.config)
        self.tri_mask = ConditionedTriMask(self.config)
        self.modulation_branch = SpectralBranchEncoder(self.config)
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
        """Return the physically aligned complex spectrum."""

        return self.front_end(values)

    def forward(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        spectrum = self.front_end(values)
        context_features = self.context_encoder(spectrum)
        condition = self.environment_encoder(context_features)
        decomposition = self.tri_mask(context_features, condition)
        modulation_mask, jammer_mask, overlap_mask = decomposition["masks"].unbind(dim=1)
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
        modulation_embedding = self.modulation_branch(modulation_spectrum)
        jammer_embedding = self.jammer_branch(jammer_spectrum)
        joint = torch.cat((condition, jammer_embedding), dim=1)
        return {
            "logits": self.modulation_head(modulation_embedding),
            "jam_logits": self.jammer_head(jammer_embedding),
            "quality": self.quality_head(joint),
            "embedding": modulation_embedding,
            "jammer_embedding": jammer_embedding,
            "spectrum": spectrum,
            "features": context_features,
            "mod_spectrum": modulation_spectrum,
            "jammer_spectrum": jammer_spectrum,
            "modulation_weight": modulation_weight,
            "jammer_weight": jammer_weight,
            **decomposition,
            "rho_predicted": predicted_rho,
            "rho": applied_rho,
        }


class DualMaskVIMDNet(nn.Module):
    """A6 two-route VIMD variant with dual branches and residual path."""

    supports_tri_mechanism = False

    def __init__(
        self,
        num_classes: int,
        num_jammers: int,
        config: ModelConfig | None = None,
    ):
        super().__init__()
        self.config = config or ModelConfig()
        self.use_residual = True
        self.mask_routes = 2
        self.front_end = ComplexSTFT(self.config.n_fft, self.config.hop_length)
        self.context_encoder = SpectralContextEncoder(self.config)
        self.environment_encoder = SpectralEnvironmentEncoder(self.config)
        self.dual_mask = ConditionedDualMask(self.config)
        self.modulation_branch = SpectralBranchEncoder(self.config)
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

    def forward(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        spectrum = self.front_end(values)
        context_features = self.context_encoder(spectrum)
        condition = self.environment_encoder(context_features)
        decomposition = self.dual_mask(context_features, condition)
        modulation_mask, non_target_mask = decomposition["masks"].unbind(dim=1)
        modulation_weight = modulation_mask + decomposition["rho"][:, :, None]
        jammer_weight = non_target_mask
        modulation_spectrum = modulation_weight * spectrum
        jammer_spectrum = jammer_weight * spectrum
        modulation_embedding = self.modulation_branch(modulation_spectrum)
        jammer_embedding = self.jammer_branch(jammer_spectrum)
        joint = torch.cat((condition, jammer_embedding), dim=1)
        return {
            "logits": self.modulation_head(modulation_embedding),
            "jam_logits": self.jammer_head(jammer_embedding),
            "quality": self.quality_head(joint),
            "embedding": modulation_embedding,
            "jammer_embedding": jammer_embedding,
            "spectrum": spectrum,
            "features": context_features,
            "mod_spectrum": modulation_spectrum,
            "jammer_spectrum": jammer_spectrum,
            "modulation_weight": modulation_weight,
            "jammer_weight": jammer_weight,
            **decomposition,
        }


__all__ = [
    "PhysicalTriMaskTeacher",
    "PhysicalDualMaskTeacher",
    "VIMDNet",
    "DualMaskVIMDNet",
]
