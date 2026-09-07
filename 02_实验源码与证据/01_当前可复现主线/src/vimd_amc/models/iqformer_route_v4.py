"""VIMD-v4: one shared IQFormer encoder with raw/route BatchNorm domains.

Every convolution, recurrent layer, attention projection, linear layer, and
classifier is instantiated exactly once.  Only BatchNorm affine parameters
and running statistics are duplicated for the received-mixture and routed
activation domains.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
from typing import Iterator, Literal

import torch
from torch import nn

from .baselines import IQFormerInspiredClassifier
from .common import ModelConfig
from .iqformer_route import ComplexSTFTOverlapAdd
from .spectral import (
    ComplexSTFT,
    ConditionedTriMask,
    SpectralContextEncoder,
    SpectralEnvironmentEncoder,
)

BatchNormDomain = Literal["raw", "route"]


class DomainSpecificBatchNorm(nn.Module):
    """Two BN states selected explicitly around one shared encoder call."""

    def __init__(self, source: nn.BatchNorm1d | nn.BatchNorm2d):
        super().__init__()
        if not isinstance(source, (nn.BatchNorm1d, nn.BatchNorm2d)):
            raise TypeError("source must be BatchNorm1d or BatchNorm2d")
        self.raw = copy.deepcopy(source)
        self.route = copy.deepcopy(source)
        self._active_domain: BatchNormDomain | None = None

    @property
    def dimensionality(self) -> int:
        return 1 if isinstance(self.raw, nn.BatchNorm1d) else 2

    def set_domain(
        self,
        domain: BatchNormDomain | None,
    ) -> BatchNormDomain | None:
        if domain not in (None, "raw", "route"):
            raise ValueError(f"unsupported BatchNorm domain: {domain!r}")
        previous = self._active_domain
        self._active_domain = domain
        return previous

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self._active_domain is None:
            raise RuntimeError(
                "DomainSpecificBatchNorm requires an explicit raw/route domain"
            )
        return getattr(self, self._active_domain)(values)


def _replace_batch_norm(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d)):
            setattr(module, name, DomainSpecificBatchNorm(child))
        else:
            _replace_batch_norm(child)


class SharedDSBNIQFormerEncoder(nn.Module):
    """One IQFormer-inspired stack whose only domain-specific state is BN."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = IQFormerInspiredClassifier(num_classes)
        _replace_batch_norm(self.backbone)

    @property
    def classifier(self) -> nn.Linear:
        return self.backbone.classifier

    def domain_norms(self) -> tuple[DomainSpecificBatchNorm, ...]:
        return tuple(
            module
            for module in self.backbone.modules()
            if isinstance(module, DomainSpecificBatchNorm)
        )

    @contextmanager
    def _using_domain(self, domain: BatchNormDomain) -> Iterator[None]:
        norms = self.domain_norms()
        previous = [normalization.set_domain(domain) for normalization in norms]
        try:
            yield
        finally:
            for normalization, prior in zip(norms, previous):
                normalization.set_domain(prior)

    def encode(
        self,
        values: torch.Tensor,
        *,
        domain: BatchNormDomain,
    ) -> torch.Tensor:
        with self._using_domain(domain):
            return self.backbone.encode(values)

    def normalization_parameter_accounting(self) -> dict[str, int]:
        norms = self.domain_norms()
        per_domain_affine = sum(
            parameter.numel()
            for normalization in norms
            for parameter in normalization.raw.parameters()
        )
        route_affine = sum(
            parameter.numel()
            for normalization in norms
            for parameter in normalization.route.parameters()
        )
        if per_domain_affine != route_affine:
            raise RuntimeError("raw and route BatchNorm affine sizes differ")
        extra_running_state = sum(
            normalization.route.running_mean.numel()
            + normalization.route.running_var.numel()
            + normalization.route.num_batches_tracked.numel()
            for normalization in norms
        )
        encoder_total = sum(
            parameter.numel() for parameter in self.backbone.parameters()
        )
        return {
            "batch_norm_layer_count": len(norms),
            "base_batch_norm_affine_parameters": per_domain_affine,
            "dsbn_extra_trainable_parameters": route_affine,
            "dsbn_extra_running_state_values": extra_running_state,
            "single_domain_equivalent_parameters": encoder_total - route_affine,
            "shared_dsbn_encoder_parameters": encoder_total,
        }


class VIMDIQFormerRouteDSBNNet(nn.Module):
    """Tri-mask route and raw safety path with a shared DSBN IQFormer stack."""

    supports_tri_mechanism = False
    requires_teacher_at_inference = False
    provenance = {
        "display_name": "VIMD-v4 shared-IQFormer route with domain BatchNorm",
        "claim_level": "pre-registered diagnostic candidate; outside paper A0--A7",
        "inference_inputs": ["received_iq_mixture"],
        "teacher_or_component_inputs_at_inference": False,
        "shared_convolution_linear_attention_weights": True,
        "domain_specific_state": "BatchNorm affine and running statistics only",
        "shared_encoder_calls_per_forward": 2,
        "replaces_a5": False,
    }

    def __init__(
        self,
        num_classes: int,
        num_jammers: int,
        config: ModelConfig | None = None,
    ):
        super().__init__()
        del num_jammers
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
        self.shared_encoder = SharedDSBNIQFormerEncoder(num_classes)
        self.condition_gate = nn.Linear(self.config.environment_dim, 1)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return self.front_end(values)

    def parameter_accounting(self) -> dict[str, int]:
        normalization = self.shared_encoder.normalization_parameter_accounting()
        total = sum(parameter.numel() for parameter in self.parameters())
        route_modules = (
            self.context_encoder,
            self.environment_encoder,
            self.tri_mask,
            self.condition_gate,
        )
        route_parameters = sum(
            parameter.numel()
            for module in route_modules
            for parameter in module.parameters()
        )
        return {
            **normalization,
            "route_and_gate_trainable_parameters": route_parameters,
            "candidate_total_parameters": total,
            "shared_encoder_calls_per_forward": 2,
        }

    def forward(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        if values.ndim != 3 or values.shape[1] != 2:
            raise ValueError("VIMD-v4 expects [batch, 2, time] received mixtures")
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

        raw_embedding = self.shared_encoder.encode(values, domain="raw")
        route_embedding = self.shared_encoder.encode(
            route_values,
            domain="route",
        )
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
    "BatchNormDomain",
    "DomainSpecificBatchNorm",
    "SharedDSBNIQFormerEncoder",
    "VIMDIQFormerRouteDSBNNet",
]
