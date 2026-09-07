"""Non-deployable physical-teacher route probes.

This module exists only to test whether a fixed component-informed mask route
retains modulation-discriminative information.  Its overlap-add inverse is a
feature adapter, not a clean-waveform estimator.
"""

from __future__ import annotations

from typing import Final

import torch
from torch import nn

from .common import ModelConfig
from .spectral import ComplexSTFT, PhysicalTriMaskTeacher


class PhysicalTeacherRouteProbe(nn.Module):
    """Apply fixed physical oracle routes to the mixture STFT.

    The two declared routes are:

    - ``ms_only``: target-dominant teacher mass ``M_s``;
    - ``ms_plus_half_mo``: ``M_s + 0.5 M_o``.

    The probe requires ground-truth clean, jammer, and unexplained components,
    so every output is non-deployable.  The returned I/Q sequences are only
    adapters for a downstream fixed-feature diagnostic.
    """

    oracle_component_access: Final[bool] = True
    deployment_eligible: Final[bool] = False
    waveform_reconstruction_claimed: Final[bool] = False
    evidence_designation: Final[str] = "diagnostic_upper_control_only"
    route_names: Final[tuple[str, ...]] = (
        "ms_only",
        "ms_plus_half_mo",
    )

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.front_end = ComplexSTFT(config.n_fft, config.hop_length)
        self.teacher = PhysicalTriMaskTeacher(config)

    def inverse_for_feature_probe(
        self,
        spectrum: torch.Tensor,
        *,
        length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Invert the exact analysis lattice by explicit weighted overlap-add.

        ``torch.istft`` rejects the repository's ``center=False`` periodic Hann
        lattice because sample zero has zero window coverage.  This explicit
        inverse reproduces every covered sample and deterministically fills
        uncovered boundary samples with zero.  The second return value is the
        Boolean sample-coverage mask and must accompany any probe result.
        """

        if not torch.is_complex(spectrum) or spectrum.ndim != 3:
            raise ValueError("spectrum must be complex [batch, frequency, frame]")
        if spectrum.shape[1] != self.config.n_fft:
            raise ValueError("spectrum frequency dimension disagrees with n_fft")
        if length <= 0:
            raise ValueError("length must be positive")

        frames = torch.fft.ifft(
            spectrum.transpose(1, 2),
            n=self.config.n_fft,
            dim=-1,
            norm="ortho",
        )
        output = torch.zeros(
            spectrum.shape[0],
            length,
            dtype=spectrum.dtype,
            device=spectrum.device,
        )
        denominator = torch.zeros(
            length,
            dtype=spectrum.real.dtype,
            device=spectrum.device,
        )
        window = self.front_end.window.to(
            dtype=spectrum.real.dtype,
            device=spectrum.device,
        )
        window_square = window.square()
        for frame_index in range(frames.shape[1]):
            start = frame_index * self.config.hop_length
            stop = min(start + self.config.n_fft, length)
            if start >= length:
                break
            frame_length = stop - start
            output[:, start:stop] += (
                frames[:, frame_index, :frame_length] * window[:frame_length]
            )
            denominator[start:stop] += window_square[:frame_length]
        threshold = 16.0 * torch.finfo(denominator.dtype).eps
        covered = denominator > threshold
        output[:, covered] = output[:, covered] / denominator[covered]
        output[:, ~covered] = 0.0
        iq = torch.stack((output.real, output.imag), dim=1)
        return iq, covered

    @torch.no_grad()
    def forward(
        self,
        mixture: torch.Tensor,
        clean: torch.Tensor,
        jammer: torch.Tensor,
        unexplained: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        shapes = {
            tuple(values.shape)
            for values in (mixture, clean, jammer, unexplained)
        }
        if len(shapes) != 1:
            raise ValueError("mixture and component tensors must share one shape")
        if mixture.ndim != 3 or mixture.shape[1] != 2:
            raise ValueError("route probe expects [batch, 2, time] I/Q tensors")
        spectrum = self.front_end(mixture)
        masks = self.teacher(clean, jammer, unexplained)
        if masks.shape[0] != spectrum.shape[0] or masks.shape[2:] != spectrum.shape[1:]:
            raise RuntimeError("teacher and mixture STFT lattices are misaligned")
        route_weights = {
            "ms_only": masks[:, 0],
            "ms_plus_half_mo": masks[:, 0] + 0.5 * masks[:, 2],
        }
        result: dict[str, torch.Tensor] = {
            "masks": masks,
            "mixture_spectrum": spectrum,
        }
        common_coverage: torch.Tensor | None = None
        for route_name, weights in route_weights.items():
            route_iq, covered = self.inverse_for_feature_probe(
                weights * spectrum,
                length=mixture.shape[-1],
            )
            result[route_name] = route_iq
            result[f"{route_name}_weights"] = weights
            if common_coverage is None:
                common_coverage = covered
            elif not torch.equal(common_coverage, covered):
                raise RuntimeError("route inverse coverage masks disagree")
        if common_coverage is None:
            raise RuntimeError("no route probe was produced")
        result["covered_samples"] = common_coverage
        return result

    def control_metadata(self) -> dict[str, object]:
        return {
            "control_name": "fixed_physical_tri_teacher_route_feature_probe",
            "oracle_component_access": self.oracle_component_access,
            "deployment_eligible": self.deployment_eligible,
            "waveform_reconstruction_claimed": self.waveform_reconstruction_claimed,
            "evidence_designation": self.evidence_designation,
            "routes": {
                "ms_only": "M_s",
                "ms_plus_half_mo": "M_s + 0.5 M_o",
            },
            "allowed_interpretation": (
                "diagnose whether oracle teacher allocation retains "
                "class-discriminative information for fixed HOC features"
            ),
            "forbidden_interpretation": (
                "deployable mask, clean waveform recovery, source separation, "
                "or paper performance evidence"
            ),
        }


__all__ = ["PhysicalTeacherRouteProbe"]
