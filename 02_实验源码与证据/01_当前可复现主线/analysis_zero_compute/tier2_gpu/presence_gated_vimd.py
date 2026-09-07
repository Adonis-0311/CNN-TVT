"""Exploratory clean-retention remediation: interference-presence-gated VIMD.

STATUS: CONTROL ARM, NOT THE PRIMARY REMEDY.  Result A3b showed that all nine
A0--A7 models -- including the mask-free backbone A0 -- collapse on exactly
QPSK / 16QAM / 64QAM on unjammed windows (mean recall < 0.01), while CSSL,
MCLDNN and IQFormer do not.  The deficit is therefore a property of the shared
spectral front end, not of the tri-route decomposition, and a gate that
interpolates towards the *backbone* path cannot by itself repair it.  This
variant is retained to test that attribution and to look for a better
jammed/clean trade-off; see ``PREREGISTRATION_TIER2.md`` arm H2-G.

Original motivation (superseded as a causal account): on no-interference
windows the frozen A5 model almost never emits QPSK, 16QAM or 64QAM (about 2
windows out of ~500 per class, averaged over the 10 algorithm seeds); those
decisions collapse onto 8PSK and 256QAM, which suggested that the tri-route
decomposition was being applied at full strength when there was nothing to
separate.

Design.  ``VIMDNet`` forms the modulation-route weight as

    w = m + lambda * o + rho

and multiplies it into the complex spectrum.  This variant interpolates that
weight towards the identity using a scalar interference-presence gate derived
from the environment embedding:

    g = sigmoid(a * s + b),      s = presence score from the condition vector
    w' = g * w + (1 - g) * 1

Under a clean window a converged gate drives g -> 0 and the modulation branch
sees the unmodified spectrum, i.e. the shared backbone route.  Because that
backbone exhibits the same clean constellation collapse, this interpolation is
not expected to repair clean retention by itself.  Under interference g -> 1
and the A5 route is recovered.  The gate adds ``environment_dim + 3``
parameters, so the compactness claim is unaffected.

Status: EXPLORATORY.  This class is deliberately outside the immutable A0--A7
registry and outside the frozen confirmatory family.  Nothing produced with it
may be merged into ``artifacts/tvt_v4r_headline_composite`` or presented as a
preregistered result.  See ``PREREGISTRATION_TIER2.md``.

Run a one-seed, one-epoch full-cache smoke before launching the preregistered
multi-seed study.
"""

from __future__ import annotations

import torch
from torch import nn

from vimd_amc.models.common import ModelConfig
from vimd_amc.models.vimd import VIMDNet


class PresenceGatedVIMDNet(VIMDNet):
    """A5 with an interference-presence gate on the modulation route."""

    supports_tri_mechanism = True
    provenance = {
        "display_name": "VIMD with interference-presence gating",
        "claim_level": "exploratory post-hoc remediation of the clean-retention gate",
        "derived_from": "a5_vimd_full",
        "frozen_family_member": False,
    }

    def __init__(
        self,
        num_classes: int,
        num_jammers: int,
        config: ModelConfig | None = None,
        *,
        use_residual: bool = True,
        gate_bias_init: float = 2.0,
    ):
        super().__init__(num_classes, num_jammers, config, use_residual=use_residual)
        self.presence_score = nn.Linear(self.config.environment_dim, 1)
        self.gate_scale = nn.Parameter(torch.tensor(1.0))
        self.gate_bias = nn.Parameter(torch.tensor(float(gate_bias_init)))

    def forward(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        spectrum = self.front_end(values)
        context_features = self.context_encoder(spectrum)
        condition = self.environment_encoder(context_features)
        decomposition = self.tri_mask(context_features, condition)
        modulation_mask, jammer_mask, overlap_mask = decomposition["masks"].unbind(dim=1)
        lambda_overlap = decomposition["lambda_overlap"][:, :, None]
        predicted_rho = decomposition["rho"]
        applied_rho = predicted_rho if self.use_residual else torch.zeros_like(predicted_rho)
        rho = applied_rho[:, :, None]

        modulation_weight = modulation_mask + lambda_overlap * overlap_mask + rho
        jammer_weight = jammer_mask + (1.0 - lambda_overlap) * overlap_mask

        score = self.presence_score(condition)
        gate = torch.sigmoid(self.gate_scale * score + self.gate_bias)
        gate_map = gate[:, :, None]
        gated_modulation_weight = gate_map * modulation_weight + (1.0 - gate_map) * torch.ones_like(
            modulation_weight
        )

        modulation_spectrum = gated_modulation_weight * spectrum
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
            # the ungated weight is exported so the frozen mask/teacher losses
            # and every mechanism metric keep their original definition
            "modulation_weight": modulation_weight,
            "jammer_weight": jammer_weight,
            "gated_modulation_weight": gated_modulation_weight,
            "presence_gate": gate.squeeze(-1),
            **decomposition,
            "rho_predicted": predicted_rho,
            "rho": applied_rho,
        }


def register(factories: dict) -> dict:
    """Add the exploratory variant to a runner factory table.

    Usage inside ``experiments/run_standard_experiment.py`` (in the block that
    already carries the ``diagnostic_*`` entries, which is explicitly outside
    the immutable A0--A7 registry)::

        from analysis_zero_compute.tier2_gpu.presence_gated_vimd import register
        factories = register(factories)
    """

    from vimd_amc.losses import VIMDLossWeights
    from vimd_amc.models.spectral import PhysicalTriMaskTeacher
    from vimd_amc.training import TrainingObjective

    def build(classes: int, jammers: int, config):
        from experiments.run_standard_experiment import BuiltStandardModel

        return BuiltStandardModel(
            model=PresenceGatedVIMDNet(classes, jammers, config),
            teacher=PhysicalTriMaskTeacher(config),
            objective=TrainingObjective(
                name="full_vimd",
                loss_family="vimd",
                use_mask_supervision=True,
                use_jammer_auxiliary=True,
                use_quality_auxiliary=True,
                use_cross_condition_contrastive=True,
                use_orthogonality=True,
            ),
            loss_weights=VIMDLossWeights(),
        )

    factories["diagnostic_vimd_v5_presence_gated"] = build
    return factories
