"""Single source of truth for the manuscript's A0--A7 table."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


PAPER_ABLATION_PROTOCOLS: dict[str, dict[str, Any]] = {
    "a0_backbone": {
        "ablation_id": "A0",
        "mask_routes": 0,
        "teacher": "none",
        "mtl": False,
        "xcc": False,
        "residual_path": False,
    },
    "a1_single_mask": {
        "ablation_id": "A1",
        "mask_routes": 1,
        "teacher": "none",
        "mtl": False,
        "xcc": False,
        "residual_path": False,
    },
    "a2_tri_no_teacher": {
        "ablation_id": "A2",
        "mask_routes": 3,
        "teacher": "none",
        "mtl": False,
        "xcc": False,
        "residual_path": True,
    },
    "a3_tri_teacher": {
        "ablation_id": "A3",
        "mask_routes": 3,
        "teacher": "fixed_physical_tri",
        "mtl": False,
        "xcc": False,
        "residual_path": True,
    },
    "a4_tri_teacher_mtl": {
        "ablation_id": "A4",
        "mask_routes": 3,
        "teacher": "fixed_physical_tri",
        "mtl": True,
        "xcc": False,
        "residual_path": True,
    },
    "a5_vimd_full": {
        "ablation_id": "A5",
        "mask_routes": 3,
        "teacher": "fixed_physical_tri",
        "mtl": True,
        "xcc": True,
        "residual_path": True,
    },
    "a6_dual_full": {
        "ablation_id": "A6",
        "mask_routes": 2,
        "teacher": "fixed_physical_dual_collapsed_from_tri",
        "mtl": True,
        "xcc": True,
        "residual_path": True,
    },
    "a7_vimd_no_residual": {
        "ablation_id": "A7",
        "mask_routes": 3,
        "teacher": "fixed_physical_tri",
        "mtl": True,
        "xcc": True,
        "residual_path": False,
    },
}

PAPER_ABLATION_NAMES = tuple(PAPER_ABLATION_PROTOCOLS)

# A3-prime is an inserted teacher-function control, not a renumbering of the
# immutable A0--A7 ladder. Keeping it in a separate registry preserves every
# existing ID, table row, and formal contrast key.
TEACHER_FORM_ABLATION_PROTOCOLS: dict[str, dict[str, Any]] = {
    "a3p_tri_proportional_teacher": {
        "ablation_id": "A3\u2032",
        "mask_routes": 3,
        "teacher": "proportional_component_power_tri",
        "mtl": False,
        "xcc": False,
        "residual_path": True,
        "comparison_role": "teacher_function_control_for_A3",
    },
}
TEACHER_FORM_ABLATION_NAMES = tuple(TEACHER_FORM_ABLATION_PROTOCOLS)

# The local-whitening variant is implemented for a pre-registered diagnostic,
# but is not silently substituted into A3--A7. Enabling it requires an
# explicit diagnostic runner name and a new evidence freeze.
LOCAL_NORMALIZATION_TEACHER_POLICY: dict[str, Any] = {
    "primary_a3_to_a7_enabled": False,
    "diagnostic_runner_name": "diagnostic_a3_local_whitened_teacher",
    "mode": "local_whitened_margin",
    "local_frequency_bins": 9,
    "local_time_frames": 5,
    "local_whitening_exponent": 1.0,
    "selection_status": (
        "frozen_disabled_pending_development_only_teacher_distribution_and_"
        "performance_comparison"
    ),
}

MODEL_ALIASES = {
    "backbone": "a0_backbone",
    "single_mask": "a1_single_mask",
    "dual_mask": "a6_dual_full",
    "vimd": "a5_vimd_full",
    "a3_ratio_teacher": "a3p_tri_proportional_teacher",
}


def canonical_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def paper_ablation_protocol(name: str) -> dict[str, Any] | None:
    canonical = canonical_model_name(name)
    protocol = PAPER_ABLATION_PROTOCOLS.get(canonical)
    if protocol is None:
        protocol = TEACHER_FORM_ABLATION_PROTOCOLS.get(canonical)
    return deepcopy(protocol) if protocol is not None else None


__all__ = [
    "PAPER_ABLATION_NAMES",
    "PAPER_ABLATION_PROTOCOLS",
    "TEACHER_FORM_ABLATION_NAMES",
    "TEACHER_FORM_ABLATION_PROTOCOLS",
    "LOCAL_NORMALIZATION_TEACHER_POLICY",
    "MODEL_ALIASES",
    "canonical_model_name",
    "paper_ablation_protocol",
]
