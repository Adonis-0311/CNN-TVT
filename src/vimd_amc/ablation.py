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
MODEL_ALIASES = {
    "backbone": "a0_backbone",
    "single_mask": "a1_single_mask",
    "dual_mask": "a6_dual_full",
    "vimd": "a5_vimd_full",
}


def canonical_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def paper_ablation_protocol(name: str) -> dict[str, Any] | None:
    protocol = PAPER_ABLATION_PROTOCOLS.get(canonical_model_name(name))
    return deepcopy(protocol) if protocol is not None else None


__all__ = [
    "PAPER_ABLATION_NAMES",
    "PAPER_ABLATION_PROTOCOLS",
    "MODEL_ALIASES",
    "canonical_model_name",
    "paper_ablation_protocol",
]
