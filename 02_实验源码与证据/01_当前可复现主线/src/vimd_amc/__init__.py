"""VIMD-Net research implementation."""

from .data.synthesis import SynthesisConfig, SignalSynthesizer
from .data.dataset import PairedAMCDataset, Regime
from .models.baselines import (
    BackboneClassifier,
    CSSLAMCSupervisedAdaptation,
    DiagnosticDualMaskCEClassifier,
    IQFormerInspiredClassifier,
    MCLDNNReimplementation,
    SingleMaskClassifier,
)
from .models.vimd import (
    DualMaskVIMDNet,
    LocalWhitenedTriMaskTeacher,
    PhysicalDualMaskTeacher,
    PhysicalTriMaskTeacher,
    ProportionalTriMaskTeacher,
    TriMaskTeacherSpec,
    VIMDNet,
    build_tri_mask_teacher,
)

__all__ = [
    "SynthesisConfig",
    "SignalSynthesizer",
    "PairedAMCDataset",
    "Regime",
    "BackboneClassifier",
    "SingleMaskClassifier",
    "DiagnosticDualMaskCEClassifier",
    "MCLDNNReimplementation",
    "CSSLAMCSupervisedAdaptation",
    "IQFormerInspiredClassifier",
    "VIMDNet",
    "DualMaskVIMDNet",
    "PhysicalTriMaskTeacher",
    "ProportionalTriMaskTeacher",
    "LocalWhitenedTriMaskTeacher",
    "PhysicalDualMaskTeacher",
    "TriMaskTeacherSpec",
    "build_tri_mask_teacher",
]
