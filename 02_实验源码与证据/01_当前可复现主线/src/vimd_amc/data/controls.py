"""Dataset controls that deliberately expose non-deployable information.

These wrappers are diagnostic instruments.  They must not be registered as
deployable baselines or mixed into headline model comparisons.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import copy
from typing import Any, Final

import numpy as np
import torch
from torch.utils.data import Dataset


def _independent_copy(value: Any) -> Any:
    """Copy a per-sample array/tensor without changing its value or dtype."""

    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    return copy(value)


class CleanOracleInputDataset(Dataset):
    """Read-only view that feeds the cached clean target as classifier input.

    Only ``view1["x"]`` and ``view2["x"]`` are replaced, each by an
    independent copy of the corresponding ``clean`` component.  Labels,
    source ordering, component tensors, and condition metadata are returned
    exactly as supplied by the wrapped dataset.

    The clean component is unavailable at deployment.  Consequently this
    wrapper is a learnability/identifiability upper control, never a deployable
    baseline and never headline evidence.
    """

    oracle_clean_input: Final[bool] = True
    deployment_eligible: Final[bool] = False
    evidence_designation: Final[str] = "diagnostic_upper_control_only"

    def __init__(self, dataset: Dataset):
        if isinstance(dataset, CleanOracleInputDataset):
            raise ValueError("a clean-oracle dataset must not be wrapped twice")
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, object]:
        raw = self.dataset[index]
        if not isinstance(raw, Mapping):
            raise TypeError("clean-oracle wrapper expects mapping-valued samples")
        item: dict[str, object] = dict(raw)
        for view_name in ("view1", "view2"):
            raw_view = raw.get(view_name)
            if not isinstance(raw_view, Mapping):
                raise KeyError(f"sample is missing mapping-valued {view_name}")
            if "clean" not in raw_view or "x" not in raw_view:
                raise KeyError(f"{view_name} must contain both x and clean")
            view = dict(raw_view)
            view["x"] = _independent_copy(raw_view["clean"])
            item[view_name] = view
        return item

    def source_ids(self) -> list[int]:
        source_ids = getattr(self.dataset, "source_ids", None)
        if not callable(source_ids):
            raise AttributeError("wrapped dataset does not expose source_ids()")
        return list(source_ids())

    def manifest(self) -> dict[str, Any]:
        """Return the underlying immutable-cache manifest without mutation."""

        manifest = getattr(self.dataset, "manifest", None)
        if not callable(manifest):
            raise AttributeError("wrapped dataset does not expose manifest()")
        return manifest()

    def control_metadata(self) -> dict[str, object]:
        """Return explicit provenance that must accompany reported results."""

        return {
            "control_name": "clean_received_target_oracle_input",
            "oracle_clean_input": self.oracle_clean_input,
            "deployment_eligible": self.deployment_eligible,
            "evidence_designation": self.evidence_designation,
            "allowed_interpretation": (
                "non-deployable learnability/identifiability upper control"
            ),
            "forbidden_interpretation": (
                "deployable baseline, fair headline comparator, or source "
                "separation performance"
            ),
        }


__all__ = ["CleanOracleInputDataset"]
