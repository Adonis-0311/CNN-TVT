"""Standards-aligned channel backends used by the VIMD-AMC evidence pipeline."""

from .cache import (
    CachedPairedAMCDataset,
    FACTOR_ISOLATED_SPLITS,
    FactorSplitPolicy,
    TDLCacheBuildConfig,
    TDLCacheBuildResult,
    build_tdl_paired_cache,
    factor_isolated_split_policies,
    validate_cached_components,
)
from .nrtdl_matlab import (
    NRTDLBatchResult,
    NRTDLConfiguration,
    apply_nrtdl_batch,
)

__all__ = [
    "CachedPairedAMCDataset",
    "FACTOR_ISOLATED_SPLITS",
    "FactorSplitPolicy",
    "NRTDLBatchResult",
    "NRTDLConfiguration",
    "TDLCacheBuildConfig",
    "TDLCacheBuildResult",
    "apply_nrtdl_batch",
    "build_tdl_paired_cache",
    "factor_isolated_split_policies",
    "validate_cached_components",
]
