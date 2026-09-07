"""Standards-aligned channel backends used by the VIMD-AMC evidence pipeline."""

from .cache import (
    CachedPairedAMCDataset,
    DEFAULT_MATLAB_BATCH_SIZE,
    FACTOR_ISOLATED_SPLITS,
    MAT_V7_MAX_VARIABLE_BYTES,
    TVT_V2_FACTOR_SPLITS,
    TVT_V2_RECEIVER_STRESS_SPLITS,
    FactorSplitPolicy,
    ReceiverStressSplitPolicy,
    TDLCacheBuildConfig,
    TDLCacheBuildResult,
    build_tdl_paired_cache,
    cache_build_execution_preflight,
    factor_isolated_split_policies,
    factor_isolated_split_policies_v2,
    validate_cached_components,
)
from .nrtdl_matlab import (
    NRTDLBatchResult,
    NRTDLConfiguration,
    apply_nrtdl_batch,
)

__all__ = [
    "CachedPairedAMCDataset",
    "DEFAULT_MATLAB_BATCH_SIZE",
    "FACTOR_ISOLATED_SPLITS",
    "MAT_V7_MAX_VARIABLE_BYTES",
    "TVT_V2_FACTOR_SPLITS",
    "TVT_V2_RECEIVER_STRESS_SPLITS",
    "FactorSplitPolicy",
    "ReceiverStressSplitPolicy",
    "NRTDLBatchResult",
    "NRTDLConfiguration",
    "TDLCacheBuildConfig",
    "TDLCacheBuildResult",
    "apply_nrtdl_batch",
    "build_tdl_paired_cache",
    "cache_build_execution_preflight",
    "factor_isolated_split_policies",
    "factor_isolated_split_policies_v2",
    "validate_cached_components",
]
