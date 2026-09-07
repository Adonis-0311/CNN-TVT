# TVT v3 8 GiB execution amendment

## Status

This is a prospective, pre-result execution-resource amendment issued on
2026-08-02. It supersedes the v2 freeze only for future compute execution.
No eligible formal v2 result was observed before the amendment.

- Base freeze: `tvt_submission/configs/formal_tvt_freeze_v2.json`
- Base SHA-256: `926e8eebdfadaab7876686f4762ec9bf2ffbb564a3e013b46932598f0a25b423`
- Amendment: `tvt_submission/configs/formal_tvt_freeze_v3_8gb.json`
- Amendment SHA-256: `8ff752ead790663f38e25e7c72a00d44d3dde2291a14a8ac2b580a4fd38dafc6`

## Frozen resource changes

The NVIDIA GeForce RTX 5060 Ti provides 8 GiB total VRAM. The amendment
therefore freezes the following execution profile:

- minimum free GPU memory: 6,000 MiB;
- per-device batch size: 32;
- AMP enabled;
- one CUDA fit at a time; and
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8`.

Any out-of-memory event is fail-closed. Adaptive batch resizing, model-width
changes, seed replacement, or outcome-dependent redesign are prohibited.

## Invariants inherited unchanged

The amendment inherits the v2 cache data, split/source identities, models,
seeds, 12-by-10 formal grid, epochs, optimizer, statistical family,
bootstrap, scientific gates, and paper provenance requirements. Only the
execution resource profile and run/artifact identities change. The new formal
run is `tvt_headline_1024_10seed_v3_8gb` and the learning-curve tag is
`v3_8gb`.

The operator entry point is `tvt_submission/run_v3_8gb_after_gpu_free.ps1`;
the exact continuation procedure is maintained in
`tvt_submission/LOCAL_EXECUTION_QUEUE.md`.
