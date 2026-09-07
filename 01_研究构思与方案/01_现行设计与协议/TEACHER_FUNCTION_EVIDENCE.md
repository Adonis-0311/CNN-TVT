# Teacher-function evidence and execution contract

This note closes the static/code portion of reviewer items M1 and M3 and fixes
the execution boundary for the remaining measurements. It does not turn a
screening cache into formal TVT evidence.

## Direct closed-form teacher

Let the three component STFT powers in one cell be \(P_s,P_j,P_u\), and let

\[
q_k=\frac{P_k}{P_s+P_j+P_u},\qquad k\in\{s,j,u\}.
\]

The primary fixed teacher is implemented directly as

\[
M_s^\star=[q_s-q_j]_+,\qquad
M_j^\star=[q_j-q_s]_+,\qquad
M_o^\star=q_u+2\min(q_s,q_j).
\]

These expressions are algebraically identical to the former
relative-dominance construction. Writing them directly exposes the low-SIR
behavior and makes the per-cell simplex identity immediate. The implementation
uses a scale-relative denominator floor, evaluates in float32 outside AMP, and
assigns a truly empty STFT cell to the overlap/unexplained route.

## A3-prime proportional teacher

The inserted control is:

- display ID: **A3′**;
- canonical runner name: `a3p_tri_proportional_teacher`;
- compatibility alias: `a3_ratio_teacher`; and
- teacher target: \((M_s,M_j,M_o)=(q_s,q_j,q_u)\).

A3′ uses the same `VIMDNet` architecture, residual path, CE objective, mask
loss weight, optimizer, data, and schedule as A3. Only the teacher function
changes. It lives in a separate teacher-form registry, so the immutable
A0--A7 IDs and their existing table/contrast keys are not renumbered.

The admissible confirmatory contrast is margin A3 minus proportional A3′ on
the predeclared hard split. No performance direction is asserted before an
eligible multi-seed run.

## Local-whitened diagnostic

For a local mean-power envelope \(E_k\) and frozen exponent \(\alpha\),

\[
\widetilde P_k=P_k/E_k^\alpha,\quad
d_{\rm loc}=
\frac{\widetilde P_s-\widetilde P_j}
{\widetilde P_s+\widetilde P_j}.
\]

The diagnostic teacher uses

\[
M_s=(q_s+q_j)[d_{\rm loc}]_+,\quad
M_j=(q_s+q_j)[-d_{\rm loc}]_+,\quad
M_o=q_u+(q_s+q_j)(1-|d_{\rm loc}|).
\]

The frozen diagnostic setting is a 9-frequency-bin by 5-frame periodic-Hann
STFT neighborhood with \(\alpha=1\). The raw \(q_u\) is retained; local
whitening therefore cannot erase the noise/receiver-artifact burden.

Full local whitening removes window-level SIR from the target-versus-jammer
comparison. It is consequently registered only as
`diagnostic_a3_local_whitened_teacher` and is explicitly disabled for primary
A3--A7 execution pending a development-only distribution and performance
comparison. It must not be silently substituted into the formal method.

## Deterministic hard-split audit

`experiments/audit_hard_split_teacher.py` traverses every selected source and
both views in manifest order. It pools every float32 target-route cell and
reports exact pooled-cell mean, NumPy-linear P90, and the exact
`M_s > 0` fraction:

```powershell
python experiments\audit_hard_split_teacher.py `
  --cache-root standards\cache_factor_screening_1024_v1 `
  --teachers closed_form_margin,proportional_power,local_whitened_margin `
  --output diagnostics\hard_split_teacher_cell_stats_screening_1024_v1.json `
  --cpu-threads 1
```

The generated artifact is bound to cache digest
`241b3aec6e74c79bac2d3ac22295098f0efe5cc79ff07acabf3593cbc32c49e3`
and cache designation `screening_not_formal_tvt_evidence`. It contains 500
sources, 1,000 views, and 3,904,000 cells per teacher. The artifact itself
sets `formal_tvt_evidence_eligible=false` and labels its scope
development-only.

### Screening diagnostic: full hard split

| Teacher | \(M_s\) mean | P90 | Nonzero-cell fraction |
|---|---:|---:|---:|
| Closed-form margin | 0.181774 | 0.895008 | 0.607694 |
| Proportional power | 0.196468 | 0.905317 | 1.000000 |
| Local-whitened margin | 0.161360 | 0.737403 | 0.640848 |

For the closed-form margin teacher, the family-stratified screening values are:

| Jammer family | Views | \(M_s\) mean | P90 | Nonzero-cell fraction |
|---|---:|---:|---:|---:|
| Tone | 184 | 0.214660 | 0.939135 | 0.785014 |
| Multitone | 165 | 0.193310 | 0.922774 | 0.683888 |
| Chirp | 154 | 0.219294 | 0.935609 | 0.758558 |
| Sweep | 170 | 0.176421 | 0.882232 | 0.550692 |
| Partial-band | 160 | 0.145557 | 0.789238 | 0.493283 |
| Comb | 167 | 0.139689 | 0.769761 | 0.365567 |

### Screening diagnostic: exact SNR = -10 dB, SIR = -15 dB

The cache contains 41 views at this exact operating point. For the closed-form
margin teacher, pooled \(M_s\) has mean 0.068658, P90 0.258110, and nonzero
fraction 0.565611.

| Jammer family | Views | \(M_s\) mean | P90 | Nonzero-cell fraction |
|---|---:|---:|---:|---:|
| Tone | 7 | 0.073732 | 0.275977 | 0.752488 |
| Multitone | 4 | 0.077530 | 0.313271 | 0.584465 |
| Chirp | 8 | 0.079251 | 0.317149 | 0.748175 |
| Sweep | 6 | 0.068684 | 0.257712 | 0.592298 |
| Partial-band | 7 | 0.072999 | 0.276951 | 0.522578 |
| Comb | 9 | 0.047957 | 0.138140 | 0.265283 |

These screening measurements confirm that the global-power thought experiment
does not imply an identically zero per-cell target route, but they also confirm
severe attenuation at the reviewer’s low-SNR/low-SIR point and substantial
jammer-family dependence. They justify the formal family-stratified audit and
A3-versus-A3′ experiment; they do not establish a method benefit or choose the
local-whitened variant.

## Pre-registered occupancy/gain mechanism test

The falsifiable mechanism hypothesis is machine-executable, not only prose.
For each source, jammer spectral occupancy uses only the single inference view
(`view1`) and is the fraction of periodic-Hann complex-STFT cells with jammer
power at least 1% of that source view's maximum jammer-cell power (a -20 dB
support threshold). Family occupancy is the arithmetic mean of those
source-level occupancies across sources in the hard-split family.

For exactly `tone`, `multitone`, `chirp`, `sweep`, `partial_band`, and `comb`,
the gain is calculated within that family's source subset as paired
A5-minus-A0 macro-F1 for each algorithm seed, then averaged arithmetically
across seeds and reported in percentage points. The predeclared direction is
non-increasing gain with increasing occupancy. The primary statistic is
Spearman rho with average ranks for ties; the one-sided alternative is rho < 0.
With six families, the p-value is exact over all 720 permutations of gain
labels. Alpha is 0.05. A supportive order check requires zero gain inversions
between strictly different occupancies using a `1e-12` tolerance. Neither
occupancy nor this test may select/tune a model.

The teacher audit emits family occupancy and a `pending` mechanism record until
formal paired A5/A0 family gains exist. Once they exist, run:

```powershell
python experiments\evaluate_occupancy_gain_mechanism.py `
  --input <formal-family-rows.json> `
  --output <formal-occupancy-gain-test.json>
```

The input contains `family_rows`, with `family`, `occupancy`, and either
`gain_pp` or both `a5_macro_f1` and `a0_macro_f1`. The evaluator rejects
missing/extra families, duplicates, nonfinite values, and occupancies outside
`[0,1]`.

For pipeline verification only, the screening cache yields mean occupancies
of 0.050 (tone), 0.121 (multitone), 0.050 (chirp), 0.150 (sweep), 0.242
(partial-band), and 0.326 (comb). These are development diagnostics; the gain
side remains `pending`, so no correlation or mechanism pass/fail is claimed.

## Related routing/loss clarifications

- `lambda_overlap`, `rho`, and mask temperature remain per-window scalars.
  The mask lattice already locates overlap per cell; a cellwise allocation gate
  would add an unsupervised lattice that can leak evidence between branches.
  Scalar allocation is retained for auditability, while its inability to vary
  within a sweep/pulse is disclosed.
- Since \(W_m\in[\rho,1+\rho]\), the spectral contrast diagnostic is bounded by
  \(20\log_{10}((1+\rho)/\rho)\), which is about 26.4 dB at
  \(\rho_{\min}=0.05\).
- Jensen--Shannon mask loss is symmetric, finite on zero-support teacher cells,
  and bounded by \(\log 2\) with natural logarithms. Those properties motivate
  it over one-way CE/KL for this sparse teacher, but the bounded objective can
  yield weak gradients far from the teacher; formal loss/teacher-agreement
  curves remain required.

## Remaining formal closure

Run this audit unchanged on the eligible formal hard cache, then execute A3
and A3′ with the frozen multi-seed protocol. The formal artifact must retain
the cache digest/designation, teacher specification, STFT settings, source/view
and cell counts, jammer-family strata, and the exact low-SNR/low-SIR operating
point. Screening numbers above must not be copied into a submission table.
