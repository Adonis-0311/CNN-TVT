# VIMD-v2 Phase-Aware Temporal Candidate (Diagnostic Only)

## Status and scope

`VIMDTemporalNet` is an internal algorithm candidate. It does not replace A5,
does not alter the locked A0--A7 ablation registry, and is not part of either
the preregistered screening or headline model suite. Any output from
`diagnostics/run_temporal_v2_diagnostic.py` is labeled
`diagnostic_non_evidence_do_not_cite`.

## Hypothesis isolated

The current A5 modulation branch applies 2-D convolutions and then pools the
whole time--frequency map. The new branch keeps the ordered STFT-frame axis:

1. apply the student tri-mask to the complex mixture spectrum;
2. retain real, imaginary, and log-magnitude values for every frequency bin;
3. project the `3*n_fft` phase-aware channels at each frame;
4. apply parallel 3/5/7-frame filters and dilated causal blocks with dilation
   1/2/4/8;
5. fuse the temporal embedding with the existing spectral embedding and
   environment embedding.

The model's `forward` signature is exactly `forward(received_iq)`. It accepts
no clean component, jammer component, teacher mask, or modulation label.

## Fairness and audit

The standard 256-sample configuration uses the same STFT, tri-mask, teacher
loss, MTL heads, XCC loss, optimizer, and checkpoint-selection path as A5.
Only the modulation representation and its explicit fusion layer are added.
The standard configuration parameter counts are:

| Model | Parameters |
|---|---:|
| A0 spectral backbone | 8,458 |
| A5 full VIMD | 39,500 |
| diagnostic VIMD-v2 temporal | 88,556 |

The extra capacity is material (2.24x A5), so a positive result would justify
a later width-matched control; it would not by itself establish that temporal
ordering, rather than capacity, caused the gain.

## Tiny diagnostic decision rule

The bounded one-seed diagnostic predeclares success as both:

- validation accuracy at least 3 percentage points above the better of A0/A5;
- held-out-channel accuracy at least 2 percentage points above the better of
  A0/A5.

Otherwise the temporal-only hypothesis is treated as unsupported at this
budget. Passing this rule only promotes the candidate to a stable multi-seed
screening cache; it never promotes these tiny results to paper evidence.

The first received-mixture run (seed 17, 160/80/160 samples, three epochs)
did **not** pass: A0/A5/VIMD-v2 validation accuracy was
7.50%/6.25%/6.25%, and held-out-channel accuracy was
10.00%/10.00%/8.75%. VIMD-v2 reduced training modulation CE from 2.380 to
2.273, faster than A5 (2.326 to 2.301), but that optimization signal did not
generalize. These numbers are diagnostic and must not be cited as experimental
evidence.

The separate transparent learnability control shows why a clean-input probe is
needed before another mixture run: its 61-D HOC/cyclostationary ridge achieved
71.29% held-out accuracy on the tracked clean target but only 12.70% on the
received mixture, while the existing A0 clean oracle achieved only 17.77% after
12 epochs. This is diagnostic context from
`artifacts/learnability_controls/screening_v1_seed20260727/run.json`, not a
paper result. The same runner can therefore replace `x` with the tracked clean
component and use CE only to test representation learnability. That
`tracked_clean` mode is an oracle control and is never an inference mode.

The locked tracked-clean run (seed 17, 500/120/250 balanced samples, 12 epochs,
CE only) passed the tiny promotion rule:

| Model | Validation acc./F1 | Held-out-channel acc./F1 |
|---|---:|---:|
| A0 | 18.33% / 16.94% | 12.80% / 13.05% |
| A5 architecture | 15.00% / 14.45% | 13.20% / 12.47% |
| diagnostic VIMD-v2 temporal | 25.83% / 23.51% | 21.20% / 19.09% |

VIMD-v2's training CE fell from 2.341 to 1.739, versus 2.325 to 2.260 for the
A5 architecture. This supports the narrow claim that the added temporal path
has greater clean-component learnability under this diagnostic budget. It
still leaves a large gap to the transparent 61-D control and does not overcome
received-mixture interference. Its 2.24x A5 parameter count also leaves a
capacity confound that requires a width-matched control before a mechanism
claim.

## Teacher-routed or clean-component curriculum decision

The first candidate deliberately does not implement component-routed training,
because the initial observation (overlap-mask correlation without
classification learning) did not distinguish a routing failure from an
encoder failure. Adding oracle routing at the same time would have confounded
the test. The tracked-clean result now clears the precondition for a separately
registered routing-curriculum diagnostic: temporal representation improved
when interference was removed, while received-mixture accuracy did not.

A later curriculum is suitable only if all of the following are locked before
running it:

- teacher/student route blending is based only on simulation component powers,
  never on the modulation label;
- the oracle coefficient reaches exactly zero before checkpoint selection;
- validation, test, checkpoint reload, export, and inference call the
  mixture-only signature;
- the route coefficient and the first student-only epoch are logged;
- an oracle-to-student transition gap is reported on a held-out validation
  subset;
- a student-only checkpoint beats the same-architecture no-curriculum control.

Reject the curriculum if oracle removal causes more than a 2-point validation
accuracy drop, if any selected checkpoint predates full annealing, or if any
evaluation code supplies component tensors. Clean-logit or representation
distillation has the same restrictions and additionally needs an explicitly
separate clean teacher whose parameters are unavailable at inference.

The implemented curriculum followed those gates. Its teacher-route
coefficients were `1.00, 0.75, 0.50, 0.25, 0.00`, then remained exactly zero.
Checkpoint selection did not begin until epoch 10. The saved state had no
teacher-routing keys and reproduced logits after strict reload with zero
numerical error. Despite that clean inference contract, it failed the
prospective mixture rule:

| Same-parameter VIMD-v2 control | Validation acc. | Held-out acc. / F1 |
|---|---:|---:|
| student-only | 9.00% | 10.50% / 9.74% |
| fully annealed teacher routing | 11.00% | 8.00% / 6.86% |

Both variants have 88,556 parameters. The curriculum gained 2 points on
validation but lost 2.5 points in held-out accuracy and 2.88 points in
held-out macro-F1. It is therefore closed as an unsupported diagnostic, not a
paper method.

## Fixed-descriptor late-fusion stop test

After curriculum failure, one final low-cost control late-fused the audited
fixed 61-D HOC/cyclostationary descriptor with the VIMD-v2 embedding. The
descriptor itself has no trainable parameters. Its projection and fusion head
increase the full candidate from 88,556 to 92,598 parameters (+4,042, +4.56%),
so even a positive result would retain a small capacity confound.

The same one-seed 320/100/200 balanced subsets and 12-epoch CE-only protocol
gave:

| Input | Candidate | Validation acc. / F1 | Held-out acc. / F1 |
|---|---|---:|---:|
| received mixture | neural VIMD-v2 | 9.00% / 6.04% | 13.50% / 12.05% |
| received mixture | + fixed 61-D descriptor | 12.00% / 11.18% | 11.00% / 9.64% |
| tracked-clean oracle | neural VIMD-v2 | 19.00% / 18.84% | 16.50% / 15.25% |
| tracked-clean oracle | + fixed 61-D descriptor | 27.00% / 26.89% | 24.00% / 23.11% |

Late fusion improves the tracked-clean arm by 8 validation points and 7.5
held-out points, showing that the neural embedding still omits useful
modulation statistics. On the deployable mixture input it gains 3 validation
points but loses 2.5 held-out points and 2.40 macro-F1 points, so it fails the
locked all-required rule. This distinguishes clean statistical sufficiency
from interference robustness: fixed descriptors help when the target is
available, but do not repair mixture identifiability/routing. The branch is
therefore stopped; it is neither the main method nor an innovation claim.

All three result families are immutable JSON diagnostics with adjacent
SHA-256 manifests under:

- `artifacts/diagnostic_vimd_v2_temporal/`
- `artifacts/diagnostic_teacher_routing/`
- `artifacts/diagnostic_descriptor_assisted/`

## Reproduction

```powershell
D:\Python\python.exe diagnostics\run_temporal_v2_diagnostic.py `
  --cache-root standards\cache_screening_v1 `
  --epochs 3 --seed 17 --cpu-threads 1
```

Clean-input representation control:

```powershell
D:\Python\python.exe diagnostics\run_temporal_v2_diagnostic.py `
  --cache-root standards\cache_screening_v1 `
  --input-mode tracked_clean --epochs 12 --seed 17 `
  --train-per-class 50 --validation-per-class 12 `
  --heldout-per-class 25 --cpu-threads 1
```

Fully annealed routing and final descriptor stop tests:

```powershell
D:\Python\python.exe diagnostics\run_teacher_routing_curriculum.py `
  --cache-root standards\cache_screening_v1 --epochs 12 `
  --anneal-epochs 5 --student-only-before-selection 3 `
  --train-per-class 32 --validation-per-class 10 `
  --heldout-per-class 20 --seed 17 --cpu-threads 1

D:\Python\python.exe diagnostics\run_descriptor_assisted_diagnostic.py `
  --cache-root standards\cache_screening_v1 --epochs 12 `
  --train-per-class 32 --validation-per-class 10 `
  --heldout-per-class 20 --seed 17 --cpu-threads 1
```
