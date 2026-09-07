# Cochannel Source-Exchange Identifiability Boundary

## Status and isolation

This is an identifiability analysis, not a model comparison. Its executable
artifact is labeled `diagnostic_identifiability_counterexample_only`. It is
not a member of the factor-isolated nine-domain protocol, is not headline
evidence, and cannot enter a performance table.

No nine-domain cache, split, A0--A7 ablation, or standard experiment runner is
changed by this branch.

## Impossibility statement

Let two emitters produce finite complex-baseband waveforms \(s_a\) and \(s_b\)
with different modulation labels \(y_a \ne y_b\). A single-antenna receiver
observes only

\[
  x = h_a s_a + h_b s_b + n.
\]

Let the hidden variable \(T\in\{a,b\}\) specify which emitter is the
pre-designated target. Without an observable pilot, preamble, grant, device
identifier, spatial signature, or another physical anchor, changing \(T=a\)
to \(T=b\) does not change \(x\). It changes the required answer from \(y_a\)
to \(y_b\).

For the balanced latent pair

\[
 (x,Y)=(x,y_a)\quad\text{or}\quad(x,y_b)
\]

with probability \(1/2\) each, every deterministic mixture-only classifier
must return the same value on both cases and can be correct on at most one.
A randomized classifier cannot improve the average: if it emits class
probabilities \(q\), its average probability of correctness is
\(\tfrac12(q_{y_a}+q_{y_b})\le\tfrac12\). Therefore its Bayes error is at
least \(1/2\) on this exchange-equivalence class. This is an information
boundary, not an optimization or network-capacity problem.

Near-equal received power does not resolve a **pre-designated identity**. The
numeric suite fixes the two physical waveforms and received gains at gaps of
0, 0.25, and 0.5 dB, then exchanges only the hidden target role. Both
assignments retain the exact same complex observation and SHA-256 checksum
while requiring QPSK versus 16QAM.

## Executable counterexample

Run from the project root:

```powershell
python diagnostics/run_cochannel_identifiability.py `
  --output artifacts/cochannel_identifiability/source_swap_v1
```

The script uses the public waveform-generation method without modifying the
generator. It writes:

- `counterexample.json`, including seeds, powers, labels, numerical tolerance,
  per-waveform and observation checksums, and the 50% lower error bound; and
- `checksums.json`, binding the generated artifact.

The runner refuses to overwrite a directory. The default absolute equality
tolerance is zero, and the two operand-order observations are also required to
have identical checksums.

## Three executable task protocols

### 1. Dominant-emitter AMC

Define the target as the emitter with maximum received component power, not as
an unobserved identity. Predeclare a minimum dominance margin
\(\Delta_{\min}=3\) dB:

\[
 y =
 \begin{cases}
 y_a,&P_a-P_b\ge 3\ {\rm dB},\\
 y_b,&P_b-P_a\ge 3\ {\rm dB},\\
 \text{ambiguous/abstain},&|P_a-P_b|<3\ {\rm dB}.
 \end{cases}
\]

The simulator must retain component powers for label construction, but the
receiver consumes only the mixture. Report accuracy outside the ambiguous
band together with coverage and ambiguous-band performance/abstention rate.
Changing the 3 dB margin is a protocol change and must be selected before test
evaluation.

Boundary: this protocol can support a claim about the strongest received
emitter. It cannot support a claim that an externally pre-designated emitter
was identified. It remains compatible with the patent-inspired
interference-robust classification objective only if “target” is explicitly
redefined as dominant emitter.

### 2. Physically anchored target AMC

Keep a pre-designated target, but provide an observable anchor: for example a
known pilot/preamble or grant association, device/RNTI identity, a
target-channel estimate, or an array-derived direction/spatial signature.
The simulator and model input must contain the same anchor available at
deployment. Evaluation swaps source order while holding the anchor on the
designated emitter and includes anchor-error/missing-anchor stress tests.

Boundary: this protocol may claim designated-target AMC conditional on the
declared anchor. It must not be described as single-antenna mixture-only AMC
unless the anchor is genuinely observable in that setting. Patent language
about target/interference components does not itself prove that the target
identity is observable.

### 3. Set-valued or multi-label collision AMC

Remove emitter identity from the output. Predict the unordered set (or
multiset, if multiplicity matters) of active modulation labels:

\[
  \{y_a,y_b\}.
\]

Use a permutation-invariant loss and report exact-set accuracy,
micro/macro-F1, cardinality error, and per-collision-pair recall. If two
emitters use the same modulation, a simple multi-hot vector cannot encode
multiplicity; a cardinality-aware multiset head is then required.

Boundary: this protocol supports collision-content recognition, not
designated-target recognition or source separation. It is adjacent to the
patent-inspired interference analysis but changes the paper's task and output
semantics.

## Recommended paper wording

Recommended precise limitation:

> Our single-antenna classifier is evaluated only where the target label is
> operationally defined by the data protocol. For two unanchored cochannel
> emitters drawn from the same modulation taxonomy, exchanging the hidden
> target identity leaves the mixture invariant while it may change the
> required label. Designated-target AMC is therefore not identifiable on this
> exchange class. We exclude such cases from headline evidence and treat
> dominant-emitter, physically anchored target, and permutation-invariant
> set-valued recognition as distinct future protocols.

Do not write that a latent mask, simulation-component teacher, confidence
score, or larger neural network resolves this symmetry. Ground-truth
components may supervise a diagnostic teacher during simulation, but they do
not provide target identity to a mixture-only receiver at inference.
