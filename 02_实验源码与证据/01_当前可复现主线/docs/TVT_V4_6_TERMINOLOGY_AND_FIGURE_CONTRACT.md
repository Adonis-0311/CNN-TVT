# TVT V4.6 Terminology and Figure Contract

Status: working contract for the prospective V4.6 revision. It does not alter
the sealed evidence family or any experiment artifact.

## Terminology ledger

| Canonical term | First-use form | Variants retired or restricted | Decision |
|---|---|---|---|
| VIMD-Net / A5 | VIMD-Net (A5) | compact model, proposed model | Use A5 in quantitative comparisons and VIMD-Net for the named configuration. |
| IQFormer-inspired | IQFormer-inspired | IQFormer, IQFormer-insp. | Use the full form in prose; the abbreviated form is allowed only in figures and narrow tables. |
| MCLDNN | MCLDNN | none | Preserve canonical capitalization. |
| received-I/Q sidecar | received-I/Q sidecar | I/Q side channel, side head | Use `sidecar` only after first definition. |
| frozen campaign | frozen source-disjoint simulation campaign | original campaign, campaign | Use `frozen campaign` when the evidence boundary matters. |
| independently generated severe-held set | independently generated severe-held set | independent severe set, severe held | Use the full form at first use in each standalone document. |
| condition-dependent rank instability | condition-dependent rank instability | rank exchange, winning region | `Rank exchange` may describe a local sign change, but not the paper-level contribution. |
| campaign-diagnostic overlap-SIR envelope | campaign-diagnostic overlap-SIR envelope | operating envelope, operating boundary, practical selector | `Operating envelope` is retained only when describing the historical fit or equation. |
| prospective falsification | prospective falsification of broad transport | failed validation | The failure is positive boundary evidence, not an exclusion. |
| representation repair | received-I/Q representation repair | solves severe interference | Repair is bounded to the tested simulation domains and is not a uniform-superiority claim. |
| sealed confirmatory family | sealed confirmatory family | sealed result family | Only A5-A0 is positive; isolated teacher and route-count contrasts remain unresolved. |
| exploratory/post-hoc | exploratory or post-hoc | validated, confirmed | SIR-cell, envelope, Tier-2, capacity, S5-A, and S5-R1 evidence stays outside the sealed family. |

## Evidence allocation

| Result | Class | V4.6 role | Location |
|---|---|---|---|
| A5-A0 +4.57 pp [3.65, 5.49] | sealed confirmatory | bounded configuration evidence | Abstract, V-A, Conclusion |
| Campaign SIR=-15 dB reversal | campaign-internal post-hoc | establishes within-campaign rank instability | Abstract, V-B, Fig. 2 |
| 32-cell fit / 80-cell held envelope | exploratory diagnostic | summarizes campaign structure only | V-C, Fig. 3, Supporting Material |
| S5-A negative transfer | preregistered post-campaign robustness | prospectively falsifies broad severe transport | Abstract, V-F, Discussion, Conclusion |
| S5-R1 +4.37 pp [2.96, 5.90] | preregistered post-campaign robustness | independent representation-repair evidence | Abstract, V-F, Discussion, Conclusion |
| probe, coverage, presence gate, capacity ladder | prospective exploratory controls | bounds the representation interpretation | V-E/V-F and Supporting Material |

## Figure contracts

Target: IEEE Transactions on Vehicular Technology. Python is the exclusive
plotting backend. Final widths are 88.9 mm (single column) or 182 mm (double
column). PDF is the manuscript vector format; SVG is the editable source; PNG
is the 600 dpi review proof. Fonts are Arial/Helvetica-compatible and embedded.

### Fig. 2, campaign-internal rank instability

- Core conclusion: A5 changes rank across the campaign SNR-SIR plane, but the
  favorable row is a post-hoc campaign-internal result.
- Archetype: quantitative grid.
- Panels: A5-MCLDNN and A5-IQFormer-inspired cell gains.
- Hero evidence: sign reversal on SIR=-15 dB.
- Risk control: shared zero-centered scale, printed values, star defined as an
  unadjusted positive lower interval, and no implication of independent
  transport.

### Fig. 3, campaign-diagnostic envelope

- Core conclusion: overlap and SIR summarize part of the within-campaign gain
  structure, with heterogeneous held-regime skill.
- Archetype: quantitative grid.
- Panels: diagnostic surface and held-cell predicted-versus-observed gain.
- Hero evidence: 6.23 pp RMSE versus 8.37 pp for the fit-domain constant.
- Risk control: the title and legend call the surface diagnostic; the caption
  states that independent severe transport later fails.

### Fig. 4, complexity and representation

- Core conclusion: the sidecar changes the representation at modest cost;
  parameter count alone does not explain the performance ordering.
- Archetype: single comparison panel.
- Hero evidence: A5 and sidecar, with the two strong I/Q references retained.
- Risk control: sealed points are filled, the exploratory sidecar is open, and
  all stochastic points show one seed standard deviation on the matched
  five-seed subset.

### Fig. 5, clean-condition transfer diagnosis

- Core conclusion: the compact spectral family loses clean-condition
  information for uncovered constellation-order classes more often than the
  I/Q baselines.
- Archetype: quantitative grid.
- Panels: compact spectral family and I/Q-domain baselines.
- Hero evidence: per-class hard-interference versus jammer-free F1.
- Risk control: the star denotes training-condition coverage only; it is not a
  significance marker. The same summary rule and scale are used in both panels.

