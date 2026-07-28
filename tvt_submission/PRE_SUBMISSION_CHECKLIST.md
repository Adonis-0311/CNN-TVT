# TVT pre-submission checklist

## Automatic gates

- [ ] Formal cache manifest has schema 2 and designation
  `headline_formal_tvt_evidence`.
- [ ] Formal cache has exactly nine protocol splits and globally disjoint
  source identities.
- [ ] File checksums, component identity, realized SNR/SIR, guard, and
  normalization audits pass.
- [ ] Five predeclared seeds complete for every required model.
- [ ] The exact formal model family includes
  `cssl_amc_supervised_adaptation` as the prospectively fixed primary
  reference; A0, MCLDNN, IQFormer-inspired, and CSSL are all present as the
  required non-oracle hard-gate baselines.
- [ ] Every fitted model selects at least one checkpoint after the complete
  objective becomes eligible; no final-state fallback is used.
- [ ] Start/end executable-source fingerprints are identical.
- [ ] Source-aligned probability bundles, metrics, paired statistics,
  mechanism probes, and complexity records exist.
- [ ] `evidence_eligibility.eligible` is exactly `true`.
- [ ] `generate_macro_values.py` derives the paper result manifest from that
  run; no performance value is typed or copied by hand.
- [ ] The macro manifest is exactly schema
  `vimd_amc.tvt.macro_values.v3`, contains 73 provenance records and the
  complete `scientific_release_gate`, and canonical artifact reconstruction
  agrees exactly.
- [ ] The generator's JSON/CSV/NPZ cross-check passes with no missing,
  duplicate, nonfinite, inconsistent, or ambiguous required cell.
- [ ] `validate_release.py --write` generates
  74 non-sentinel commands plus `\EligibleLockedResults` only with an intact
  schema-v2 release lock, and revalidation confirms the 75-command released
  file, sentinel name/value, and `results_auto.tex` hash.
- [ ] Latency commands are exactly `VIMDLatencyPFifty`,
  `VIMDLatencyPNinetyFive`, and `VIMDLatencyDevice`; no digit-bearing `P50` or
  `P95` TeX command name is accepted.
- [ ] The final LaTeX build has no undefined citation/reference, fatal error,
  overfull box, or internal-review banner.

## Scientific gates

- [ ] Primary endpoint and minimum meaningful paired effect were fixed before
  opening test results.
- [ ] `PrimaryReference` is exactly the predeclared CSSL supervised adaptation;
  it is never relabeled “strongest,” and no removed post-hoc winner contract is
  reintroduced.
- [ ] CSSL-AMC is labeled only as an official-architecture supervised
  adaptation; it is not described as a complete reproduction, official result,
  structured-interference-specific method, or MFENet substitute.
- [ ] Any strongest-published or strongest-interference-specific claim remains
  prohibited unless the open comparator and architecture-specific tuning
  sensitivity gates are prospectively closed.
- [ ] A0--A7 causal interpretations match the actual controls; A6 remains a
  composite control.
- [ ] Hierarchical seed/source bootstrap is the headline inference.
- [ ] All per-seed McNemar tests are labeled supplemental and their
  multiplicity family is explicit.
- [ ] Supported-label masks govern jammer auxiliary metrics; held-out or
  excluded family logits are not presented as trained recognition.
- [ ] Clean retention is stratified by seen A/C/D and held B/E profiles.
- [ ] A5 hard macro-F1 is at least +5 pp versus each required non-oracle
  baseline and strictly greater than A1 and A6; at least two of three held
  factors gain +3 pp versus CSSL; both clean strata satisfy the -1 pp point and
  -2 pp 95% lower-bound floors.
- [ ] Required mechanism fields are finite, required correlations are
  nonnegative, and the oracle-conditioned spectral component ratio is strictly
  positive without being described as waveform SIR, SDR, or separation.
- [ ] No cochannel, mixed-jammer, SDR, real-time, source-separation, waveform
  recovery, or V2X-compliance claim exceeds the frozen evidence.
- [ ] If no SDR layer is added, title, abstract, conclusion, and cover letter
  all state the simulation-only boundary.

## Human-author gates

- [ ] Names, affiliations, corresponding author, ORCID, funding,
  acknowledgments, and conflicts are supplied.
- [ ] Patent/publication timing is reviewed by a qualified professional.
- [ ] Every equation, table, figure, and claim is checked by a human author.
- [ ] Every citation, DOI, year, volume, pages, 3GPP release/version, and
  claimed implementation provenance is verified against a primary source.
- [ ] Current TVT page limits, templates, fees, disclosure policy, and
  submission instructions are rechecked on the upload date.
- [ ] Any required disclosure of generative-AI-assisted language editing is
  prepared by the human authors.
- [ ] The human authors approve the final PDF and source archive.
