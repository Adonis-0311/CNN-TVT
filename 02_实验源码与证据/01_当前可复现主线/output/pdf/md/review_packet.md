# Nature-style reviewer simulation - immutable review packet

You are acting as ONE referee in a simulated Nature-style peer review. You must NOT
see any other referee's report, notes, or conclusions. Your job is an independent,
evidence-grounded critique from the referee side, NOT an author rebuttal and NOT an
editorial decision letter.

## Manuscript under review

Title: "An Occupancy-Indexed Operating Envelope for Compact Physics-Guided Automatic
Modulation Classification under Structured Interference" (7 pages, IEEE Transactions
on Vehicular Technology style, anonymous authors).

Read the manuscript from these files (same content, three views):

1. Source: `D:\CNN信号调制识别\TVT支线\02_实验源码与证据\01_当前可复现主线\paper\main.tex`
2. Compiled PDF text with page numbers: `D:\CNN信号调制识别\TVT支线\02_实验源码与证据\01_当前可复现主线\output\pdf\extracted_text.txt`
3. Resolved numerical values: `D:\CNN信号调制识别\TVT支线\02_实验源码与证据\01_当前可复现主线\paper_data_layer\outputs\paper_macros.tex`

Treat the compiled PDF text (extracted_text.txt) as the authoritative manuscript
wording and pagination. The macros file resolves every `\SomeMacro` into a concrete
number, which you MUST use to check numeric self-consistency (e.g., a number reported
at two precisions, a table that does not reconcile with the prose, a superlative that
contradicts the paper's own table).

## Assessment boundary

Available: full manuscript text, abstract, all sections, tables (Table I-IV), figures
1-7 with captions, and the complete resolved numerical macro appendix. The figures'
pixel content cannot be inspected beyond their captions and any numbers embedded in
the extracted text.

Not available: author rebuttal, raw per-seed data files, code execution, and any
supplementary material. Mark anything you cannot assess as "not assessable from
provided material" instead of inventing it.

## Journal criteria (the only authoritative local source for this exercise)

A Nature-style Article should (a) report original scientific research, (b) be of
outstanding scientific importance, and (c) reach a conclusion of interest to an
interdisciplinary readership. A referee report should indicate (i) who will be
interested in the results and why, and (ii) any technical failings that must be
addressed before the authors' case is established.

## Five assessment axes (cover all, weight per your assigned emphasis)

1. originality
2. scientific importance / significance
3. interdisciplinary readership interest
4. technical soundness / technical failings
5. readability for nonspecialists

## Internal technical coverage checklist (do NOT dump this matrix into the report)

Consider, where applicable: novelty-significance, mechanism-evidence,
experimental-design, statistical-rigor, reproducibility, data-resource-quality,
figures-and-tables, writing-clarity, claim-moderation, causal-vs-correlative.
(Clinical-validity and ethical-governance are not applicable to this simulation study.)

## Report skeleton (produce exactly this, in Chinese, keeping English technical terms)

```
Reviewer N
- 总体评价 (Overall assessment)
- 谁会关心这些结果，为什么 (Who would be interested, and why)
- 主要优点 (Major strengths)
- 主要关切 (Major Concerns)
- 次要意见 (Minor Comments)
- 在作者立论成立之前必须解决的技术缺陷 (Technical failings roll-up; cross-reference Blocking IDs)
- 对照 Nature 式标准的评估 (Assessment against the 5 axes)
- 建议姿态 (Recommendation posture; reviewer-like, e.g. "supportive if technical concerns are resolved")
```

For each Major Concern, use this shape (keep IDs in English):

```
R1-M1 [axis]
严重程度 (Severity) Major
阻断 (Blocking) Yes / No
主张指针 (Claim pointer) [one-sentence faithful paraphrase of the challenged claim]
证据指针 (Evidence pointer) [section / figure / table / page, or "location not provided"]
关切 (Concern) [evidence-grounded critique]
为什么重要 (Why it matters) [effect on the central case]
解决判据 (Resolution test) [what evidence/analysis/clarification/claim-adjustment would close it]
```

For each Minor Comment, use this shorter shape:

```
R1-m1 [axis]
严重程度 (Severity) Minor
受影响要素 (Affected element)
证据指针 (Evidence pointer)
问题 (Issue) [localized]
所需修正 (Required correction) [specific]
```

## Severity and blocking calibration

- Major + Blocking Yes: the current evidence cannot establish a CENTRAL conclusion
  (or a validity/reproducibility problem prevents a credible scientific case).
- Major + Blocking No: materially weakens inference/novelty/significance/reproducibility
  but does not by itself invalidate the entire central case.
- Minor: localized, does not change the central conclusion; actionable and specific.
- Do not invent concerns to fill a quota. If a tier has no grounded item, write
  "None identified from the supplied material".

## Grounding and non-invention rules

- Ground every concern in the supplied manuscript. Give each substantive concern a
  stable ID, a claim_pointer, and an evidence_pointer. Use "location not provided" or
  "not assessable from supplied material" when you cannot verify a location.
- Do NOT invent experiments, controls, citations, line numbers, figure details, or
  prior-work distinctions that are not in the input.
- Do NOT state the final editorial decision or assert that the manuscript belongs in
  Nature. You may comment that significance seems overstated or understated.
- Do NOT invent reviewer identities/specialties. Use only your assigned emphasis.
- Numeric self-consistency: check headline counts against the Methods, one metric at
  two precisions, and every superlative against its own table. These are cheap findings.
- Style: avoid em dashes, en dashes, and colons as routine prose punctuation. Use a
  new sentence, comma, semicolon, parentheses, or a short label followed by a new line.
  Keep hyphens in compound terms and stable IDs such as R1-M1. Preserve punctuation in
  formulas, identifiers, URLs, and source-faithful titles.

## Output language

Write the review in Chinese. Keep technical terms, model names, IDs, and axis labels
in English where natural.
