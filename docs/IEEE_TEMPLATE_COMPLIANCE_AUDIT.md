# IEEE Transactions Template Compliance Audit

Date: 2026-07-28  
Scope: `paper/main.tex` against the user-supplied
`bare_jrnl_new_sample4.tex`, `New_IEEEtran_how-to.tex`, and bundled
`IEEEtran.cls`. This is a format and fail-closed-boundary audit; it is not a
claim of IEEE TVT acceptance or an audit of experimental evidence.

## Executive Finding

The internal-review manuscript now conforms to the structural requirements
demonstrated by the supplied IEEE Transactions journal template. It compiles
with the bundled local class on US Letter paper in journal mode, with the
abstract and index terms in the correct location, IEEE-style figure/table
caption ordering, and the IEEEtran bibliography style.

The public/submission build is intentionally fail-closed. Switching
`internalreview` to false is insufficient by itself: verified author metadata
and an eligible locked result export must also be supplied. This prevents an
anonymous author placeholder or dash-valued result macro from entering a
submission PDF.

## Source and Class Provenance

- Template reference directory:
  `tmp/ieee_transactions_template_reference`.
- Manuscript class:
  `paper/IEEEtran.cls`.
- SHA-256 of both the extracted reference class and manuscript-local class:
  `B0EB3567B81AEC7FE98144A3AD283EEAC2D31035BB19E0D9DCBA7DA190F18D9D`.
- The final `.fls` dependency record contains `INPUT IEEEtran.cls`, confirming
  that compilation resolved the manuscript-local class rather than the MiKTeX
  installation copy.
- The class reports IEEEtran V1.8b (2015/08/26).

The sample uses the option `lettersize`, but this bundled V1.8b class does not
declare that option and reports it as unused. The manuscript therefore uses
the class-supported equivalent:

```tex
\documentclass[letterpaper,journal]{IEEEtran}
```

The compile log confirms `8.5in x 11in (letter) paper`, journal two-column
output, and the default 10-point journal size.

## Itemized Audit

| Area | Status | Evidence or action |
|---|---|---|
| Title | Pass | Uses `\title{...}` without display math or manual line breaks. `AMC` is retained as a field-standard acronym; spelling it out is an editorial option, not a requirement in the supplied template. |
| Internal author placeholder | Pass for internal review | `Anonymous Author(s)` and its internal footnote occur only inside `\ifinternalreview`. |
| Submission authors | Fail-closed, pending human input | The public branch loads `authors_verified.tex`; the missing file deliberately prevents a submission build until names, affiliations, funding, membership, and disclosures are verified. |
| `\maketitle` | Pass | Occurs immediately after `\begin{document}` and before the abstract. |
| Abstract | Pass | Uses the IEEEtran `abstract` environment immediately after `\maketitle`; it contains no displayed equations. |
| Index terms | Pass | Uses `IEEEkeywords` immediately after the abstract. |
| Initial paragraph | Pass | Uses `\IEEEPARstart` as demonstrated by the template. |
| Figures | Pass | Figure content precedes `\caption`; every figure `\label` follows its caption. Double-column figures use `figure*`. |
| Tables | Pass after repair | Every table caption precedes the tabular material, every label follows the caption, and captions were converted to the title case requested by the supplied sample. |
| Equations | Pass | Standard `equation`, `align`, and `align*` environments are used; no `eqnarray` or `$$` display delimiters were found. |
| Citations | Pass | Uses `cite` and symbolic citation keys; citations are resolved in the final build. |
| Bibliography | Pass after repair | Uses `\bibliographystyle{IEEEtran}` and an external `.bib`; the non-template negative `\IEEEbibitemsep` override was removed so class defaults control reference spacing. |
| US Letter page | Pass | `letterpaper` is recognized and confirmed by the log; no unused global option remains. |
| Copyright/publication ID | Correctly omitted | The supplied how-to says a journal copyright line is not needed at submission; production adds it. |
| Running headers | Acceptable for submission | No fabricated volume, issue, date, or author running head was added. IEEE production/editorial metadata can be added when supplied. |
| Biographies | Not required here | Example biographies are illustrative, not mandatory content to copy into an anonymous internal draft. |
| Internal review material | Pass | The red evidence-lock box and Internal Submission Note are conditional and disappear from the public branch. |
| Public quantitative claims | Fail-closed | A public build requires `\EligibleLockedResults` from the generated result export. The present placeholder `results_auto.tex` cannot silently produce a submission PDF. |

## Repairs Applied

1. Made the author front matter conditional and required
   `authors_verified.tex` for the public build.
2. Added a public-build evidence gate requiring
   `\EligibleLockedResults`.
3. Converted all six table captions to IEEE template title case.
4. Removed the manual negative bibliography item spacing override.

No sample publication identifiers, dummy headers, acknowledgments,
biographies, unrelated packages, or example prose were copied.

## Verification

Command:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error `
  -file-line-error -outdir=build main.tex
```

Result:

- output: `paper/build/main.pdf`;
- pages: 7;
- local `IEEEtran.cls`: confirmed;
- unused global options: none;
- undefined citations/references: none;
- overfull boxes: none;
- fatal errors: none.

Only ordinary underfull-box diagnostics remain; visual inspection showed no
clipped or overlapping content.

## Human Submission Blockers

Before setting `internalreviewfalse`, the authors must:

1. create `paper/authors_verified.tex` containing the final IEEEtran
   `\author{...}` block and verified affiliations/disclosures;
2. regenerate `results_auto.tex` from an eligible frozen result bundle that
   defines `\EligibleLockedResults`;
3. remove or resolve every `pending` result cell through that generated export;
4. perform the final TVT-specific portal and current author-policy check.

The supplied generic IEEE Transactions template alone cannot establish current
TVT page-charge, disclosure, review-anonymity, or submission-portal rules.
