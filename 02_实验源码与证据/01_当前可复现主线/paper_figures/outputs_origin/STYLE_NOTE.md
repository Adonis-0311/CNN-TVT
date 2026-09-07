# Origin-style figure set (draft — not wired into the manuscript)

These files are a **presentation-only** restyle of the main-manuscript figures.
They are written to `paper_figures/outputs_origin/` and **do not replace**
anything in `outputs/`, `outputs_v46/`, or `paper/figures/`. The manuscript
still points at its existing figures; switching over is a one-line change per
`\includegraphics` path, to be made only if you want this style.

Generator: `paper_figures/make_figs_origin.py`
Preview at true manuscript widths: `origin_style_preview.pdf`

## What is and is not touched

No model is trained, no frozen artifact is refitted, and no number changes.
Every mark is read from the same CSV/JSON analysis layer the current figures
use; the SHA-256 of each source file is recorded in `provenance_origin.json`.

**Fig. 1 (the simulation-component teacher) is deliberately not regenerated.**
The existing `paper/figures/physical_teacher_example.pdf` is kept as is. The
generator can restyle it with `--teacher-figure` if that is ever wanted, but
that flag is off by default.

## The house style

Applied identically to every panel, so the figures read as one set.

| Element | Rule |
|---|---|
| Typeface | Arial, one scale: 8 pt labels, 7.5 pt ticks, 7 pt legend |
| Frame | four-sided black box, 0.9 pt |
| Ticks | inward, mirrored on the top and right axes, minor ticks on |
| Grid | none |
| Legend | framed, 0.7 pt black rule, white fill |
| Marks | solid fills for the fitted/sealed class, open outlines for held-out and prospective classes |
| Panel key | bold `(a)`, `(b)` above the frame at the left |
| Export | PDF and SVG vector, plus 600 dpi PNG and LZW TIFF |

## Colour

Three categorical slots plus neutral black and gray. No hue is used for two
different meanings across figures.

| Slot | Hex | Carries |
|---|---|---|
| Blue | `#0072B2` | A5 / the compact spectral family / envelope-fitting cells / jammer-free bars |
| Vermilion | `#D55E00` | channel-side held regimes / hard-interference bars / the prospective Tier-2 sidecar |
| Green | `#009E73` | interference-side held regimes |
| Black / gray | `#000000`, `#5A5A5A` | reference geometry, I/Q comparators |

The signed gain map in Fig. 2 is the diverging pair built from the same two
poles, vermilion through white to blue, so a reader who has learned "blue
favours A5" in one figure reads the other the same way.

The three categorical slots were checked with the all-pairs colour-vision
criteria rather than by eye: worst-case CVD ΔE 11.0, worst-case
normal-vision ΔE 18.7, and every slot at or above 3:1 contrast against a white
page. Shape is always redundant with colour, so the set also survives
greyscale printing.

## Known layout note

The manuscript includes Fig. 3 at `0.75\textwidth` while Fig. 2 is at
`0.95\textwidth`. Both are generated at the same physical size, so Fig. 3 is
scaled down about 20% more and its type prints correspondingly smaller. If you
adopt this set, raising Fig. 3 to `0.95\textwidth` would make the type sizes
match across the two double-column figures.
