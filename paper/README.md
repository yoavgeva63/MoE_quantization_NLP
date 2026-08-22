# The paper

LaTeX source for the final report.

## Compiling

A TeX Live 2026 tree is now installed under the project account, so the paper can be
built in place. It is not on `PATH`; one export is enough:

```bash
export PATH="/home/morg/NLP_2526b/yoavgeva1/texlive/2026/bin/x86_64-linux:$PATH"
cd paper && latexmk -pdf main.tex     # ~4 seconds from clean
```

`pdfinfo` and `pdftotext` come with it, which is how the page split and the figure
contents below were checked. Build products (`main.pdf`, `*.aux`, `*.log`, ...) are not
meant to be committed.

**Overleaf still works** and is the fallback if the local tree goes away. The report has
to be in ACL format anyway, and the guidelines link the official template:
<https://www.overleaf.com/latex/templates/association-for-computational-linguistics-acl-conference/jvxskxpnznfj>

## What to upload to Overleaf

Upload these, preserving the `figures/` subdirectory:

```
main.tex             the paper (this is the file to compile)
custom.bib           the bibliography — READ ITS HEADER COMMENT
acl.sty              official ACL style file
acl_natbib.bst       official ACL bibliography style
figures/*.pdf        all eight figure PDFs
```

Do **not** upload:

- `acl_latex_example.tex` — the upstream ACL example document, kept here only as a
  reference for the citation macros (`\citet`, `\citep`, `\citealp`, `\citeposs`) and the
  author-block formatting options. It is not part of our paper and will not compile
  alongside `main.tex` as a second `\documentclass`.
- `README.md` — this file.

`anthology.bib` was **not** downloaded. It is tens of megabytes and every work we cite is
already in `custom.bib` with a verified identifier, so `\bibliography{custom}` is
sufficient. If you later want to cite something from the ACL Anthology by its Anthology
key, add the Anthology bib through Overleaf's own template rather than committing it to
this repo.

### Overleaf settings

- Compiler: **pdfLaTeX** (the ACL style files recommend it).
- Main document: `main.tex`.
- Run it twice, or let Overleaf handle it, so that BibTeX resolves the citations and the
  `\ref` cross-references settle. A single pass will leave `??` in place of table numbers.

The style files came from the official repository at commit-time HEAD of `master`:

```
https://raw.githubusercontent.com/acl-org/acl-style-files/master/acl.sty
https://raw.githubusercontent.com/acl-org/acl-style-files/master/acl_natbib.bst
```

If Overleaf's own ACL template is newer, prefer Overleaf's copies of `acl.sty` and
`acl_natbib.bst` and upload only `main.tex`, `custom.bib` and `figures/`.

## The skeleton is a skeleton

`main.tex` has the full section structure, all tables populated with the real numbers, and
the figures wired up. **The prose is not written.** Every gap is a `\todo{...}`, which
renders in red in the compiled PDF, and each section opens with a comment block stating
the claim that section has to land and which table or number supports it.

Before submitting:

```bash
grep -c 'todo{' main.tex     # must be 0
grep -c 'note{' main.tex     # must be 0
```

Then delete the `\todo` and `\note` definitions from the preamble, so a stray one becomes
a compile error rather than red text nobody noticed.

Two structural notes:

- `\usepackage[final]{acl}` produces the non-anonymous camera-ready layout, which is what
  the course wants (the guidelines ask for names, IDs and emails). Switch to `[review]` if
  you want anonymised, line-numbered output for internal review.
- The page limit is 8 pages excluding references and appendix. `Limitations` is an
  unnumbered starred section per ACL convention; the course guidelines do not exempt it,
  so budget it inside the 8 pages.
- **The page budget, measured.** With the `\todo` blocks rendering, the document is 16
  pages: body ~11.6, references ~1.0, appendix ~3.4. The notes are far longer than the
  prose they stand in for, so that number says nothing about the real budget. Suppress
  them and the whole document is **8 pages: body ~2.9, references ~0.4, appendix ~4.7**:

```bash
sed 's|^\\long\\def\\todo#1{.*$|\\long\\def\\todo#1{}|; s|^\\long\\def\\note#1{.*$|\\long\\def\\note#1{}|' \
    main.tex > _b.tex && latexmk -pdf _b.tex && pdfinfo _b.pdf | grep Pages
```

  So every table, figure, caption and factual sentence currently in the paper occupies
  under 3 body pages, leaving roughly 5 body pages for prose. Re-measure this way after
  large edits; do not read the budget off the noted build.

## Figures

All eight are PDFs, as the guidelines require (they explicitly rule out JPEG and PNG, and
ask for legible fonts). They were copied out of `results/`, which is gitignored for PDFs
via `results/**/*.pdf` — `paper/figures/` is not ignored, so these copies are tracked and
the Overleaf upload is self-contained.

| File | Pages | Contents |
|------|-------|----------|
| `headline.pdf` | 1 | **The page-1 figure.** Three panels: routing KL (log scale), top-1 flip rate, and WikiText-2 perplexity (log scale), all four policies, both models, with bootstrap error bars. The third panel exists to show the routing-versus-perplexity divergence, which the first two cannot. |
| `comparison.pdf` | 4 | Cross-model detail: (1) perplexity, (2) top-*k* Jaccard distance, (3) output KL over the gold top-*M* support, (4) normalized expert-usage entropy. |
| `collapse_olmoe.pdf` | 7 | Expert collapse on OLMoE: (1–3) starved experts by layer at INT8/INT4/INT3, (4–6) per-layer usage entropy at the same three, (7) expert-load concentration. |
| `collapse_qwen.pdf` | 7 | Same page order for Qwen. Page 3 is the INT3 collapse panel — the one cell of the sweep where collapse actually happens, and the one the paper includes. |
| `olmoe_figures.pdf` | 9 | Per-model Part 1: (1) perplexity, (2) routing KL, (3) top-1 flip rate, (4) Jaccard distance, (5) expert-usage entropy, (6) output KL, (7–9) layer-wise routing drift at INT8, INT4, INT3. |
| `qwen_figures.pdf` | 9 | Same panel order as `olmoe_figures.pdf`. |
| `olmoe_attribution.pdf` | 4 | Part 2: (1) mechanism bar chart across bit-widths, (2–4) per-layer mechanism split at INT8, INT4, INT3. |
| `qwen_attribution.pdf` | 4 | Same page order as `olmoe_attribution.pdf`. |

**Multi-page PDFs matter here.** `\includegraphics` silently takes page 1. To pull any
other panel, pass the page explicitly:

```latex
\includegraphics[width=\columnwidth,page=7]{olmoe_figures.pdf}
```

After the first compile, check that the panel you intended actually appeared. Getting this
wrong produces a plausible-looking figure with the wrong caption, which is the worst kind
of error.

### Regenerating the figures

The venv is not on `PATH`; call it by absolute path.

```bash
cd /home/morg/NLP_2526b/yoavgeva1/MoE_quantization_NLP

# Cross-model figure and table (headline.pdf, comparison.pdf, comparison.md)
/home/morg/NLP_2526b/yoavgeva1/venv/bin/python scripts/compare_models.py \
    --results-dir results/olmoe results/qwen

# Per-model Part 1 figures and summary.md
/home/morg/NLP_2526b/yoavgeva1/venv/bin/python scripts/analyze.py --results-dir results/olmoe
/home/morg/NLP_2526b/yoavgeva1/venv/bin/python scripts/analyze.py --results-dir results/qwen

# Part 2 attribution figures and attribution.md
/home/morg/NLP_2526b/yoavgeva1/venv/bin/python scripts/attribute.py --results-dir results/olmoe
/home/morg/NLP_2526b/yoavgeva1/venv/bin/python scripts/attribute.py --results-dir results/qwen

# Expert collapse: collapse.md, collapse.json, collapse.pdf
/home/morg/NLP_2526b/yoavgeva1/venv/bin/python scripts/collapse.py --results-dir results/olmoe
/home/morg/NLP_2526b/yoavgeva1/venv/bin/python scripts/collapse.py --results-dir results/qwen

# Paired-difference bootstrap (appendix table); no figures
/home/morg/NLP_2526b/yoavgeva1/venv/bin/python scripts/paired_bootstrap.py --results-dir results/olmoe
/home/morg/NLP_2526b/yoavgeva1/venv/bin/python scripts/paired_bootstrap.py --results-dir results/qwen

# Offline verification: router bit-identity, quantization levels, placebo FQNs. CPU only.
/home/morg/NLP_2526b/yoavgeva1/venv/bin/python scripts/verify_offline.py --results-dir results/olmoe
/home/morg/NLP_2526b/yoavgeva1/venv/bin/python scripts/verify_offline.py --results-dir results/qwen
```

These write into `results/`. Copy the ones the paper uses back into `figures/`:

```bash
cp results/paper/headline.pdf        paper/figures/headline.pdf
cp results/paper/comparison.pdf      paper/figures/comparison.pdf
cp results/olmoe/figures.pdf         paper/figures/olmoe_figures.pdf
cp results/qwen/figures.pdf          paper/figures/qwen_figures.pdf
cp results/olmoe/attribution.pdf     paper/figures/olmoe_attribution.pdf
cp results/qwen/attribution.pdf      paper/figures/qwen_attribution.pdf
cp results/olmoe/collapse.pdf        paper/figures/collapse_olmoe.pdf
cp results/qwen/collapse.pdf         paper/figures/collapse_qwen.pdf
```

**If you regenerate results, regenerate the tables too.** Every number in `main.tex` is
transcribed by hand from `results/*/summary.md`, `results/*/attribution.md` and
`results/paper/comparison.md`, and cross-checked against the source `metrics.json` and
`attribution.json`. Nothing in the `.tex` updates itself.

## Citation caveat — do not reintroduce this error

**Two distinct papers were conflated in our own project notes.** An earlier version of
`OVERVIEW.md` claimed that the proposal's "Anonymous (2026)" entry *was* the Fang & Huang
paper. It is not. They are separate submissions with different theses:

| Bib key | Paper | Status | Thesis |
|---------|-------|--------|--------|
| `routequant2026` | "Beyond Freezing the Router: Rank-Aligned Post-Training Quantization for Mixture-of-Experts Models" | **Anonymous**, under review | A full-precision router is *insufficient*: expert-output quantization error still shifts the next layer's router logits. Quantizes the router and corrects with Rank-Aware Jaccard Loss + Gap Hinge Loss. |
| `fang2026router` | "Router Choice Matters: Rank-Aware Post-Training Quantization for MoE Models" | **Yi-Zeng Fang and Juinn-Dar Huang**, ICLR 2026 submission, **withdrawn** | Most routing errors are near-neighbour rank flips around the top-*k* boundary. |

`routequant2026` is the proposal's "Anonymous (2026)" and **must stay anonymous** — do not
invent authors for it. `fang2026router` is the real "Fang & Huang" and the source of the
rank-flip finding. Both are cited separately in `main.tex`, and the distinction is recorded
as a comment at the top of `custom.bib` so it survives future edits.

Relatedly: the proposal cited VSRAQ as "Park et al. (2026)". That attribution is
**correct** — arXiv confirms the first author is Hancheol Park.

### Verification status of the bibliography

Every arXiv identifier in `custom.bib` was resolved against the live arXiv API and the
author lists are the ones arXiv returned. Three entries could not be machine-verified,
because OpenReview returns HTTP 403 to unauthenticated API clients and puts its public
forum pages behind a browser check:

- `routequant2026` — metadata from our verified notes. Open
  <https://openreview.net/forum?id=bPsPPI65hf> in a browser and confirm.
- `fang2026router` — metadata from our verified notes. Open
  <https://openreview.net/forum?id=kPgLp47bJf> in a browser and confirm authors and the
  withdrawal.
- `sramoe` ("Output-Aware Selective Router Alignment for MoE Quantization") — **no
  identifier resolved at all.** OpenReview search returned nothing. It is currently an
  authorless entry with no URL and a `TODO VERIFY` comment. Either find the forum and fill
  it in, or delete the entry and its `\citep{sramoe}` call. Do not ship it as-is.

One title changed upstream and the bib follows the current one: arXiv:2406.08155 circulated
as "Examining Post-Training Quantization for Mixture-of-Experts: A Benchmark" but v2 is
titled "QuantMoE-Bench: Examining Post-Training Quantization for Mixture-of-Experts".

## Known gaps in the results, reflected in the skeleton

Both gaps that this section used to list are now closed. Kept here in resolved form so
nobody reintroduces the old wording:

- **The INT8 placebo landed and passed.** All three bit-widths of the placebo are now in
  `results/` on both models, and the `running; not yet available` placeholder rows are
  gone from Table 1. OLMoE INT8 placebo is identical to `uniform` on every reported
  metric (9.3941 / 0.008535 / 18.01%); Qwen is 9.3461 against `uniform`'s 9.3465 with
  identical routing KL and flip rate.
- **The placebo's protected module is now named**, by
  `scripts/verify_offline.py`, which re-derives the selection from a meta-device skeleton
  because `metrics.json` records only `num_protected_modules`. It is one expert
  down-projection per model — `model.layers.14.mlp.experts.10.down_proj` on OLMoE
  (1.000000x of the router budget) and `model.layers.19.mlp.experts.45.down_proj` on Qwen
  (0.977778x) — and the selection is a single draw, not the greedy accumulation the code's
  loop suggests. See `results/*/verification.md`.

What replaced them is a sharper version of the same worry, and it is the live one:

- **The placebo is parameter-matched but not exposure-matched.** A router is read by every
  token in every layer; the protected expert projection is read by 11.43% of tokens in one
  layer on OLMoE and 5.04% in one layer on Qwen, a token × layer exposure gap of **140x**
  and **477x**. The `attention` control is what addresses this — same every-token,
  every-layer exposure as the routers, ~128x their parameter count — and the pair of
  controls is what the Limitations section now argues from. At MoE granularity no single
  control can be matched on count and exposure at once.
