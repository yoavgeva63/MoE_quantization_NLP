# The paper

LaTeX source of the final report for the NLP course (2025b, Tel Aviv University), in ACL
format. The compiled report is [`main.pdf`](main.pdf).

## Contents

| File | Role |
|------|------|
| `main.tex` | The paper. This is the file to compile. |
| `custom.bib` | Bibliography, 22 entries. Its header comment records how each entry was verified. |
| `acl.sty`, `acl_natbib.bst` | Official ACL style and bibliography style, from <https://github.com/acl-org/acl-style-files>. |
| `figures/` | The eight figure PDFs (see below). |
| `main.pdf` | The compiled report. Tracked so the result is readable without a build. Other build products are ignored. |
| `wc_sections.py` | `python wc_sections.py` prints prose words per section and the size of every float. |
| `acl_latex_example.tex` | The upstream ACL example, kept as a reference for the citation macros. Not part of the paper; do not upload it. |

## Compiling

**Overleaf.** Upload `main.tex`, `custom.bib`, `acl.sty`, `acl_natbib.bst` and the
`figures/` directory, keeping the directory name. Compiler pdfLaTeX, main document
`main.tex`. Let Overleaf run the passes it needs; a single pass leaves `??` where BibTeX
and the cross-references have not settled. If Overleaf's ACL template ships newer style
files, prefer those and upload only `main.tex`, `custom.bib` and `figures/`.

**Locally.** Either of

```bash
cd paper && tectonic --keep-logs main.tex      # Tectonic (XeTeX-based), runs BibTeX itself
cd paper && latexmk -pdf main.tex              # TeX Live
```

writes `main.pdf` next to the source. The paper compiles with no overfull boxes and no
undefined references or citations; check `main.log` after edits.

## Figures

All figures are vector PDFs, as the course guidelines require. The two the paper includes
are drawn at the width they are printed (6.3 in for the text width, 3.03 in for a column),
so their text lands at 8 to 9.5 pt on the page rather than being scaled down.

| File | Pages | Contents |
|------|-------|----------|
| `headline.pdf` | 1 | **Figure 1.** Routing KL (log scale), top-1 flip rate and WikiText-2 perplexity (log scale) against expert precision, all policies, both models, with bootstrap intervals. |
| `collapse_qwen.pdf` | 6 | Expert collapse on Qwen: (1–3) starved experts by layer at INT8/INT4/INT3, (4–6) per-layer usage entropy. **Figure 2 is page 3**, included with `trim={0 0 0 21},clip` to drop the in-figure title. The expert-load concentration panel is absent because the PDF was regenerated with `--skip-reconstruction` (see below). |
| `collapse_olmoe.pdf` | 7 | Same for OLMoE, plus (7) expert-load concentration. Not included in the paper. |
| `comparison.pdf` | 4 | Cross-model detail: perplexity, top-*k* Jaccard distance, output KL over the gold top-*M* support, normalized expert-usage entropy. |
| `olmoe_figures.pdf`, `qwen_figures.pdf` | 9 | Per-model results: perplexity, routing KL, top-1 flip rate, Jaccard distance, usage entropy, output KL, and layer-wise routing drift at INT8, INT4, INT3. |
| `olmoe_attribution.pdf`, `qwen_attribution.pdf` | 4 | Attribution: mechanism bar chart across bit-widths, then the per-layer mechanism split at INT8, INT4, INT3. |

`\includegraphics` takes page 1 of a multi-page PDF unless told otherwise. To include
another panel pass `page=`, and check after compiling that the intended panel appeared:

```latex
\includegraphics[width=\columnwidth,page=3]{figures/collapse_qwen.pdf}
```

## Regenerating figures and tables

Every figure and every number in the paper comes from a committed artifact under
`results/`; the top-level README maps each table and figure to the script and file that
produce it. From the repository root, with the project environment active:

```bash
python scripts/compare_models.py --results-dir results/olmoe results/qwen --out-dir results/paper
python scripts/analyze.py          --results-dir results/olmoe     # and results/qwen
python scripts/attribute.py        --results-dir results/olmoe
python scripts/collapse.py         --results-dir results/qwen      # --skip-reconstruction without the .pt captures
python scripts/paired_bootstrap.py --results-dir results/olmoe
python scripts/verify_offline.py   --results-dir results/olmoe
```

`collapse.py` needs the captured router inputs (`artifacts.pt`, `router_inputs.pt`) for
its expert-load concentration panel; without them pass `--skip-reconstruction`, which
drops that one page. Copy the outputs the paper uses back into `figures/`:

```bash
cp results/paper/headline.pdf     paper/figures/headline.pdf
cp results/qwen/collapse.pdf      paper/figures/collapse_qwen.pdf
```

The numbers in `main.tex` are transcribed from `results/*/summary.md`,
`results/*/attribution.md` and `results/paper/comparison.md` and checked against the
source `metrics.json`. Nothing in the `.tex` updates itself, so regenerating results means
re-checking the tables.

## Bibliography

`custom.bib` is cited through `\citet` and `\citep` with the ACL natbib style. Every
arXiv identifier was resolved against the arXiv API and the author lists are the ones
arXiv returned. One entry cannot be machine-verified because OpenReview requires a
browser: Fang and Huang (2026), *Beyond Freezing the Router*, cited as TMLR from the
published PDF (<https://openreview.net/forum?id=bPsPPI65hf>). Their earlier, withdrawn
ICLR draft *Router Choice Matters* is the same line of work and is not cited separately.
Confirm the OpenReview page once before submitting.

## Related notes

Project notes that informed the paper live in `../others/`: `prior_art.md` (the prior-art
check behind Section 3), `method_audit.md` (the methodology audit behind Sections 4 to 6
and the Limitations), `STATUS.md` (project status as of 22 August 2026) and the course
guidelines PDF.
