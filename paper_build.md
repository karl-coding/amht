# Paper Build Notes

## Files

- `paper.tex`
- `references.bib`
- `figures_plan.md`

## Build

If `latexmk` is available:

```bash
latexmk -pdf paper.tex
```

If not, use:

```bash
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

## Generate Figures

The manuscript includes figures automatically if the expected PDF files exist under `figures/`.

Generate them with:

```bash
python3 scripts/plot_paper_figures.py \
  --glob-amht-fast "results/amht_v2_8k_fast_seed*_eval.json" \
  --glob-amht-accurate "results/amht_v2_8k_accurate_seed*_eval.json" \
  --glob-transformer "results/transformer_8k_seed*_eval.json" \
  --outdir figures
```

## Current Status

The manuscript is written as a clean arXiv-style preprint draft based on the currently verified AMHT V2 results.

Before submission, the highest-value improvements are:

1. add author/affiliation information
2. replace placeholder related-work citations with a fuller bibliography if needed
3. add architecture and benchmark figures
4. add one real-text benchmark or one more baseline family

## Figure Roadmap

The planned figure set is documented in `figures_plan.md`.

Recommended first-pass figures:

1. `figures/amht_v2_architecture.pdf`
2. `figures/niah_by_depth.pdf`
3. `figures/throughput_scaling.pdf`
4. `figures/efficiency_quality_frontier.pdf`
