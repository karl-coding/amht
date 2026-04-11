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

## Two Commands

For iterative work on Colab, use one command for train+benchmark and one for paper generation.

Training and benchmark only:

```bash
bash scripts/run_train_benchmark.sh
```

Paper bundle from existing runs:

```bash
bash scripts/run_paper.sh
```

These wrappers default to:

- `scripts/run_train_benchmark.sh` -> `--preset stage1_tuning --skip-figures`
- `scripts/run_paper.sh` -> `--preset paper_v4 --skip-train --skip-eval`

Current preset coverage:

- `stage1_tuning` runs `AMHT-V4-Fast`, `Transformer`, and `Mamba-3-Inspired Hybrid` with full comparison summary for adjustment work
- `colab_quick` runs `AMHT-V4-Fast`, `Transformer`, and `Mamba-3-Inspired Hybrid`
- `paper_v4` runs `AMHT-V4-Fast`, `AMHT-V4-Accurate`, `Transformer`, and `Mamba-3-Inspired Hybrid`

Naming note for paper artifacts:

- use `Mamba-3-inspired hybrid baseline` or `Mamba-3-Inspired Hybrid` in tables and figures
- do not present this baseline as an exact reproduction of the original Mamba-3 paper

You can override the preset, device, and output directory with environment variables:

```bash
AMHT_PRESET=stage1_tuning AMHT_DEVICE=cuda AMHT_OUTDIR=paper_runs/stage1_tuning bash scripts/run_train_benchmark.sh
AMHT_PRESET=paper_v4 AMHT_DEVICE=cuda AMHT_OUTDIR=paper_runs/paper_v4 bash scripts/run_paper.sh
```

The underlying Python entrypoint remains:

```bash
python3 scripts/run_colab_paper.py ...
```

Paper generation produces:

- per-seed checkpoints and raw train/eval outputs under `paper_runs/.../runs/`
- `summary.md` and `paper_tables.tex` under `paper_runs/.../report/`
- paper figures under `paper_runs/.../figures/`

Stage-one tuning additionally produces:

- `report/summary.json`
- `report/summary.md`
- `report/adjustments.md`

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

For new benchmarks on the current main branch, use `scripts/run_v3_multiseed.sh`. The legacy `scripts/run_v2_multiseed.sh` is intentionally frozen so new runs are not written into old V2 filenames.

Before submission, the highest-value improvements are:

1. add author/affiliation information
2. replace placeholder related-work citations with a fuller bibliography if needed
3. sync generated paper figures and tables into the final paper build directory
4. add one real-text benchmark or one more baseline family
5. keep the baseline naming precise so the paper does not imply an exact Mamba-3 reproduction

## Figure Roadmap

The planned figure set is documented in `figures_plan.md`.

Recommended first-pass figures:

1. `figures/amht_v2_architecture.pdf`
2. `figures/niah_by_depth.pdf`
3. `figures/throughput_scaling.pdf`
4. `figures/efficiency_quality_frontier.pdf`
