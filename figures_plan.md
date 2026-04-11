# Figure Plan For The AMHT Paper

## Goal

Add the minimum figure set needed to turn `paper.tex` from a clean draft into a credible arXiv preprint.

The current paper is text-complete, but still missing the visuals that make the architecture and benchmark results easy to parse.

## Naming Note

If later figure legends, tables, or captions add the hybrid comparison model introduced in the V4 benchmark plan, label it as `Mamba-3-inspired hybrid baseline` or `Mamba-3-Inspired Hybrid`.

Do not label it as `Mamba-3 baseline` unless the implementation is upgraded to a strict paper-faithful reproduction.

## Figure 1: AMHT V2 Architecture

**Filename**

- `figures/amht_v2_architecture.pdf`

**Purpose**

- show the three-path hybrid structure clearly
- explain why AMHT V2 is hybrid in computation, not only in modules

**Must include**

- token sequence entering layer
- block partitioning
- block router scores and top-k selection
- SSM path on all blocks
- sparse attention path on routed blocks only
- latent memory read/write path
- residual merge / feed-forward

**Caption draft**

AMHT V2 uses three coordinated computation paths. An SSM backbone processes all blocks, routed sparse attention runs only on selected blocks, and latent memory provides compressed global communication through read/write updates. This converts AMHT from module-level hybridization into computation-level hybridization.

## Figure 2: NIAH Accuracy By Depth

**Filename**

- `figures/niah_by_depth.pdf`

**Purpose**

- visualize the retrieval gap between AMHT and Transformer

**Must include**

- x-axis: needle depth
- y-axis: accuracy
- separate curves or grouped bars for:
  - AMHT
  - Transformer
- error bars from seed variation

**Data to plot**

| Depth | AMHT mean | AMHT std | Transformer mean | Transformer std |
| --- | --- | --- | --- | --- |
| 0.05 | 0.8333 | 0.4082 | 0.6667 | 0.5774 |
| 0.15 | 0.6667 | 0.5164 | 0.3333 | 0.5774 |
| 0.30 | 0.8333 | 0.4082 | 0.6667 | 0.5774 |
| 0.50 | 1.0000 | 0.0000 | 0.3333 | 0.5774 |
| 0.70 | 0.6667 | 0.5164 | 0.3333 | 0.5774 |
| 0.85 | 0.8333 | 0.4082 | 0.6667 | 0.5774 |
| 0.95 | 0.8333 | 0.4082 | 0.6667 | 0.5774 |

**Caption draft**

Needle-in-a-Haystack retrieval accuracy by depth at 8K context. AMHT improves mean retrieval accuracy relative to the matched local-attention Transformer baseline and is especially stronger at several shallow and intermediate retrieval depths.

## Figure 3: Throughput Scaling

**Filename**

- `figures/throughput_scaling.pdf`

**Purpose**

- show that AMHT gains are not limited to one context length

**Must include**

- x-axis: sequence length
- y-axis: tokens per second
- AMHT vs Transformer with error bars

**Data to plot**

| Seq Len | AMHT mean | AMHT std | Transformer mean | Transformer std |
| --- | --- | --- | --- | --- |
| 1024 | 98300.5394 | 7567.1875 | 102736.7469 | 722.5284 |
| 2048 | 131019.1113 | 14728.2585 | 115244.4212 | 561.7383 |
| 4096 | 137312.5889 | 9475.0074 | 104229.2280 | 2769.8869 |
| 8192 | 136201.4159 | 7054.8371 | 103857.9270 | 1086.9657 |

**Caption draft**

Throughput scaling from 1K to 8K context. AMHT is competitive at 1K and substantially faster than the Transformer baseline at longer sequence lengths, consistent with its routed selective-computation design.

## Figure 4: Efficiency-Quality Frontier

**Filename**

- `figures/efficiency_quality_frontier.pdf`

**Purpose**

- make the Fast vs Accurate operating-point story obvious

**Must include**

- x-axis: throughput
- y-axis: mean NIAH accuracy
- points:
  - Transformer
  - AMHT-Fast
  - AMHT-Accurate

**Suggested values**

- Transformer: use the multi-seed aggregate from the baseline study
- AMHT-Fast: use the multi-seed fast aggregate
- AMHT-Accurate: use the multi-seed accurate aggregate

**Caption draft**

Efficiency-quality frontier for the evaluated 8K operating points. AMHT exposes two useful regimes: a faster configuration with higher throughput than the Transformer baseline, and a more accurate configuration that further improves retrieval while remaining throughput-competitive.

## Table Plan

The current paper already contains three tables in LaTeX form. These should remain in the main text unless figures fully replace them.

Recommended table set:

- Table 1: Main comparison
- Table 2: Retrieval by depth
- Table 3: Throughput scaling

If the figure set is added, Table 2 or Table 3 can move to the appendix.

## Packaging Recommendation

For arXiv, the minimum clean package should contain:

- `paper.tex`
- `references.bib`
- `figures/amht_v2_architecture.pdf`
- `figures/niah_by_depth.pdf`
- `figures/throughput_scaling.pdf`
- `figures/efficiency_quality_frontier.pdf`

## Next Practical Step

The highest-value next step is to generate placeholder figure files and update `paper.tex` so the figures are referenced, even if the first version uses simple chart exports rather than polished final artwork.
