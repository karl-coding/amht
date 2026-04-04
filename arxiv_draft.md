# Adaptive Memory Hybrid Transformer: Routed Hybrid Sequence Modeling for Long-Context Retrieval and Throughput

## Abstract

Long-context modeling remains constrained by the cost of attention over long sequences. Pure attention architectures provide strong token-level retrieval but scale poorly, while pure recurrent or state-space models improve efficiency at the cost of precise selective interaction. We present the Adaptive Memory Hybrid Transformer (AMHT), a hybrid long-context architecture that combines three computation paths: a state-space backbone for universal low-cost sequence processing, routed sparse attention for selective high-resolution interaction, and latent memory for compressed global communication. We further introduce AMHT V2, which replaces token-level routing with block-level routing and skips attention computation for non-routed blocks. This converts the model from a hybrid in module composition into a hybrid in computation pathways. On an 8K synthetic long-context retrieval benchmark, AMHT V2 improves both efficiency and retrieval quality relative to a matched local-attention Transformer baseline. Across three seeds, AMHT-Fast reaches 122,862 +- 5,774 tokens/s with mean Needle-in-a-Haystack accuracy 0.6190 +- 0.0825, while AMHT-Accurate reaches perfect mean retrieval accuracy at 121,642 +- 2,486 tokens/s. The matched Transformer baseline reaches 96,123 +- 4,098 tokens/s with mean accuracy 0.5238 +- 0.0825. These results suggest that routed hybrid sequence models can preserve retrieval quality while improving long-context throughput when the routing policy controls actual attention computation rather than merely modulating its outputs.

## 1. Introduction

Attention-based architectures remain the dominant paradigm for sequence modeling because they provide direct token-to-token interaction and strong retrieval behavior. Their main weakness is computational: long-context inference and training become expensive as attention is applied broadly across the sequence. In parallel, recurrent and state-space approaches offer favorable scaling but often struggle with the sharp selective interactions needed for retrieval-heavy workloads.

This tradeoff motivates hybrid architectures. A useful hybrid architecture should not merely concatenate multiple modules in a single layer. It should allocate different computations to different roles. In the long-context setting, this suggests a three-path decomposition:

- a low-cost universal path that processes all tokens
- a selective high-resolution path used only when needed
- a compressed global path for long-range communication

AMHT is designed around exactly this principle. The model combines a state-space model (SSM) backbone, sparse routed attention, and latent memory. The initial AMHT design established the functional combination of these ingredients, but still paid attention cost too broadly because attention was computed before routing sparsity was fully enforced. AMHT V2 addresses this limitation by routing at the block level and skipping attention computation for non-routed blocks.

The central claim of this work is that a routed hybrid architecture can improve long-context retrieval and throughput simultaneously when the routing mechanism controls actual computation rather than only post-hoc attention weighting.

The contributions of this draft are:

1. We formulate AMHT V2 as a three-path hybrid architecture with explicit division of labor between SSM dynamics, routed sparse attention, and latent compressed memory.
2. We implement block-level routing that limits sparse attention to selected blocks while keeping the SSM path active over the full sequence.
3. We evaluate AMHT against a matched local-attention Transformer baseline on an 8K synthetic retrieval benchmark and show consistent throughput and retrieval gains in the current setup.
4. We identify an explicit efficiency-quality frontier within AMHT, exposing useful fast and accurate operating points rather than a single fixed configuration.

## 2. Related Direction

This work is motivated by three broad families of long-context architectures.

First, Transformers and their efficient variants provide direct interaction between tokens, which is especially useful for retrieval tasks. Local and sparse attention methods reduce cost but still depend on attention as the primary mechanism for sequence interaction.

Second, state-space and recurrent architectures improve scaling by replacing broad attention with compressed sequential dynamics. These methods provide attractive throughput properties but can struggle on tasks that require exact selective recall.

Third, memory-augmented architectures add explicit latent or compressed memory paths to improve long-range communication. AMHT builds on this idea but integrates memory with both recurrent dynamics and selective attention.

The design target is not to replace all attention with recurrence, or vice versa. It is to assign each mechanism a coherent systems role:

- SSM for default low-cost processing
- sparse attention for exceptions that require precise interaction
- latent memory for global cross-block communication

## 3. Architecture

### 3.1 AMHT Overview

AMHT processes a sequence with three coupled paths:

- `SSM backbone`: runs on every block and provides continuous recurrent state propagation
- `Routed sparse attention`: runs only on selected blocks and provides high-resolution local interaction
- `Latent memory`: stores compressed global information and supports read/write communication across blocks

This architecture is intended to preserve retrieval behavior without paying universal attention cost across the full sequence.

### 3.2 From AMHT to AMHT V2

The original AMHT design was hybrid in components but not yet strongly hybrid in compute allocation. Attention was still computed broadly, and routing mostly modulated the resulting outputs.

AMHT V2 makes two structural changes:

1. routing is performed at the block level rather than the token level
2. non-routed blocks skip attention computation entirely

This changes the model from a soft mixture of modules into an architecture with explicit computational branching.

### 3.3 Block-Level Routing

The input sequence is partitioned into fixed-size contiguous blocks. For each block, the router computes a summary representation and outputs a routing score. A top-k policy, determined by the target router ratio, selects the routed blocks. Only these routed blocks enter the sparse attention path.

This design provides several advantages:

- lower routing overhead than token-level gating
- improved memory locality on GPU hardware
- clearer budget control
- easier complexity analysis

### 3.4 Selective Attention and Universal SSM

The SSM path remains active for all blocks. This gives the model a universal low-cost computation path that preserves order-dependent information even when attention is skipped.

The attention path becomes selective. Routed blocks attend to a limited local context while non-routed blocks remain on the SSM path only. The intended division of labor is:

- SSM handles continuity and default state propagation
- attention handles sparse exceptions requiring explicit interaction

### 3.5 Latent Memory

Latent memory acts as a compressed global channel. Each layer reads from latent state and writes an updated compressed summary back into that state. This supports cross-block information flow even when the majority of blocks bypass attention.

In this sense AMHT V2 is a three-path hybrid architecture rather than a two-path attention-recurrence interpolation.

## 4. Experimental Setup

### 4.1 Task

We evaluate on an 8K synthetic Needle-in-a-Haystack style retrieval benchmark with multiple needle depths:

- 0.05
- 0.15
- 0.30
- 0.50
- 0.70
- 0.85
- 0.95

The benchmark is configured to avoid immediate saturation and expose differences in both retrieval quality and throughput.

### 4.2 Models

We compare three effective operating points:

- `Transformer`: matched local-attention baseline
- `AMHT-Fast`: shorter optimization schedule, efficiency-oriented
- `AMHT-Accurate`: longer optimization schedule, stronger retrieval-oriented operating point

The AMHT-Fast and AMHT-Accurate configurations are produced from the same architecture family and define an explicit efficiency-quality frontier.

### 4.3 Protocol

All major comparisons are run at sequence length 8192. We report:

- tokens per second
- milliseconds per step
- mean Needle-in-a-Haystack accuracy
- accuracy by depth
- scaling throughput from 1K to 8K

Experiments are repeated across seeds 42, 43, and 44. Multi-seed execution is automated through the repository benchmark script.

## 5. Results

### 5.1 Main Comparison

Across AMHT V2 operating points and seeds, the comparison against the Transformer baseline is:

- `AMHT-Fast`
  - Throughput: `122862.4010 +- 5773.6282` tokens/s
  - Latency: `66.7724 +- 3.0694` ms/step
  - Mean NIAH accuracy: `0.6190 +- 0.0825`
- `AMHT-Accurate`
  - Throughput: `121642.1870 +- 2486.2791` tokens/s
  - Latency: `67.3639 +- 1.3837` ms/step
  - Mean NIAH accuracy: `1.0000 +- 0.0000`
- `Transformer`
  - Throughput: `96122.7907 +- 4098.2288` tokens/s
  - Latency: `85.3266 +- 3.5999` ms/step
  - Mean NIAH accuracy: `0.5238 +- 0.0825`

These results support the main empirical claim of this draft: the routed hybrid architecture exposes a useful efficiency-quality frontier while improving on the matched local-attention Transformer baseline in both throughput and retrieval quality at 8K.

### 5.2 Retrieval by Depth

AMHT-Accurate reaches perfect retrieval at every evaluated depth, while AMHT-Fast improves over the Transformer baseline most clearly at depth 0.50 and remains competitive elsewhere:

| Depth | AMHT-Fast mean+-std | AMHT-Accurate mean+-std | Transformer mean+-std |
| --- | --- | --- | --- |
| 0.05 | 0.6667 +- 0.5774 | 1.0000 +- 0.0000 | 0.6667 +- 0.5774 |
| 0.15 | 0.3333 +- 0.5774 | 1.0000 +- 0.0000 | 0.3333 +- 0.5774 |
| 0.3 | 0.6667 +- 0.5774 | 1.0000 +- 0.0000 | 0.6667 +- 0.5774 |
| 0.5 | 1.0000 +- 0.0000 | 1.0000 +- 0.0000 | 0.3333 +- 0.5774 |
| 0.7 | 0.3333 +- 0.5774 | 1.0000 +- 0.0000 | 0.3333 +- 0.5774 |
| 0.85 | 0.6667 +- 0.5774 | 1.0000 +- 0.0000 | 0.6667 +- 0.5774 |
| 0.95 | 0.6667 +- 0.5774 | 1.0000 +- 0.0000 | 0.6667 +- 0.5774 |

The strongest separation appears in the accurate operating point, which reaches perfect accuracy even at shallow and intermediate retrieval depths.

### 5.3 Scaling Throughput

AMHT-Fast is essentially tied with the Transformer at 1K and clearly faster at 2K, 4K, and 8K. AMHT-Accurate gives up some short-context throughput but retains a large advantage at longer sequence lengths:

| Seq Len | AMHT-Fast mean+-std | AMHT-Accurate mean+-std | Transformer mean+-std |
| --- | --- | --- | --- |
| 1024 | 102373.8495 +- 2612.6417 | 94227.2293 +- 9303.4593 | 102736.7469 +- 722.5284 |
| 2048 | 137688.3755 +- 5603.9558 | 124349.8471 +- 19428.3921 | 115244.4212 +- 561.7383 |
| 4096 | 144233.1649 +- 4243.4056 | 130392.0129 +- 7921.4776 | 104229.2280 +- 2769.8869 |
| 8192 | 141236.5166 +- 5952.1760 | 131166.3151 +- 3597.4572 | 103857.9270 +- 1086.9657 |

This pattern is important. The hybrid model does not uniformly dominate at the shortest context, but once sequence length increases beyond 1K, both AMHT operating points produce a substantial throughput advantage.

### 5.4 Efficiency-Quality Frontier

The AMHT-Fast and AMHT-Accurate configurations define two useful operating points:

- `AMHT-Fast`: optimized for higher throughput with lower optimization budget
- `AMHT-Accurate`: optimized for stronger retrieval quality while retaining a throughput advantage over the Transformer baseline

This is stronger than a single merged comparison because it shows that the same hybrid architecture can be tuned toward systems efficiency or retrieval quality depending on the target regime.

## 6. Discussion

The central lesson from these experiments is that hybrid sequence architectures become substantially more compelling when they are hybrid in actual computation rather than only in module composition.

The original AMHT formulation combined recurrence, sparse attention, and memory, but did not fully realize the intended efficiency benefit because attention cost was not sufficiently constrained. AMHT V2 changes that by:

- routing at the block level
- skipping attention for non-routed blocks
- preserving an always-on recurrent path
- using latent state for compressed global communication

The result is an architecture that is easier to justify both computationally and empirically.

These experiments do not show that AMHT dominates every long-context architecture. They show something narrower and more defensible: when compared to a matched local-attention Transformer on the present 8K retrieval benchmark, routed hybrid computation improves the quality-efficiency frontier.

## 7. Limitations

This draft has several important limitations.

First, the benchmark is synthetic and retrieval-oriented. It demonstrates long-context selective recall, but it is not yet a full natural-language long-context benchmark.

Second, the baseline set is still incomplete. The current comparison includes a local-attention Transformer but not yet a denser Transformer baseline, a sparse Transformer baseline, or a memory-centric SSM baseline such as Mamba-family models.

Third, although the architecture now provides clear throughput benefits, the routed sparse attention implementation is still relatively simple and likely not yet systems-optimal.

Fourth, the paper currently supports a strong preprint claim, but not yet the broadest possible claim about long-context generality.

## 8. Conclusion

We introduced AMHT V2, a routed hybrid long-context architecture that combines an SSM backbone, block-routed sparse attention, and latent compressed memory. By moving from module-level hybridization to computation-level hybridization, AMHT V2 makes routing control real attention cost rather than only output weighting.

On the current 8K synthetic retrieval benchmark, AMHT V2 outperforms a matched local-attention Transformer baseline through a useful efficiency-quality frontier: AMHT-Fast is the most efficient operating point, while AMHT-Accurate delivers perfect retrieval on the current benchmark while retaining a throughput advantage.

These results suggest that routed hybrid architectures are a promising direction for long-context modeling, especially when recurrence, selective attention, and compressed memory are given distinct computational roles.

## 9. Reproducibility Notes

The repository contains:

- explicit AMHT and Transformer configs
- multi-seed benchmarking scripts
- structured JSON and JSONL logging
- benchmark aggregation utilities

These components are sufficient to reproduce the core tables in this draft from committed code and explicit configuration files.
