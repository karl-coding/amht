# AMHT Architecture V2

## Goal

Redesign AMHT so that it is not only a hybrid architecture in module composition, but also a hybrid architecture in computation pathways.

The main objective is to make the hybrid structure produce:

- clearer division of labor
- real compute savings
- better long-context scaling
- stronger paper-quality architectural justification

## Problem With The Current Version

The current AMHT combines:

- SSM
- sparse attention
- router
- latent memory

but the hybrid structure is still weak in one important sense:

- attention is still computed for all tokens
- the gate scales the output after attention is already paid for

That means the model is hybrid in components, but not yet strongly hybrid in compute allocation.

## Core V2 Idea

AMHT V2 should be organized around three roles:

### 1. SSM Backbone

SSM runs on all blocks and provides:

- continuous recurrent memory
- low-cost default state propagation
- a fallback path when attention is skipped

### 2. Routed Sparse Attention

Attention should run only on selected blocks.

The router should:

- score blocks, not individual tokens
- choose a limited budget of blocks
- send only selected blocks through attention

This makes attention a selective high-resolution path rather than a universal cost.

### 3. Latent Global Memory

Latent memory should become a true cross-block global channel.

Blocks should:

- write compressed summaries into latent slots
- read latent summaries from previous blocks

This lets the model preserve global information even when many blocks skip attention.

## New Computation Structure

At a high level, the sequence should be processed like this:

1. Partition sequence into fixed-size blocks.
2. Compute block-level router scores.
3. Select top-k or thresholded routed blocks.
4. Run SSM over every block.
5. Run sparse local attention only on routed blocks.
6. Let each block read from latent memory.
7. Let each block write an updated summary back to latent memory.
8. Merge outputs into the next layer.

## Why Block-Level Routing

Block-level routing is preferable to token-level routing because:

- lower routing overhead
- fewer irregular memory accesses
- better GPU efficiency
- simpler budget control
- easier complexity analysis in the paper

Token-level routing is more flexible, but too expensive and fragmented for the current system goals.

## Proposed Block Definition

A block is a contiguous chunk of tokens, for example:

- 64 tokens
- 128 tokens
- 256 tokens

The exact block size should be configurable.

The model should expose:

- `block_size`
- `block_router_ratio`
- `attention_window_blocks`

## Router V2

## Router Input

The router should operate on a block summary, not raw per-token states.

Examples of block summary:

- mean pooled token states
- first token state
- learned pooling projection

## Router Output

The router should output one score per block.

Two candidate routing modes:

### Soft Budget Mode

- continuous scores
- top-k selected for attention
- auxiliary loss keeps routed fraction near target

### Hard Budget Mode

- direct block selection
- possibly with straight-through estimation

For implementation stability, start with top-k block routing.

## Attention V2

Attention should be computed only for routed blocks.

Each routed block may attend to:

- itself
- nearby routed blocks
- optional local context neighborhood

This gives:

- real compute skipping
- cleaner sparse complexity
- easier comparison to local/sparse Transformer baselines

## SSM V2

SSM remains the universal path and should:

- process every block
- preserve order-dependent memory
- provide low-cost long-range state propagation

The design intent is:

- SSM handles continuity
- attention handles exceptions

This is the main hybrid principle of V2.

## Latent Memory V2

Latent memory should no longer be only a learned additive bias.

Instead, each layer should include:

- latent read
- latent write

### Latent Read

Each block reads a compressed global summary from latent slots.

### Latent Write

Each block writes a compressed update into latent slots.

This enables:

- long-range communication across distant blocks
- global summary persistence
- reduced dependence on dense long-range attention

## Layer Structure V2

Each AMHT V2 layer should approximately look like:

1. Normalize block/token states
2. Compute block summaries
3. Router scores blocks
4. SSM update on all blocks
5. Attention update on routed blocks only
6. Latent read path
7. Latent write path
8. Feed-forward update
9. Residual merge

## Complexity Story

The paper should be able to explain V2 in complexity terms.

### Current Version

- pays attention cost for all tokens
- router mostly modulates outputs

### V2

- SSM cost applies everywhere
- attention cost applies only to routed blocks
- latent memory cost grows with latent size, not sequence length

The intended result is:

- lower effective attention cost
- better scaling at longer contexts

## Hybrid Interpretation

V2 makes AMHT hybrid in a stronger sense:

- SSM is the default computation path
- attention is the exception path
- latent memory is the global path

This is a true division of labor.

In paper language:

> AMHT V2 is a three-path hybrid long-context architecture in which recurrent state-space dynamics provide universal low-cost sequence processing, routed sparse attention provides selective high-resolution interaction, and latent memory provides compressed global communication across blocks.

## Baseline Implications

Once V2 is implemented, comparisons become cleaner:

- Transformer baseline tests pure attention path
- Mamba-family baseline tests pure recurrent/memory path
- AMHT V2 tests whether the combination is better than either alone

## Implementation Stages

### Stage 1

Add block partitioning and block summaries.

### Stage 2

Replace token routing with block routing.

### Stage 3

Skip attention entirely for non-routed blocks.

### Stage 4

Add latent read/write path across blocks.

### Stage 5

Run AMHT V2 vs Transformer benchmark again.

### Stage 6

Only after V2 stabilizes, add Mamba-family baseline.

## Success Criteria

AMHT V2 is successful if it achieves at least one of the following clearly:

- better retrieval quality than Transformer at matched compute
- better throughput than Transformer at matched retrieval quality
- better efficiency-quality tradeoff frontier than either baseline alone

## Immediate Next Coding Goal

The next concrete implementation step is:

- add block partitioning and block-level routing in the router path

This should be done before adding more baselines or more paper text.
