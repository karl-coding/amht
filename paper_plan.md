# AMHT Paper Plan

## Working Goal

Produce a paper-style essay, in the spirit of *Attention Is All You Need*, backed by reproducible and verified experimental data for AMHT (Adaptive Memory Hybrid Transformer).

The essay should not be framed as a pure implementation note. It should make a specific architectural claim and support it with controlled experiments, comparisons, and scaling evidence.

## Candidate Thesis

AMHT combines:

- SSM memory backbone
- sparse router-controlled attention
- latent compressed memory

to improve long-context efficiency while retaining strong retrieval behavior.

## Primary Research Question

Can AMHT preserve long-context retrieval quality while achieving better scaling characteristics than stronger attention baselines under comparable model budgets?

## Candidate Claims

Choose one primary claim and treat the rest as supporting claims.

### Claim A

AMHT achieves comparable or better long-context retrieval accuracy than the Transformer baseline and the Mamba-3-inspired hybrid baseline while maintaining lower effective attention cost.

### Claim B

AMHT scales more favorably in throughput as context length grows from 4K to 32K because most tokens avoid dense attention.

### Claim C

AMHT can maintain stable sparse routing near a target ratio of `0.1` without collapsing retrieval quality.

## Evidence Requirements

The paper should not rely on a single toy result. It needs at least three classes of evidence.

### 1. Controlled Synthetic Long-Context Retrieval

Purpose:

- verify that AMHT can solve retrieval tasks under known difficulty controls
- study where it fails as context, distractors, and query difficulty increase

Required:

- easy retrieval benchmark
- harder retrieval benchmark
- held-out retrieval distribution
- multiple depth settings
- multiple distractor counts
- multiple key-space sizes

### 2. Efficiency and Scaling

Purpose:

- show that AMHT’s architectural choices produce measurable performance advantages

Required:

- throughput vs context length
- memory usage vs context length
- scaling from short to long context
- router sparsity statistics across runs

### 3. Baseline Comparisons

Purpose:

- convert internal validation into paper-quality evidence

Required baseline families:

- Transformer baseline
- Mamba-3-inspired hybrid baseline
- optional pure recurrence baseline if feasible

## Current Verified Status

Already verified in this repo:

- checkpoint-loaded NIAH works correctly
- 8K GPU toy retrieval can reach perfect NIAH on aligned tasks
- harder 8K curriculum can reduce saturation and expose failure depths
- 4K CPU retrieval path is validated end to end

Not yet sufficient for paper claims:

- no true baseline comparisons
- no held-out distribution study
- no repeated runs by seed
- no real dataset benchmark
- no structured experiment logging
- no figure/table generation pipeline

## Required Metrics

Every major experiment should report:

- retrieval accuracy
- mean accuracy across depths
- accuracy by depth
- throughput tokens/sec
- milliseconds/step
- router mean
- router penalty
- sequence length
- number of distractor pairs
- key-space size
- device
- seed

For larger-scale experiments also report:

- peak memory if available
- checkpoint path
- wall-clock training time

## Minimum Figures and Tables

The paper should eventually contain at least:

### Figure 1

AMHT architecture diagram:

- token stream
- SSM memory path
- sparse router path
- latent memory path

### Figure 2

NIAH accuracy by depth for:

- AMHT
- Transformer baseline
- Mamba-3-inspired hybrid baseline

### Figure 3

Throughput vs context length:

- 1K
- 2K
- 4K
- 8K
- 16K
- 32K if feasible

### Figure 4

Router mean over training steps.

### Table 1

Model configuration comparison:

- parameters
- heads
- layers
- latent tokens
- router ratio

### Table 2

Main benchmark results:

- NIAH mean accuracy
- throughput
- memory usage

## Experimental Tiers

## Tier 1: Verified Toy Retrieval

Goal:

- ensure code path correctness
- validate checkpoint loading
- validate evaluation logic

Status:

- largely complete

## Tier 2: Harder Synthetic Retrieval

Goal:

- stress generalization under harder retrieval patterns

Requirements:

- larger key space
- more distractor pairs
- held-out depth combinations
- train/eval distribution mismatch
- multiple seeds

Status:

- partially complete

## Tier 3: Real Long-Context Benchmark

Goal:

- support a stronger paper claim beyond synthetic tasks

Possible directions:

- long document retrieval
- long-range question answering
- passkey retrieval with held-out formats
- chunked language modeling benchmark

Status:

- not started

## Baselines to Implement

### Baseline 1: Transformer Baseline

Purpose:

- establish the non-recurrent attention baseline used in the default paper runs

### Baseline 2: Mamba-3-Inspired Hybrid Baseline

Purpose:

- compare AMHT against a fixed-period hybrid that uses the stronger recurrent backbone without AMHT's router or latent-memory advantages
- label this baseline as Mamba-3-inspired rather than Mamba-3 to avoid implying a strict paper-faithful reproduction

## Reproducibility Requirements

Every result used in the essay should be reproducible from committed code and explicit configs.

Must add:

- checkpoint resume
- fixed config files per experiment
- structured JSON result logs
- seed control
- scriptable table generation
- experiment naming convention

## Writing Structure

The essay should eventually follow this shape:

1. Problem
2. Why dense attention struggles at long context
3. AMHT architectural idea
4. Mathematical or conceptual description of AMHT
5. Experimental setup
6. Main results
7. Scaling and efficiency analysis
8. Failure cases and limitations
9. Conclusion

## Immediate Implementation Priorities

### Priority 1

Add checkpoint resume and structured experiment logging.

Why:

- required for longer runs
- required for paper-grade reproducibility

### Priority 2

Add held-out harder synthetic benchmark.

Why:

- current synthetic task is still too easy to saturate

### Priority 3

Implement baseline Transformer variants.

Why:

- without baselines there is no paper-quality comparative claim

### Priority 4

Add experiment report generation.

Why:

- need figures, tables, and reusable summaries

## Immediate Next Step

The single best next coding task is:

- add checkpoint resume plus structured JSON metrics logging to `train/train.py` and `eval/benchmark.py`

This improves:

- reproducibility
- long-run practicality
- paper evidence quality

## Exit Criteria For A First Paper Draft

Do not draft the essay until all of the following are true:

- at least one baseline is implemented
- harder synthetic benchmark is stable
- results are repeatable across multiple seeds
- throughput and retrieval results are both logged
- figures/tables can be regenerated from saved outputs
- at least one non-toy long-context benchmark exists
