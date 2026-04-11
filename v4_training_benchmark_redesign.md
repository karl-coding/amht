# AMHT V4 Training And Benchmark Redesign

## Goal

Turn the current V4 feature list into a training and benchmark plan that can support:

- iterative bug-fix and architecture tuning during development,
- controlled ablations during model selection,
- paper-grade multi-seed final results for arXiv submission.

This document assumes the V4 feature target is:

1. real recurrent SSM backbone,
2. richer state update including optional complex-state mode,
3. optional stronger SSM variants,
4. router trained by task signal,
5. router uses recurrent and latent-memory state,
6. true top-k plus neighborhood expansion,
7. stronger per-layer latent memory,
8. broader train/eval coverage including `16K` and `32K`.

## Feature Review -> Training Implications

| V4 Feature | Why It Changes Training | Required Training Change | Required Benchmark Change |
| --- | --- | --- | --- |
| Real recurrent SSM backbone | Optimization is no longer mostly about sparse attention; stability of recurrent state matters | add shorter-context warmup and long-context scaling curriculum | add state-tracking and long-context scaling benchmarks |
| Richer / complex-valued state update | More expressive dynamics can improve recall but can also destabilize optimization | log gradient norms, loss spikes, and recurrent-state-sensitive failures | add ablation against real-only recurrent state |
| Stronger SSM variant | Backbone quality can improve without more attention | isolate backbone gains before router gains | compare against local Transformer and SSM-family baselines |
| Router trained by task signal | Router quality is now part of main-task optimization, not only budget regularization | log router stats per run and per depth | add router-quality metrics and routing ablations |
| Router uses recurrent + latent state | Router depends on backbone and memory quality | train memory and router jointly after backbone is stable | evaluate misses at shallow/intermediate retrieval depths |
| True top-k + neighborhood expansion | Compute budget is structured, not just soft-biased | tune expansion radius and keep total routed ratio near `0.1` | ablate expansion radius and bonus policy |
| Per-layer latent memory | Memory path becomes a real architectural contributor | train with tasks that force long-range compressed communication | ablate shared vs per-layer memory I/O |
| Broader `16K/32K` train/eval | Short aligned retrieval is no longer enough | add curriculum and held-out difficulty shift | report scaling, memory, and generalization tables |

## Training Redesign

### Principle

Do not train V4 as one monolithic retrieval-only run from the start.

The training system should separate:

1. architecture bring-up,
2. backbone stabilization,
3. router + memory specialization,
4. paper-grade final multi-seed runs.

### Training Stages

| Stage | Purpose | Context | Data / Task | Output |
| --- | --- | --- | --- | --- |
| Stage 0 | smoke / bug-fix | `512` to `2K` | synthetic retrieval only | verify forward, backward, checkpoint, benchmark |
| Stage 1 | backbone stabilization | `1K` to `8K` | state tracking + retrieval mix | stable recurrent backbone before heavy router tuning |
| Stage 2 | hybrid specialization | `4K` to `16K` | harder retrieval + held-out retrieval shift | tune router, memory, expansion radius |
| Stage 3 | scaling validation | `8K` to `32K` | retrieval + long-context LM / chunked document task | pick publishable V4 configs |
| Stage 4 | final paper runs | target context from paper | fixed configs, fixed seeds, frozen code | reportable tables and figures |

### Stage 0: Smoke / Bug-Fix

Use this stage for iteration. These results are not publishable.

- steps: minimal
- seeds: `42` only
- tasks: synthetic retrieval only
- goal: catch shape bugs, instability, checkpoint conflicts, benchmark errors

Exit criteria:

- train and eval complete without runtime conflicts,
- router ratio stays near target,
- no exploding loss,
- no repeated architecture changes needed for basic correctness.

### Stage 1: Backbone Stabilization

This stage exists because V4’s main novelty is the stronger recurrent backbone.

Operationally, the repo should treat this as the first serious comparison loop rather than as paper generation. A practical default is a single-seed `AMHT-V4-Fast` vs `Transformer` vs `Mamba-3-inspired hybrid baseline` run with full train + retrieval + scaling summary, followed by one explicit adjustment note for the next coding iteration.

Train on a mixture of:

- retrieval,
- state tracking,
- simple algorithmic sequence tasks that stress recurrence.

Recommended task mix:

- `50%` retrieval,
- `25%` state tracking / parity / modular carry,
- `25%` generic LM-style next-token synthetic or real-text chunk task.

Why:

- retrieval alone can over-credit sparse attention,
- state tracking reveals whether the SSM path is actually helping,
- LM-style next-token loss stabilizes tokenwise representations.

Use Mamba-3 as a reference for parameter search directions here, especially:

- stronger recurrent state size,
- richer or complex-valued recurrent dynamics,
- larger local mixing kernel before turning to more router complexity.

### Stage 2: Hybrid Specialization

Once the backbone is stable, train the hybrid behavior harder.

Increase:

- distractor count,
- key-space size,
- held-out retrieval depths,
- distribution mismatch between train and eval.

Tune here:

- `router_neighbor_radius`,
- `router_neighbor_bonus`,
- `router_straight_through_temperature`,
- latent token count,
- per-layer memory I/O vs shared memory.

### Stage 3: Scaling Validation

Do not claim `32K` support from `8K`-only training.

At minimum, the publishable path should include:

- train or continue-train at `16K`,
- benchmark at `8K`, `16K`, `32K`,
- at least one held-out task that is not identical to the retrieval training setup.

### Stage 4: Final Paper Runs

Freeze:

- code,
- configs,
- seeds,
- report script,
- figure generation script.

Then run multi-seed only after no further adjustments are expected.

## Training Objectives

### Main Loss

Use a mixed objective instead of retrieval-only when possible:

- retrieval final-token loss,
- optional next-token LM loss,
- optional state-tracking auxiliary loss.

### Router Loss

The router should no longer be optimized mainly by a score-mean penalty.

Keep:

- compute-budget regularization near target ratio.

Do not rely on:

- budget regularization as the main router training signal.

The primary router learning signal should come from task loss through the routing surrogate path.

### What To Log Every Run

- total loss,
- main loss,
- router loss,
- router mean,
- tokens per second,
- seed,
- config path,
- checkpoint path,
- sequence length,
- device.

For V4 final runs also log:

- routed block count,
- routed ratio by layer,
- memory token count,
- wall-clock train time,
- peak memory if available.

## Benchmark Redesign

### Benchmark Tiers

| Tier | Use | Publishable | Tasks |
| --- | --- | --- | --- |
| A | debug / iteration | no | throughput, one retrieval run |
| B | model selection | partly | retrieval by depth, scaling, router stats |
| C | paper main results | yes | multi-seed retrieval, scaling, long-context task, baseline comparison |

### Required Main Benchmarks

1. Controlled retrieval

- NIAH / passkey style,
- accuracy by depth,
- mean accuracy,
- held-out distribution shift.

2. Efficiency and scaling

- throughput vs sequence length,
- milliseconds per step,
- memory usage if available,
- routed ratio statistics.

3. State-sensitive benchmark

- parity / modular tracking / associative recall,
- this is needed because V4’s backbone claim is stronger than V3’s.

4. Long-context non-toy task

- long document retrieval,
- chunked QA,
- or long-context LM-style perplexity proxy.

### Baseline Redesign

For paper-grade evidence, compare against:

| Family | Required | Why |
| --- | --- | --- |
| Transformer baseline | yes | efficient-attention anchor with no recurrent backbone |
| Mamba-3-inspired hybrid baseline | yes | fixed hybrid baseline with stronger recurrence but no AMHT router or latent-memory path |
| Pure recurrence baseline | optional later | useful if compute allows, but not required for the simplified default paper path |

For AMHT V4, the clean comparison set is:

1. Transformer baseline,
2. Mamba-3-inspired hybrid baseline,
3. AMHT V4.

That simplified set is enough to test the main claim: whether AMHT's explicit routing plus latent memory beats both a pure attention baseline and a fixed hybrid baseline built on the stronger recurrent backbone.

For reporting, name the hybrid baseline as `Mamba-3-inspired hybrid baseline` rather than `Mamba-3 baseline`. The current repo implementation is a comparison-oriented inspired variant, not a strict reproduction of the original paper architecture.

### Required Ablations

| Ablation | Question Answered |
| --- | --- |
| V4 recurrent SSM vs surrogate SSM | does the real backbone matter |
| real-only vs complex-state SSM | does complex-state help enough to justify cost |
| router with vs without recurrent features | are backbone features helping routing |
| router with vs without latent-memory features | is latent memory useful for routing |
| top-k only vs top-k + expansion | does neighborhood expansion improve retrieval |
| shared memory I/O vs per-layer memory I/O | is the stronger memory path actually needed |
| router ratio `0.05`, `0.10`, `0.20` | quality-efficiency frontier |

## Publishable Result Gate

Results are ready for paper use only when all of the following are true:

1. No open training-instability bugs remain.
2. No further architecture adjustments are planned for the reported configs.
3. The same code and configs can be rerun from scratch.
4. Multi-seed results are available.
5. At least one non-toy benchmark beyond aligned retrieval is included.
6. At least one recurrence-first baseline family is included if feasible.
7. Figures and tables are generated from committed scripts, not hand-edited.

## Recommended Final V4 Report Table

| Model | Params | Seq Len | Train Steps | Mean Retrieval | Throughput | Latency | Routed Ratio | Peak Memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AMHT-V4-Fast | ... | ... | ... | ... | ... | ... | ... | ... |
| AMHT-V4-Accurate | ... | ... | ... | ... | ... | ... | ... | ... |
| Transformer | ... | ... | ... | ... | ... | ... | n/a | ... |
| Mamba-3-Inspired Hybrid | ... | ... | ... | ... | ... | ... | architecture-dependent | ... |

## Recommended Final Figure Set

1. architecture diagram
2. retrieval accuracy by depth
3. throughput scaling from `1K` to `32K`
4. training curves
5. efficiency-quality frontier
6. routed-ratio or layerwise routing profile

## Practical Use With Current Repo

Use the repo in two modes:

1. iteration mode

- short train+benchmark runs,
- single seed,
- no paper claim,
- explicit architecture adjustment after each comparison bundle.

2. report mode

- frozen configs,
- multi-seed,
- full report generation,
- paper tables and figures only after no more adjustments are expected.

That means the correct workflow is:

- iterate until architecture and bugs stabilize,
- freeze V4 configs,
- run full multi-seed benchmark bundle,
- publish only the frozen bundle results.
