# AMHT Architecture V4

## Goal

AMHT V4 upgrades the recurrent backbone first, then makes routing depend on stronger sequence state, while keeping the core AMHT thesis unchanged:

- low-cost recurrent processing on all tokens,
- sparse attention on a small routed subset,
- compressed latent memory for global communication.

The main V4 change relative to V3 is that the SSM path becomes a real recurrent state-space layer instead of a tokenwise surrogate.

## V4 Features Compared With V3

1. Grouped selective recurrent SSM backbone

- recurrent state update across sequence positions,
- data-dependent input and output projections,
- causal short-convolution front-end,
- optional complex-state mode for richer dynamics.

2. Router uses stronger signals

- block summaries from the main hidden stream,
- recurrent features from the SSM path,
- latent-memory summary,
- true top-k plus neighborhood expansion under a fixed routing budget.

3. Better router optimization path

- hard routing at inference remains intact,
- selected attention blocks use a straight-through score gate so task loss reaches router scores,
- explicit sparsity is still controlled near `router_ratio ~= 0.1`.

4. Stronger latent-memory structure

- shared latent state across the model,
- separate read/write projections per layer,
- residual latent-state evolution across layers.

5. Cleaner evaluation path

- dedicated V4 configs,
- scaling targets for `16K` and `32K`,
- retrieval plus state-tracking style follow-up tasks.

## No-Conflict Rollout Rules

To avoid conflicts with existing V3 results:

- keep the old SSM implementation available behind `ssm_impl: surrogate`,
- keep the old router neighborhood behavior available behind `router_expand_mode: bonus`,
- add V4 through new config flags and new config files,
- do not remove V3 code paths until V4 is benchmarked.

## Dependency Order

Checkpoint order is determined by data dependencies, not coding convenience.

1. Config scaffolding

- add new V4 flags with V3-safe defaults.

2. Latent-memory interface split

- required before per-layer read/write projections can be used cleanly.

3. Selective recurrent SSM

- required before the router can consume recurrent features.

4. Router upgrade

- depends on recurrent features from checkpoint 3,
- also depends on stable latent-memory interfaces from checkpoint 2.

5. V4 configs and evaluation expansion

- depends on checkpoints 2 through 4.

This ordering avoids one recurring conflict from V3 planning: improving the router before the recurrent and memory signals it should actually use.

## Acceptance Criteria

V4 is only considered structurally complete when all of the following are true:

- V3 configs still run unchanged,
- V4 configs instantiate the selective recurrent SSM path,
- routed attention still respects an explicit budget close to `0.1`,
- latent state evolves across layers with layer-specific I/O projections,
- benchmark scripts can run V4 without code-path forks outside config.
