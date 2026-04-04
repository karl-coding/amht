# AMHT Architecture V3

## Goal

AMHT V2 established the right direction: hybrid computation paths can outperform a matched local-attention Transformer on the current synthetic long-context retrieval benchmark. AMHT V3 should turn that directional result into a stronger architecture and a stronger paper claim.

The V3 target is not merely ``higher numbers''. It is:

- stronger systems efficiency from better routed-attention execution,
- stronger global memory behavior,
- better routing decisions,
- broader experimental support.

## What V2 Already Established

AMHT V2 introduced four important properties:

- block-level routing,
- attention skipped for non-routed blocks,
- an SSM path on all blocks,
- latent read/write support.

This was enough to show a meaningful result:

- on the current 8K synthetic retrieval benchmark, and more generally at sequence lengths of 2048 and above, AMHT V2 outperforms the matched local-attention Transformer baseline in both throughput and retrieval quality.

That claim should be stated carefully. AMHT is not uniformly faster at every tested sequence length. In the current aggregate results it is slightly slower at 1024, but faster at 2048, 4096, and 8192.

## What Still Limits V2

Despite the positive result, V2 still has structural limits:

- routed attention is implemented with Python-level block iteration, which limits GPU efficiency,
- latent memory is too weak and too global, relying on simple pooled updates rather than content-aware memory access,
- routing decisions are based on limited summaries and do not yet exploit richer memory state,
- evaluation is still dominated by synthetic retrieval rather than broader long-context workloads.

## V3 Optimization Axes

The V3 work should be understood as four optimization axes. These axes are not ordered by implementation priority.

### Axis A: Systems Optimization

This is the highest-return engineering path.

Target:

- replace Python-loop routed attention with batched routed attention

Concrete changes:

- pack routed blocks into contiguous tensors,
- compute Q/K/V in batch for all routed blocks,
- use block-index maps instead of per-block Python iteration,
- scatter outputs back in batch,
- precompute routing windows per block.

Expected effect:

- lower kernel-launch overhead,
- better GPU occupancy,
- stronger throughput scaling at longer sequence lengths.

### Axis B: Latent Memory Upgrade

This should come before any router upgrade that depends on memory state.

Current issue:

- latent memory is still too weak to serve as a real global workspace,
- current read/write behavior is closer to pooled summary injection than content-aware memory.

Target:

- upgrade latent memory into a content-addressable compressed global channel

Concrete changes:

- latent read via attention from block state into latent slots,
- latent write via gated updates into latent slots,
- separate read and write projections per layer,
- residual latent-state evolution across layers.

Expected effect:

- stronger long-range communication without increasing routed attention budget,
- better support for difficult retrieval positions,
- a much cleaner ``third path'' in the architecture story.

### Axis C: Router Upgrade

This axis depends on Axis B if the router is going to consume memory-aware signals.

Current issue:

- the router is still relatively shallow,
- block summary alone may be insufficient for robust importance prediction.

Target:

- make the router predict importance better while retaining explicit sparsity control

Concrete changes:

- use a small MLP router instead of a single linear scorer,
- feed router inputs with block summary plus optional latent read and local recurrent state,
- add routing persistence or neighborhood expansion so routed blocks can bring adjacent blocks with them,
- optionally move from pure top-k to top-k plus local expansion.

Expected effect:

- fewer routing misses,
- lower seed variance,
- better retrieval quality at shallow and intermediate depths.

Important dependency:

- if the router is extended with latent-read features, Axis B should be implemented first.

### Axis D: Evaluation Expansion

V3 should not rely on the same synthetic benchmark alone.

Target:

- broaden the claim beyond one synthetic retrieval setup

Concrete additions:

- held-out retrieval distributions,
- longer contexts such as 16K and 32K,
- passkey-style retrieval with formatting shift,
- one longer-form document retrieval or chunked QA benchmark,
- one stronger non-Transformer baseline family.

Expected effect:

- stronger external credibility,
- clearer evidence that the architecture generalizes beyond one toy regime.

## Why Add A Mamba-Family Baseline

The strongest missing baseline is not another minor Transformer variant. It is a memory-centric baseline.

Why Mamba-like models matter:

- AMHT claims value from combining recurrence or state-space dynamics with selective attention,
- a Mamba-family baseline isolates whether the hybrid design is better than a strong recurrence-first alternative,
- without that comparison, the paper mostly shows that AMHT is better than a local-attention Transformer, not that the hybrid is necessary.

This comparison is risky, and that should be stated explicitly.

Risk:

- because Mamba-family models are strong linear-time sequence models, AMHT may underperform them on some tasks,
- if that happens, the paper claim must narrow rather than overstate general superiority.

Why the risk is still worth taking:

- if AMHT wins, the paper becomes much stronger,
- if AMHT loses on some settings but wins on retrieval-specific settings, the paper can still make a more precise claim about when hybridization helps.

## Recommended Implementation Order

The implementation order should follow dependency and leverage, not the axis labels above.

1. Batched routed attention

- this is the highest-leverage systems change,
- it strengthens the speed story without changing the scientific claim too much.

2. Latent memory upgrade

- router improvements should not depend on weak memory features,
- the global memory path should become real before the router is asked to use it.

3. Router upgrade

- once latent memory is stronger, the router can safely use richer inputs.

4. Evaluation expansion

- after the architecture stabilizes, add 16K and 32K runs and at least one held-out retrieval setting.

5. Mamba-family baseline

- add this once the V3 architecture is stable enough that a negative result would still be interpretable.

## V3 Claim Scope

The paper claim should evolve carefully.

### V2 Claim

AMHT V2 improves throughput and retrieval quality relative to a matched local-attention Transformer baseline on the current synthetic long-context retrieval benchmark, especially at sequence lengths of 2048 and above.

### V3 Claim Target

A routed hybrid sequence model with explicit division of labor between recurrent dynamics, selective attention, and compressed memory improves the quality-efficiency frontier at longer context lengths and remains competitive beyond a single synthetic benchmark.

## A Better V3 Architecture Target

The clean V3 target is:

- SSM backbone on every block,
- content-aware block router,
- batched routed sparse attention,
- attention-based latent read/write,
- local neighborhood expansion around routed blocks.

That version would support a much stronger architectural story than V2 while remaining consistent with the hybrid-computation thesis already validated by the current experiments.
