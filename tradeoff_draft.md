# AMHT Tradeoff Draft

## Speed-Accuracy Tradeoff

The current AMHT experiments suggest that the architecture should not be summarized by a single operating point.

Instead, two practically useful configurations emerge:

### AMHT-Fast

This configuration uses the shorter 8K training schedule and the lighter default sparse-attention settings.

Observed behavior:

- stronger throughput
- lower latency
- weaker retrieval accuracy on the harder 8K benchmark

### AMHT-Accurate

This configuration uses:

- longer 500-step optimization
- lower learning rate
- stronger router regularization
- larger sparse-attention chunk size

Observed behavior:

- substantially better retrieval accuracy
- perfect accuracy on the current 8K tuned seed-42 run
- lower throughput than the faster AMHT preset

## Interpretation

This split is useful for the paper because it shows that the current AMHT implementation exposes an explicit efficiency-quality tradeoff rather than a single universally dominant point.

That is still a meaningful result:

- if the paper emphasizes retrieval quality, the accurate configuration is the correct reference
- if the paper emphasizes systems efficiency, the fast configuration is the correct reference

## Paper-Safe Claim

A careful paper-safe formulation is:

> AMHT supports multiple operating points along an efficiency-quality frontier. Under shorter optimization schedules it remains competitive in throughput, while longer tuned optimization substantially improves retrieval accuracy at the cost of reduced end-to-end speed.

## Recommended Use In The Paper

The essay should present both configurations in a single table:

- AMHT-Fast
- AMHT-Accurate
- Transformer baseline

This makes the paper stronger than presenting only a single AMHT run, because it demonstrates that the hybrid architecture can be tuned toward either efficiency or accuracy depending on the target operating regime.
