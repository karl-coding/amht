# Results Draft

## AMHT vs Transformer at 8K Context

We compared AMHT against a matched local-attention Transformer baseline on a harder 8K synthetic retrieval benchmark across three random seeds.

### Main Comparison

AMHT achieved higher mean Needle-in-a-Haystack accuracy than the Transformer baseline:

- AMHT: `0.6190 +- 0.0825`
- Transformer: `0.5238 +- 0.0825`

This indicates that the hybrid memory-plus-sparse-routing design improves retrieval quality under the current long-context benchmark.

However, the efficiency results were mixed rather than uniformly favorable to AMHT:

- Throughput:
  - AMHT: `96984.7348 +- 7343.9002` tokens/s
  - Transformer: `101121.9142 +- 935.4014` tokens/s
- Milliseconds per step:
  - AMHT: `84.7868 +- 6.3429`
  - Transformer: `81.0157 +- 0.7480`

Therefore, the current implementation supports a quality advantage for AMHT, but not yet a clean 8K throughput advantage over the Transformer baseline.

### Per-Depth Retrieval Behavior

Per-depth analysis shows that most depths remain noisy under the current three-seed study, but one depth displays a clear separation:

- Depth `0.85`
  - AMHT: `1.0000 +- 0.0000`
  - Transformer: `0.3333 +- 0.5774`

Other depths are either tied in expectation or show high variance:

| Depth | AMHT mean+-std | Transformer mean+-std |
| --- | --- | --- |
| 0.05 | 0.6667 +- 0.5774 | 0.6667 +- 0.5774 |
| 0.15 | 0.3333 +- 0.5774 | 0.3333 +- 0.5774 |
| 0.3 | 0.6667 +- 0.5774 | 0.6667 +- 0.5774 |
| 0.5 | 0.6667 +- 0.5774 | 0.6667 +- 0.5774 |
| 0.7 | 0.3333 +- 0.5774 | 0.3333 +- 0.5774 |
| 0.85 | 1.0000 +- 0.0000 | 0.3333 +- 0.5774 |
| 0.95 | 0.6667 +- 0.5774 | 0.6667 +- 0.5774 |

These results suggest that AMHT’s main advantage may emerge most clearly at specific deep-context retrieval positions, although a larger number of seeds is still needed to reduce uncertainty.

### Scaling Behavior

AMHT showed stronger throughput than the Transformer baseline at short and medium context lengths, but this advantage narrowed and reversed at 8K:

| Seq Len | AMHT mean+-std | Transformer mean+-std |
| --- | --- | --- |
| 1024 | 118445.9784 +- 6921.2147 | 108008.2512 +- 8090.8488 |
| 2048 | 136105.0687 +- 8962.4779 | 115253.4598 +- 6155.7595 |
| 4096 | 122737.6711 +- 8299.7776 | 107570.8273 +- 6197.0073 |
| 8192 | 103933.7242 +- 5815.2476 | 106341.3171 +- 5327.9055 |

This pattern supports a nuanced interpretation:

- AMHT appears to scale favorably through the mid-range context regime.
- At the full 8K setting, the present implementation still carries enough overhead that the Transformer baseline becomes slightly faster on average.

### Summary Interpretation

Taken together, these experiments support the following limited but defensible conclusion:

> AMHT improves long-context retrieval quality relative to a matched local-attention Transformer baseline on the current 8K synthetic benchmark, but its implementation does not yet deliver a consistent end-to-end throughput advantage at the longest evaluated context.

This is a useful result for the project because it separates two questions that are often conflated:

1. whether the architecture improves retrieval behavior
2. whether the current implementation realizes the intended efficiency benefit

The present evidence supports the first claim more strongly than the second.

## Immediate Follow-Up

The next experiments should focus on:

1. reducing AMHT overhead at 8K
2. increasing the number of seeds
3. adding at least one additional baseline or benchmark

Only after those follow-up experiments should the essay make a stronger efficiency claim.
