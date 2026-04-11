# Stage 1 Training Focus

## Goal

Stage 1 is not paper work.

The goal is to run training and benchmark loops, compare AMHT against the two active baselines, and keep adjusting code and architecture until the direction is clearly favorable.

Current comparison set:

- `AMHT-V4-Fast`
- `Transformer`
- `Mamba-3-Inspired Hybrid`

## One Command

Use:

```bash
bash scripts/run_train_benchmark.sh
```

Default behavior:

- preset: `stage1_tuning`
- single-seed comparison for fast iteration
- train + throughput + NIAH + scaling
- summary written to `paper_runs/stage1_tuning/report/summary.md`
- adjustment note written to `paper_runs/stage1_tuning/report/adjustments.md`

## What To Optimize First

Use Mamba-3 as a parameter reference for performance improvement, not as a reproduction target.

The first search axes are:

| Priority | Knob | Why |
| --- | --- | --- |
| P0 | `ssm_state_size` | strongest direct lever for recurrent capacity |
| P0 | `ssm_complex` | closest current knob to richer Mamba-style state dynamics |
| P0 | `ssm_conv_kernel` | improves short-range mixing around the recurrent path |
| P1 | `router_straight_through_temperature` | affects whether task signal reaches routing cleanly |
| P1 | `router_neighbor_radius` / `router_neighbor_bonus` | controls sparse attention cost-quality tradeoff |
| P1 | `latent_tokens` | controls compressed memory capacity |
| P2 | `block_size` | affects routing granularity and runtime |

## Iteration Rule

Do not tune everything at once.

Use this order:

1. backbone first
2. rerun same baseline comparison
3. router and memory next
4. only then move to longer context or paper-freeze

## Decision Rule

Use the generated `adjustments.md` after every run.

Interpret it as:

- if AMHT loses on quality to the Mamba-3-inspired hybrid, strengthen the recurrent backbone first
- if AMHT loses on throughput without a quality gain, reduce routed-attention cost first
- if AMHT beats both baselines, keep architecture stable and scale context or task difficulty
