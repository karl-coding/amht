# AMHT Project Instructions

## Goal
Implement and optimize Adaptive Memory Hybrid Transformer (AMHT)

## Architecture
- SSM = memory backbone
- Router = dynamic attention gating
- Attention = sparse (10%)
- Latent memory = compressed history

## Tasks

### Training
- Run training with config.yaml
- Optimize loss (main + router)

### Evaluation
- Run:
  - NIAH test
  - throughput benchmark
  - scaling test

## Constraints
- Avoid full attention
- Keep router ratio ~0.1
- Optimize memory usage

## Commands
- train: python train/train.py
- eval: python eval/benchmark.py
