# AMHT Todo

## Model

### `model/ssm.py`

- [x] Add `SSMBlock`.
- [x] Accept input shape `[batch, seq, dim]`.
- [x] Project into a smaller state space.
- [x] Apply gated state update.
- [x] Project back to model dimension.
- [x] Keep the block lightweight for CPU smoke tests.
- [x] Verify forward pass preserves input shape.

### `model/router.py`

- [x] Add token-wise router gate.
- [x] Output routing scores in `[0, 1]`.
- [x] Implement chunked local sparse attention.
- [x] Limit attention context using `router_ratio`.
- [x] Avoid full `seq x seq` attention at 8K.
- [x] Verify sparse attention preserves input shape.

### `model/memory.py`

- [x] Add latent memory module.
- [x] Store learned latent tokens.
- [x] Expand latent memory across batch.
- [x] Return compressed latent summary.

### `model/amht.py`

- [x] Define `AMHTBlock`.
- [x] Combine normalization, SSM, sparse router attention, and feed-forward layers.
- [x] Define `AMHTModel`.
- [x] Add token embeddings.
- [x] Add position embeddings.
- [x] Inject latent memory summary.
- [x] Stack multiple AMHT blocks.
- [x] Add LM head projection.
- [x] Return logits and router stats.
- [x] Add `compute_loss`.
- [x] Include main loss and router loss.
- [x] Report router mean.
- [x] Add `synthetic_batch` helper.

## Training

### `train/config.yaml`

- [x] Set small-scale model hyperparameters.
- [x] Set `router_ratio` near `0.1`.
- [x] Set `max_seq_len` to `8192`.
- [x] Add training defaults.
- [x] Add evaluation defaults.
- [x] Add scaling lengths.
- [x] Add NIAH settings.

### `train/distributed.py`

- [x] Add `maybe_init_distributed()` stub.
- [x] Return safe default behavior.

### `train/train.py`

- [x] Load YAML config.
- [x] Set random seed.
- [x] Select device automatically.
- [x] Call distributed stub.
- [x] Build AMHT model.
- [x] Build optimizer.
- [x] Build synthetic batches or synthetic dataset batches.
- [x] Run forward pass.
- [x] Run backward pass.
- [x] Clip gradients.
- [x] Step optimizer.
- [x] Save checkpoint.
- [x] Print total loss.
- [x] Print main loss.
- [x] Print router loss.
- [x] Print router mean.
- [x] Print tokens per second.
- [x] Support `--config`.
- [x] Support `--seq-len`.
- [x] Support `--steps`.
- [x] Support `--device`.

## Data

### `data/dataset.py`

- [x] Add `SyntheticDataset`.
- [x] Generate fixed-length token sequences.
- [x] Use configurable vocab size.
- [x] Return `torch.long` tensors.

### `data/tokenizer.py`

- [x] Add minimal tokenizer stub.
- [x] Implement `encode`.
- [x] Implement `decode`.
- [x] Keep behavior deterministic.

## Eval

### `eval/niah.py`

- [x] Add NIAH batch builder.
- [x] Insert needle token at configurable depth.
- [x] Insert expected answer token.
- [x] Compare predicted final token to expected token.
- [x] Report per-depth accuracy.
- [x] Report mean accuracy.

### `eval/scaling.py`

- [x] Run throughput benchmark across configured sequence lengths.
- [x] Include `1024`.
- [x] Include `2048`.
- [x] Include `4096`.
- [x] Include `8192`.
- [x] Return structured scaling results.

### `eval/benchmark.py`

- [x] Load config.
- [x] Set seed.
- [x] Select device.
- [x] Build model.
- [x] Implement throughput benchmark.
- [x] Import NIAH benchmark.
- [x] Import scaling benchmark.
- [x] Support `throughput`.
- [x] Support `niah`.
- [x] Support `scaling`.
- [x] Support `all`.
- [x] Print JSON output.

## Scripts

### `scripts/run_train.sh`

- [x] Call `train/train.py`.
- [x] Use `train/config.yaml`.
- [x] Default to `8192` context.
- [x] Forward extra CLI args.

### `scripts/run_eval.sh`

- [x] Call `eval/benchmark.py`.
- [x] Use `train/config.yaml`.
- [x] Default to `--task all`.
- [x] Default to `8192` context.
- [x] Forward extra CLI args.

## Docs

### `README.md`

- [x] Document repo layout.
- [x] Document install command.
- [x] Document train command.
- [x] Document throughput command.
- [x] Document NIAH command.
- [x] Document scaling command.
- [x] Document full eval command.

## Verification

- [x] Run one training step at 8K.
- [x] Run throughput benchmark at 8K.
- [x] Run NIAH benchmark.
- [x] Run scaling benchmark.
- [x] Run `--task all`.
- [x] Confirm training logs router metrics.
- [x] Confirm eval output is JSON.

## Next

- [ ] Replace synthetic data with a real dataset pipeline.
- [ ] Tighten router regularization toward `0.1` for longer CPU/GPU runs.
- [ ] Improve harder-curriculum NIAH retrieval behavior beyond the current partial-depth saturation.
- [ ] Add checkpoint resume.
- [ ] Add proper distributed training.
