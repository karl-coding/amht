# Small-Scale AMHT Code Tasks

## Goal

Turn the small AMHT baseline into a concrete implementation plan organized file by file.

## Global Constraints

- Do not use full-sequence dense attention.
- Keep router activity near `0.1`.
- Keep the project runnable on CPU.
- Support `8192` as the configured max context length.
- Use synthetic data first.

## File-by-File Tasks

### `model/ssm.py`

Purpose:

- Implement the memory backbone block.

Tasks:

- Add an `SSMBlock` module.
- Accept hidden states of shape `[batch, seq, dim]`.
- Project input into a smaller state space.
- Apply a gated update.
- Project back to model dimension.
- Keep the block lightweight enough for CPU smoke tests.

Done when:

- The block can be imported by `model/amht.py`.
- Forward pass preserves input shape.

### `model/router.py`

Purpose:

- Implement sparse routing and sparse attention.

Tasks:

- Add a router gate that outputs one routing score per token.
- Convert router scores to values in `[0, 1]`.
- Implement chunked local sparse attention.
- Limit attention context based on `router_ratio`.
- Avoid materializing full `seq x seq` attention for 8K length.

Done when:

- Router returns token-wise gating scores.
- Sparse attention output preserves input shape.
- The implementation avoids full attention over the whole sequence.

### `model/memory.py`

Purpose:

- Hold compressed latent memory.

Tasks:

- Add a latent memory module.
- Store a learned bank of latent tokens.
- Expand latent memory across batch.
- Return a compressed summary that can be added to token states.

Done when:

- `model/amht.py` can inject latent memory into the forward pass.

### `model/amht.py`

Purpose:

- Assemble the full AMHT model.

Tasks:

- Define `AMHTBlock`.
- Combine:
  - normalization
  - SSM backbone
  - sparse router attention
  - feed-forward block
- Define `AMHTModel`.
- Add token embeddings.
- Add position embeddings.
- Inject latent memory summary.
- Stack multiple AMHT blocks.
- Add LM head projection.
- Return both logits and router stats.
- Add a `compute_loss` helper.
- Include:
  - main cross-entropy loss
  - router regularization loss
  - router mean reporting
- Add a `synthetic_batch` helper for eval.

Done when:

- Model accepts token ids and returns logits.
- Loss helper returns:
  - total loss
  - main loss
  - router loss
  - router mean

### `train/config.yaml`

Purpose:

- Hold small-scale defaults.

Tasks:

- Set model size to a small runnable baseline.
- Keep `router_ratio` near `0.1`.
- Set `max_seq_len` to `8192`.
- Add training defaults:
  - batch size
  - learning rate
  - weight decay
  - grad clip
  - checkpoint dir
- Add evaluation defaults:
  - warmup steps
  - benchmark steps
  - scaling lengths
  - NIAH settings

Done when:

- Training and evaluation can run without extra config edits.

### `train/distributed.py`

Purpose:

- Reserve a place for future distributed support.

Tasks:

- Add a minimal `maybe_init_distributed()` stub.
- Return a harmless default for now.
- Do not add full distributed logic yet.

Done when:

- `train/train.py` can import and call it safely.

### `train/train.py`

Purpose:

- Run the small-scale training loop.

Tasks:

- Load YAML config.
- Set random seed.
- Select device automatically.
- Call the distributed stub.
- Build the AMHT model.
- Build optimizer.
- Build or read a synthetic dataset batch.
- Run forward and backward passes.
- Clip gradients.
- Step optimizer.
- Save checkpoint after training.
- Print:
  - step
  - total loss
  - main loss
  - router loss
  - router mean
  - tokens per second
- Accept CLI arguments:
  - `--config`
  - `--seq-len`
  - `--steps`
  - `--device`

Done when:

- One training step runs successfully at `8192`.
- A checkpoint file is written.

### `data/dataset.py`

Purpose:

- Provide a synthetic dataset for bring-up.

Tasks:

- Add a `SyntheticDataset`.
- Generate token sequences of length `seq_len`.
- Use vocab size from config-compatible inputs.
- Return `torch.long` tensors.

Done when:

- `train/train.py` can build batches from it.

### `data/tokenizer.py`

Purpose:

- Provide a simple placeholder tokenizer.

Tasks:

- Add a minimal tokenizer stub.
- Implement `encode`.
- Implement `decode`.
- Keep it simple and deterministic.

Done when:

- The file is importable and usable for future data pipeline work.

### `eval/niah.py`

Purpose:

- Provide needle-in-a-haystack evaluation.

Tasks:

- Add a batch builder for NIAH prompts.
- Insert a needle token at configurable depth.
- Place expected answer token in a known position.
- Run final-token prediction comparison.
- Report:
  - per-depth accuracy
  - mean accuracy

Done when:

- NIAH runs without crashing at configured sequence length.

### `eval/scaling.py`

Purpose:

- Provide sequence-length scaling evaluation.

Tasks:

- Run throughput evaluation repeatedly across configured sequence lengths.
- Aggregate results for:
  - `1024`
  - `2048`
  - `4096`
  - `8192`
- Return a single structured result object.

Done when:

- Scaling benchmark reports per-length throughput and latency.

### `eval/benchmark.py`

Purpose:

- Main evaluation entrypoint.

Tasks:

- Load config.
- Set seed.
- Select device.
- Build AMHT model.
- Implement throughput benchmark.
- Import and call NIAH benchmark.
- Import and call scaling benchmark.
- Support tasks:
  - `throughput`
  - `niah`
  - `scaling`
  - `all`
- Print machine-readable JSON output.

Done when:

- Throughput runs at `8192`.
- `--task all` returns throughput, NIAH, and scaling results.

### `scripts/run_train.sh`

Purpose:

- Provide a simple train launcher.

Tasks:

- Call `train/train.py`.
- Use `train/config.yaml`.
- Default to `8192` context.
- Forward extra CLI args.

Done when:

- The script runs the training entrypoint directly.

### `scripts/run_eval.sh`

Purpose:

- Provide a simple eval launcher.

Tasks:

- Call `eval/benchmark.py`.
- Use `train/config.yaml`.
- Default to `--task all`.
- Default to `8192` context.
- Forward extra CLI args.

Done when:

- The script runs the eval entrypoint directly.

### `README.md`

Purpose:

- Document the small-scale baseline.

Tasks:

- Document the repo layout.
- Document install command.
- Document train command.
- Document throughput command.
- Document NIAH command.
- Document scaling command.
- Document full eval command.

Done when:

- A new user can run the project from the README alone.

## Execution Order

1. Implement `model/ssm.py`, `model/router.py`, and `model/memory.py`.
2. Assemble `model/amht.py`.
3. Set defaults in `train/config.yaml`.
4. Add `data/dataset.py` and `data/tokenizer.py`.
5. Add `train/distributed.py`.
6. Finish `train/train.py`.
7. Implement `eval/niah.py` and `eval/scaling.py`.
8. Wire `eval/benchmark.py`.
9. Add `scripts/run_train.sh` and `scripts/run_eval.sh`.
10. Update `README.md`.

## Verification Tasks

- Run one training step at 8K.
- Run throughput benchmark at 8K.
- Run NIAH benchmark.
- Run scaling benchmark.
- Run `--task all`.
- Confirm router metrics are printed during training.
- Confirm outputs are JSON for eval.

## Expected Small-Scale Outcome

- The project is runnable end to end.
- The model is structurally AMHT-like.
- The pipeline is ready for later upgrades:
  - real dataset
  - better router calibration
  - stronger long-context retrieval
  - distributed training
