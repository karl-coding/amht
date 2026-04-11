# AMHT V4 Checkpoints

## Checkpoint 0: Docs

- Files: `architecture_v4.md`, `v4_checkpoints.md`
- Depends on: none
- Exit criteria: feature scope, ordering, and non-conflict rules are written down

## Checkpoint 1: Config-Safe V4 Scaffolding

- Files: `model/amht.py`, `model/router.py`, `model/ssm.py`, `train/config_amht_v4_*.yaml`
- Depends on: checkpoint 0
- Exit criteria:
- V3 configs keep old behavior by default
- V4 behavior is enabled only by new config flags

## Checkpoint 2: Shared State, Per-Layer Memory I/O

- Files: `model/memory.py`, `model/amht.py`
- Depends on: checkpoint 1
- Exit criteria:
- latent state is initialized once per forward pass
- each AMHT block has its own read/write projections

## Checkpoint 3: Selective Recurrent SSM Backbone

- Files: `model/ssm.py`, `model/amht.py`
- Depends on: checkpoint 1
- Exit criteria:
- recurrent state is updated over sequence positions
- causal short-convolution preprocessing is available
- recurrent features can be returned to the router

## Checkpoint 4: Router Upgrade

- Files: `model/router.py`, `model/amht.py`
- Depends on: checkpoints 2 and 3
- Exit criteria:
- router consumes latent and recurrent context
- neighborhood expansion is budgeted, not only score-biased
- selected attention path has a straight-through training gate

## Checkpoint 5: V4 Configs

- Files: `train/config_amht_v4_8k_fast.yaml`, `train/config_amht_v4_8k_accurate.yaml`
- Depends on: checkpoints 2 through 4
- Exit criteria:
- one efficiency-oriented config
- one quality-oriented config

## Checkpoint 6: Smoke Validation

- Files: none required
- Depends on: checkpoints 2 through 5
- Exit criteria:
- model forward pass works for V3 and V4 configs
- a short train step and benchmark step complete without runtime conflicts

## Dependency Check

The dependency graph is conflict-free:

- checkpoint 2 does not require the new router,
- checkpoint 3 does not require the new router,
- checkpoint 4 is the first point where recurrent and memory upgrades meet,
- checkpoint 5 only consumes finished model interfaces,
- checkpoint 6 validates both old and new paths after integration.

This means the implementation can proceed in order without circular dependencies.
