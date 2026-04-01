model:
  dim: 1024
  layers: 24

training:
  batch_size: 32
  lr: 2e-4
  steps: 300k

context:
  max_len: 32768

router:
  target_ratio: 0.1