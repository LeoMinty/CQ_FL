# Experiment 5: persistent training-state storage

This primary experiment table uses physically packed 2-bit persistent weights for the two quantized-weight methods and FP32 persistent weights for the FP32/SignSGD baselines.

Transient gradients, activations, decoded work buffers, Python objects and non-trainable BatchNorm moving state are excluded.

## ravdess (standard)

Trainable scalar parameters: `6,372,680`

| Method | Weight MiB | Optimizer MiB | Total MiB | B/param |
|---|---:|---:|---:|---:|
| FedAvg (FP32) | 24.310 | 48.620 | 72.930 | 12.000000 |
| BitFL | 24.310 | 48.620 | 72.930 | 12.000000 |
| SignSGD | 24.310 | 0.000 | 24.310 | 4.000000 |
| 2-bit W + FP32 Adam | 1.514 | 48.620 | 50.134 | 8.249101 |
| CQ-FL | 1.514 | 6.830 | 8.344 | 1.372968 |
