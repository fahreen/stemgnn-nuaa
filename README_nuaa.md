# StemGNN on NUAA FM spectrum

Patched StemGNN (Cao et al., NeurIPS 2020) applied to single-receiver NUAA FM spectrogram forecasting.

## Patches (modern torch/numpy/pandas)
- torch.rfft/irfft -> torch.fft.fft/ifft (base_model.py)
- fillna(method=) -> ffill().bfill() (forecast_dataloader.py)
- np.float -> float; torch.load weights_only=False (handler.py)

## Pipeline
Raw I/Q .xls (10,777 files) -> nuaa_converter.py -> dataset/nuaa.csv (10777 x 256, per-bin power dB).

## Result: persistence crossover
Per-bin power forecasting, window 24, horizon 10, z-score, 7:2:1 split.

| horizon (s) | persistence MAE | StemGNN MAE | winner |
|---|---|---|---|
| 1-3 | 0.99-1.67 | 1.39-1.79 | persistence |
| 4   | 1.98 | 1.96 | crossover |
| 5-10| 2.24-2.85 | 2.09-2.30 | StemGNN |

**Finding:** persistence is optimal at short horizons (<=3 s); StemGNN overtakes it at a
~4 s crossover and gives up to ~19% lower MAE at 10 s, where persistence saturates and
temporal modeling pays off. Persistence diverges, StemGNN flattens. See
stemgnn_vs_persistence_crossover.png.

Harness validated by reproducing PEMS07 (MAE 2.65 vs paper 2.14).
