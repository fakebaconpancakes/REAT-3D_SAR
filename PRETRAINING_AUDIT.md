# Pretraining audit: findings and readiness

## Findings and severity

| Finding | Severity | Correction | Retraining required? |
|---|---|---|---|
| Short clips were zero-padded before velocity features were computed. The last real frame therefore saw a false transition into zero padding, while the temporal transformer could attend to padded frames. | Critical | Velocity is now zero unless both adjacent frames are real. The dataset returns a valid-frame mask and the transformer masks padded temporal keys. | **Yes.** Inputs and attention outputs for short clips change. Retrain every expert and regenerate XAI. |
| The two-stream tuner loaded the joints checkpoint while feeding bone features. Ensemble composition, tuning, final evaluation, and XAI also used separate hardcoded configurations. | Critical | The tuner loads the bones expert; validation tuning writes one manifest-bound `ensemble_config.json`. Final test evaluation and XAI load and verify that exact configuration. Prediction arrays are count-checked while all streams process the same ordered validation batches. | **Yes** for the corrected expert and ensemble runs. Test results must be generated after validation selection. |
| Training initialization and shuffled sample order were not recorded, best epoch/metrics were absent from metadata, and interrupted runs could not resume. | Recommended | Python, NumPy, PyTorch, and worker seeds are configurable. Metadata records seed, deterministic mode, software versions, protocol, and split manifest hash. `--resume` restores model, optimizer, scheduler, data-loader generator, and RNG state. | Use a new run for reproducible experiments. Resume does not change the model architecture. |
| The analytic complexity estimate omitted the classifier and treated temporal attention as dense despite the sliding-window mask. It did not report measured inference time. | Recommended | Classifier parameters/MACs and allowed windowed attention pairs are included. The profiler now times inference and reports CUDA peak allocation when run on a GPU. MAC estimates exclude bias, normalization, softmax, activations, and data movement; they differ from hardware measurements. | No. This affects reporting only. |
| Zero-height bodies were divided by a small epsilon. This remained finite for zeros but could amplify tiny numerical noise. | Recommended | Zero-height bodies use a unit denominator and remain zero. Non-finite raw tensors now fail with a sample-specific error. | No expected change for exact zero bodies; retraining is already required for the temporal-mask correction. |

## Reproducibility notes

Training defaults to seed 42. Use `--seed N` to select another seed. Add
`--deterministic` to request deterministic PyTorch algorithms; on CUDA this sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, enables cuDNN deterministic behavior, and
disables cuDNN benchmarking. Deterministic mode can be slower and can raise an
error when a used CUDA operation has no deterministic implementation. Exact
bitwise identity across different GPU models, CUDA/PyTorch builds, or platforms
is not guaranteed. Resume restores the main-process RNG streams and loader
generator; worker-local random states are not captured, though the current
dataset has no stochastic worker transforms.

Checkpoint selection remains the highest internal validation accuracy. The
best checkpoint's `metadata.json` records the best epoch and validation loss
and accuracy. `last_training_state.pth` is saved at each epoch boundary and is
only loaded when `--resume` is supplied. New experiments use
`run18_official_v2`; old runs remain under their original IDs.

## Pretraining verification checklist

- [ ] Build all required `*_official_v2` splits from complete original raw data.
- [ ] Review `split_summary.txt` and `split_manifest.csv`; confirm the expected
  official test counts and no train/validation group leakage.
- [ ] Convert every split and confirm each has non-empty `binary_pt` contents.
- [ ] Run `python -m unittest discover -s tests -v`.
- [ ] Start each expert with the intended protocol and `--seed 42`; use
  `--deterministic` when the installed CUDA stack supports it.
- [ ] Confirm each checkpoint metadata file has the expected manifest hash,
  seed, protocol, best epoch, and validation metrics.
- [ ] Run validation ensemble selection and preserve
  `results/<dataset>/<run_id>/ensemble_config.json`.
- [ ] Run final test evaluation and XAI only after ensemble selection; confirm
  both scripts load the same manifest-bound ensemble configuration.
- [ ] Keep test results out of all model, ensemble, and XAI development choices.
