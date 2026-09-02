# LOSO Results, All Seven Architectures

Leave-one-subject-out cross-validation, 17 folds per row, dynamic dataset. Aggregated from per-fold JSON output produced by `dynamic_dataset/train.py`. R2 is reported as mean ± standard deviation across the 17 held-out subjects; RMSE and PCC are fold means.

## Single-joint, EMG only

Elbow flexion torque, 12-channel EMG input, no kinematics.

| Model | Family | R2 | RMSE (N m) | PCC |
|---|---|---:|---:|---:|
| iTransformer | attention over channels | 0.886 ± 0.049 | 2.23 | 0.963 |
| TCN to iTransformer | hybrid | 0.863 ± 0.122 | 2.31 | 0.966 |
| PatchTST | attention over time | 0.844 ± 0.090 | 2.55 | 0.963 |
| GRU | recurrent | 0.818 ± 0.214 | 2.54 | 0.964 |
| Mamba | selective state space | 0.813 ± 0.296 | 2.49 | 0.968 |
| TCN | causal convolutional | 0.808 ± 0.341 | 2.44 | 0.970 |
| LSTM | recurrent | 0.801 ± 0.221 | 2.64 | 0.963 |

Range across all seven: 0.085 R2. Between-subject standard deviation within a single model ranges from 0.049 (iTransformer) to 0.341 (TCN), wider than the spread between models.

## Single-joint, EMG plus kinematics

Same target and folds, joint angle and angular velocity appended to the EMG input. Only tested on the causal TCN and LSTM, the two ends of the R2 range above.

| Model | Input | R2 | RMSE (N m) | PCC |
|---|---|---:|---:|---:|
| TCN | EMG + kin | 0.943 ± 0.047 | 1.51 | 0.983 |
| LSTM | EMG + kin | 0.917 ± 0.099 | 1.75 | 0.981 |
| TCN | EMG only | 0.808 ± 0.341 | 2.44 | 0.970 |
| LSTM | EMG only | 0.801 ± 0.221 | 2.64 | 0.963 |

Fusion narrows the between-subject spread as well as raising the mean: TCN standard deviation falls from 0.341 to 0.047.

## Multi-joint, EMG only, elbow flexion target

Same seven architectures, two degrees of freedom (elbow and shoulder), still predicting elbow flexion torque.

| Model | R2 | RMSE (N m) | PCC |
|---|---:|---:|---:|
| iTransformer | 0.687 ± 0.116 | 2.22 | 0.879 |
| PatchTST | 0.680 ± 0.113 | 2.24 | 0.884 |
| TCN to iTransformer | 0.643 ± 0.189 | 2.32 | 0.880 |
| Mamba | 0.600 ± 0.243 | 2.42 | 0.873 |
| TCN | 0.595 ± 0.309 | 2.40 | 0.878 |
| GRU | 0.655 ± 0.153 | 2.31 | 0.882 |
| LSTM | 0.572 ± 0.391 | 2.46 | 0.871 |

Every architecture drops by roughly 0.15 to 0.20 R2 relative to its single-joint number. The drop is uniform across model families, consistent with a shared cause (shoulder-elbow co-contraction ambiguity in a 12-channel montage) rather than a limitation specific to one architecture.

## Multi-joint, EMG plus kinematics, elbow flexion target

Tested on LSTM and TCN only.

| Model | Input | R2 | RMSE (N m) | PCC |
|---|---|---:|---:|---:|
| TCN | EMG + kin | 0.690 ± 0.230 | 2.07 | 0.913 |
| LSTM | EMG + kin | 0.665 ± 0.236 | 2.17 | 0.902 |
| TCN | EMG only | 0.595 ± 0.309 | 2.40 | 0.878 |
| LSTM | EMG only | 0.572 ± 0.391 | 2.46 | 0.871 |

Fusion still helps under the multi-joint condition, though the gain is smaller than in the single-joint case, since kinematics disambiguate joint contribution less completely once two joints share the same channel set.

## Multi-joint, EMG only, shoulder elevation target

Second regression target, tested on four architectures.

| Model | R2 | RMSE (N m) | PCC |
|---|---:|---:|---:|
| GRU | 0.379 ± 0.585 | 5.25 | 0.804 |
| LSTM | 0.431 ± 0.384 | 5.22 | 0.798 |
| TCN | 0.142 ± 1.705 | 5.46 | 0.808 |
| Mamba | 0.287 ± 0.805 | 5.50 | 0.805 |

Median R2 across subjects is substantially higher than the mean for every model here; the wide standard deviations are driven by a small number of subjects with strongly negative fold-level R2, the same subjects showing the amplitude-mismatch signature described in the main analysis. Positive R2 on 14 to 16 of 17 subjects per model.

---

## Known discrepancy: TCN to iTransformer hybrid, single-joint

This table gives 0.863 ± 0.122 (SJ) and 0.643 ± 0.189 (MJ) for the hybrid model, taken from the aggregated fold JSONs. The written report gives 0.888 ± 0.044 and 0.625 for the same two conditions. Every other model in this table matches its corresponding report figure exactly.

The hybrid's array tasks failed once during the study with `ModuleNotFoundError: No module named 'train'`, traced to the cross-file import that has since been fixed (`TemporalBlock` now lives in `layers.py`, imported by both `train.py` and `models_ext.py`). Which of the two numbers reflects the run that completed after the fix, and which reflects a partial or stale run, has not been confirmed. Before citing either figure, locate the source JSON (`tcn_it_SJ_elbow_flexion_loso_emg.json` and its MJ counterpart) and check the fold count and completion timestamp against the fix.

---

## Source

Fold-level JSON per run, `{model}_{cond}_{target}_loso_{emg|kin}.json`, produced by `dynamic_dataset/train.py` and aggregated by a collection step. Full per-fold breakdown, including per-subject R2 and the affine-parameter calibration results, is in the accompanying report where the source data is confirmed.
