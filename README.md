# EMG-to-Torque Estimation for Upper-Limb Exoskeleton Control

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 <AUTHOR>. All rights reserved.

Cross-subject deep learning for joint torque estimation from surface electromyography, developed as the sensing front end of an EMG-driven assist-as-needed controller for an upper-limb rehabilitation exoskeleton.

Seven neural architectures spanning four model families, evaluated on two public datasets with deliberately opposite structure, under three evaluation protocols of increasing rigor, with per-subject calibration, EMG-kinematics sensor fusion, and a second regression target at the shoulder.

---

## Headline results

Leave-one-subject-out cross-validation, 17 subjects, dynamic dataset, single-joint elbow flexion torque, EMG input only.

| Model | Family | LOSO R2 | RMSE (N m) | PCC |
|---|---|---:|---:|---:|
| iTransformer | attention over channels | 0.886 ± 0.049 | 2.23 | 0.963 |
| TCN to iTransformer | hybrid | 0.863 ± 0.122 | 2.31 | 0.966 |
| PatchTST | attention over time | 0.844 ± 0.090 | 2.55 | 0.963 |
| GRU | recurrent | 0.818 ± 0.214 | 2.54 | 0.964 |
| Mamba | selective state space | 0.813 ± 0.296 | 2.49 | 0.968 |
| TCN | causal convolutional | 0.808 ± 0.341 | 2.44 | 0.970 |
| LSTM | recurrent | 0.801 ± 0.221 | 2.64 | 0.963 |

Best deployable configuration, causal TCN with EMG plus joint kinematics: **R2 = 0.943, RMSE 1.51 N m**, strictly causal, no look-ahead.

Effect of per-subject calibration on the cross-subject gap:

| Condition | Cross-subject R2 | Between-subject SD |
|---|---:|---:|
| zero-shot, no calibration | 0.80 | 0.33 |
| affine, 2 parameters from 2 trials | 0.94 | 0.02 |
| full fine-tuning | 0.97 | 0.02 |

---

## What this project establishes

**1. Published-style sequence splits leak, and the leak costs roughly one full unit of R2.** Pooling overlapping sliding windows and splitting them at random puts windows from the same recording into both train and test. On the isometric dataset this yields R2 = 0.950 under the leaky split and R2 = -0.400 under a leakage-free recording split, with correlation falling from 0.975 to 0.197. A stratified split guaranteeing full task coverage does not recover it, which narrows the diagnosis to effort-level extrapolation specifically.

**2. The collapse is a property of the data, not of recurrent models.** A causal TCN pushed through the identical pipeline collapses on both leakage-free splits (R2 = -0.15 and -0.11). Discrete constant-effort recordings supply too few distinct torque values per subject for continuous regression, and any model degenerates into recognizing a finite set of conditions rather than estimating a continuous quantity.

**3. Architecture is not the bottleneck on data that does support the mapping.** On the dynamic dataset, seven architectures spanning recurrent, causal convolutional, time attention, channel attention, hybrid, and state-space families fall inside a 0.085 R2 band. The between-subject standard deviation within a single model ranges from 0.049 to 0.341, so which subject is held out moves the result more than which model is used. Three of the seven were published in 2023 to 2024 and represent current practice in multivariate time-series modeling, which makes the flat result informative rather than a sign of an under-explored search space.

Two independent arguments explain it. Physiologically, the EMG-to-torque relationship is dominated by the electromechanical delay, roughly 50 to 150 ms, which at 100 Hz is 5 to 15 samples, an order of magnitude shorter than the 200-sample window every model already covers. Information-theoretically, recovering a slowly varying amplitude from a noise-like modulated carrier is a power-estimation problem whose variance is bounded below by a quantity that scales with the inverse square root of the bandwidth-time product. Once a model's receptive field exceeds the torque correlation time, further accuracy is limited by the signal, not by network capacity.

**4. The residual cross-subject gap is an amplitude mismatch, not a representation failure.** On subjects where the population model produces negative R2, Pearson correlation stays around 0.97. High correlation paired with negative R2 is the specific signature of a gain and offset error rather than a failure to learn the shape of the relationship. This is an instance of a broader identifiability confound: per-subject gain and true torque amplitude enter the observation multiplicatively and cannot be separated from a single unlabeled recording.

**5. Two parameters close most of it.** A scale-and-offset affine correction, fitted on two trials of the held-out subject with test trials kept disjoint from calibration trials and normalization statistics drawn from the pretraining pool only, raises cross-subject R2 from 0.80 to 0.94 and collapses the between-subject standard deviation from 0.33 to 0.02. Full fine-tuning reaches 0.96 to 0.97 but needs per-user training. That two parameters are sufficient, and that more calibration data adds little beyond that, is the predicted consequence of the identifiability argument rather than an empirical accident: the worst-performing subject has a fitted scale near 0.5, meaning the population model over-predicts that subject by roughly a factor of two, and correcting that single gain term alone recovers R2 to about 0.95.

**6. The same calibration procedure fails on the isometric dataset**, plateauing near R2 = 0 regardless of calibration budget. Calibration needs continuous torque structure in the target signal to have something to adapt to; it does not help when the underlying regression problem itself is ill-posed for the data.

**7. The multi-joint condition costs about 0.20 R2 uniformly across all seven architectures**, consistent with shoulder-elbow co-contraction ambiguity in a twelve-channel montage under two degrees of freedom, rather than a limitation specific to any one model.

**8. Shoulder elevation torque is estimable as a second target**, with median cross-subject R2 of 0.59 to 0.67 and a positive result on 14 to 16 of 17 subjects, extending the system from one degree of freedom to two. The subjects where it fails are the same subjects showing the amplitude mismatch described above.

**9. EMG-kinematics fusion improves every configuration tested**, most sharply in the single-joint case (TCN 0.808 to 0.943), where joint angle determines muscle length and moment arm and therefore resolves a genuine physical ambiguity rather than adding redundant information. The required sensing, joint encoders and an IMU, is already present on the target hardware.

**10. Against the dataset's originating publication**, which evaluates only within-subject, these models match the reference single-joint correlation (0.96 to 0.97 versus 0.971) while generalizing across subjects, a strictly harder setting, and without requiring a force or torque sensor at calibration time. The within-subject recording-split result on the isometric dataset separately reproduces the most directly comparable published LSTM number (0.986 here versus 0.981 published), which validates the pipeline against the literature before any cross-subject claim is built on it.

---

## Datasets

Neither dataset is redistributed in this repository. Both are public.

| | Isometric | Dynamic |
|---|---|---|
| Role | reproduction target and methodological diagnostic | primary object of study |
| Participants | 12 healthy male | 17 healthy, 11 male |
| Regime | isometric, held constant | continuous dynamic tracking against viscous resistance |
| Conditions | 4 tasks: flexion, extension, pronation, supination | single-joint (elbow), multi-joint (elbow and shoulder) |
| Effort structure | 3 discrete levels: 10, 30, 50 percent MVC | continuous, spline trajectories |
| Recordings | 144 (12 x 4 x 3), 10 s each | 340 trials (17 x 2 x 10), about 30 s each |
| EMG | 5 muscles via three 2D HD arrays, monopolar, 2048 Hz | 12 bipolar SENIAM channels, released at 100 Hz |
| Conditioning | raw monopolar, all filtering done here | pre-filtered 20 to 450 Hz, rectified, 3 Hz envelope, MVC-normalized |
| Torque ground truth | two transducers at the brace, elbow axis | ATI force/torque sensor at the wrist interface, propagated by inverse dynamics through a subject-scaled model |
| Kinematics | none, limb fixed | joint angle and velocity from optical motion capture |
| Targets | elbow flexion/extension, pronation/supination | elbow flexion, shoulder elevation (multi-joint only) |
| Source | figshare | Zenodo, DOI 10.5281/zenodo.11209324 |

The decisive contrast is effort structure. The isometric dataset provides a small number of constant torque values per participant; the dynamic dataset varies torque continuously inside every trial. That difference alone determines whether a continuous EMG-to-torque mapping exists to be learned, and the two datasets produce opposite outcomes under identical leakage-free evaluation.

---

## Method

### Preprocessing, isometric arm

Per recording: load monopolar `.bin` files (biceps, triceps, three-region forearm), filter each channel zero-phase with a 50 Hz notch, a 5 Hz high-pass and a 500 Hz low-pass. Build five per-muscle representative signals by averaging channels inside each anatomical region, with forearm channel ranges read from the dataset's own metadata and verified to partition the full channel set before processing continues. Slide a 100-sample window at step 1, extract 14 time-domain features per window per muscle for a 70-dimensional vector per time step. Average the two torque transducers, smooth, and align the target to the last sample of its window so no look-ahead exists.

Two departures from the reference publication are deliberate. Five per-muscle signals rather than three grid-averaged signals, because the forearm grid spans muscles that are functionally distinct and partly antagonistic across the four tasks, and averaging them destroys the activation differences that distinguish one task from another. No frequency-domain features at this window length, since at 2048 Hz a 100-sample FFT has roughly 20 Hz resolution and stable spectral estimation needs 200 to 400 ms windows.

### Preprocessing, dynamic arm

EMG arrives already conditioned and co-sampled with kinematics and torque at 100 Hz. Twelve channels are read directly, optionally with joint angle and angular velocity appended in a joint-set-aware manner, and windowed into 200-sample sequences, stride 100 during training and 200 at evaluation so evaluation windows never overlap.

### Architectures

All seven share one harness with matched training settings: Adam, learning rate 5e-3 with step decay, MSE loss, gradient clipping, early stopping on validation loss, sequence-to-sequence output at every time step.

- **LSTM, GRU.** Single layer, 128 hidden units.
- **TCN.** Dilated causal convolutions, receptive field exceeding the input window, strictly causal by construction.
- **PatchTST.** Overlapping patches along time, self-attention over the patch sequence, channels processed independently through a shared encoder.
- **iTransformer.** Attention axis inverted so each EMG channel is a token whose embedding encodes that channel's entire window, modeling inter-muscle correlation directly. Structurally motivated here because EMG channels are co-activated by shared neural drive.
- **TCN to iTransformer hybrid.** Causal TCN front end supplies local temporal structure that channel attention alone cannot represent, feeding the channel-attention stage. Causality is preserved in the temporal path.
- **Mamba.** Selective state-space model with input-dependent transitions, causal by construction, with a fused CUDA kernel where available and a portable scan otherwise.

### Evaluation protocols

- **sequence.** All windows pooled and split at random. Leaks by construction; retained as an optimistic reference and as the mechanism under study.
- **recording.** Whole recordings or trials held out per subject; every subject still appears in training. Leakage-free at the trial level.
- **loso.** One test subject and one rotated validation subject held out entirely per fold. Leakage-free at the subject level and the realistic deployment condition.

A stratified variant on the isometric arm additionally isolates effort-level extrapolation by guaranteeing every subject's training set covers all four tasks.

Normalization statistics are computed from training data only throughout, and torque targets are always aligned to the last sample of their window.

### Metrics

R2, RMSE in newton-meters, and Pearson correlation. The separation between the R2 family and the correlation family is load-bearing: high correlation with low or negative R2 is the signature of an amplitude mismatch, which is what motivates the calibration analysis above.

---

## Repository layout

```
.
├── README.md
├── LICENSE
├── .gitignore
│
├── dynamic_dataset/          Quesada dataset, primary study arm
│   ├── layers.py             TemporalBlock, shared causal residual block
│   ├── models_ext.py         iTransformer, PatchTST, hybrid, Mamba
│   ├── train.py              unified harness: 7 models, 3 splits, 2 conditions, 2 targets
│   ├── calibrate.py          zero-shot vs affine vs fine-tune calibration
│   └── snr_sweep.py          noise injection on raw 2 kHz EMG, then full refilter
│
├── isometric_dataset/        Rojas-Martinez dataset, reproduction and diagnostic arm
│   ├── layers.py             same module, kept local so each folder is self-contained
│   ├── preprocess.py         5-muscle time-domain feature extraction
│   ├── preprocess_fd.py      dual-window TD + FD ablation at a spectrally valid window
│   ├── check_features.py     pre-flight validation gate, non-zero exit on failure
│   ├── train.py              LSTM, GRU, causal TCN, 3 split protocols
│   ├── snr_sweep.py          model-agnostic noise-robustness sweep
│   ├── plot_predictions.py   scatter, residual, and time-series overlay figures
│   └── collect_results.py    aggregation to CSV and comparison figures
│
└── results/
    └── loso_7models.md       collected LOSO comparison across all seven architectures
```

`layers.py` is intentionally duplicated in both folders rather than imported across a package boundary. The repository is two flat scripts-plus-shared-module directories, not an installable package, so each folder stays runnable on its own without a `sys.path` adjustment or an `__init__.py`.

---

## Environment

Python 3.11, PyTorch 2.6 with a CUDA 12.4 build, NumPy, SciPy, pandas. Mamba additionally requires `causal-conv1d` and `mamba-ssm` built against a matching CUDA toolkit; there is no Metal path, so on Apple Silicon it falls back to the portable scan in `models_ext.py`, which is correct but slow. All scripts read data and output paths from environment variables rather than hardcoding a machine-specific location.

---

## References

Datasets:

- Rojas-Martinez M. et al. (2020). High-density surface electromyography signals during isometric contractions of elbow muscles of healthy humans. *Scientific Data* 7:397. doi:10.1038/s41597-020-00717-6
- Quesada L. et al. (2024). A dataset for the investigation of upper limb torque prediction from EMG signals. *Zenodo*. doi:10.5281/zenodo.11209324

Reference methods reproduced or compared against:

- Shakeriaski F., Mohammadian M. (2025). Enhancing upper limb exoskeletons using sensor-based deep learning torque prediction and PID control. *Sensors* 25(11):3528.
- Quesada L. et al. (2026). EMG-to-torque models for exoskeleton assistance: a framework for the evaluation of in situ calibration. *IJRR*. doi:10.1177/02783649251414884
- Hoang D. et al. (2026). EMG-based torque prediction for assistive exoskeleton control using neural networks with bounded generalization error. HAL hal-05556797.
- Quesada L. et al. (2025). EMG feature extraction and muscle selection for continuous upper limb movement regression. *BSPC* 103:107323.

Architectures:

- Hochreiter S., Schmidhuber J. (1997). Long short-term memory. *Neural Computation* 9(8).
- Cho K. et al. (2014). Learning phrase representations using RNN encoder-decoder. *EMNLP*.
- Bai S., Kolter J.Z., Koltun V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv:1803.01271.
- Nie Y. et al. (2023). A time series is worth 64 words. *ICLR*.
- Liu Y. et al. (2024). iTransformer: inverted transformers are effective for time series forecasting. *ICLR*.
- Gu A., Dao T. (2023). Mamba: linear-time sequence modeling with selective state spaces. arXiv:2312.00752.

Method grounding:

- Cavanagh P.R., Komi P.V. (1979). Electromechanical delay in human skeletal muscle. *Eur. J. Appl. Physiol.* 42(3).
- Potvin J.R., Brown S.H.M. (2004). Less is more: high pass filtering to remove up to 99% of the surface EMG signal power improves EMG-based biceps brachii muscle force estimates. *JEK* 14(3).
- Staudenmann D. et al. (2006, 2007). Improving EMG-based muscle force estimation using a high-density EMG grid and PCA.
- Phinyomark A., Phukpattaranont P., Limsakul C. (2012). Feature reduction and selection for EMG signal classification. *ESWA*.
- Sarker P., Mirka G. (2019). The effect of sampling frequency on the analysis of EMG median frequency. *HFES*.
- Hajian G., Morin E., Etemad A. (2022). Multimodal estimation of endpoint force during quasi-dynamic and dynamic muscle contractions using deep learning. *IEEE TIM* 71:2513111.
- Hajian G. et al. (2024). Generalizing upper limb force modeling with transfer learning. *IEEE TNSRE* 32.
- Xiong D. et al. (2021). Deep learning for EMG-based human-machine interaction: a review. *IEEE/CAA JAS* 8(3).

---

## Status

Offline study complete across both datasets, all seven architectures, both conditions, both targets, EMG-only and fused inputs, and the full calibration comparison. All results are on healthy participants; the calibrated cross-subject accuracy reported here is a healthy-subject ceiling, and clinical populations with altered activation patterns require separate validation before the same figures can be claimed for a rehabilitation setting.

Two properties established here carry directly into on-device design. The identifiability analysis implies a brief known-amplitude calibration burst at the start of a session is the correct hedge against session-to-session gain drift, at negligible cost. The information-theoretic floor implies the estimator's window should be sized to the torque correlation time and no longer, since accuracy past that point is limited by the signal rather than the model.

---

## Acknowledgements

Compute provided by the RWTH Aachen University IT Center on the CLAIX-2023 cluster.
