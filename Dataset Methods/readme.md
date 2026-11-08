# Penalty trajectory benchmark pipeline

Python 3.10+. All scripts in this folder are required. Install with:

```bash
python -m pip install -r requirements.txt
python verify.py
```

## Use your existing 79,893 accepted kicks (no resimulation)

```bash
python data.py --source kicks_358 --output kicks_split --seed 42
python train.py --dataset kicks_split --models models --components 16
python evaluate.py --dataset kicks_split --models models --output metrics.json
python predict.py --models models --method both --r1 1.50 0.45 0.44 --r2 2.96 0.86 0.74 --output prediction
```

That row count yields 63,914 training shots and 15,979 test shots. Train count is floor(0.8*N); remaining rows form test. The shuffled split is made once and stored, with seed 42 by default. Old fitted models must be retrained because they may contain test shots and lack split fingerprints.

For a quick benchmark before running the full test set:

```bash
python evaluate.py --dataset kicks_split --models models --limit 200 --output metrics_pilot.json
```

`--limit` chooses a reproducible random subset of the saved test partition. It never creates a new training split. Do not tune parameters on test scores: use a validation subset of training for tuning and keep test for the final benchmark.

## New generation

```bash
python generate.py --power 20 --workers 8 --output kicks_new
python train.py --dataset kicks_new --models models_new --components 16
python evaluate.py --dataset kicks_new --models models_new --output metrics_new.json
```

Generation retains the original RK45 equations, observations at 0.05 and 0.10 s, and terminal goal/ground events. It automatically writes `train/`, `test/`, and `split.json` after all candidate chunks complete. Top-level raw partitions are retained for checkpoint/resume compatibility; model code never loads them. This duplicates accepted Parquet storage. Resume with the same command. Use a fresh generation folder for this version; use `data.py` to migrate old data. An interrupted split (train/test folders without split.json) must be retried into a fresh split destination using data.py; trajectory regeneration is not needed.

Sampling defaults remain Team 358 polar 0–90 degrees from +z, azimuth 55–125 degrees from +x toward +y; speed sqrt(9.8*11)–129/3.6 m/s; spin x [-8,8], y [-5,5], z [0,15] rad/s. All bounds remain configurable. The spin bounds are the previously selected Team 480 modeling intervals, not separate human records. Negative z spin requires explicitly changing --spin-min.

## Responsibilities

- `generate.py`: simulate, filter accepted airborne goals, and persist 80/20 split.
- `data.py`: migrate existing data and validate partition fingerprints.
- `train.py`: reads **only train**, fits the joint 8D full-covariance GMM, saves `cgmm.npz`, saves the training reference pool and adaptive ESS setting in `bayes.npz`, and records provenance. CGMM means *conditional Gaussian mixture model*. Bayesian conditioning has no parametric fit.
- `predict.py`: loads saved models only, never fits or loads test data. Writes JSON and Matplotlib heatmaps, mean, KDE mode and oriented ellipse. Input defaults to problem coordinates; use `--frame paper` for Team 358 input and `--output-frame paper` for displayed x,z. Internal JSON fields explicitly remain in paper frame, with converted output coordinates and covariance supplied separately.
- `evaluate.py`: loads saved models and **only test**, verifies split fingerprints and disjoint IDs, produces benchmark JSON. No training happens here. A prediction failure aborts the run rather than silently dropping hard cases.
- `physics_predictors.py`, `models.py`: shared implementation used by prediction and evaluation.

Both predictors always condition on airborne goal entry because only those shots are retained. They cannot estimate the probability of missing the goal. The models reflect the simulation sampling distribution, not a measured population distribution of human kicks.

## Standard metrics

Primary point estimator: posterior mean. Let prediction error be e=(predicted horizontal - true horizontal, predicted z - true z), and let posterior covariance be C.

| Metric | Definition | Units |
|---|---|---|
| Horizontal/vertical MSE | mean(e_h²), mean(e_z²) over test shots | m² |
| MSE per coordinate | (MSE_h + MSE_z)/2 | m² |
| Radial MSE | mean(e_h² + e_z²) = MSE_h + MSE_z | m² |
| Horizontal/vertical RMSE | square root of each axis MSE | m |
| Radial RMSE | sqrt(radial MSE) | m |
| Mean radial error | mean(sqrt(e_h² + e_z²)) | m |
| Directional uncertainty | sigma_h=sqrt(C_hh), sigma_z=sqrt(C_zz) | m |
| Radial RMS uncertainty | sqrt(trace(C)) | m |
| 95% radial radius | posterior 95th percentile distance from posterior mean | m |
| Marginal 95% intervals | posterior 2.5%–97.5% quantiles for each axis | m |
| Marginal/radial coverage | fraction of true crossings inside corresponding intervals/circles | fraction |
| Joint 95% ellipse coverage | fraction of true crossings inside the nominal 95% covariance-oriented ellipse | fraction |
| Ellipse area | pi * major semiaxis * minor semiaxis | m² |

Uncertainty is summarized across test shots by mean and median; interval width by mean. Radial RMS uncertainty is NOT the standard deviation of the scalar radius, and does not include ground-truth error. It measures the model's predicted spread. Its 95% circle is reported separately. All benchmark horizontal coordinates are Team 358 x-right; reflecting to the problem's y-left leaves these error/coverage metrics unchanged.

Ellipse construction: orient using posterior covariance eigenvectors; scale using the posterior weighted 95th percentile of squared Mahalanobis distance. It is a moment-based summary, **not necessarily a highest-density region**, particularly for multimodal posteriors. Report availability, coverage when available, coverage over all queries (missing ellipses count as uncovered), and area. Coverage near 95% by itself does not imply a useful narrow prediction.

## Noise, adaptivity, and numerical details

Default measurement likelihood: independent Gaussian sigma=0.005/sqrt(3) m for six coordinates. Evaluation adds independent Gaussian noise with that sigma, identically for both methods. This matches the variance of centimetre rounding but is not the same likelihood as uniform rounding. `--clean` disables added noise while retaining likelihood sigma. `--sigma` controls both likelihood and injected noise. Seed controls reproducible noise/subset and posterior draws.

Bayesian conditioning **always adapts** its Gaussian observation bandwidth upward until the saved `--min-ess` target (default 30) is met, unless measurement-noise support already exceeds the target. Adaptive bandwidth is smoothing, not a claim about actual sensor noise. Benchmark records raw ESS, final ESS and effective bandwidth. `train.py --min-ess` configures and saves it; prediction has no fixed-bandwidth switch.

CGMM fits standardized joint observations and targets on training only. Measurement-noise covariance is added during analytical conditioning. The conditional Gaussian mixture is sampled and rejected outside the goal rectangle to preserve the existing goal-conditioned prediction behavior. Mean, covariance, quantiles, and ellipse therefore have Monte Carlo error. Default is 10,000 draws per query (`--draws` in predict/evaluate). For final numerical convergence checks, increase this. Predict JSON reports retained sample fraction; all predictions remain simulation-model dependent.

The primary benchmark skips KDE mode optimization for speed. `--include-mode` adds mode MSE metrics using the same 0.05 m KDE bandwidth as prediction plots. Plotting always includes the mode, mean, heatmap and uncertainty ellipse. Changing KDE bandwidth is not a change to posterior covariance.

## Validation performed

`verify.py` checks original force components, launch axes, signed spin, solver refinement, events, Parquet round trip, split disjointness, split repeatability, modified-partition rejection, analytic MSE values, and known covariance/uncertainty values. A separate end-to-end smoke run generated 2,048 candidates (166 goals), split them 132/34, trained both models, evaluated test queries, and rendered both heatmaps. These are software checks, not accuracy claims for the full dataset.
