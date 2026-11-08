# Analytical penalty predictors

Four entry scripts, with shared support files (keep this folder together):

| Script | Purpose |
|---|---|
| evaluate_linear.py | Fixed test-partition benchmark, first-order drag linearization |
| evaluate_taylor.py | Fixed test-partition benchmark, physics-constrained cubic Taylor trajectory |
| predict_linear.py | One-shot prediction, JSON and Matplotlib heatmap |
| predict_taylor.py | One-shot prediction, JSON and Matplotlib heatmap |

No model training is required. Each query solves an inverse launch problem using only its two observed positions, the physical constants, and the launch prior. Test launch parameters and target crossings are never used during inference. Targets are used only to score predictions. The shared data loader enforces the saved test fingerprint. Keep the original dataset-based pipeline separate; this package does not modify it.

## Installation and commands (Windows PowerShell compatible)

```powershell
python -m pip install -r requirements.txt
python verify_analytic.py

# Quick comparison on the same reproducible 200-shot subset as the existing evaluator.
python evaluate_linear.py --dataset ../kicks_split --limit 200 --workers 4 --output metrics_linear_pilot.json
python evaluate_taylor.py --dataset ../kicks_split --limit 200 --workers 4 --output metrics_taylor_pilot.json

# Full test partition: omit --limit.
python evaluate_linear.py --dataset ../kicks_split --workers 8 --output metrics_linear.json
python evaluate_taylor.py --dataset ../kicks_split --workers 8 --output metrics_taylor.json

# Original problem observations are the defaults; no dataset is needed here.
python predict_linear.py --samples 16384 --output prediction_linear
python predict_taylor.py --samples 16384 --output prediction_taylor

# Explicit input, if desired:
python predict_linear.py --r1 1.50 0.45 0.44 --r2 2.96 0.86 0.74 --samples 16384
```

`--dataset` is the root containing `split.json`, `train/`, and `test/`. It may be called `kicks`, `kicks_split`, or anything else; use your actual path. `--frame paper` switches input to Team 358 coordinates; `--output-frame paper` switches displayed output to X,Z. Default inputs and displayed output use the problem's forward/left/up frame. Internal summary JSON remains explicitly labeled Team 358 X-right,Z-up and includes separately transformed output covariance.

## Physical assumptions and approximations

Both methods use m=.430 kg, radius=.108 m, density=1.27 kg/m³, drag coefficient=.4, g=9.8 m/s², constant spin, initial center (0,0,.11), and Team 358's velocity-cross-spin Magnus sign. Launch parameters are uniform independently in speed [sqrt(9.8*11),129/3.6], polar [0,90] degrees from +Z, azimuth [55,125] degrees from +X toward +Y, and spins [-8,8],[-5,5],[0,15] rad/s. These are the final pipeline defaults, hard-coded in LOW/HIGH in analytic_core.py. If your generation used different bounds, update them before evaluation. The scripts do not infer custom bounds from split.json.

Writing f(v)=-kD*|v|*v+B*v-g_eff*eZ, with B*v=kM*(v cross omega):

- Linear: linearize the full drag vector at candidate v0. A=-kD*(|v0|*I+v0*v0.T/|v0|)+B and b=kD*|v0|*v0-g_eff*eZ. The trajectory solves v'=A*v+b analytically through a matrix exponential, using an eigendecomposition when well-conditioned and a direct matrix-exponential fallback otherwise. It does not freeze drag as a scalar damping coefficient.
- Taylor: a0=f(v0), j0=A*a0, and r(t)=r0+v0*t+a0*t²/2+j0*t³/6. The jerk is constrained by the force law, not freely fitted to two frames.

The models agree through the cubic short-time position expansion. They are approximations of the original nonlinear dynamics; no RK solver is used in prediction. Taylor crossing/contact times use polynomial roots. Linear crossings use sign-change brackets with a default .02 s grid and Brent root refinement. Very closely spaced crossings within a grid cell can in principle be missed; check convergence with --event-step .01 or .005. No analytical event routine can make the approximate dynamics exact.

The first forward Y=11 crossing must occur after .1 s and by 15 s, before descending contact at Z=.108. Its center must lie strictly inside |X|<3.66, 0<Z<2.44. Selection is based on approximate dynamics, which may disagree with the original generator near acceptance boundaries. Whole-ball mode is not implemented here.

## Inverse inference and uncertainty

Each method performs multiple bounded least-squares fits in the six-dimensional unit launch box (default 4 starts). Fits initialize a defensive importance proposal: 90% is an equal mixture of inflated local Gaussians based on fit Jacobians, and 10% is the original uniform prior. Out-of-bound Gaussian draws are discarded, without clipping or resampling them. Each retained launch receives weight proportional to:

    exp(-0.5 * ||predicted_observations - measurements||² / sigma²)
    / proposal_density(launch)

Goal-invalid trajectories receive zero weight. The complete mixture density, including its uniform component, appears in the denominator. Thus local targeting is corrected by importance weights; it is not a new launch prior or an adaptive observation likelihood. The observation sigma stays fixed at .005/sqrt(3) m. Posterior samples propagate parameter ambiguity, correlated motion, and uncertain goal time into crossing uncertainty.

The local Jacobian eigenvalue floor and covariance inflation only control the proposal. Reported uncertainty is the covariance of weighted crossing samples, not a clipped inverse-Hessian covariance. Multiple modes and weakly constrained directions can nevertheless be poorly sampled with finite budgets. The default budget is 4096 proposals per query. Increase --samples and --starts, and compare independent --seed runs before treating intervals as converged. ESS below 100 produces a warning; 100 is an operational diagnostic, not a proof of convergence. A high ESS alone cannot rule out a missed posterior mode.

These fits and posterior sampling run separately for each test observation, so full evaluation is substantially slower than a saved CGMM prediction. Start with --limit 20 or 200. --workers parallelizes independent queries. Only aggregate JSON is written by default. Optional --per-shot records are flushed for auditing; automatic resume is not implemented; use a new output name when rerunning.

No artificial approximation-error covariance or training-set calibration is added. Uncertainties reflect measurement noise and launch ambiguity under each approximate model. They exclude drag-linearization/Taylor truncation error and uncertainty in physical constants. Poor test coverage can therefore be physically meaningful rather than a sampling defect; inspect ESS as well.

## Benchmark equivalence and output

The input loader, noise generation, subset selection, and metric formulas match the previous final dataset pipeline. The same seed (42 by default) and subset limit produce the same selected test rows and noisy observations. --clean disables injected noise but keeps the likelihood sigma. Primary prediction is the posterior mean; --include-mode adds the more expensive 0.05 m KDE mode to evaluation. Prediction plots always include it.

Output metrics include horizontal/vertical MSE and RMSE, MSE per coordinate, radial MSE/RMSE, mean radial error, mean/median directional and radial RMS uncertainties, 95% marginal interval widths and coverage, radial 95% coverage, and ellipse coverage, availability and area. The ellipse uses the posterior 95th percentile of Mahalanobis distance, matching the previous implementation. Sampling diagnostics include ESS, proposal counts, observation residual chi-square, and optimizer success counts.

The JSON places aggregates in `metrics_on_successful_queries`. It explicitly reports requested/successful/failed counts and `coverage_all_requested`, where missing predictions count as uncovered. Do not compare successful-only MSE blindly if failure rates differ. Optional `--per-shot` JSONL includes IDs and reports. Otherwise full reports are discarded immediately after aggregation. At most twice the worker count is queued. Five scalar sequences are retained for exact medians, roughly a few MB for 16,000 shots. Test arrays and per-worker inference memory remain. Low-ESS predictions remain in scores and are flagged, rather than being silently excluded. If a method fails often, inspect failures and rerun with a larger budget before interpreting accuracy.

The optional KDE mode is approximate; all plotted covariance/interval summaries use the weighted posterior without adding KDE smoothing variance. Heatmap density and posterior moments therefore answer slightly different questions.

## Validation

verify_analytic.py compares the matrix-exponential trajectory to high-accuracy numerical integration of the affine equation; checks cubic initial velocity and agreement with the shared short-time expansion; checks event-grid refinement and ground-first rejection. Both prediction CLIs and both parallel evaluators were smoke-tested on a small generated, split Parquet dataset. Full 15,979-shot benchmarking remains for your machine. Small smoke-test scores are not reported as model-performance evidence.
