# Analytical approximations for penalty-trajectory prediction

## 1. Scope and principal finding

This report documents the final `analytic_kicks` implementation and the supplied full-test results. Two forward approximations are evaluated: first-order linearization of the nonlinear drag vector, and a physics-constrained cubic Taylor expansion of position. Both use the same bounded Bayesian inverse calculation, implemented with importance sampling. Neither is a learned regression model, and neither requires a dataset-wide training step.

“Analytical” describes the forward trajectory expressions. Parameter estimation, matrix functions, crossing roots, and posterior integration are evaluated numerically. These methods are not exact closed-form solutions of the original nonlinear flight equations.

Both methods returned predictions for all 15,979 test shots. Radial RMSE was approximately 1.12 m for each. Their nominal 95% ellipses covered only 46.53% and 45.17% of all requested shots, respectively. More than 90% of queries had importance-sampling ESS below 100. The reported distributions are therefore **not calibrated 95% uncertainty regions across this test set**. The results characterize the complete implemented approximation-and-inference pipeline, not the irreducible accuracy of the analytical forward models alone.

## 2. Coordinates and prediction task

Internally use Team 358's right-handed frame: X right, Y toward goal, Z up. The problem frame has x forward, y left, z up:

\[
(X,Y,Z)=(-y,x,z),\qquad(x,y,z)=(Y,-X,Z).
\]

The ball center starts at \(\mathbf r_0=(0,0,0.11)\) m. Observations occur at 0.05 and 0.10 s. The internal observation vector is

\[
\mathbf d=(X_1,Y_1,Z_1,X_2,Y_2,Z_2)^T.
\]

The goal plane is Y=11 m; the target is \(\mathbf q=(X_g,Z_g)\). In problem coordinates the answer is \((11,-X_g,Z_g)\). Default plots display y-left and z-up. Internal JSON fields remain in the paper frame; `mean_output_m`, `mode_output_m`, and `output_covariance_m2` carry the displayed-frame quantities. Reflecting the horizontal axis changes the sign of cross-covariance but not axis variances, radial errors, ellipse areas, or coverage.

## 3. Physical constants and original ODE

The adopted force model and parameter table come from Team 358 [1]. The initial center height and observation times are from the problem statement.

| Constant | Symbol | Value |
|---|---|---:|
| Ball mass | m | 0.430 kg |
| Ball radius | a | 0.108 m |
| Air density | rho | 1.27 kg/m³ |
| Drag coefficient | C_D | 0.4 |
| Gravity | g | 9.8 m/s² |
| Initial center height | Z_0 | 0.11 m |
| Goal distance | L | 11 m |
| Goal width / height | W / H | 7.32 / 2.44 m |

Define

\[
k_D=\frac{C_D\rho\pi a^2}{2m},\qquad
k_M=\frac{8\pi\rho a^3}{3m},\qquad
g_e=g\left(1-\frac{4\pi\rho a^3}{3m}\right).
\]

Here k_D has units m⁻¹, k_M is dimensionless, and g_e has units m/s². Gravity and buoyancy have been combined. With constant angular velocity \(\boldsymbol\omega\), the original equations are

\[
\dot{\mathbf r}=\mathbf v,\qquad
\dot{\mathbf v}=-k_D\|\mathbf v\|\mathbf v
+k_M(\mathbf v\times\boldsymbol\omega)-g_e\mathbf e_Z.
\]

Write the Magnus term as Bv, with

\[
B=k_M\begin{pmatrix}
0&\omega_Z&-\omega_Y\\
-\omega_Z&0&\omega_X\\
\omega_Y&-\omega_X&0
\end{pmatrix}.
\]

Thus the components are

\[
\begin{aligned}
\dot v_X&=-k_D\|\mathbf v\|v_X+k_M(v_Y\omega_Z-v_Z\omega_Y),\\
\dot v_Y&=-k_D\|\mathbf v\|v_Y+k_M(v_Z\omega_X-v_X\omega_Z),\\
\dot v_Z&=-k_D\|\mathbf v\|v_Z+k_M(v_X\omega_Y-v_Y\omega_X)-g_e.
\end{aligned}
\]

The implementation deliberately preserves the selected velocity-cross-spin sign. Substituting omega-cross-velocity without redefining spin would reverse its effect. Assumptions are a rigid spherical ball, uniform stationary air, constant aerodynamic coefficients and spin, and no ground interaction during retained flight. Wind, spin decay, seams, drag crisis, bounce, rolling, goalkeeper contact, and uncertain physical constants are excluded. Radius 0.108 and initial center height 0.11 are distinct: the center begins 2 mm above the model's ground-contact level.

## 4. Launch parameterization and prior

The unknown launch vector is

\[
\boldsymbol\lambda=(v_0,\theta,\phi,\omega_X,\omega_Y,\omega_Z).
\]

Polar theta is measured from +Z; azimuth phi is measured from +X toward +Y. The initial velocity is

\[
\mathbf v_0=v_0(\sin\theta\cos\phi,\sin\theta\sin\phi,\cos\theta)^T.
\]

The code assumes independent uniform distributions in the following intervals:

| Variable | Bounds | Basis |
|---|---|---|
| Speed | sqrt(9.8×11)–129/3.6 m/s | Ballistic lower reference; Guinness-listed upper reference [3] |
| Polar angle | 0–90 degrees | Chosen upward launch envelope |
| Azimuth | 55–125 degrees | Chosen symmetric ±35° envelope about forward |
| Spin X | -8 to +8 rad/s | Team 480 section 4.2 [2] |
| Spin Y | -5 to +5 rad/s | Team 480 section 4.2 [2] |
| Spin Z | 0 to +15 rad/s | Team 480 section 4.2 [2] |

The lower speed, approximately 10.383 m/s, follows from the ideal same-height vacuum range \(R=v_0^2\sin(2\alpha)/g\), maximized at 45°. It is a reference cutoff, not a rigorous minimum under drag and lift. The upper value is 35.833 m/s from the public 129 km/h Guinness listing, not an asserted absolute human capability limit.

The angle envelope pads beyond Team 358's upper-corner example and symmetrizes azimuth around 90°; it is a pipeline design choice rather than a measured universal bound [1]. Team 480's spin intervals are modeling choices, not separate human records. Its appendix also explores negative vertical spin; the final pipeline retains the stated nonnegative interval [2]. No spin-decay model from that paper is used.

Uniform polar angle does not mean uniform solid angle. Uniform Cartesian spin components do not give isotropic spins. These choices match the dataset's configured reference measure and are consequential priors, not estimates of how people usually kick.

Computations use normalized parameters \(\mathbf u\in[0,1]^6\), with

\[
\boldsymbol\lambda=\boldsymbol\ell+
(\boldsymbol h-\boldsymbol\ell)\odot\mathbf u.
\]

The affine map makes the prior constant on the unit cube. Bounds are hard-coded in `analytic_core.py`; custom generation bounds would require changing them. The scripts do not infer bounds from split metadata.

## 5. Method A: first-order drag linearization

Let \(D(\mathbf v)=-k_D\|\mathbf v\|\mathbf v\). For nonzero velocity,

\[
\frac{\partial D}{\partial\mathbf v}
=-k_D\left(\|\mathbf v\|I+
\frac{\mathbf v\mathbf v^T}{\|\mathbf v\|}\right).
\]

Expand about each candidate launch velocity \(\mathbf v_0\), with \(s_0=\|\mathbf v_0\|\):

\[
D(\mathbf v)\approx D(\mathbf v_0)+D'_0(\mathbf v-\mathbf v_0).
\]

Since \(D'_0\mathbf v_0=-2k_Ds_0\mathbf v_0\), the acceleration becomes

\[
\boxed{\dot{\mathbf v}=A\mathbf v+\mathbf b},
\]

\[
A=-k_D\left(s_0I+\frac{\mathbf v_0\mathbf v_0^T}{s_0}\right)+B,
\qquad \mathbf b=k_Ds_0\mathbf v_0-g_e\mathbf e_Z.
\]

This linearizes the complete drag vector and keeps constant-spin Magnus acceleration exactly linear in velocity. It is not merely freezing the scalar drag coefficient. The reference velocity is inferred for each candidate, not fixed from the true test launch.

Set \(\mathbf d_A=A^{-1}\mathbf b\). The affine system has trajectory

\[
\mathbf v_L(t)=e^{At}(\mathbf v_0+\mathbf d_A)-\mathbf d_A,
\]

\[
\boxed{
\mathbf r_L(t)=\mathbf r_0+
A^{-1}(e^{At}-I)(\mathbf v_0+\mathbf d_A)-t\mathbf d_A.
}
\]

Although inverse notation is used mathematically, the code uses linear solves. A is nonsingular for the permitted positive speeds: its symmetric part is strictly negative definite, while B is skew-symmetric. The code diagonalizes A for efficient repeated time evaluation, using `expm1` for small exponential differences. When the eigenvector matrix condition number is at least 10⁸, it switches position evaluation to SciPy's matrix exponential [5]. The threshold is an implementation heuristic, not a physical constant.

The affine solution remains exact for the approximated ODE, but the drag approximation worsens as velocity moves away from its launch value. It must not be described as exact nonlinear flight physics.

## 6. Method B: cubic short-time Taylor expansion

Expand position at t=0:

\[
\mathbf r(t)=\mathbf r_0+\mathbf v_0t+
\tfrac12\mathbf a_0t^2+\tfrac16\mathbf j_0t^3+O(t^4).
\]

The original nonlinear acceleration at launch is

\[
\mathbf a_0=-k_Ds_0\mathbf v_0+B\mathbf v_0-g_e\mathbf e_Z.
\]

Because spin and gravity are constant, the chain rule gives

\[
\mathbf j_0=\left.\frac{\partial\mathbf f}{\partial\mathbf v}\right|_0\mathbf a_0
=A\mathbf a_0.
\]

The implemented approximation and its derivative are therefore

\[
\boxed{\mathbf r_T(t)=\mathbf r_0+\mathbf v_0t+
\tfrac12\mathbf a_0t^2+\tfrac16A\mathbf a_0t^3},
\]

\[
\mathbf v_T(t)=\mathbf v_0+\mathbf a_0t+\tfrac12A\mathbf a_0t^2.
\]

The coefficients are constrained by the force law. An arbitrary cubic through the initial position and two observations would be underdetermined. This approach instead infers six physical launch parameters and derives acceleration and jerk from them.

Both approximations have the same position expansion through t³ for the same launch. The affine trajectory contains higher-order terms generated by its linear dynamics; the cubic stops at t³. The original nonlinear fourth and higher derivatives contain additional drag-curvature terms. This explains why the two methods can produce similar answers without either accurately capturing all later dynamics. A short observation interval does not guarantee that a short-time expansion remains accurate until goal crossing.

## 7. Goal events and selection

For each candidate the predictor seeks the first forward Y=11 crossing after 0.10 s and no later than 15 s. A candidate is excluded if a descending crossing of Z=0.108 occurs first. The crossing center must satisfy

\[
|X_g|<3.66,\qquad 0<Z_g<2.44.
\]

This is center-entry selection, not whole-ball clearance; these analytical scripts do not implement the optional whole-ball mode. Ground and goal events refer to each approximation, so the same selection rule can disagree with the original nonlinear simulator near a boundary.

For Taylor trajectories, the goal and ground equations are polynomials of degree at most three. `numpy.roots` finds their roots; the code retains positive roots up to 15 s with imaginary magnitude below 10⁻⁸ and the required velocity sign. For linear trajectories, positions on a 0.02 s grid bracket sign-changing events, then Brent's method refines each first event with absolute root tolerance 10⁻¹¹ s [6]. This grid is an event search, not an ODE time step. Closely spaced crossings within one cell can be missed; refinement checks with 0.01 or 0.005 s are appropriate. Polynomial roots are also evaluated numerically, not symbolically.

Each retained candidate supplies its own crossing time before extracting X,Z. Thus crossing-time uncertainty is propagated through the sampled trajectories rather than holding time fixed across samples.

## 8. Observation likelihood and inverse problem

For M equal to linear or Taylor, define

\[
F_M(\mathbf u)=
\begin{pmatrix}\mathbf r_M(0.05;\mathbf u)\\\mathbf r_M(0.10;\mathbf u)\end{pmatrix}.
\]

The measurement model is independent Gaussian coordinate error:

\[
\mathbf d=F_M(\mathbf u)+\boldsymbol\epsilon,
\quad\boldsymbol\epsilon\sim\mathcal N(0,\sigma^2I_6),
\quad\sigma=\frac{0.005}{\sqrt3}=0.0028867513\ \mathrm m.
\]

This sigma matches the variance of uniform rounding error on ±5 mm; the Gaussian likelihood is not identical to a bounded rounding model. Positions at launch and observation times are treated as exact. Sensor correlations and time uncertainty are excluded.

Let \(\chi_M^2(\mathbf u)=\|F_M(\mathbf u)-\mathbf d\|^2/\sigma^2\). The posterior target is

\[
p_M(\mathbf u\mid\mathbf d,S_M)\propto
\mathbf1_{[0,1]^6}(\mathbf u)
\mathbf1_{S_M}(\mathbf u)\exp[-\chi_M^2(\mathbf u)/2],
\]

where S_M denotes accepted airborne goal entry under the approximation. This conditions on entry, rather than estimating its unconditional probability. The finite launch box and goal selection can truncate or distort the posterior.

Six observations and six parameters do not ensure stable inversion. Spin nearly parallel to initial velocity has little immediate Magnus effect, and early motion may weakly distinguish different launches. Multiple solutions, boundary-constrained solutions, and thin curved posterior regions can result.

## 9. Implemented posterior inference

### 9.1 Local fits initialize proposals

A kinematic quadratic gives an initial velocity estimate for h=0.05 s:

\[
\widehat{\mathbf v}_0=
\frac{4\mathbf r_1-\mathbf r_2-3\mathbf r_0}{2h}.
\]

This is only an initializer; it does not replace the physical equations. Speed and angles are derived from it; initial spin is (0,0,7.5). The normalized starting vector is clipped into [10⁻⁶,1−10⁻⁶]. Four bounded least-squares fits are attempted by default. The first uses that initializer; the other three randomize only normalized spin coordinates in [0.01,0.99]. They are not four independent random initializations of all six parameters.

SciPy `least_squares` minimizes standardized observation residuals subject to the unit-box bounds, with `max_nfev=150` [4]. Default trust-region reflective and finite-difference Jacobian behavior is used. These fits do not impose the goal condition; goal filtering happens on sampled proposals. Fits are used even if they hit the iteration limit, with optimizer success count reported. The best fit chi-square is a diagnostic, not proof of posterior convergence or correct endpoint prediction.

### 9.2 Defensive proposal mixture

For each fit k, let J_k be the Jacobian of standardized residuals with respect to unit parameters. If

\[
J_k^TJ_k=V_k\operatorname{diag}(\eta_{kj})V_k^T,
\]

the Gaussian proposal covariance is

\[
C_k^{\rm prop}=V_k\operatorname{diag}
\left(\frac4{\max(\eta_{kj},4)}\right)V_k^T.
\]

The factor four broadens locally concentrated directions; the eigenvalue floor limits excessive proposal variance in poorly identified directions. Both are sampling heuristics in normalized parameter space, not physical priors or uncertainty corrections. The proposal is

\[
q(\mathbf u)=0.1\,U_{[0,1]^6}(\mathbf u)
+\frac{0.9}{K}\sum_{k=1}^K
\mathcal N(\mathbf u;\hat{\mathbf u}_k,C_k^{\rm prop}),\quad K=4.
\]

The 10% prior component gives support across the bounded prior, but finite random draws can still miss important regions. The mixture is sampled with NumPy's seeded random generator; this inverse sampling is **not Sobol sampling**.

Out-of-box draws and goal-invalid trajectories are discarded, without clipping or replacement. The complete mixture density is evaluated for surviving draws. The importance log-weight is

\[
\ell_i=-\tfrac12\chi_M^2(\mathbf u_i)-\log q(\mathbf u_i),
\qquad w_i=\exp[\ell_i-\operatorname{logsumexp}(\ell)].
\]

The unit-box prior has constant density one. Rejection removes zero-target samples; self-normalization cancels the common retention normalization. Omitting the proposal-density denominator would incorrectly treat targeted sampling as the physical prior.

This is self-normalized importance sampling, not MCMC, a Laplace-only approximation, or adaptive-likelihood Bayesian conditioning. The measurement sigma is never broadened. Posterior covariance is computed from weighted crossings, not from the proposal's regularized inverse curvature.

### 9.3 Sampling diagnostics

\[
\mathrm{ESS}=\frac1{\sum_iw_i^2}.
\]

The default benchmark uses 4,096 attempted proposals per query. ESS is computed after bounds and goal selection. ESS below 100 produces a warning; 100 is a practical threshold, not a convergence theorem. A large ESS cannot exclude an unvisited posterior mode. Small ESS can yield a sharply concentrated but unstable estimate. Increasing proposals, changing seeds, and using more starts are convergence checks, not automatic remedies for physical approximation error.

## 10. Point estimates, uncertainty, and plots

For retained crossing samples \(\mathbf q_i\), use

\[
\bar{\mathbf q}=\sum_iw_i\mathbf q_i,\qquad
C_q=\sum_iw_i(\mathbf q_i-\bar{\mathbf q})(\mathbf q_i-\bar{\mathbf q})^T.
\]

The posterior mean is the primary squared-error predictor. Covariance is a weighted predictive moment, so no unbiased-sample correction is applied. Directional and radial RMS uncertainties are

\[
\sigma_h=\sqrt{C_{11}},\qquad\sigma_Z=\sqrt{C_{22}},\qquad
u_r=\sqrt{\operatorname{tr}C_q}.
\]

Radial RMS uncertainty measures posterior distance from its own mean, not error relative to the true crossing. The 95% radial radius is the weighted 95th percentile of that distance. Marginal 95% intervals use separate 2.5% and 97.5% weighted quantiles.

For covariance eigenvalues lambda_j and eigenvectors V, calculate posterior squared Mahalanobis distances. Let c_95 be their weighted 95th percentile. The ellipse is

\[
(\mathbf q-\bar{\mathbf q})^TC_q^{-1}(\mathbf q-\bar{\mathbf q})\le c_{95},
\]

with semiaxes \(\sqrt{c_{95}\lambda_j}\) and eigenvector orientation. It is available only if ESS is at least three and the smaller eigenvalue exceeds 10⁻¹² times the larger. It is a covariance-oriented summary, not necessarily a highest-density region. It may extend outside the physical opening; reported area is the full ellipse area.

The plotted star is an approximate mode of a 0.05 m Gaussian KDE, numerically optimized from multiple starts. The heatmap uses weighted 0.02 m bins over a 366×122 goal grid and Gaussian smoothing. KDE mode and rendered density are approximations; posterior covariance does not include the additional KDE smoothing variance. Prediction scripts always compute this mode; the supplied benchmarks disabled it.

Uncertainty includes measurement noise and launch ambiguity under each approximate model. It omits approximation error, uncertain aerodynamic coefficients, and discrepancies between synthetic physics and real football. The earlier proposed inverse-Hessian covariance propagation is not the final uncertainty method: weighted posterior trajectory propagation is what the scripts actually use.

## 11. Dataset, split, noise, and aggregate evaluation

The evaluation dataset was generated with the original nonlinear ODE using adaptive RK45, not either analytical approximation. The existing generation used a scrambled six-dimensional Sobol design, filtered to airborne goal entries. The reported 1,048,576 candidates yielded 79,893 accepted kicks. Parquet stores launch IDs, parameters, two snapshots, and goal coordinates rather than whole trajectories.

The fixed split sorts by ID, applies a seeded permutation (42), and assigns floor(0.8N) rows to training: 63,914 train and 15,979 test. Analytical evaluation loads only the test partition and validates its stored fingerprint. The shared training split is unused by these predictors. A launch prior shared with the generator is not itself training on held-out targets.

Both analytical runs evaluated every test row. Independent Gaussian coordinate noise with the stated sigma is generated once using seed 42. The same noise realization and subset-selection procedure as the dataset-based benchmark are used. Per-query posterior seed is 42 plus the test row index. The true endpoint is used only after inference for scoring; true launch speed/spin fields are not passed to the inverse estimator.

Aggregate mode discards each full report after accumulating errors, coverage, and area. Only five scalar sequences are retained for exact medians: four uncertainty quantities and ESS. The parallel queue holds at most twice the worker count. Default output is one metrics JSON; optional `--per-shot` writes JSONL. This controls report/queue memory growth, but each worker still has inference arrays and the loader still materializes the test table. Automatic resume is not implemented.

## 12. Metric definitions and reasons

For error \(\mathbf e_i=\bar{\mathbf q}_i-\mathbf q_i^{\rm true}\):

| Metric | Definition | Interpretation |
|---|---|---|
| Horizontal/vertical MSE | mean(e_h²), mean(e_Z²) | Axis-specific squared error, m² |
| MSE per coordinate | (MSE_h+MSE_Z)/2 | Coordinate-averaged loss |
| Radial MSE | MSE_h+MSE_Z | Squared 2D displacement error |
| Axis RMSE | sqrt(axis MSE) | Axis error in metres |
| Radial RMSE | sqrt(radial MSE) | Combined squared-error scale in metres |
| Mean radial error | mean(norm(e)) | Average distance error, less dominated by outliers |
| Predicted sigmas / radial RMS | Posterior moments, averaged and medianed across queries | Sharpness rather than observed correctness |
| Marginal widths | Mean width of 95% intervals | Directional precision |
| Marginal/radial coverage | Fraction of truths inside intervals/circles | Calibration of reported uncertainty |
| Ellipse coverage | Fraction inside covariance-oriented region | Joint correlated uncertainty calibration |
| Ellipse availability | Valid ellipse count / query count | Detects unsupported uncertainty outputs |
| Mean ellipse area | Average pi×major×minor over available ellipses | Precision, interpreted together with coverage |
| ESS statistics | Median, minimum, fraction below 100 | Finite-sampling support |

Missing ellipses count as uncovered in all-query coverage. Coverage-when-available uses only valid ellipses. Prediction failures are counted separately; MSE uses successful queries. In these runs failures were zero, so all-query MSE has no failed-prediction selection bias. A returned prediction is not equivalent to converged inference or an available ellipse.

## 13. Results from the supplied files

The following tables are populated directly from `metrics_linear.json`, `metrics_taylor.json`, `prediction(1).json` (linear), and `prediction(2).json` (Taylor). No pilot results are substituted.

### 13.1 Full-test performance

| Metric | Linearized drag | Cubic Taylor |
|---|---:|---:|
| Horizontal MSE (m²) | 1.081369 | 1.078482 |
| Vertical MSE (m²) | 0.178444 | 0.185142 |
| MSE per coordinate (m²) | 0.629906 | 0.631812 |
| Radial MSE (m²) | 1.259813 | 1.263623 |
| Horizontal RMSE (m) | 1.039889 | 1.038500 |
| Vertical RMSE (m) | 0.422426 | 0.430281 |
| Radial RMSE (m) | 1.122414 | 1.124110 |
| Mean radial error (m) | 0.763211 | 0.765533 |
| Mean horizontal sigma (m) | 0.208609 | 0.201061 |
| Mean vertical sigma (m) | 0.190553 | 0.187581 |
| Mean radial RMS uncertainty (m) | 0.293818 | 0.285671 |
| Mean radial 95% radius (m) | 0.508217 | 0.494259 |
| Horizontal marginal 95% coverage | 44.114% | 42.412% |
| Vertical marginal 95% coverage | 78.128% | 75.580% |
| Radial 95% coverage | 49.784% | 48.514% |
| Ellipses available | 14182 | 14061 |
| Ellipses covering truth | 7435 | 7217 |
| Ellipse availability | 88.754% | 87.997% |
| Ellipse coverage when available | 52.426% | 51.326% |
| Ellipse coverage over all requested | 46.530% | 45.166% |
| Mean available ellipse area (m²) | 1.128158 | 1.105123 |
| Median ESS | 30.875 | 30.224 |
| Minimum ESS | 1.000 | 1.000 |
| Queries with ESS below 100 | 90.412% | 90.588% |
| Reported elapsed time (minutes) | 38.79 | 16.37 |

Both runs used 4,096 proposals, four starts, no mode benchmark, sigma 0.0028867513 m, seed 42, and event-step setting 0.02 s (used only by linear event bracketing). Hardware and worker count are not stored in these result files, so timings are run-specific rather than controlled hardware-normalized performance claims.

### 13.2 Original problem prediction

Inputs in the problem frame are (1.50,0.45,0.44) at 0.05 s and (2.96,0.86,0.74) at 0.10 s. Reported output below is (y,z), with x=11 m fixed.

| Quantity | Linearized drag | Cubic Taylor |
|---|---:|---:|
| Mean crossing (m) | (2.628733, 1.958994) | (2.662095, 1.972450) |
| Approximate KDE mode (m) | (2.626379, 1.941570) | (2.617097, 1.978097) |
| Horizontal sigma (m) | 0.123328 | 0.116537 |
| Vertical sigma (m) | 0.133296 | 0.131819 |
| Radial RMS uncertainty (m) | 0.181597 | 0.175947 |
| 95% radial radius (m) | 0.314524 | 0.303199 |
| Posterior ESS | 491.990272 | 526.527481 |
| Best observation chi-square | 0.325340 | 0.299990 |
| Attempted proposals | 16384.000000 | 16384.000000 |
| Bounded proposals | 4560.000000 | 4646.000000 |
| Goal-valid proposals | 3093.000000 | 3177.000000 |
| 95% y interval (m) | [2.438365, 2.894196] | [2.476489, 2.916148] |
| 95% z interval (m) | [1.679678, 2.213548] | [1.703303, 2.228555] |

The linear output covariance is `[[0.015209792, 0.000392566], [0.000392566, 0.017767802]]` m²; ellipse semiaxes are 0.327510 and 0.301931 m.

The Taylor output covariance is `[[0.013580988, -0.000557543], [-0.000557543, 0.017376255]]` m²; ellipse semiaxes are 0.320181 and 0.281576 m.

## 14. Interpretation of the results

The original-shot means are approximately (11,2.629,1.959) m and (11,2.662,1.972) m in problem coordinates. Their difference is small, and both plots show compact regions near the same goal location. These single-query runs used 16,384 proposals, four times the benchmark budget, and achieved ESS approximately 492 and 527. All four local optimizers reported success. They therefore have materially better importance-sampling support than the typical benchmark query, although neither ESS nor a low early-position residual proves convergence or later trajectory accuracy. The actual endpoint of this problem instance is not known, so agreement between methods is not validation.

The full test results tell a different story from the compact example plots. Both methods severely under-cover: only about half of truths lie inside nominal 95% ellipses even when an ellipse is available. Missing ellipses further reduce all-query coverage. Mean radial posterior uncertainty is about 0.29 m, while radial RMSE is about 1.12 m. These quantities are not mathematically identical calibration targets, but the coverage statistics independently establish overconfidence in the reported regions.

There are at least two unresolved error sources:

1. **Approximate dynamics.** Matching 0.05 and 0.10 s observations does not guarantee accurate extrapolation to the goal. Both models neglect higher-order nonlinear drag effects, and posterior intervals include no allowance for that discrepancy.
2. **Finite inverse sampling.** Median ESS is about 30, minimum ESS is one, and over 90% of queries fall below the warning threshold. A limited mixture proposal can miss curved or alternative parameter regions. Narrow apparent uncertainty can therefore reflect incomplete sampling, not strong information.

These aggregate files cannot apportion error between those causes. It would be unjustified to claim that the linearization alone causes all undercoverage, or that simply increasing samples would repair it. The small RMSE difference between the two methods does not establish a meaningful ranking without convergence checks. The scalar means being close on one shot likewise does not establish equivalence across the dataset.

Zero prediction failures means that every inverse run returned a weighted point estimate. It does not mean every posterior was well supported. Ellipse failures occur separately when ESS or covariance conditions fail; the linear run lacked 1,797 ellipses and the Taylor run lacked 1,918.

A defensible next validation sequence would compare more proposals and independent seeds on a fixed diagnostic subset, refine the linear event grid, and compare forward approximations to nonlinear trajectories at known launch parameters on training data. This would separate forward-model error from inverse-estimation error. Any discrepancy calibration or hyperparameter selection should use training/validation data, not the final test scores. These are proposed follow-up checks, not modifications already included in the reported results.

## 15. Numerical settings and their status

| Setting | Value | Reason/status |
|---|---|---|
| Benchmark proposals | 4,096/query | Computational budget; does not guarantee sufficient ESS |
| Original-shot proposals | 16,384 | Larger query-specific budget recorded in supplied JSON |
| Least-squares starts | 4 | Multiple spin initializations; not exhaustive |
| Fit evaluation cap | 150 | Runtime safeguard |
| Uniform proposal mass | 0.1 | Defensive global support |
| Local Gaussian mass | 0.9 total | Concentration near observation-fitting regions |
| Curvature covariance inflation | 4 | Broader local exploration |
| Curvature eigenvalue floor | 4 in normalized units | Bounds proposal variance only |
| ESS warning threshold | 100 | Practical diagnostic, not a guarantee |
| Linear eigensystem condition threshold | 10⁸ | Selects matrix-exponential fallback |
| Linear event bracket spacing | 0.02 s | Root search resolution; refine to test stability |
| Brent absolute tolerance | 10⁻¹¹ s | Event-root numerical precision |
| Polynomial imaginary tolerance | 10⁻⁸ | Treats nearly real numerical roots as real |
| Ellipse minimum ESS | 3 | Minimum support rule |
| Ellipse eigenvalue ratio threshold | 10⁻¹² | Rejects numerically singular covariance |
| KDE bandwidth | 0.05 m | Display/mode smoothing, not posterior noise |
| Heatmap bin width | 0.02 m | Rendering resolution |

The constants in this table are implementation choices. They should not be attributed to Team 358, Team 480, or experimental measurements. Numerical precision of a root or fit does not imply comparable physical accuracy.

## 16. Software roles, verification, and reproduction

| File | Role |
|---|---|
| `analytic_core.py` | Force coefficients, two forward approximations, events, inverse importance sampling |
| `analytic_cli.py` | Prediction/evaluation commands, bounded worker queue, aggregate orchestration |
| `stream_metrics.py` | Running metric sums and compact scalar median storage |
| `data.py` | Load and fingerprint-check fixed test partition |
| `physics_predictors.py` | Shared posterior moments, quantiles, KDE mode, heatmap and ellipse plotting |
| `evaluate_linear.py`, `evaluate_taylor.py` | Separate test benchmark entry points |
| `predict_linear.py`, `predict_taylor.py` | Separate single-shot entry points |
| `verify_analytic.py` | Forward equations and event checks |

The tests compare the matrix-exponential trajectory to high-accuracy integration of the affine ODE; verify cubic launch derivatives and shared short-time expansion; check event-grid refinement and ground-first rejection. Both prediction commands and parallel evaluators were smoke-tested. Streaming aggregates were compared against the preceding batch metric calculation. These checks verify implementation consistency, not posterior convergence for every test case or empirical fidelity of the ball model.

Example commands, run from the extracted package:

```powershell
python -m pip install -r requirements.txt
python verify_analytic.py

python evaluate_linear.py --dataset ../kicks_split --workers 4 --output metrics_linear.json
python evaluate_taylor.py --dataset ../kicks_split --workers 4 --output metrics_taylor.json

python predict_linear.py --samples 16384 --output prediction_linear
python predict_taylor.py --samples 16384 --output prediction_taylor
```

The original point is built into the prediction defaults. Explicit observations can be passed with `--r1` and `--r2`. Use `--limit 200` for the same seeded test subset as the earlier pipeline; `--clean` disables injected Gaussian noise while retaining the likelihood sigma. `--include-mode` enables additional mode errors in evaluation. `--per-shot` enables optional detailed logging. No model `.npz` or `train.py` is needed for these analytical inverse predictors.

The result JSON records method, split fingerprint, noise, proposal settings, counts, and aggregate metrics. It does not record all dependency versions, hardware, or worker count; those should accompany a fully archived reproduction. The attached full-test files match source fingerprint `a90ca8cc9cf32a1cc8ef149ae4f4c5bce2222335fac094401e74eb678e0b4e6c` and test fingerprint `affad316537ec73597c38697f933cdc46aecee8275350a8bfbe8c7e91466956c`.

## 17. Sources and attribution

The force model and selected bounds are externally sourced; the linearization, Taylor derivation, proposal construction, and uncertainty workflow above describe the delivered implementation and elementary derivations from that model. They should not be presented as methods reported by the competition teams.

1. [Team 358, *The Best Strategy for Penalty Kicks*, 2022 University Physics Competition](https://www.uphysicsc.com/2022-GM-B-358.pdf). Reference for the adopted force model, physical parameter table, coordinate convention, and motivating angle example. This implementation uses the selected table values and equations, not a verbatim copy of the paper's appendix code.
2. [Team 480, *Analysis of the optimal initial velocities and spins of soccer in Penalty Kicks*, 2022](https://www.uphysicsc.com/2022-GM-B-480.pdf). Section 4.2, page 7, supplies the modeling spin intervals. Its other dynamical assumptions are not imported here.
3. [Guinness World Records, *Fastest football kick*](https://www.guinnessworldrecords.com/world-records/fastest-football-kick). Public 129 km/h reference used by the existing dataset for its speed ceiling; not a universal physiological bound.
4. [SciPy, `least_squares`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html). Bounded nonlinear least-squares API and its default solver/Jacobian behavior. Exact installed versions were not recorded in the attached runs.
5. [SciPy, `scipy.linalg.expm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html). Matrix exponential used for velocity evaluation and the position fallback. Documentation cites Al-Mohy and Higham's scaling-and-squaring work.
6. [SciPy, `brentq`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html). Bracketed scalar event-root refinement.

Primary run evidence: `metrics_linear.json`, `metrics_taylor.json`, `prediction(1).json`, `prediction(2).json`, and their associated heatmaps supplied on 8 November 2026. Code evidence: the final aggregate-output version of `analytic_kicks.zip`. Analytical derivations in sections 5–6 follow by differentiating the stated drag law, solving an affine linear ODE, and applying the chain rule; no additional empirical coefficients are introduced by those derivations.
