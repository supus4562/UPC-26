# Dataset-based prediction of penalty trajectories

## 1. Scope and reproducibility

This document describes the final `kick_pipeline` implementation: a physics-generated dataset, a persistent 80/20 split, adaptive Bayesian conditioning, and a conditional Gaussian mixture model (CGMM). It reports the supplied `metrics(1).json` and `prediction.json`. Earlier fixed-bandwidth Bayesian experiments, on-query model fitting, changing evaluation splits, spin-decay models, and alternative launch conventions are not part of this pipeline.

The forward integrator is **adaptive RK45**, not classical fixed-step RK4. Team 358 supplies the adopted flight equations; the integration and inference implementation described here is our pipeline. The probabilistic models estimate crossing location **conditional on airborne goal entry under the chosen simulation distribution**. They do not predict the probability that an arbitrary shot scores, nor whether a goalkeeper saves it.

Reported benchmark results use 63,914 training shots and all 15,979 test shots. The supplied prediction confirms a converged 16-component CGMM. Generation defaults and implementation details below come from the delivered source code; the original generation manifest was not attached to this results message, so defaults should not be mistaken for independently recovered run metadata.

## 2. Coordinates, observations, and prediction target

We retain Team 358's right-handed coordinate convention throughout simulation and storage. Let uppercase coordinates denote this frame and lowercase coordinates denote the 2026 problem frame.

| Direction | Team 358 frame | Problem frame |
|---|---|---|
| Forward toward goal | +Y | +x |
| Left when facing goal | -X | +y |
| Up | +Z | +z |
| Goal plane | Y=11 m | x=11 m |

The transformation is

\[
(X,Y,Z)=(-y,x,z),\qquad(x,y,z)=(Y,-X,Z).
\]

It is a rotation, preserving handedness. Velocity and angular velocity transform by the same matrix if conversion is needed. Internally, the initial ball center is `(0,0,0.11)` m. The goal opening is `|X|<3.66`, `0<Z<2.44` at `Y=11`.

The six-dimensional input is

\[
\mathbf u=(X_{0.05},Y_{0.05},Z_{0.05},X_{0.10},Y_{0.10},Z_{0.10}),
\]

and the two-dimensional target is \(\mathbf q=(X_g,Z_g)\). The plane coordinate is always 11 and therefore is not a learned target. Crossing time is stored but is not predicted by these two models.

`predict.py` accepts problem-frame observations by default, converts them internally, and displays `(y,z)` unless `--output-frame paper` requests `(X,Z)`. JSON internal summaries retain the paper frame; output coordinates and output covariance are separately labeled. Reflection of the horizontal coordinate preserves axis variances, distances, MSE, ellipse area, and coverage, but reverses the sign of horizontal–vertical covariance.

## 3. Launch variables and sampling bounds

The six independently sampled launch parameters are speed, two direction angles, and three Cartesian spin components:

\[
\boldsymbol\lambda=(v_0,\theta,\phi,\omega_X,\omega_Y,\omega_Z).
\]

Polar angle \(\theta\) is measured from +Z. Azimuth \(\phi\) is measured from +X toward +Y in the horizontal plane. Consequently,

\[
\mathbf v_0=v_0(\sin\theta\cos\phi,\sin\theta\sin\phi,\cos\theta).
\]

At `theta=90°, phi=90°` the launch is horizontal toward goal. Smaller theta adds upward velocity. Phi 55° aims right of forward; phi 125° aims left. These endpoints are symmetric about 90°.

| Parameter | Final default interval | Basis and interpretation |
|---|---|---|
| Speed | sqrt(9.8×11) to 129/3.6 m/s, approximately 10.383–35.833 | Ideal ballistic lower reference; Guinness-listed upper reference [3] |
| Polar theta | 0–90 degrees | Forward/upward search envelope, padded to horizontal |
| Azimuth phi | 55–125 degrees | Symmetric ±35° envelope around goal direction |
| Spin omega_X | -8 to +8 rad/s | Team 480 section 4.2 modeling interval [2] |
| Spin omega_Y | -5 to +5 rad/s | Team 480 section 4.2 modeling interval [2] |
| Spin omega_Z | 0 to +15 rad/s | Team 480 section 4.2 modeling interval [2] |

These are independent Cartesian component bounds, not bounds on spin magnitude or body-fixed Euler rotation rates. All spin units are rad/s. The largest magnitude allowed by this box is sqrt(8²+5²+15²)=17.72 rad/s, approximately 2.82 revolutions/s. The final default deliberately retains nonnegative vertical spin; it does not include both lateral-curvature signs through omega_Z. This can make the accepted distribution asymmetric even though the azimuth interval is symmetric.

### 3.1 Source qualifications

The minimum-speed expression follows from the ideal same-height vacuum range

\[
R=\frac{v_0^2\sin(2\alpha)}{g},\quad \max_\alpha R=\frac{v_0^2}{g}.
\]

Setting `R=11` gives the adopted lower reference. It is **not a proven minimum** for the actual model: release height, allowed crossing height, drag, and Magnus acceleration change reachability. Simulated trajectories still have to pass the event-based selection.

Guinness's public “Fastest football kick” listing gives 129 km/h, achieved by Francisco Javier Galan Marin in 2001 [3]. This is the cited operational upper bound, not proof of an absolute biological limit; the page itself notes that public records may lag updates.

Team 358 reports upper-corner angle windows for a particular speed/spin example. Their maxima of 79.8° polar and 110.5° azimuth motivated padding; the final 90° polar cap and 55–125° symmetric azimuth envelope are our search choices, not experimentally established universal bounds [1].

Team 480 explicitly states the three spin intervals above on page 7. They are selected modeling ranges, **not separately verified human records**. Its appendix also explores negative vertical spin. This pipeline follows the stated intervals, not that appendix's full search [2]. Only these bounds are borrowed from Team 480; its rotational decay dynamics are not adopted.

## 4. Forward dynamics implemented in the generator

The following equations reproduce the implemented Team-358-style model [1]. Define state \(\mathbf s=(\mathbf r,\mathbf v)\), speed \(v=\|\mathbf v\|\), and constant spin \(\boldsymbol\omega=\boldsymbol\omega_0\).

| Constant | Value |
|---|---:|
| Ball mass m | 0.430 kg |
| Ball radius a | 0.108 m |
| Air density rho | 1.27 kg/m³ |
| Drag coefficient C_D | 0.4 |
| Gravitational acceleration g | 9.8 m/s² |
| Initial center height | 0.11 m |

With \(A=\pi a^2\), \(V=4\pi a^3/3\), define

\[
k_D=\frac{C_D\rho A}{2m},\qquad
k_M=\frac{8\pi\rho a^3}{3m},\qquad
g_{\rm eff}=g\left(1-\frac{\rho V}{m}\right).
\]

The ODE is

\[
\dot{\mathbf r}=\mathbf v,\qquad
\dot{\mathbf v}=-k_Dv\mathbf v-g_{\rm eff}\hat{\mathbf Z}
+k_M(\mathbf v\times\boldsymbol\omega).
\]

In component form:

\[
\begin{aligned}
\dot v_X&=-k_Dvv_X+k_M(v_Y\omega_Z-v_Z\omega_Y),\\
\dot v_Y&=-k_Dvv_Y+k_M(v_Z\omega_X-v_X\omega_Z),\\
\dot v_Z&=-g_{\rm eff}-k_Dvv_Z+k_M(v_X\omega_Y-v_Y\omega_X).
\end{aligned}
\]

Gravity and buoyancy combine into the effective downward acceleration; quadratic drag opposes motion. The code preserves the selected paper's `v cross omega` convention. Reversing the cross-product order without changing the spin convention would reverse the modeled Magnus acceleration. The sign is explicitly tested against the vector expression.

The model assumes a rigid spherical ball, still uniform air, fixed coefficients, constant spin, and no contact during retained flight. There is no wind, spin decay, seam-dependent force, drag crisis, variable lift coefficient, bounce, rolling, goalkeeper collision, or uncertain aerodynamic parameter. The 0.108 m physical radius and 0.11 m initial height are distinct values: the center starts 2 mm above the modeled contact height.

## 5. RK45 integration and terminal events

For each launch, `scipy.integrate.solve_ivp` integrates from 0 to a maximum of 15 s, with `method='RK45'`, `rtol=1e-8`, `atol=1e-10`, and `max_step=0.02` s. `t_eval=[0.05,0.10]` requests the two input snapshots. RK45 uses an embedded Runge–Kutta 5(4) scheme with adaptive error control and quartic dense interpolation [4]. It is neither a fixed 0.02 s time grid nor classical RK4; 0.02 s is only the largest allowed step.

Two terminal event functions are supplied:

\[
f_{\rm goal}=Y-11,\quad\text{direction}=+1;
\qquad f_{\rm ground}=Z-0.108,\quad\text{direction}=-1.
\]

The solver locates an event root using its continuous step interpolant. `y_events` supplies the crossing state and `t_events` the event time. Thus no separate spline fitted to stored trajectory samples is needed. “Event location” means a numerically resolved crossing within solver/interpolation accuracy, not an exact analytical solution.

`dense_output=False` avoids retaining a global continuous solution object; local interpolation can still be used internally for events and requested output times. Only the snapshots and crossing are written to the dataset.

## 6. Scrambled Sobol design and shot selection

The generator constructs one six-dimensional scrambled Sobol sequence with default seed 358. For `--power 20`, it generates \(2^{20}=1,048,576\) candidate launches. Each unit-cube coordinate is mapped affinely to its parameter interval. Sobol points are a low-discrepancy space-filling design, rather than independent random draws; power-of-two totals preserve their intended balance properties [5].

Chunks continue the same sequence instead of restarting a sampler. Default chunk size is 4,096. Resumption regenerates the deterministic sequence and skips completed chunks with matching checkpoints. A manifest guards against incompatible configurations. Workers integrate separate candidates independently.

The implied reference distribution is uniform in the six selected parameters before filtering. In particular, **uniform polar angle is not uniform solid angle**, and independent Cartesian spins are not an isotropic spin distribution. These modeling choices determine the prior used by the predictors. They are not an empirical distribution of professional players' kicks. Filtering also destroys independence between parameters and the original rectangular low-discrepancy structure of the accepted subset.

### 6.1 Acceptance rule

A candidate is retained only if integration succeeds, reaches the forward goal plane before ground contact, includes both observation snapshots, and crosses strictly inside the goal opening. By default the criterion concerns the center: `abs(X_goal)<3.66` and `0<Z_goal<2.44`. Ground contact is separately excluded, so a retained trajectory cannot pass through the pitch before reaching the goal. An optional `--whole-ball` mode shrinks lateral and vertical opening bounds by 0.108 m; it is not the default selection.

Rejected categories include solver failure, ground-first, timeout, crossing before the second observation, and outside-goal. There is no rebound or roll continuation. Goal-center acceptance is not identical to full-ball clearance of posts/crossbar.

The reported generation counts in this conversation were:

| Outcome | Count |
|---|---:|
| Outside goal | 614,151 |
| Ground first | 354,532 |
| Accepted | 79,893 |
| Total | 1,048,576 |

Approximately 7.62% of candidates survived. These fractions describe the configured synthetic launch design, not human scoring probabilities. All later prediction and testing use only the accepted subset.

### 6.2 Parquet storage

Each accepted row contains an integer launch ID; speed and angles; three spin components; three initial velocity components; six snapshot coordinates; crossing time; and three crossing coordinates. Numerical fields are float64. Zstandard-compressed Parquet partitions preserve a typed, columnar representation without storing complete flight histories. Parquet allows consumers to read relevant columns and avoids CSV parsing and textual precision issues. The current split helper nevertheless materializes tables in memory; storage format alone does not make that operation streaming.

## 7. Fixed train/test split and leakage prevention

`data.py` sorts accepted rows by unique ID, forms `numpy.random.default_rng(42).permutation(N)`, assigns the first `floor(0.8*N)` indices to training, and assigns the remainder to test. For N=79,893, this gives 63,914/15,979. It is an unstratified random split of accepted shots, not an 80/20 split of attempted launches.

Both partitions are saved with `split.json`, including seed, counts, and SHA-256 fingerprints of IDs, observations, and targets. A repeated compatible split request validates and reuses the existing partition. The fingerprints guard against accidental data changes; they do not prove physical correctness or independence of correlated near-neighbor trajectories.

`train.py` reads only training rows. Standardization, mixture fitting, and the Bayesian reference pool use no test observations or targets. Saved model files carry training IDs and partition fingerprints. `predict.py` loads only model artifacts. `evaluate.py` loads only the test partition plus model artifacts and rejects mismatched fingerprints or overlapping test/training IDs. It does not fit or resplit.

New generation creates the split automatically. Existing Parquet data can be split without resimulation. Hyperparameters should be selected using a validation subset within training, not by repeatedly optimizing test performance. The supplied benchmark is a within-simulator generalization test, not an external test on recorded football trajectories.

## 8. Shared observation model

Both predictors assume

\[
\mathbf d=\mathbf u+\boldsymbol\epsilon,\qquad
\boldsymbol\epsilon\sim\mathcal N(0,\sigma^2I_6),\quad
\sigma=0.005/\sqrt3=0.0028867513\ \text{m}.
\]

This standard deviation equals that of uniform error on a ±5 mm rounding interval. A Gaussian with the same variance is not identical to a bounded rounding likelihood. The implementation assumes independent, equally precise coordinate errors and exact timestamps; temporal correlation and timing uncertainty are absent.

Evaluation adds Gaussian noise with this sigma to clean simulated test inputs. Both methods receive the same noisy observations. The default seed is 42. `--clean` suppresses injected noise but retains the likelihood sigma. No noise is added to the crossing target. The uncertainties below are conditional model predictions and do not include uncertainty in the chosen physics or launch distribution.

## 9. Adaptive Bayesian conditioning

### 9.1 Reference model and posterior weights

The training set represents an empirical joint distribution on observation/target pairs. For training shot i, with input \(\mathbf u_i\) and target \(\mathbf q_i\), the normalized weight for measured input d is

\[
w_i(h)=\frac{\exp[-\|\mathbf u_i-\mathbf d\|^2/(2h^2)]}
{\sum_j\exp[-\|\mathbf u_j-\mathbf d\|^2/(2h^2)]}.
\]

The posterior over targets is the weighted discrete measure \(\sum_iw_i\delta_{\mathbf q_i}\). No parametric optimization is required; `train.py` saves the training pool and adaptive setting in `bayes.npz`. Equal prior row weights reflect the accepted simulation design. Distances are computed in the six raw position coordinates, all measured in metres.

For numerical stability, the implementation subtracts the minimum squared distance before exponentiating. This cancels during normalization and prevents all likelihoods underflowing together.

### 9.2 Why and how adaptation is used

Effective sample size is

\[
\mathrm{ESS}(h)=\frac1{\sum_iw_i(h)^2}.
\]

At the measurement sigma, a finite library may contain almost no close matches in six-dimensional input space. The final method starts at `h=sigma` and, if ESS is below 30, expands h and solves for `ESS(h)=30` using a bracketed root finder. It never deliberately reduces the bandwidth below the assumed measurement noise. There is no fixed-bandwidth evaluation option in the final pipeline.

This stabilizes inference by averaging over more trajectories, but changes the effective likelihood. It is best understood as adaptive kernel-based approximate Bayesian conditioning, not an exact posterior under the original 2.9 mm measurement model. The extra smoothing scale is reported as \(\sqrt{h^2-\sigma^2}\). It must not be interpreted as measured sensor error.

The full benchmark's raw ESS median is about 1 and maximum only 3.46. Adapted ESS is approximately 30 throughout. Median effective sigma is 0.03338 m, with maximum 0.26250 m. Thus finite-library sparsity materially affects this method.

## 10. Conditional Gaussian mixture model

### 10.1 Joint density training

CGMM here means **conditional Gaussian mixture model**. A joint full-covariance Gaussian mixture is fitted to the eight-vector \(\mathbf z=(\mathbf u,\mathbf q)\):

\[
p(\mathbf z)=\sum_{k=1}^{16}\pi_k\mathcal N(\mathbf z;\boldsymbol\mu_k,\Sigma_k).
\]

The eight columns are standardized using training-only means and standard deviations. Zero standard deviations are replaced by one. A full covariance allows correlations between early positions and final location, as well as between the two output coordinates. Multiple components approximate nonlinear and potentially multimodal relationships; a single Gaussian would impose one global linear conditional mean.

The implementation uses scikit-learn `GaussianMixture` with expectation-maximization, 16 components, full covariance, two initializations, seed 42, maximum 500 iterations, and covariance regularization `1e-6` in standardized units [6]. It refuses to save a nonconverged fit. The fitted parameters, scaling, convergence flag, training size, IDs, and fingerprints are saved in `cgmm.npz`. The supplied example confirms convergence. Convergence does not establish optimal component count or physical validity.

### 10.2 Analytical conditioning with measurement noise

Partition each standardized component into input/output blocks. Let \(\tilde{\mathbf d}\) be the standardized observation, and let \(R=\operatorname{diag}[(\sigma/s_1)^2,\ldots,(\sigma/s_6)^2]\), where s are training input scales. Then

\[
A_k=\Sigma_{uu,k}+R,\qquad
B_k=\Sigma_{qu,k}A_k^{-1},
\]

\[
\mathbf m_k=\boldsymbol\mu_{q,k}+B_k(\tilde{\mathbf d}-\boldsymbol\mu_{u,k}),
\qquad C_k=\Sigma_{qq,k}-B_k\Sigma_{uq,k},
\]

\[
\alpha_k\propto\pi_k\mathcal N(\tilde{\mathbf d};\boldsymbol\mu_{u,k},A_k).
\]

The code uses linear solves rather than explicit inverses and normalizes component weights in log space. Thus prediction conditions a previously trained model; it does not refit the GMM.

### 10.3 Goal truncation and Monte Carlo summaries

A fitted Gaussian mixture has unbounded support even when training targets lie within the goal. The implementation draws 10,000 conditional samples, transforms targets back into metres, and rejects samples outside the center-defined opening. Summaries are calculated from the retained samples. This implements an additional truncation to goal bounds and is consistent with the intended goal-conditioned prediction, though it modifies the raw Gaussian conditional distribution.

Posterior means, covariance, quantiles, and ellipses therefore contain Monte Carlo error. The draw count is configurable and the RNG is seeded. If no draws remain, prediction raises an error. The example retained all 10,000. No nearest-training-input diagnostic is computed in CGMM-only prediction, so an empty warnings list is not evidence of strong local dataset support.

## 11. Point estimates, heatmaps, and uncertainty summaries

For weighted targets, the posterior mean and covariance are

\[
\bar{\mathbf q}=\sum_iw_i\mathbf q_i,\qquad
C=\sum_iw_i(\mathbf q_i-\bar{\mathbf q})(\mathbf q_i-\bar{\mathbf q})^T.
\]

There is no sample-unbiased covariance correction: the weights define the predictive distribution itself. CGMM retained draws have equal weights.

The posterior mean minimizes expected squared Euclidean error under the predicted distribution and is therefore the primary MSE estimator. The “most likely” plotted point is an approximate mode of a target-space Gaussian KDE with bandwidth 0.05 m, optimized from multiple starting points within goal bounds. It is bandwidth-dependent and not guaranteed to be the global mode of the unsmoothed mixture or discrete measure.

Heatmaps use a weighted 366×122 histogram over the opening, giving 0.02 m bins, followed by Gaussian smoothing with the same 0.05 m nominal scale and density normalization. The star marks the approximate KDE mode and the cross the mean. This visual smoothing is separate from the observation-space adaptive bandwidth and is not added to the reported posterior covariance.

### 11.1 Directional and radial uncertainty

\[
\sigma_h=\sqrt{C_{11}},\quad \sigma_z=\sqrt{C_{22}},\quad
u_r=\sqrt{\operatorname{tr}(C)}.
\]

The radial RMS uncertainty \(\nu_r\) is the square root of the expected squared distance from the posterior mean. It is not the standard deviation of the scalar radius, and not an observed prediction error. The radial 95% radius is the posterior 95th percentile of \(\|\mathbf q-\bar{\mathbf q}\|\). Marginal intervals use the weighted 2.5th and 97.5th percentiles independently; two marginal 95% intervals do not imply a joint 95% rectangle.

### 11.2 Joint ellipse

For eigendecomposition \(C=V\Lambda V^T\), calculate each posterior point's squared Mahalanobis distance. Let c be its weighted 95th percentile. The ellipse is

\[
(\mathbf q-\bar{\mathbf q})^TC^{-1}(\mathbf q-\bar{\mathbf q})\le c.
\]

Semiaxes are \(\sqrt{c\lambda_1}\) and \(\sqrt{c\lambda_2}\), oriented by covariance eigenvectors. The code requires ESS at least 3 and a nonsingular covariance. The scaling is estimated from the posterior rather than imposed using a Gaussian chi-square constant.

This is a covariance-oriented summary, not necessarily a highest-density region for a multimodal distribution. It may extend outside the opening even though posterior support is confined inside; plots clip to the displayed goal. Reported area is the full ellipse area, not its intersection with the goal.

## 12. Benchmark definitions and rationale

For n test shots, let \(e_i=\hat{\mathbf q}_i-\mathbf q_i^{\rm true}\). The benchmark reports:

| Metric | Definition | Why it is included |
|---|---|---|
| Horizontal/vertical MSE | mean(e_h²), mean(e_z²), m² | Distinguishes lateral and height performance; penalizes large misses |
| MSE per coordinate | (MSE_h+MSE_z)/2 | Single coordinate-averaged squared-error score |
| Radial MSE | MSE_h+MSE_z, m² | Expected squared two-dimensional crossing error |
| Axis RMSE | sqrt(axis MSE), m | Error scale in physical units |
| Radial RMSE | sqrt(radial MSE), m | Combined crossing error in metres |
| Mean radial error | mean(norm(e)), m | Typical distance loss, less sensitive to extremes than squared error |
| Directional and radial uncertainty | Posterior standard deviations and radial RMS; mean and median across queries | Measures predicted spread separately from actual error |
| Marginal interval widths | Average 95% interval width in each axis | Direction-specific precision |
| Marginal/radial coverage | Fraction of truths inside predicted intervals/circles | Tests whether stated uncertainty matches held-out outcomes |
| Ellipse coverage | Fraction of truths inside nominal 95% ellipses | Tests correlated two-dimensional uncertainty |
| Ellipse availability | Number/fraction of valid ellipses | Prevents selective reporting of only easy cases |
| Mean ellipse area | mean(pi×a×b), m² | Measures sharpness; very broad ellipses can achieve coverage trivially |
| Bayesian ESS and bandwidth | Raw/final ESS and effective sigma | Reveals how strongly approximation depends on smoothing |

Coverage is reported both among available ellipses and over all queries, counting missing ellipses as uncovered. A prediction failure aborts evaluation rather than silently excluding the shot. The full benchmark contains posterior-mean errors only; mode errors require `--include-mode` and were not supplied here.

## 13. Full held-out results

These values come from the supplied `metrics(1).json`, not earlier pilot or changing-split experiments. Both methods evaluated the same 15,979 noisy held-out shots with sigma 0.0028867513 m. Every query had an ellipse.

| Metric | Adaptive Bayes | CGMM |
|---|---:|---:|
| Horizontal MSE (m²) | 0.800802 | 0.417782 |
| Vertical MSE (m²) | 0.141674 | 0.099728 |
| MSE per coordinate (m²) | 0.471238 | 0.258755 |
| Radial MSE (m²) | 0.942476 | 0.517511 |
| Horizontal RMSE (m) | 0.894875 | 0.646361 |
| Vertical RMSE (m) | 0.376396 | 0.315798 |
| Radial RMSE (m) | 0.970812 | 0.719382 |
| Mean radial error (m) | 0.782714 | 0.492666 |
| Mean predicted horizontal sigma (m) | 0.769173 | 0.448197 |
| Mean predicted vertical sigma (m) | 0.358990 | 0.251700 |
| Mean predicted radial RMS uncertainty (m) | 0.858976 | 0.518893 |
| Median predicted radial RMS uncertainty (m) | 0.732077 | 0.382219 |
| Mean 95% radial radius (m) | 1.400747 | 0.925445 |
| Mean horizontal 95% interval width (m) | 2.597547 | 1.739419 |
| Mean vertical 95% interval width (m) | 1.244809 | 0.972313 |
| Horizontal 95% coverage | 94.030% | 93.911% |
| Vertical 95% coverage | 95.300% | 93.917% |
| Radial 95% coverage | 95.444% | 94.186% |
| Joint ellipse covered / total | 15,158 / 15,979 | 15,054 / 15,979 |
| Joint 95% ellipse coverage | 94.862% | 94.211% |
| Mean ellipse area (m²) | 4.647161 | 2.815999 |

CGMM reduces radial RMSE by approximately 25.9% and radial MSE by 45.1% relative to adaptive Bayes, with approximately 39.4% smaller mean ellipse area. Adaptive Bayes's joint coverage is closer to the nominal 95%. CGMM shows modest aggregate undercoverage: it is more precise and more accurate on this split, but its narrower distributions are not perfectly calibrated. These results do not justify calling either model's uncertainty exact.

Averages do not establish calibration for every speed, spin, goal region, or observation pattern. Both methods have greater horizontal than vertical error. This can reflect dynamics, parameter support, and the wider lateral target domain; the aggregate benchmark alone does not identify the causal contribution of each.

## 14. Prediction for the supplied penalty

Problem-frame observations are `(1.50,0.45,0.44)` m at 0.05 s and `(2.96,0.86,0.74)` m at 0.10 s. Internally they become `(-0.45,1.50,0.44)` and `(-0.86,2.96,0.74)`.

The following output is in the **problem frame**, with x=11 m fixed:

| Quantity | Adaptive Bayes | CGMM |
|---|---:|---:|
| Mean (y,z), m | (2.940893, 1.904353) | (2.645236, 1.913777) |
| Approximate KDE mode (y,z), m | (2.524617, 1.937254) | (2.653129, 1.914937) |
| Horizontal sigma, m | 0.372096 | 0.135454 |
| Vertical sigma, m | 0.262254 | 0.121485 |
| Radial RMS uncertainty, m | 0.455228 | 0.181951 |
| 95% radial radius, m | 0.717219 | 0.314358 |
| Marginal 95% y interval, m | [2.360058, 3.595267] | [2.376082, 2.908216] |
| Marginal 95% z interval, m | [1.397124, 2.409153] | [1.677393, 2.148793] |
| Ellipse major/minor semiaxes, m | 0.781848 / 0.531694 | 0.331286 / 0.297008 |

The output-frame covariance matrices, in m², are

\[
C_{\rm Bayes}=\begin{pmatrix}0.138455&-0.015384\\-0.015384&0.068777\end{pmatrix},\qquad
C_{\rm CGMM}=\begin{pmatrix}0.018348&-0.000151\\-0.000151&0.014759\end{pmatrix}.
\]

For this query, adaptive Bayes has raw ESS 1.000007 and expands its observation bandwidth to 0.040727 m to reach ESS 30. The nearest training observation is 0.026898 m away in combined six-coordinate Euclidean distance, and no training shot lies within all six ±5 mm rounding intervals. This explains why the plotted posterior is patchy and why its mean differs noticeably from its mode: it combines several locally supported crossing clusters after substantial smoothing.

CGMM gives a compact, approximately unimodal region in the supplied plot. All 10,000 sampled conditional crossings remained inside the goal. Its mean and mode nearly coincide. The ground-truth endpoint for this particular problem instance is not supplied, so the narrower ellipse cannot itself prove that the CGMM answer is correct.

Using CGMM as the stronger benchmarked estimator, the predicted mean crossing is approximately **(11, 2.645, 1.914) m**, and its approximate most-likely crossing is **(11, 2.653, 1.915) m**. These are conditional model predictions, not experimentally measured endpoints.

## 15. Executable workflow

Keep the package files together. Dependencies are NumPy, SciPy, PyArrow, scikit-learn, Matplotlib, and threadpoolctl; Python 3.10 or later is required.

```bash
python -m pip install -r requirements.txt
python verify.py

# Existing accepted dataset: split once without rerunning physics.
python data.py --source kicks_358 --output kicks_split --seed 42

# Fit once; stores both model artifacts from training only.
python train.py --dataset kicks_split --models models --components 16

# Full saved test partition; no fitting here.
python evaluate.py --dataset kicks_split --models models --output metrics.json

# Repeated inference; no dataset or fitting required.
python predict.py --models models --method both --r1 1.50 0.45 0.44 --r2 2.96 0.86 0.74
```

For a new generation run, `python generate.py --power 20 --workers 8 --output kicks_new` automatically saves the accepted train/test partitions. `--limit 200` in evaluation produces a reproducible test subset for a quick operational check; it does not change the saved split. `--include-mode` adds slower mode-error benchmarks. `--draws` controls CGMM Monte Carlo resolution. Existing models fitted before the fixed split must be retrained.

Generation stores chunk checkpoints and keeps raw accepted partitions in addition to split copies. Model directories must be empty at training time to avoid accidental overwrites. Prediction writes a JSON summary and Matplotlib images. Benchmark output is aggregate JSON; this implementation does not save per-shot benchmark records.

## 16. Validation and limits of the evidence

The delivered verification script checks force components, angle directions, signed spins, solver refinement, goal events against a ballistic case, ground termination, rejection behavior, Parquet round trips, split disjointness and repeatability, fingerprint rejection after modification, known MSE values, and known covariance-derived uncertainty values. End-to-end smoke testing exercised generation, fitting, both predictors, benchmarking, and plotting. Those checks establish software consistency, not empirical validation of the force law.

The final results support choosing CGMM's posterior mean for squared-error prediction within this synthetic domain. Adaptive Bayes remains a useful direct reference method and exposes sparse input-space support explicitly. Neither method includes aerodynamic model uncertainty, realistic human launch frequencies, unobserved kick types excluded by the bounds, or empirical tracking-system error. Their apparent precision must be interpreted within those assumptions.

## 17. Sources and audit trail

External sources checked on 8 November 2026. Equations and algorithm descriptions above document the delivered implementation; the sources justify the adopted physical references, bound provenance, and library behavior.

1. [Team 358, *The Best Strategy for Penalty Kicks*, 2022 University Physics Competition](https://www.uphysicsc.com/2022-GM-B-358.pdf). Physical model and constants, spherical launch convention, and motivating upper-corner angle ranges. The final generator uses its selected equations and table values, not a verbatim copy of its appendix solver.
2. [Team 480, *Analysis of the optimal initial velocities and spins of soccer in Penalty Kicks*, 2022](https://www.uphysicsc.com/2022-GM-B-480.pdf). Section 4.2, page 7, for the adopted Cartesian spin intervals. Appendix contains a broader signed vertical-spin search; spin decay from that paper is not used here.
3. [Guinness World Records, *Fastest football kick*](https://www.guinnessworldrecords.com/world-records/fastest-football-kick). Public 129 km/h reference used for the speed cap.
4. [SciPy, `solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html). RK45 order, dense interpolation, output times, and event semantics.
5. [SciPy, `scipy.stats.qmc.Sobol`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html). Scrambling, low-discrepancy sampling, and power-of-two balance guidance.
6. [scikit-learn, `GaussianMixture`](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html). Full covariance fitting, EM configuration, and convergence metadata.

Local evidence: final `kick_pipeline.zip` source; supplied `metrics(1).json`, `prediction.json`, and the two associated heatmaps. The full benchmark source fingerprint is `a90ca8cc9cf32a1cc8ef149ae4f4c5bce2222335fac094401e74eb678e0b4e6c`; training fingerprint is `e5914ba671ad894d23893404c2fb85905df135d7db207113d5350e5174782094`; test fingerprint is `affad316537ec73597c38697f933cdc46aecee8275350a8bfbe8c7e91466956c`.
