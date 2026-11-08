# Projected goal-grid lookup tables for penalty prediction

## 1. Purpose and scope

This method compresses early ball positions into a small number of directional categories and uses simulated examples to associate those categories with eventual goal locations. Its intended use is a simple, interpretable decision aid that could be memorized or tabulated. It replaces the need to perform numerical trajectory inversion during a shot with one or two zone classifications and a lookup.

The experiment compares four predictors using the same training and test split. The first uses the zone at 0.05 s, the second uses the zone at 0.10 s, the third uses their ordered pair, and the control always returns the global training mean. Each predictor also supplies probabilities and spatial uncertainty.

The results in this report come from the supplied `metrics_grid.json`, `prediction_grid.json`, and `prediction_grid.png`. Algorithm details were checked against the delivered `grid.py` and `data.py`. The training model and original generation manifest were not supplied with these results. Consequently, generation settings and smoothing settings below describe the delivered defaults; the output files alone do not independently prove which command-line overrides were used.

The target population consists of simulated airborne shots that enter the goal. The probabilities describe this selected population under its simulation prior. Goal-cell accuracy is not a goalkeeper save rate. The study does not model a goalkeeper's reaction delay, reach, diving dynamics, or visual perception.

## 2. Coordinates and underlying data

Throughout this report, the reporting coordinates are defined by +x toward the goal, +y to the right when facing the goal, and +z upward. The goal lies at x=11 m, with -3.66<y<3.66 and 0<z<2.44. The initial ball center is (0,0,0.11) m.

The stored Team 358 coordinates use X right, Y forward, and Z up. The implemented positional correspondence is

\[
(X,Y,Z)=(y,x,z),\qquad (x,y,z)=(Y,X,Z).
\]

Thus the stored target (X_goal,Z_goal) and reported target (y_goal,z_goal) have the same numerical values. The forward plane coordinate changes its name from Y to x. This is an explicit choice of reporting axes. The force equations and spin components remain in the internal Team 358 convention. The positional relabeling should not be used to transplant a cross-product formula without also respecting the chosen frame's handedness.

The Parquet observations are `x_005,y_005,z_005,x_010,y_010,z_010` in the internal frame. The target is `x_goal,z_goal`, with `y_goal=11`. `predict.py` accepts reporting coordinates by default and swaps the first two components before looking up the internal table. `--frame paper` accepts the stored convention directly. Training and evaluation operate entirely in the stored frame.

### 2.1 Inherited simulation assumptions

The lookup classifier learns from the existing physics dataset. It does not integrate another trajectory at prediction time. The generator adopts constant spin, gravity with buoyancy, quadratic drag, and a Magnus term from the Team-358-style model used earlier in this study. With internal velocity v and spin omega,

\[
\dot{\mathbf r}=\mathbf v,\qquad
\dot{\mathbf v}=-k_D\|\mathbf v\|\mathbf v+k_M(\mathbf v\times\boldsymbol\omega)-g_{\rm eff}\mathbf e_Z,
\]

\[
k_D=\frac{C_D\rho\pi a^2}{2m},\quad
k_M=\frac{8\pi\rho a^3}{3m},\quad
 g_{\rm eff}=g\left(1-\frac{4\pi\rho a^3}{3m}\right).
\]

The adopted constants are m=0.430 kg, a=0.108 m, rho=1.27 kg/m³, C_D=0.4, and g=9.8 m/s². These values and the force model are inherited assumptions, rather than quantities estimated by the lookup tables. Team 358 is the physical-model source; Team 480 motivates the adopted spin component intervals. [1,2]

| Sampled variable | Delivered interval | Interpretation |
|---|---|---|
| Initial speed | sqrt(9.8×11) to 129/3.6 m/s | Approximately 10.383–35.833 m/s |
| Polar angle theta | 0–90 degrees | From internal +Z |
| Azimuth phi | 55–125 degrees | From internal +X toward +Y |
| Spin omega_X | -8 to +8 rad/s | Internal Cartesian component |
| Spin omega_Y | -5 to +5 rad/s | Internal Cartesian component |
| Spin omega_Z | 0 to +15 rad/s | Internal Cartesian component |

Initial velocity is v0(sin(theta)cos(phi), sin(theta)sin(phi), cos(theta)). The lower speed is an ideal vacuum-range reference, obtained from R_max=v0²/g with R=11 m. It does not establish the exact minimum speed of the aerodynamic model. The upper speed is the operational bound adopted from the earlier study's cited 129 km/h record. The angle padding is a modeling choice. Team 480's spin intervals are modeling ranges, not separate human spin records. [1,2,3]

The generator maps scrambled six-dimensional Sobol points into these intervals, giving uniform marginal sampling of the listed parameters. Uniform angles do not imply uniform solid angle. Sobol sampling improves coverage of the launch box; it does not establish a real-player launch distribution. The original run used 2^20 candidate launches. It retained 79,893 shots, rejected 614,151 outside the goal, and rejected 354,532 for reaching the ground first. Filtering changes the distribution from the original independent launch prior to that prior conditional on airborne goal entry. [4]

Trajectories use `solve_ivp` with adaptive RK45, observations at 0.05 and 0.10 s, and an event at the goal plane. Ground contact is the downward crossing of center height a=0.108 m; a 15 s timeout bounds the simulated flight. The stored crossing is obtained from the solver's event interpolation. The accepted center satisfies the rectangular goal limits. The implementation uses center-in-opening criteria rather than subtracting a ball radius from the side and upper boundaries. The RK45 method and event machinery are described in SciPy's documentation. [5]

### 2.2 Fixed split and data integrity

The delivered split procedure sorts records by ID, applies a seeded random permutation, and assigns floor(0.8N) records to training. For N=79,893, this gives 63,914 training records and 15,979 test records. The default split seed is 42. This is a fixed random split, without class stratification.

`data.py` verifies fingerprints of IDs, observations, and crossings against `split.json`. Training loads the training partition only and stores its IDs and split metadata in the model. Evaluation loads the test partition, requires matching split metadata, and rejects any training/test ID overlap. These checks prevent accidental reuse of training shots as test shots. They measure interpolation to held-out shots from the same simulator distribution; they do not establish generalization to a new ball, aerodynamic model, or population of players.

## 3. Geometry of the projected grid

At the goal, divide the 7.32 m width into three 2.44 m columns and the 2.44 m height into three approximately 0.813333 m rows. The interior lateral cuts are -1.22 and +1.22 m. The interior vertical cuts are 2.44/3 and 4.88/3 m.

Extend rays backward from the goal grid to the initial ball center r0=(0,0,0.11). For an observation (x,y,z), the line is

\[
\mathbf r(\lambda)=\mathbf r_0+\lambda[(x,y,z)-\mathbf r_0].
\]

At x=11, lambda=11/x. Therefore

\[
y_p=\frac{11y}{x},\qquad
z_p=0.11+\frac{11(z-0.11)}{x}.
\]

This follows directly from similar triangles. The 0.11 m offset matters because the cone's apex is the initial ball center. At distance x, a goal-height boundary b becomes z=0.11+(x/11)(b-0.11), and a lateral boundary a becomes y=(x/11)a. Thus the internal grid shrinks linearly toward the starting point.

The projection is a directional feature. It is not a ballistic prediction of the crossing. Curved paths with similar early directions can end in different goal cells; the training outcomes estimate that spread. Forward distance must be positive, and the code rejects nonpositive observed forward distances.

### 3.1 Infinite outer zones

Only the two interior cuts per axis are used to assign input zones. The low and high vertical bands and the left and right lateral bands extend to infinity.

| Band | Lateral interval | Vertical interval |
|---|---|---|
| First | y_p<-1.22 | z_p<2.44/3 |
| Second | -1.22<=y_p<1.22 | 2.44/3<=z_p<4.88/3 |
| Third | y_p>=1.22 | z_p>=4.88/3 |

Every observation therefore belongs to one of nine zones, even if its ray intersects outside the goal opening. An early ray can be above the bar while the actual curved trajectory enters the goal. This is compatible with the dataset containing only successful goal-plane crossings.

Exact interior-boundary values belong to the higher band through `searchsorted(..., side='right')`. With row r and column c numbered 0,1,2, the cell ID is k=3r+c. IDs 0–2 are low left/center/right, 3–5 are middle left/center/right, and 6–8 are high left/center/right. Actual target cells use the same cuts, but retained targets lie within the goal frame.

## 4. Why use a lookup table?

A projected cell compresses three measured coordinates into one label. A single-look table has nine entries and can be inspected directly. It is inexpensive to query, its support counts are visible, and its decision can be expressed as a location on the goal. This suits the requested goal of a simple rule a goalkeeper could learn.

The cost is information loss. Within a zone, the method discards exact bearing, forward distance, speed cues, and the precise change in position. All observations with the same key return the same distribution. A pair retains some direction-change information, but changes that stay within the same zones remain invisible.

The grid is defined relative to the penalty-point apex. A goalkeeper observing from the goal has a different viewpoint. Turning the geometric feature into a visual skill requires a perspective mapping, reliable timing, and practical validation. The present experiment demonstrates a compact statistical rule using measured coordinates. Human execution within 50–100 ms has not been measured.

## 5. Training the four predictors

Let training shot i have projected zones A_i and B_i, crossing q_i=(y_i,z_i), and target cell C_i. The four keys are A_i, B_i, 9A_i+B_i, and a single constant key. There are 9, 9, 81, and 1 possible rows respectively.

For row g, define I_ig=1 if shot i belongs to that row and 0 otherwise, with n_g=sum_i I_ig. Raw frequencies estimate the conditional categorical distribution by counts. The coordinate distribution is the empirical distribution of the corresponding crossings.

To stabilize sparse groups, the implementation uses weights

\[
w_{ig}=\frac{I_{ig}+\alpha b_{ig}}{n_g+\alpha},\qquad \sum_i b_{ig}=1.
\]

The delivered default is alpha=5, configurable through `--smoothing`. This is a pseudo-count strength chosen for modest shrinkage, without a reported validation search. It has little effect on a group containing thousands of shots and a large effect on a group containing only a few.

For t1 and t2, b_ig=1/N_train. For a paired row, b_ig is uniform over the raw training shots sharing its t2 label, if any exist; otherwise it is global. Thus an uncertain paired row borrows information from the later observation. It does not assume the two timestamps are independent. The parent is the raw t2 distribution, not the already-smoothed t2 output.

For an empty row, the code uses the parent distribution directly, including when alpha=0. For a nonempty row with alpha=0, it uses raw empirical frequencies. The `count` field records actual row membership, excluding pseudo-counts. `fallback` is true only when the row is empty. A zero fallback rate does not show that smoothing had no effect.

The control has I_ig=1 for every training shot and global parent weights, so its normalized weights are exactly 1/N_train for any alpha. No optimization iterations or gradient training are needed. The model stores precomputed summaries and can predict without reopening the dataset.

### 5.1 Cell probabilities and point estimates

For each row,

\[
p_{gk}=\sum_iw_{ig}\mathbf1(C_i=k),\qquad
\boldsymbol\mu_g=\sum_iw_{ig}\mathbf q_i.
\]

The predicted cell is argmax_k p_gk. This minimizes expected zero-one cell loss under the fitted distribution. Ties use the first ID. The mean crossing minimizes expected squared coordinate error because

\[
E\|Q-a\|^2=E\|Q-EQ\|^2+\|a-EQ\|^2.
\]

A cell probability mode and the cell containing the mean can differ. The figure's orange rectangle marks the categorical decision; its cross marks the coordinate mean. A probability of 33% can be the largest of nine probabilities without implying high certainty.

The control's hard classification is defined as the cell containing its mean. Its probability scores use the full global cell frequencies. The code also stores a global modal-cell field, which is conceptually distinct. In the supplied run both identify middle center.

## 6. Why the global-mean control is necessary

The control answers how well one can do using only the overall simulated goal distribution. Its constant mean is the best constant point predictor for training squared error. Its empirical probabilities are the corresponding uninformed-by-observations baseline for categorical probability prediction.

Comparing against this baseline quantifies information gained by observing the ball. A uniformly random cell baseline would ignore unequal class frequencies and would not provide a principled coordinate estimate. The global control accounts for the simulation's actual target distribution, including selection effects.

The control also demonstrates why uncertainty coverage alone is insufficient. It can cover approximately 95% of crossings using a very broad ellipse, despite having no shot-specific information. Coverage must be considered together with area, cell-set size, accuracy, and error. The control's large spread is meaningful predictive uncertainty about an unspecified retained shot.

## 7. Where the uncertainty comes from

For a fixed input key, many training launches remain possible. Different velocities and spin vectors can produce the same coarse labels while crossing at different positions. The fitted distribution preserves this variation. The coarseness of the grid is itself an important source of conditional uncertainty.

The stored covariance is

\[
\Sigma_g=\sum_iw_{ig}(\mathbf q_i-\boldsymbol\mu_g)(\mathbf q_i-\boldsymbol\mu_g)^T+10^{-10}I\quad[\mathrm{m}^2].
\]

The weights sum to one. This is the second moment of the fitted empirical predictive distribution; the implementation does not apply an N-1 correction. The small diagonal floor permits numerical inversion in degenerate groups and is negligible at metre scales.

Directional standard deviations are sigma_y=sqrt(Sigma_11) and sigma_z=sqrt(Sigma_22). The off-diagonal term describes joint lateral-height variation. Positive covariance tilts the main axis toward higher height at more positive lateral positions; negative covariance tilts it the other way.

Radial RMS uncertainty is

\[
\sigma_{\rm radial}=\sqrt{\operatorname{tr}\Sigma_g}
\approx\sqrt{E_g\|Q-\boldsymbol\mu_g\|^2}.
\]

It is the RMS distance from the predicted mean, including the negligible covariance floor. It is not the standard deviation of the scalar distance and does not by itself specify a 95% circle. The implemented outputs do not calculate a 95% radial-radius quantile or marginal quantile intervals.

These spreads describe variation in future crossings within a lookup group. They are not standard errors of the estimated mean, so they are not divided by sqrt(n_g). More training shots can estimate a broad distribution more precisely without making the underlying spread small. Finite-training uncertainty, simulator mismatch, and human perceptual errors are not separately quantified.

### 7.1 Prediction ellipses

For each training crossing, compute its squared Mahalanobis distance from the row mean,

\[
d_{ig}^2=(\mathbf q_i-\boldsymbol\mu_g)^T\Sigma_g^{-1}(\mathbf q_i-\boldsymbol\mu_g).
\]

Mahalanobis distance scales departures by directional covariance. Its contours stretch along directions of greater spread and rotate with correlated variation. [6]

Sort these distances and accumulate the same training weights. Let c_g be the first distance at which cumulative weight reaches 0.95. The nominal 95% region is

\[
E_g=\{q:(q-\mu_g)^T\Sigma_g^{-1}(q-\mu_g)\le c_g\}.
\]

This uses an empirical cutoff for each row. It does not assume that the crossings follow a bivariate Gaussian or automatically use a chi-square cutoff. The ellipse is a compact geometric summary of a distribution that can be bounded, irregular, or multimodal. It may include low-probability areas between clusters.

If Sigma has eigenvalues lambda_1,lambda_2 and corresponding orthonormal eigenvectors, the semiaxes are sqrt(c_g lambda_1) and sqrt(c_g lambda_2). The eigenvectors give their directions. The full ellipse area is

\[
A_g=\pi c_g\sqrt{\det\Sigma_g}\quad[\mathrm{m}^2].
\]

The training-based 95% construction does not guarantee exact test coverage. The same training observations determine the center, covariance, and cutoff. Held-out coverage is therefore essential, particularly for sparse rows. This is not a conformal calibration procedure.

The PNG clips the ellipse visually at the goal frame. Evaluation uses the full ellipse equation and reports full ellipse area. Area is not the area of the intersection with the goal. Because test targets lie inside the goal, clipping does not alter their inclusion, but the reported area may exceed the goal area of 17.8608 m². That occurs for the global control.

### 7.2 Categorical uncertainty sets

The code sorts cells by decreasing predicted probability and adds cells until their cumulative probability first reaches 0.95. Ties use ID order. This produces a minimum-cardinality set for that fitted probability vector, with the chosen tie rule. Its probability can exceed 95% because whole cells are added.

The evaluation reports both empirical set coverage and average number of included cells. A nine-cell set necessarily covers every retained shot, so reporting coverage without its size would conceal a completely uninformative answer. A cell set and an ellipse summarize different aspects of the same weighted crossing distribution and need not trace the same boundary.

## 8. Evaluation protocol and metric definitions

The full evaluation uses all 15,979 test shots. All four predictors see the same shots and their true retained crossings. It reports clean inputs and one shared Gaussian perturbation realization with seed 42. Noise is added independently to the six observation coordinates, before projection, with

\[
\sigma_{\rm obs}=0.005/\sqrt3=0.0028867513\ \mathrm m.
\]

This matches the variance of a uniform rounding error between -0.005 and +0.005 m. It is a Gaussian approximation to centimetre rounding. Gaussian noise is unbounded; it is not an exact rounding operation. Target coordinates remain unchanged. Training uses clean observations, and prediction performs hard zone assignment; it does not integrate over uncertainty in zone membership.

Small noise can move a sample across an interior boundary and select another row. The noisy benchmark tests this effect. It does not establish robustness to human visual errors, uncertain timestamps, or correlated tracking errors. The single seed is a reproducible perturbation trial, not an estimate of variability across repeated noise realizations.

Let n be test size, C_j the true cell, hat C_j the hard prediction, q_j the crossing, and mu_j its predicted mean. Let r(k)=floor(k/3) and c(k)=k mod 3.

| Metric | Definition and reason |
|---|---|
| Exact-cell accuracy | Mean of 1(hat C=C). Primary score for selecting the correct one of nine regions. |
| Row accuracy | Mean of 1(r(hat C)=r(C)). Measures vertical-band identification. |
| Column accuracy | Mean of 1(c(hat C)=c(C)). Measures lateral-band identification. |
| Mean grid distance | Mean of abs(r(hat C)-r(C))+abs(c(hat C)-c(C)). Distinguishes adjacent mistakes from opposite-corner mistakes; range 0–4. |
| Macro recall | Average of per-class true-positive fractions over classes present in test. Prevents common classes from dominating the score. |
| Confusion matrix | Counts with true classes on rows and predicted classes on columns, ordered 0–8. Reveals systematic missed classes. |
| Multiclass Brier score | Mean over shots of sum_k (p_jk-1(C_j=k))². Uses the full probability vector, rewarding probabilities consistent with outcomes; range 0–2. |
| 95% cell-set coverage and size | Fraction whose true class is included, and mean number of included cells. Measures uncertainty coverage alongside informativeness. |
| Axis MSE | Mean of (mu_y-y)² and (mu_z-z)² in m². Separates lateral and height error. |
| Radial MSE | Sum of axis MSEs, equivalently mean squared Euclidean error. |
| Radial RMSE | Square root of radial MSE, in metres. Gives a spatial error scale comparable with prior coordinate predictors. |
| Mean predicted axis/radial uncertainty | Average of each shot's sigma values. Summarizes the spread the tables report. |
| 95% ellipse coverage | Fraction satisfying the predicted ellipse inequality. Evaluates the joint spatial region. |
| Mean ellipse area | Average full ellipse area in m². Measures sharpness, interpreted alongside coverage. |
| Fallback rate | Fraction assigned to a row with no direct training shots. Exposes dependence on broader parent distributions. |

Standard classification, regression, and probability scores follow the definitions in the cited metric documentation. The grid-distance construction and combination of metrics are task-specific choices. Brier score measures overall probabilistic quality, combining calibration and discrimination; it is not a standalone proof of calibration. [7,8]

Average predicted radial uncertainty is mean_j sqrt(trace(Sigma_j)). Radial RMSE is sqrt(mean_j ||q_j-mu_j||²). These different orders of averaging mean that matching values are not required even for a well-calibrated predictor. Joint coverage is an aggregate measure and can hide undercoverage in individual input zones.

The evaluation writes a compact aggregate JSON with four confusion matrices per scenario, rather than thousands of individual prediction records. It loads test arrays in memory but retains no per-shot JSON log. The current script does not report confidence intervals for the benchmark scores or statistical significance of pairwise differences.

## 9. Full held-out results

Values below are computed directly from the supplied JSON. Percentages are rounded to two decimal places; errors retain three decimals.

### 9.1 Clean observations

| Predictor | Cell accuracy | Row accuracy | Column accuracy | Macro recall | Grid distance | Brier |
|---|---:|---:|---:|---:|---:|---:|
| t1 | 36.81% | 46.51% | 76.62% | 37.13% | 0.9632 | 0.73526 |
| t2 | 38.56% | 47.71% | 78.34% | 38.95% | 0.9213 | 0.72179 |
| both | 38.55% | 47.71% | 78.34% | 38.93% | 0.9205 | 0.72064 |
| control | 12.19% | 34.87% | 34.50% | 11.11% | 1.3063 | 0.88866 |

| Predictor | Lateral MSE (m²) | Height MSE (m²) | Radial RMSE (m) | Ellipse coverage | Ellipse area (m²) | Cell-set coverage | Cell-set size |
|---|---:|---:|---:|---:|---:|---:|---:|
| t1 | 1.327 | 0.356 | 1.297 | 94.82% | 11.070 | 97.41% | 6.124 |
| t2 | 1.250 | 0.346 | 1.263 | 94.79% | 10.763 | 97.58% | 6.049 |
| both | 1.248 | 0.346 | 1.263 | 94.81% | 10.720 | 97.57% | 6.009 |
| control | 4.282 | 0.453 | 2.176 | 95.04% | 18.989 | 100.00% | 9.000 |

### 9.2 Noisy observations

| Predictor | Cell accuracy | Row accuracy | Column accuracy | Macro recall | Grid distance | Brier |
|---|---:|---:|---:|---:|---:|---:|
| t1 | 36.77% | 46.50% | 76.56% | 37.08% | 0.9638 | 0.73549 |
| t2 | 38.61% | 47.70% | 78.43% | 39.00% | 0.9205 | 0.72135 |
| both | 38.58% | 47.68% | 78.43% | 38.96% | 0.9199 | 0.72041 |
| control | 12.19% | 34.87% | 34.50% | 11.11% | 1.3063 | 0.88866 |

| Predictor | Lateral MSE (m²) | Height MSE (m²) | Radial RMSE (m) | Ellipse coverage | Ellipse area (m²) | Cell-set coverage | Cell-set size |
|---|---:|---:|---:|---:|---:|---:|---:|
| t1 | 1.329 | 0.356 | 1.298 | 94.74% | 11.069 | 97.43% | 6.122 |
| t2 | 1.246 | 0.346 | 1.262 | 94.82% | 10.761 | 97.57% | 6.048 |
| both | 1.245 | 0.346 | 1.261 | 94.81% | 10.706 | 97.51% | 5.998 |
| control | 4.282 | 0.453 | 2.176 | 95.04% | 18.989 | 100.00% | 9.000 |

### 9.3 What the results support

All observation-based tables improve substantially over the global control. With clean observations, t2 raises exact-cell accuracy from 12.19% to 38.56% and reduces radial RMSE from 2.176 m to 1.263 m. Its lateral MSE drops from 4.282 to 1.250 m², while height MSE drops from 0.453 to 0.346 m². The directional classification is correspondingly stronger across columns than across rows. Thus the coarse feature carries more useful lateral information than height information in this simulated population.

The t1 table gives an earlier decision by 50 ms. Its clean accuracy is 36.81%, compared with 38.56% at t2. Waiting provides a 1.75 percentage-point improvement in cell accuracy, but the experiment does not value the lost response time in terms of actual saves. That tradeoff requires a goalkeeper motion model or human study.

The paired table has almost the same clean accuracy as t2, differing by one correctly classified shot out of 15,979. Its radial RMSE improves by approximately 0.00067 m and Brier score decreases by about 0.00115. These are small descriptive improvements, with no significance test reported. On this benchmark, the nine-entry t2 table is a defensible compact choice if the decision is made at 0.10 s. The paired table remains useful for studying uncommon zone transitions and for slightly refined probability outputs.

The confusion matrices show that t1 and t2 never choose a middle-height cell as their top class. Their recall for all three middle-row classes is therefore zero. The paired table chooses middle center occasionally, but never middle left or middle right in the clean test. These tables still assign substantial probability to middle-height outcomes. A broad conditional distribution can have its largest individual cell probability in the high row while retaining much of its total mass in the middle row. Reporting only overall accuracy would hide this failure of the hard decision rule.

Ellipse coverage is approximately 94.8% for all three informed predictors and 95.0% for the control. The informed ellipses are smaller, averaging about 10.7–11.1 m² rather than 19.0 m². These are broad regions relative to the 17.8608 m² opening. Near-nominal coverage should be interpreted together with this limited precision. The 95% categorical sets contain approximately six cells, with about 97.4–97.6% test coverage because whole cells are added. The global control includes all nine cells and necessarily reaches 100% cell-set coverage.

All clean test queries have direct training support. Under noisy inputs, two paired queries fall back to their parent distribution, a rate of 0.0125%; the other predictors have zero fallback. The small changes in noisy scores indicate stability to the particular millimetre-scale perturbation tested. Slight improvements in some noisy scores do not establish that measurement noise helps prediction. Individual boundary crossings and finite test sampling can move scores in either direction.

## 10. Supplied penalty example

The observations in the reporting frame are (1.50,0.45,0.44) m at 0.05 s and (2.96,0.86,0.74) m at 0.10 s. Their projected locations are (3.300,2.530) m and approximately (3.196,2.451) m. Both are classified high right. The projections being above the bar do not imply the actual trajectory misses; the high input band extends upward without a bound.

The predictions below reproduce the uploaded original-shot JSON under the declared +y-right convention.

| Predictor | Training count | Mean y (m) | Mean z (m) | sigma_y (m) | sigma_z (m) | Radial RMS (m) | High-right probability |
|---|---:|---:|---:|---:|---:|---:|---:|
| t1 | 17954 | 2.0648 | 1.3827 | 1.1580 | 0.6404 | 1.3233 | 31.85% |
| t2 | 17466 | 2.0879 | 1.4000 | 1.1404 | 0.6372 | 1.3063 | 33.20% |
| both | 17272 | 2.0865 | 1.3999 | 1.1440 | 0.6372 | 1.3095 | 33.11% |
| control | 63914 | -0.0028 | 1.2557 | 2.0615 | 0.6723 | 2.1684 | 10.90% |

For t2, high right has probability 33.20%, middle right 30.75%, and low right 16.16%. The right-column total is approximately 80.11%. This is a stronger statement about lateral side than about the precise height band. The mean crossing (2.0879,1.4000) m lies in middle right even though the most probable individual cell is high right.

The paired high-right/high-right group contains 17,272 shots, compared with 17,466 in the t2 high-right group. For this query, adding t1 removes relatively few training examples, helping explain why the two distributions are so similar. The training counts provide direct evidence for this explanation without requiring independence assumptions between timestamps.

The paired covariance is approximately

[
Sigma=egin{pmatrix}1.308799&0.028991\\0.028991&0.406015end{pmatrix} mathrm{m}^2.
]

Its lateral spread is substantially larger than its height spread. The small positive off-diagonal term gives a slight upward tilt of the long axis. Its empirical squared-distance cutoff is 5.078606 and its full ellipse area is 11.6214 m². The plotted curve is generated from these stored quantities, rather than hand-selected to surround a chosen cell.

The control mean is approximately (-0.00275,1.25565) m, with radial RMS uncertainty 2.16836 m. This expresses the spread across all 63,914 training crossings. The informed tables narrow the example's radial spread to roughly 1.31 m, while leaving substantial height ambiguity. Thousands of supporting shots make the broadness credible as a feature of this coarse conditioning scheme; support count alone does not make the conditional outcome concentrated.

The image colors show probability mass per cell, with a common scale across all four panels. They do not display a fitted continuous probability density within each cell. The orange outline is the selected class, the cross is the mean, and the red curve is the nominal 95% ellipse. All panels use the same metre scale and goal dimensions.

## 11. Reproduction and interpretation limits

Run the following commands from the extracted program folder.

```bash
python -m pip install numpy pyarrow matplotlib
python train.py --dataset ../kicks_split --output grid_models
python predict.py --model grid_models/model.json
python evaluate.py --dataset ../kicks_split --model grid_models/model.json --output metrics_grid.json
```

Training writes `model.json` and the four CSV tables. Prediction writes `prediction_grid.json` and `prediction_grid.png`. Evaluation reports clean and noisy aggregate metrics. `--smoothing 0` requests raw frequencies for supported rows, `--limit 200` selects a pilot test prefix, and `--png` changes the prediction image path. A final full benchmark uses the complete test partition. Numerical values in this report refer to the supplied full run.

The largest limitation is the compression of continuous motion into nine zones. Two shots anywhere inside the same zones are indistinguishable to this method. The lookup probabilities also inherit the simulator's launch prior, positive-only omega_Z interval, constant aerodynamic coefficients, and selection of airborne goals. The reported test performance establishes internal performance on that simulation distribution. Its transfer to real players and human judgments remains unmeasured.

The nominal uncertainty includes observed training variation and smoothing. It excludes a separate estimate of model discrepancy, uncertainty in the training distribution, and uncertain timestamps. Millimetre Gaussian perturbations do not encompass all of these effects. In addition, aggregate coverage does not guarantee 95% coverage within every input group, particularly sparse ones.

The scheme is best understood as a compact direction-based statistical aid. The evaluation supports a large gain over the global-mean control and only a small gain from remembering both zones over using t2 alone. Any practical recommendation about diving should additionally account for time remaining, goalkeeper position, reach, and the unequal costs of different errors. Those factors are outside the present experiment.

## 12. Sources and computation provenance

The project-specific algorithms were derived and implemented for this study. No external source is claimed for the exact nine-zone geometry, alpha=5 smoothing strength, parent-selection rule, or empirical ellipse cutoff procedure. Their formulas and reasons are specified above so that the calculations are reproducible. Statistical references support the general definitions, rather than the particular numerical results.

| Component | Provenance |
|---|---|
| Goal dimensions, release height, observation times | Supplied competition problem |
| Internal dynamics and constants | Adopted Team 358 model and earlier generator implementation |
| Spin component intervals | Team 480 modeling ranges |
| Sobol generation and RK45 events | Existing generator, with SciPy algorithm documentation |
| Projection and internal grid cuts | Similar-triangle derivation and the requested equal thirds |
| Infinite outer zones | User-specified input categorization |
| Counts, smoothing, covariance, empirical cutoffs | Delivered `grid.py` implementation |
| Split and integrity checks | Delivered `data.py` implementation |
| Classification, Brier, and error definitions | Equations above and standard metric documentation |
| Numerical benchmark and example | Uploaded `metrics_grid.json` and `prediction_grid.json` |
| Rendering | Uploaded `prediction_grid.png` and Matplotlib plotting code |

1. University Physics Competition, Team 358, 2022 Problem B. Physical model adopted in the original simulation. [Paper](https://www.uphysicsc.com/2022-GM-B-358.pdf)
2. University Physics Competition, Team 480, 2022 Problem B, section 4.2. Source of the selected spin intervals. [Paper](https://www.uphysicsc.com/2022-GM-B-480.pdf)
3. Guinness World Records, “Fastest football kick.” Source cited by the earlier generator for its operational speed upper bound. [Record listing](https://www.guinnessworldrecords.com/world-records/fastest-football-kick)
4. SciPy documentation, `scipy.stats.qmc.Sobol`. Reference for Sobol sequences and power-of-two sampling. [Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html)
5. SciPy documentation, `scipy.integrate.solve_ivp`. Reference for RK45 and event detection. [Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)
6. NIST Dataplot reference, “Matrix Distance.” Definition of covariance-scaled Mahalanobis distance. [Reference](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/matrdist.htm)
7. Scikit-learn documentation, “Metrics and scoring” and `brier_score_loss`. Standard metric definitions and multiclass Brier convention. [Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html) · [Brier score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html)
8. Scikit-learn documentation, “Probability calibration.” Explains that probabilistic scores reflect calibration and discrimination jointly. [Documentation](https://scikit-learn.org/stable/modules/calibration.html)

References 1–5 provide inherited simulation background. References 6–8 were consulted for the statistical explanation. Numerical results are traced to the supplied files, independently of these external references.
