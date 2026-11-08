"""Condition the airborne goal library on two measured ball positions."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.ndimage import gaussian_filter
from scipy.special import logsumexp

OBSERVATIONS = [f"{axis}_{time}" for time in ("005", "010") for axis in "xyz"]
CROSSING = ["x_goal", "y_goal", "z_goal"]


def load_kicks(path):
    import pyarrow.dataset as ds

    files = sorted(path.glob("part-*.parquet")) if path.is_dir() else [path]
    if not files:
        raise ValueError("No part-*.parquet files found.")
    table = ds.dataset(files, format="parquet").to_table(columns=OBSERVATIONS+CROSSING)
    if not table.num_rows:
        raise ValueError("The dataset contains no retained kicks.")
    data = np.column_stack([table[name].to_numpy() for name in table.column_names])
    if not np.isfinite(data).all() or not np.allclose(data[:, 7], 11, rtol=0, atol=1e-8):
        raise ValueError("Expected finite observations and Team 358 crossings at y_goal=11.")
    return data[:, :6], data[:, [6, 8]]


def weighted_quantile(values, weights, probabilities):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    cumulative[-1] = 1
    return values[order[np.searchsorted(cumulative, probabilities)]]


def density_mode(points, weights, bandwidth, mean):
    # Evaluate the same Gaussian kernel at each retained crossing.
    def objective(location):
        offset = location-points
        terms = weights*np.exp(-np.sum(offset**2, axis=1)/(2*bandwidth**2))
        gradient = (terms @ offset)/bandwidth**2
        return -terms.sum(), gradient

    # Include starts from distinct goal regions as well as the posterior mean.
    cells = np.floor(points/bandwidth).astype(int)
    _, labels = np.unique(cells, axis=0, return_inverse=True)
    mass = np.bincount(labels, weights=weights)
    starts = [mean]
    for label in np.argsort(mass)[-8:]:
        selected = labels == label
        if mass[label] > 0:
            starts.append(np.average(points[selected], axis=0, weights=weights[selected]))
    fits = [minimize(objective, start, jac=True, method="L-BFGS-B",
                     bounds=[(-3.66, 3.66), (0, 2.44)]) for start in starts]
    return min(fits, key=lambda fit: fit.fun).x


def predict(observations, points, measured, sigma, min_ess, adapt, mode_bandwidth):
    error = observations-measured
    distance2 = np.sum(error**2, axis=1)
    nearest = np.argmin(distance2)

    def posterior(scale):
        weight = np.exp(-(distance2-distance2[nearest])/(2*scale**2))
        weight /= weight.sum()
        return weight, float(1/np.dot(weight, weight))

    weights, raw_ess = posterior(sigma)
    effective_sigma = sigma
    if adapt and raw_ess < min_ess < len(points):
        upper = max(sigma*2, float(np.sqrt(distance2.max())))
        while posterior(upper)[1] < min_ess:
            upper *= 2
        effective_sigma = brentq(lambda s: posterior(s)[1]-min_ess, sigma, upper)
        weights, _ = posterior(effective_sigma)

    ess = float(1/np.dot(weights, weights))
    warnings = []
    if raw_ess < min_ess:
        warnings.append("Low support at the requested measurement noise.")
    if effective_sigma > sigma:
        warnings.append("Adaptive smoothing broadens the likelihood and can bias the prediction.")
    nearest_sigma = float(np.sqrt(distance2[nearest])/sigma)
    if nearest_sigma > 5:
        warnings.append("The closest simulated observation is over five combined noise units away.")
    if min_ess >= len(points) and adapt and raw_ess < min_ess:
        warnings.append("Requested effective sample size exceeds usable library support.")

    report = {
        "conditioned_on": "airborne goal entry under the generator's launch distribution",
        "goal_coordinates": ["x_right_m", "z_up_m"],
        "retained_shots": len(points),
        "measurement_sigma_m": sigma,
        "effective_sigma_m": effective_sigma,
        "additional_smoothing_sigma_m": float(np.sqrt(effective_sigma**2-sigma**2)),
        "measurement_only_ess": raw_ess,
        "effective_sample_size": ess,
        "nearest_observation_distance_m": float(np.sqrt(distance2[nearest])),
        "nearest_observation_error_m": error[nearest].tolist(),
        "centimetre_rounding_box_matches": int(np.count_nonzero(np.all(abs(error) <= .005, axis=1))),
        "warnings": warnings,
    }
    report.update(summarize(points, weights, mode_bandwidth))
    return report, weights


def summarize(points, weights, mode_bandwidth):
    ess = float(1/np.dot(weights, weights))
    mean = weights @ points
    centered = points-mean
    covariance = (centered.T*weights) @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    joint = None
    if ess >= 3 and eigenvalues[0] > eigenvalues[-1]*1e-12:
        radii2 = np.sum((centered @ eigenvectors)**2/eigenvalues, axis=1)
        threshold = weighted_quantile(radii2, weights, [.95])[0]
        joint = {"center": mean.tolist(),
                 "semi_axes_m": np.sqrt(threshold*eigenvalues[::-1]).tolist(),
                 "major_axis_unit_vector": eigenvectors[:, -1].tolist()}

    radius = np.linalg.norm(centered, axis=1)
    return {
        "sigma_horizontal_m": float(np.sqrt(covariance[0,0])),
        "sigma_vertical_m": float(np.sqrt(covariance[1,1])),
        "radial_rms_uncertainty_m": float(np.sqrt(np.trace(covariance))),
        "radial_95_radius_m": float(weighted_quantile(radius, weights, [.95])[0]),
        "mean_crossing_m": mean.tolist(),
        "approximate_mode_crossing_m": (density_mode(points, weights, mode_bandwidth, mean).tolist() if mode_bandwidth else None),
        "mode_bandwidth_m": mode_bandwidth,
        "covariance_m2": covariance.tolist(),
        "marginal_95_percent_m": {
            axis: weighted_quantile(points[:, i], weights, [.025, .975]).tolist()
            for i, axis in enumerate(("x_right", "z_up"))
        },
        "joint_95_percent_ellipse": joint,
    }

def predict_cgmm(model_path, observations, measured, sigma, bandwidth, draws_count=10000, seed=42):
    with np.load(model_path, allow_pickle=False) as saved:
        try:
            if saved["version"].item() != 1:
                raise ValueError("Unsupported CGMM model version.")
            center, scale = saved["center"], saved["scale"]
            priors, fitted_means = saved["weights"], saved["means"]
            fitted_covariances = saved["covariances"]
            converged = bool(saved["converged"].item())
            retained_shots = int(saved["retained_shots"].item())
        except KeyError as exc:
            raise ValueError(f"Missing CGMM model field {exc}.") from exc
    residual = (measured-center[:6])/scale[:6]-fitted_means[:, :6]
    noise = np.diag((sigma/scale[:6])**2)
    log_weights, means, covariances = [], [], []
    for prior, mean, covariance, delta in zip(
            priors, fitted_means, fitted_covariances, residual):
        observed_cov = covariance[:6, :6]+noise
        gain = np.linalg.solve(observed_cov, covariance[:6, 6:]).T
        log_weights.append(np.log(prior)-.5*(np.linalg.slogdet(observed_cov)[1]
                                            +delta @ np.linalg.solve(observed_cov, delta)))
        means.append(mean[6:]+gain @ delta)
        conditional = covariance[6:, 6:]-gain @ covariance[:6, 6:]
        covariances.append((conditional+conditional.T)/2)

    component_weights = np.exp(log_weights-logsumexp(log_weights))
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(draws_count, component_weights)
    draws = np.concatenate([rng.multivariate_normal(mean, cov, size=count)
                            for mean, cov, count in zip(means, covariances, counts) if count])
    draws = center[6:]+draws*scale[6:]
    inside = (abs(draws[:, 0]) < 3.66) & (draws[:, 1] > 0) & (draws[:, 1] < 2.44)
    crossing = draws[inside]
    if not len(crossing):
        raise ValueError("The CGMM put no sampled crossings inside the goal for these observations.")
    weights = np.full(len(crossing), 1/len(crossing))
    nearest = (float(np.linalg.norm(observations-measured, axis=1).min())
               if observations is not None else None)
    warnings = []
    if nearest is not None and nearest > 5*sigma:
        warnings.append("The closest simulated observation is over five combined noise units away.")
    if not converged:
        warnings.append("The CGMM fit did not converge.")
    if inside.mean() < .95:
        warnings.append("The fitted mixture places appreciable mass outside the goal.")
    report = {
        "conditioned_on": "airborne goal entry under the generator's launch distribution",
        "goal_coordinates": ["x_right_m", "z_up_m"],
        "retained_shots": retained_shots,
        "measurement_sigma_m": sigma,
        "components": len(priors),
        "fit_converged": converged,
        "nearest_observation_distance_m": nearest,
        "cgmm_draw_fraction_inside_goal": float(inside.mean()),
        "cgmm_draws_used": len(crossing),
        "warnings": warnings,
    }
    report.update(summarize(crossing, weights, bandwidth))
    return report, crossing, weights


def plot_crossing(points, weights, report, method, frame, path, show):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    sign = 1 if frame == "paper" else -1
    horizontal = points[:, 0]*sign
    bins = (366, 122)
    mass, x_edges, z_edges = np.histogram2d(
        horizontal, points[:, 1], bins=bins, range=[[-3.66, 3.66], [0, 2.44]],
        weights=weights)
    step = x_edges[1]-x_edges[0]
    density = gaussian_filter(mass, report["mode_bandwidth_m"]/step, mode="constant")
    density /= density.sum()*step**2

    fig, ax = plt.subplots(figsize=(10, 4.3), layout="constrained")
    heat = ax.pcolormesh(x_edges, z_edges, density.T, cmap="viridis", shading="flat")
    fig.colorbar(heat, ax=ax, label="Smoothed density (m⁻²)")
    mean_x, mean_z = report["mean_crossing_m"]
    mode_x, mode_z = report["approximate_mode_crossing_m"]
    ax.scatter(sign*mode_x, mode_z, marker="*", s=155, color="white",
               edgecolors="black", label="Most likely crossing", zorder=4)
    ax.scatter(sign*mean_x, mean_z, marker="x", s=60, color="orange",
               label="Mean crossing", zorder=4)
    ellipse = report["joint_95_percent_ellipse"]
    if ellipse:
        major, minor = ellipse["semi_axes_m"]
        direction = ellipse["major_axis_unit_vector"]
        angle = np.degrees(np.arctan2(direction[1], sign*direction[0]))
        ax.add_patch(Ellipse((sign*mean_x, mean_z), 2*major, 2*minor, angle=angle,
                             fill=False, color="orange", lw=1.7, label="Joint 95% ellipse"))
    else:
        ax.text(.02, .04, "Insufficient support for a joint ellipse",
                transform=ax.transAxes, color="white")
    ax.set(xlim=(-3.66, 3.66), ylim=(0, 2.44), aspect="equal",
           xlabel="x (m, right)" if frame == "paper" else "y (m, left)",
           ylabel="z (m, up)", title=f"{method.upper()} crossing conditional on goal entry")
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)

