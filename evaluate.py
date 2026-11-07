"""Compare saved-kick predictors on held-out simulated goals.

Place this file beside predict.py. Example:
    python evaluate.py --dataset kicks_358 --test-size 200 --adaptive
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

import predict


def score(truth, estimates):
    if not estimates:
        return None
    errors = np.asarray(estimates)-np.asarray(truth)
    per_axis = np.mean(errors**2, axis=0)
    return {
        "mse_x_m2": float(per_axis[0]),
        "mse_z_m2": float(per_axis[1]),
        "mse_per_coordinate_m2": float(per_axis.mean()),
        "mean_squared_crossing_distance_m2": float(per_axis.sum()),
        "radial_rmse_m": float(np.sqrt(per_axis.sum())),
    }


def evaluate(args):
    observations, crossings = predict.load_kicks(args.dataset)
    if args.test_size >= len(crossings):
        raise ValueError("--test-size must leave training rows.")
    if args.method in ("cgmm", "both") and len(crossings)-args.test_size < max(2, args.components):
        raise ValueError("Too few training rows for the requested CGMM.")
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(crossings))
    test, train = order[:args.test_size], order[args.test_size:]
    train_obs, train_crossings = observations[train], crossings[train]
    queries = observations[test].copy()
    if not args.clean:
        queries += rng.normal(0, args.sigma, size=queries.shape)

    methods = ("bayes", "cgmm") if args.method == "both" else (args.method,)
    results = {}
    with TemporaryDirectory(prefix="kick_eval_") as directory, threadpool_limits(limits=1):
        model = Path(directory)/"holdout.npz"
        if "cgmm" in methods:
            parquet = Path(directory)/"train.parquet"
            matrix = np.column_stack((train_obs, train_crossings[:, 0],
                                      np.full(len(train), 11.), train_crossings[:, 1]))
            columns = predict.OBSERVATIONS+predict.CROSSING
            pq.write_table(pa.table(dict(zip(columns, matrix.T))), parquet)
            # Use the production trainer with only the training partition.
            import os
            env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1")
            subprocess.run([sys.executable, predict.__file__, "train",
                            "--dataset", str(parquet), "--model", str(model),
                            "--components", str(args.components)], env=env,
                           stdout=sys.stderr, check=True)

        for method in methods:
            truth, means, modes, ess = [], [], [], []
            covered = 0
            ellipse_count = 0
            failures = []
            converged = None
            for number, (query, actual) in enumerate(zip(queries, crossings[test]), 1):
                try:
                    if method == "bayes":
                        report, _ = predict.predict(train_obs, train_crossings, query,
                                                    args.sigma, args.min_ess, args.adaptive, .05)
                        ess.append(report["measurement_only_ess"])
                    else:
                        report, _, _ = predict.predict_cgmm(model, None, query, args.sigma, .05)
                        converged = report["fit_converged"]
                except ValueError as exc:
                    failures.append({"dataset_row": int(test[number-1]), "error": str(exc)})
                    continue
                truth.append(actual)
                means.append(report["mean_crossing_m"])
                modes.append(report["approximate_mode_crossing_m"])
                ellipse = report["joint_95_percent_ellipse"]
                if ellipse and min(ellipse["semi_axes_m"]) > 0:
                    major = np.array(ellipse["major_axis_unit_vector"])
                    minor = np.array([-major[1], major[0]])
                    delta = actual-np.array(ellipse["center"])
                    coords = np.array([delta @ major, delta @ minor])
                    covered += int(np.sum((coords/ellipse["semi_axes_m"])**2) <= 1)
                    ellipse_count += 1
                if number % 50 == 0:
                    print(f"{method}: {number}/{len(test)} queries", file=sys.stderr)
            results[method] = {
                "evaluated": len(truth),
                "failed": len(failures),
                "failure_examples": failures[:5],
                "mean_crossing": score(truth, means),
                "mode_crossing": score(truth, modes),
                "joint_95_ellipse_available": ellipse_count,
                "joint_95_ellipse_coverage_when_available": (
                    covered/ellipse_count if ellipse_count else None),
            }
            if method == "bayes":
                results[method]["median_measurement_only_ess"] = float(np.median(ess)) if ess else None
            else:
                results[method]["fit_converged"] = converged
    return {
        "coordinates": ["paper_x_right", "z_up"],
        "seed": args.seed,
        "training_rows": len(train),
        "test_rows": len(test),
        "likelihood_sigma_m": args.sigma,
        "test_noise_sigma_m": 0 if args.clean else args.sigma,
        "bayes_adaptive": args.adaptive,
        "bayes_min_ess": args.min_ess,
        "cgmm_components": args.components if "cgmm" in methods else None,
        "conditioned_on": "airborne goal entry",
        "methods": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--method", choices=("both", "bayes", "cgmm"), default="both")
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--components", type=int, default=16)
    parser.add_argument("--sigma", type=float, default=.005/np.sqrt(3))
    parser.add_argument("--clean", action="store_true", help="Use noiseless test observations")
    parser.add_argument("--adaptive", action="store_true", help="Adapt Bayesian likelihood bandwidth")
    parser.add_argument("--min-ess", type=float, default=30)
    args = parser.parse_args()
    if (min(args.test_size, args.components) < 1 or args.seed < 0
            or not np.isfinite([args.sigma, args.min_ess]).all()
            or args.sigma <= 0 or args.min_ess < 1):
        parser.error("Use positive counts and sigma, min ESS >= 1, and a nonnegative seed.")
    try:
        report = evaluate(args)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
