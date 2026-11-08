"""Generate goal-conditioned football flights. Python 3.10+."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
from scipy.integrate import solve_ivp
from scipy.stats import qmc

VERSION = 5
FLOAT_COLUMNS = [
    "speed_m_s", "polar_deg", "azimuth_deg",
    "omega_x_rad_s", "omega_y_rad_s", "omega_z_rad_s",
    "vx0_m_s", "vy0_m_s", "vz0_m_s",
    "x_005", "y_005", "z_005", "x_010", "y_010", "z_010",
    "t_goal", "x_goal", "y_goal", "z_goal",
]
SCHEMA = pa.schema([("id", pa.int64())] + [(n, pa.float64()) for n in FLOAT_COLUMNS])


def hit_goal_plane(t, state):
    return state[1] - 11.0


hit_goal_plane.terminal = True
hit_goal_plane.direction = 1


def hit_ground(t, state):
    return state[2] - 0.108


hit_ground.terminal = True
hit_ground.direction = -1


def integrate(velocity, omega, config):
    mass, radius, density = 0.430, 0.108, 1.27
    drag = 0.4 * density * math.pi * radius**2 / (2 * mass)
    magnus = 8 * math.pi * density * radius**3 / (3 * mass)
    gravity = 9.8 * (1 - density * 4 * math.pi * radius**3 / (3 * mass))
    wx, wy, wz = omega

    def rhs(t, state):
        vx, vy, vz = state[3:]
        damping = drag * math.sqrt(vx*vx + vy*vy + vz*vz)
        # Team 358's v cross omega sign, in a right-handed world frame.
        return [
            vx, vy, vz,
            -damping*vx + magnus*(vy*wz-vz*wy),
            -damping*vy + magnus*(vz*wx-vx*wz),
            -gravity-damping*vz + magnus*(vx*wy-vy*wx),
        ]

    return solve_ivp(
        rhs, (0, config["timeout"]), [0, 0, 0.11, *velocity],
        method="RK45", t_eval=[0.05, 0.10],
        events=(hit_goal_plane, hit_ground), dense_output=False,
        rtol=config["rtol"], atol=config["atol"], max_step=config["max_step"],
    )


def simulate(job):
    identifier, unit, config = job
    speed = config["speed_min"] + unit[0]*(config["speed_max"]-config["speed_min"])
    theta = np.deg2rad(unit[1]*config["polar_max"])
    phi = np.deg2rad(config["azimuth_min"] + unit[2]*(config["azimuth_max"]-config["azimuth_min"]))
    # Team 358 Eq. 11: polar from +z, azimuth from +x toward +y.
    velocity = speed*np.array([
        np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
    ])
    omega = np.array(config["spin_min"]) + unit[3:]*(
        np.array(config["spin_max"])-np.array(config["spin_min"]))
    result = integrate(velocity, omega, config)
    if not result.success:
        return "solver_failed", None
    if len(result.t_events[1]):
        return "ground_first", None
    if not len(result.t_events[0]):
        return "timeout", None
    if len(result.t) != 2:
        return "crossed_before_t2", None
    crossing = result.y_events[0][0]
    margin = 0.108 if config["whole_ball"] else 0
    if not (abs(crossing[0]) < 3.66-margin and margin < crossing[2] < 2.44-margin):
        return "outside_goal", None
    values = [
        speed, np.rad2deg(theta), np.rad2deg(phi), *omega, *velocity,
        *result.y[:3, 0], *result.y[:3, 1],
        result.t_events[0][0], *crossing[:3],
    ]
    return "accepted", dict(id=identifier, **dict(zip(FLOAT_COLUMNS, values)))


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("kicks"))
    parser.add_argument("--power", type=int, default=20, help="Generate 2**power candidates")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=358)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)-1))
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--speed-min", type=float, default=math.sqrt(9.8*11))
    parser.add_argument("--speed-max", type=float, default=129/3.6)
    parser.add_argument("--polar-max", type=float, default=90)
    parser.add_argument("--azimuth-min", type=float, default=55)
    parser.add_argument("--azimuth-max", type=float, default=125)
    parser.add_argument("--spin-min", nargs=3, type=float, default=[-8,-5,0],
                        metavar=("WX", "WY", "WZ"), help="Lower spin bounds in rad/s")
    parser.add_argument("--spin-max", nargs=3, type=float, default=[8,5,15],
                        metavar=("WX", "WY", "WZ"), help="Upper spin bounds in rad/s")
    parser.add_argument("--whole-ball", action="store_true")
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--max-step", type=float, default=0.02)
    args = parser.parse_args()
    config = vars(args).copy()
    output = config.pop("output")
    workers = config.pop("workers")
    if not (0 <= args.power <= 30 and 1 <= args.chunk_size and workers >= 1
            and 0 < args.speed_min < args.speed_max
            and 0 < args.polar_max <= 90 and 0 <= args.azimuth_min < args.azimuth_max <= 180
            and np.isfinite([args.spin_min,args.spin_max]).all()
            and np.all(np.array(args.spin_min) <= np.array(args.spin_max))
            and np.isfinite([args.timeout,args.rtol,args.atol,args.max_step]).all()
            and args.timeout > .1 and min(args.rtol,args.atol,args.max_step) > 0):
        parser.error("Invalid sampling or solver limits.")
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "version": VERSION, "config": config, "scipy_version": scipy.__version__,
        "numpy_version": np.__version__, "pyarrow_version": pa.__version__,
        "reference_measure": "Independent uniform coordinates in the six configured intervals",
        "selection": "Airborne through t2 and goal-plane crossing, within selected opening",
        "spin_bounds_status": "Team 480 section 4.2 modeling ranges in rad/s",
        "coordinates": "Team 358: x right, y forward, z up; goal y=11",
        "angles": "theta from +z; phi from +x toward +y",
    }
    manifest = output/"manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != metadata:
            raise SystemExit("Existing manifest differs. Use a new output directory.")
    elif any(output.iterdir()):
        raise SystemExit("Output directory must be empty or contain this generator's manifest.")
    else:
        write_json(manifest, metadata)

    # A single globally scrambled Sobol sequence, regenerated deterministically on resume.
    sampler = qmc.Sobol(d=6, scramble=True, seed=args.seed)
    counts = Counter()
    started = time.monotonic()
    total = 2**args.power
    context = ProcessPoolExecutor(workers) if workers > 1 else nullcontext(None)
    with context as executor:
        for start in range(0, total, args.chunk_size):
            stop = min(start+args.chunk_size, total)
            # Arbitrary chunk sizes preserve the same overall power-of-two sequence.
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The balance properties of Sobol")
                unit = sampler.random(stop-start)
            stem = f"part-{start:010d}"
            parquet = output/(stem+".parquet")
            checkpoint = output/(stem+".json")
            if checkpoint.exists() and parquet.exists():
                counts.update(json.loads(checkpoint.read_text())["counts"])
                continue
            jobs = ((start+i, row, config) for i, row in enumerate(unit))
            results = executor.map(simulate, jobs, chunksize=32) if executor else map(simulate, jobs)
            rows, chunk_counts = [], Counter()
            for status, row in results:
                chunk_counts[status] += 1
                if row is not None:
                    rows.append(row)
            table = pa.Table.from_pylist(rows, schema=SCHEMA)
            temporary = output/(stem+".parquet.tmp")
            pq.write_table(table, temporary, compression="zstd", row_group_size=4096)
            os.replace(temporary, parquet)
            write_json(checkpoint, {"start": start, "stop": stop, "counts": dict(chunk_counts)})
            counts.update(chunk_counts)
            print(f"{stop:,}/{total:,} candidates; {counts['accepted']:,} goals; "
                  f"{time.monotonic()-started:.1f}s this run", flush=True)
    write_json(output/"summary.json", {"candidates": total, "counts": dict(counts)})
    from data import create_split
    create_split(output, output, args.split_seed)
    print(json.dumps(dict(counts), indent=2))


if __name__ == "__main__":
    main()
