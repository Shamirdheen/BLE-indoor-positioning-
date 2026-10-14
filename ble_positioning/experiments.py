"""The three experiments. Each returns plain data; no plotting here."""

import numpy as np

from . import config
from .estimators import build_radio_map, knn, trilateration, weighted_centroid
from .kalman import Kalman2D
from .radio_model import distances, rssi
from .scenarios import random_static_points, walking_path


def error_metrics(err):
    """Mean, median, 90th percentile and RMSE of an error array (metres)."""
    err = np.asarray(err)
    return {
        "mean": float(np.mean(err)),
        "median": float(np.median(err)),
        "p90": float(np.percentile(err, 90)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
    }


def exp1_tracking(beacons, rng):
    """Track a walking user with all four methods."""
    grid, fps = build_radio_map(beacons, config.RADIO_MAP_SAMPLES,
                                config.SHADOW_SIGMA, rng)
    truth = walking_path()
    kf = Kalman2D()
    est = {m: [] for m in config.METHODS}
    for pos in truth:
        r = rssi(pos, beacons, config.SHADOW_SIGMA, rng)
        knn_pos = knn(r, grid, fps)
        est["Trilateration"].append(trilateration(r, beacons))
        est["Weighted centroid"].append(weighted_centroid(r, beacons))
        est["kNN"].append(knn_pos)
        est["kNN+Kalman"].append(kf.step(knn_pos))
    est = {m: np.array(v) for m, v in est.items()}
    errors = {m: np.linalg.norm(v - truth, axis=1) for m, v in est.items()}
    metrics = {m: error_metrics(e) for m, e in errors.items()}
    return {"truth": truth, "estimates": est, "errors": errors, "metrics": metrics}


def exp2_noise(beacons, rng):
    """Mean error of the static estimators vs shadowing sigma."""
    grid, fps = build_radio_map(beacons, config.RADIO_MAP_SAMPLES,
                                config.SHADOW_SIGMA, rng)   # sigma=4 map reused
    points = random_static_points(config.N_STATIC_POINTS, rng)
    methods = ("Trilateration", "Weighted centroid", "kNN")
    mean_err = {m: [] for m in methods}
    for sigma in config.SIGMAS:
        err = {m: [] for m in methods}
        for pos in points:
            r = rssi(pos, beacons, sigma, rng)
            err["Trilateration"].append(np.linalg.norm(trilateration(r, beacons) - pos))
            err["Weighted centroid"].append(np.linalg.norm(weighted_centroid(r, beacons) - pos))
            err["kNN"].append(np.linalg.norm(knn(r, grid, fps) - pos))
        for m in methods:
            mean_err[m].append(float(np.mean(err[m])))
    return {"sigmas": list(config.SIGMAS), "mean_error": mean_err}


def exp3_beacons(rng):
    """kNN mean error for each beacon layout in config.BEACON_CONFIGS."""
    points = random_static_points(config.N_STATIC_POINTS, rng)
    counts, mean_err = [], []
    for count, layout in config.BEACON_CONFIGS.items():
        beacons = np.array(layout, dtype=float)
        grid, fps = build_radio_map(beacons, config.EXP3_MAP_SAMPLES,
                                    config.SHADOW_SIGMA, rng)
        err = [np.linalg.norm(knn(rssi(p, beacons, config.SHADOW_SIGMA, rng),
                                  grid, fps) - p)
               for p in points]
        counts.append(count)
        mean_err.append(float(np.mean(err)))
    return {"counts": counts, "mean_error": mean_err}


def path_loss_samples(beacons, rng, n=600):
    """Noisy RSSI vs distance to beacon 0 at random room positions (for fig 3)."""
    points = random_static_points(n, rng, margin=0.0)
    d = distances(points, beacons[:1])[:, 0]
    r = rssi(points, beacons[:1], config.SHADOW_SIGMA, rng)[:, 0]
    return {"distance": d, "rssi": r}
