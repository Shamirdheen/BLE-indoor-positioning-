"""Position estimators: trilateration, weighted centroid, kNN fingerprinting."""

import numpy as np

from . import config
from .radio_model import rssi, rssi_to_distance


def trilateration(rssi_vec, beacons):
    """Linearised least-squares trilateration (last beacon is the reference).

    Subtracting the circle equation of the reference beacon r from beacon i:
      2(x_r - x_i) x + 2(y_r - y_i) y = d_i^2 - d_r^2 - |b_i|^2 + |b_r|^2
    """
    beacons = np.asarray(beacons, dtype=float)
    d = rssi_to_distance(rssi_vec)
    ref, d_ref = beacons[-1], d[-1]
    A = 2.0 * (ref - beacons[:-1])
    b = (d[:-1] ** 2 - d_ref ** 2
         - np.sum(beacons[:-1] ** 2, axis=1) + np.sum(ref ** 2))
    pos, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.clip(pos, [0.0, 0.0], [config.ROOM_W, config.ROOM_H])


def weighted_centroid(rssi_vec, beacons):
    """Beacon positions averaged with weights 1 / estimated distance."""
    beacons = np.asarray(beacons, dtype=float)
    w = 1.0 / rssi_to_distance(rssi_vec)
    return (w[:, None] * beacons).sum(axis=0) / w.sum()


def grid_points(step=config.GRID_STEP):
    """Fingerprint grid covering the room, (M, 2)."""
    xs = np.arange(0.0, config.ROOM_W + 1e-9, step)
    ys = np.arange(0.0, config.ROOM_H + 1e-9, step)
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel()])


def build_radio_map(beacons, n_samples, sigma, rng, step=config.GRID_STEP):
    """Offline site survey: averaged RSSI vector at every grid point.

    Returns (grid (M, 2), fingerprints (M, B)).
    """
    grid = grid_points(step)
    samples = rssi(np.repeat(grid[:, None, :], n_samples, axis=1),
                   beacons, sigma, rng)          # (M, n_samples, B)
    return grid, samples.mean(axis=1)


def knn(rssi_vec, grid, fingerprints, k=config.K):
    """Weighted kNN in signal space: average the K closest grid points."""
    dist = np.linalg.norm(fingerprints - rssi_vec, axis=1)
    idx = np.argsort(dist)[:k]
    w = 1.0 / (dist[idx] + 1e-6)
    return (w[:, None] * grid[idx]).sum(axis=0) / w.sum()
