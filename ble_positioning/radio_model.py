"""Log-distance path-loss model with log-normal shadowing."""

import numpy as np

from . import config


def distances(points, beacons):
    """Euclidean distance from each point to each beacon.

    points: (..., 2), beacons: (B, 2) -> (..., B)
    """
    points = np.asarray(points, dtype=float)
    beacons = np.asarray(beacons, dtype=float)
    return np.linalg.norm(points[..., None, :] - beacons, axis=-1)


def rssi(points, beacons, sigma, rng,
         tx_power=config.TX_POWER, n=config.PATH_LOSS_N):
    """Noisy RSSI (dBm) seen at each point from each beacon.

    RSSI = TX_POWER - 10 n log10(d) + N(0, sigma), with d >= MIN_DIST.
    points: (..., 2) -> (..., B)
    """
    d = np.maximum(distances(points, beacons), config.MIN_DIST)
    mean = tx_power - 10.0 * n * np.log10(d)
    return mean + rng.normal(0.0, sigma, size=mean.shape)


def rssi_to_distance(rssi_dbm, tx_power=config.TX_POWER, n=config.PATH_LOSS_N):
    """Invert the path-loss model: d = 10^((TX_POWER - RSSI) / (10 n))."""
    return 10.0 ** ((tx_power - np.asarray(rssi_dbm)) / (10.0 * n))
