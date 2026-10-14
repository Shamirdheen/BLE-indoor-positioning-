"""Ground-truth positions used by the experiments."""

import numpy as np

from . import config


def walking_path(waypoints=config.PATH_WAYPOINTS, n_steps=config.PATH_STEPS):
    """n_steps points evenly spaced (by distance) along the waypoint polyline."""
    wp = np.asarray(waypoints, dtype=float)
    seg_len = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    s = np.linspace(0.0, cum[-1], n_steps)
    x = np.interp(s, cum, wp[:, 0])
    y = np.interp(s, cum, wp[:, 1])
    return np.column_stack([x, y])


def random_static_points(n, rng, margin=config.WALL_MARGIN):
    """n uniformly random points at least `margin` metres from every wall."""
    x = rng.uniform(margin, config.ROOM_W - margin, n)
    y = rng.uniform(margin, config.ROOM_H - margin, n)
    return np.column_stack([x, y])
