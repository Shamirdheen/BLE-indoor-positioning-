"""Constant-velocity Kalman filter in 2-D, state [x, y, vx, vy]."""

import numpy as np

from . import config


class Kalman2D:
    def __init__(self, dt=config.KF_DT, q=config.KF_Q, r=config.KF_R):
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=float)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)
        # Process noise from a random (white-noise) acceleration of intensity q
        q1 = np.array([[dt ** 4 / 4, dt ** 3 / 2],
                       [dt ** 3 / 2, dt ** 2]])
        self.Q = q * np.kron(q1, np.eye(2))
        self.R = r * np.eye(2)
        self.x = None
        self.P = None

    def step(self, z):
        """Feed one position measurement z = [x, y]; return filtered [x, y]."""
        z = np.asarray(z, dtype=float)
        if self.x is None:                       # first measurement initialises
            self.x = np.array([z[0], z[1], 0.0, 0.0])
            self.P = np.diag([self.R[0, 0], self.R[1, 1], 1.0, 1.0])
            return self.x[:2].copy()

        # Predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        # Update
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[:2].copy()
