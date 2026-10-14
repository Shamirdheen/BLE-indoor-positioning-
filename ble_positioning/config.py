"""All simulation parameters in one place.

Values are immutable (numbers and tuples). Code that needs a numpy array
converts with np.array(...) locally, so nothing can mutate shared state.
"""

SEED = 42

# Room (metres)
ROOM_W = 20.0
ROOM_H = 15.0

# Beacon layouts (x, y) in metres
BEACONS = ((0.0, 0.0), (20.0, 0.0), (20.0, 15.0), (0.0, 15.0), (10.0, 7.5))

BEACON_CONFIGS = {
    3: ((0.0, 0.0), (20.0, 0.0), (0.0, 15.0)),
    4: ((0.0, 0.0), (20.0, 0.0), (20.0, 15.0), (0.0, 15.0)),
    5: BEACONS,
    6: BEACONS + ((10.0, 0.0),),
    8: BEACONS + ((10.0, 0.0), (10.0, 15.0), (0.0, 7.5)),
}

# Radio model (log-distance path loss + log-normal shadowing)
TX_POWER = -59.0      # dBm, RSSI at 1 m
PATH_LOSS_N = 2.2     # path-loss exponent
SHADOW_SIGMA = 4.0    # dB, std of shadowing noise
MIN_DIST = 0.1        # m, distance clamp to avoid log10(0)

# Fingerprinting
GRID_STEP = 1.0             # m -> 21 x 16 = 336 grid points
RADIO_MAP_SAMPLES = 10      # RSSI samples averaged per grid point
K = 4                       # neighbours in kNN

# Kalman filter (constant-velocity model)
KF_DT = 1.0
KF_Q = 0.05
KF_R = 2.5 ** 2

# Experiment 1: walking path
PATH_WAYPOINTS = ((2.0, 2.0), (18.0, 2.0), (18.0, 13.0), (2.0, 13.0), (2.0, 2.0))
PATH_STEPS = 200

# Experiment 2: noise sensitivity
N_STATIC_POINTS = 150
WALL_MARGIN = 1.0
SIGMAS = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)

# Experiment 3: beacon density
EXP3_MAP_SAMPLES = 5

# Output
RESULTS_DIR = "results"
DPI = 150
COLORS = {
    "Trilateration": "#d62728",
    "Weighted centroid": "#ff7f0e",
    "kNN": "#1f77b4",
    "kNN+Kalman": "#2ca02c",
}
METHODS = tuple(COLORS)
