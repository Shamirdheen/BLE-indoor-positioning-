# BLE Indoor Positioning: RSSI Fingerprinting + Kalman Tracking

Course project for **Short Range Wireless Communication (SRWC)**.

## Overview
This project simulates a 20 m × 15 m room with five Bluetooth Low Energy (BLE)
beacons. It generates realistic noisy RSSI with a log-distance path-loss model
and log-normal shadowing, then compares four ways to locate a user:
(1) linearised least-squares trilateration, (2) a weighted centroid,
(3) kNN fingerprinting against an offline radio map, and (4) the proposed
method, which feeds the kNN output into a constant-velocity Kalman filter. Three
experiments measure tracking accuracy on a walking path, sensitivity to
shadowing noise, and the effect of beacon density. Everything uses NumPy and
Matplotlib only, with a fixed seed (42), so every number reproduces exactly.

## Radio model
Log-distance path loss with log-normal shadowing:

```
RSSI(d) = P_tx - 10 · n · log10(d) + X_σ ,   X_σ ~ N(0, σ²)
```

with `P_tx = -59 dBm` (RSSI at 1 m), `n = 2.2`, `σ = 4 dB`, and `d` clamped to ≥ 0.1 m.
Inverse (used by trilateration and the weighted centroid):

```
d̂ = 10^((P_tx - RSSI) / (10 · n))
```

## How to run
```bash
pip install -r requirements.txt && python main.py
```
All outputs are written to `results/`.

## Project structure
```
ble_positioning/
  config.py        all parameters in one place
  radio_model.py   rssi(), rssi_to_distance()
  estimators.py    trilateration, weighted_centroid, build_radio_map, knn
  kalman.py        Kalman2D class
  scenarios.py     walking_path(), random static test points
  experiments.py   exp1_tracking, exp2_noise, exp3_beacons (data only)
  plots.py         one function per figure
main.py            runs everything, prints tables, saves figures + CSVs
results/           generated output
```

## Results (seed 42)

**Exp 1: tracking along a 200-step rectangular walk (error in metres)**

| Method                    | Mean | Median | P90   | RMSE |
|---------------------------|-----:|-------:|------:|-----:|
| Trilateration             | 5.62 | 4.98   | 10.09 | 6.42 |
| Weighted centroid         | 4.62 | 4.59   | 6.54  | 4.87 |
| kNN                       | 3.27 | 2.52   | 6.89  | 4.07 |
| **kNN+Kalman (proposed)** | **2.16** | **1.82** | **4.16** | **2.70** |

kNN+Kalman cuts mean error by about 34 % compared with plain kNN, and by 61 % compared with trilateration.

**Exp 2: mean error (m) vs shadowing σ, 150 static points (radio map built at σ = 4 dB)**

| σ (dB) | Trilateration | Weighted centroid | kNN  |
|-------:|--------------:|------------------:|-----:|
| 1      | 1.78          | 3.58              | 1.08 |
| 2      | 3.28          | 3.68              | 1.85 |
| 3      | 4.79          | 3.78              | 2.65 |
| 4      | 6.01          | 3.73              | 3.26 |
| 6      | 7.30          | 4.31              | 4.72 |
| 8      | 9.12          | 4.83              | 5.76 |

**Exp 3: kNN mean error vs number of beacons (σ = 4 dB, 5 survey samples per point)**

| Beacons | Mean error (m) |
|--------:|---------------:|
| 3       | 4.16           |
| 4       | 3.51           |
| 5       | 3.13           |
| 6       | 2.74           |
| 8       | 2.28           |

## Figures
- `results/fig1_trajectories.png`: room, beacons, true path, and the trilateration, kNN and kNN+Kalman estimates
- `results/fig2_error_cdf.png`: error CDF for all four methods
- `results/fig3_path_loss.png`: noisy RSSI vs distance with the model curve
- `results/fig4_noise_sensitivity.png`: mean error vs σ
- `results/fig5_beacon_count.png`: kNN error vs beacon count

Tables are also saved as `metrics_exp1.csv`, `metrics_exp2.csv`, `metrics_exp3.csv`
and `console_output.txt`.

## Limitations
- **Single path-loss exponent.** One `n` is used for the whole room. Real
  buildings have walls, furniture and people, so `n` changes with position and
  direction, and multipath causes fast fading that this model leaves out.
- **Independent noise.** Shadowing is drawn i.i.d. for every sample and beacon.
  Real shadowing is spatially and temporally correlated (a person standing
  still sees a persistent bias), so averaging helps less in practice.
- **Needs a site survey.** kNN fingerprinting requires an offline radio map
  measured at every grid point. That is labour-intensive, and the map goes
  stale when the environment changes. The simulation also builds its map from
  the same model it is tested on, which flatters fingerprinting.
- The Kalman filter assumes constant velocity, so it lags at sharp turns (see
  the corners in Fig. 1). Its noise parameters (q, R) were set by hand rather
  than tuned on data.
- Simulation only: there is no device heterogeneity, advertising-channel
  differences, or packet loss.
