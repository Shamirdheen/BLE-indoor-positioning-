"""Run all experiments, print tables, save figures and CSVs to results/."""

import csv
import os

import numpy as np

from ble_positioning import config, plots
from ble_positioning.experiments import (exp1_tracking, exp2_noise,
                                         exp3_beacons, path_loss_samples)


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    out = config.RESULTS_DIR
    os.makedirs(out, exist_ok=True)
    beacons = np.array(config.BEACONS, dtype=float)
    lines = []

    def log(text=""):
        print(text)
        lines.append(text)

    # Each experiment gets its own generator seeded with SEED, so every
    # experiment reproduces on its own regardless of run order.
    exp1 = exp1_tracking(beacons, np.random.default_rng(config.SEED))
    exp2 = exp2_noise(beacons, np.random.default_rng(config.SEED))
    exp3 = exp3_beacons(np.random.default_rng(config.SEED))
    pl = path_loss_samples(beacons, np.random.default_rng(config.SEED))

    # ---- Exp 1 ----
    log("Exp 1: Tracking along walking path (200 steps), error in metres")
    log(f"{'Method':<20}{'Mean':>8}{'Median':>8}{'P90':>8}{'RMSE':>8}")
    rows = []
    for m in config.METHODS:
        k = exp1["metrics"][m]
        log(f"{m:<20}{k['mean']:>8.2f}{k['median']:>8.2f}{k['p90']:>8.2f}{k['rmse']:>8.2f}")
        rows.append([m] + [f"{k[x]:.4f}" for x in ("mean", "median", "p90", "rmse")])
    write_csv(os.path.join(out, "metrics_exp1.csv"),
              ["method", "mean_error_m", "median_error_m", "p90_error_m", "rmse_m"], rows)

    # ---- Exp 2 ----
    log()
    log("Exp 2: Mean error (m) vs shadowing sigma, 150 static points")
    methods2 = list(exp2["mean_error"])
    log(f"{'Sigma (dB)':<12}" + "".join(f"{m:>20}" for m in methods2))
    rows = []
    for i, s in enumerate(exp2["sigmas"]):
        vals = [exp2["mean_error"][m][i] for m in methods2]
        log(f"{s:<12g}" + "".join(f"{v:>20.2f}" for v in vals))
        rows.append([s] + [f"{v:.4f}" for v in vals])
    write_csv(os.path.join(out, "metrics_exp2.csv"),
              ["sigma_db", "trilateration_mean_m", "weighted_centroid_mean_m", "knn_mean_m"],
              rows)

    # ---- Exp 3 ----
    log()
    log("Exp 3: kNN mean error (m) vs number of beacons")
    log(f"{'Beacons':<10}{'Mean error':>12}")
    rows = []
    for c, e in zip(exp3["counts"], exp3["mean_error"]):
        log(f"{c:<10}{e:>12.2f}")
        rows.append([c, f"{e:.4f}"])
    write_csv(os.path.join(out, "metrics_exp3.csv"),
              ["n_beacons", "knn_mean_error_m"], rows)

    # ---- Figures ----
    figs = [
        plots.fig1_trajectories(exp1, beacons, out),
        plots.fig2_error_cdf(exp1, out),
        plots.fig3_path_loss(pl, out),
        plots.fig4_noise_sensitivity(exp2, out),
        plots.fig5_beacon_count(exp3, out),
    ]
    log()
    log("Saved figures:")
    for f in figs:
        log(f"  {f}")

    with open(os.path.join(out, "console_output.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
