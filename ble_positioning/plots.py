"""One function per figure. Each takes experiment data and saves a PNG."""

import os

import matplotlib
matplotlib.use("Agg")              # no display needed
import matplotlib.pyplot as plt
import numpy as np

from . import config

C = config.COLORS


def _save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=config.DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _draw_room(ax, beacons):
    ax.plot([0, config.ROOM_W, config.ROOM_W, 0, 0],
            [0, 0, config.ROOM_H, config.ROOM_H, 0], color="grey", lw=1)
    ax.scatter(beacons[:, 0], beacons[:, 1], marker="^", s=140, color="purple",
               edgecolor="black", zorder=5, label="Beacons")
    ax.set_xlim(-1, config.ROOM_W + 1)
    ax.set_ylim(-1, config.ROOM_H + 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def fig1_trajectories(exp1, beacons, out_dir):
    fig, ax = plt.subplots(figsize=(9, 7.5))
    _draw_room(ax, beacons)
    est = exp1["estimates"]
    ax.plot(*exp1["truth"].T, color="black", lw=2, label="True path")
    ax.scatter(*est["Trilateration"].T, s=10, color=C["Trilateration"],
               alpha=0.5, label="Trilateration")
    ax.scatter(*est["kNN"].T, s=10, color=C["kNN"], alpha=0.6, label="kNN")
    ax.plot(*est["kNN+Kalman"].T, color=C["kNN+Kalman"], lw=2,
            label="kNN+Kalman (proposed)")
    ax.set_title("Exp 1: walking-path tracking")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    return _save(fig, out_dir, "fig1_trajectories.png")


def fig2_error_cdf(exp1, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in config.METHODS:
        e = np.sort(exp1["errors"][m])
        ax.plot(e, np.arange(1, len(e) + 1) / len(e), color=C[m], lw=2, label=m)
    ax.set_xlabel("Positioning error (m)")
    ax.set_ylabel("CDF")
    ax.set_ylim(0, 1)
    ax.set_xlim(left=0)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_title("Exp 1: error CDF")
    return _save(fig, out_dir, "fig2_error_cdf.png")


def fig3_path_loss(samples, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(samples["distance"], samples["rssi"], s=8, alpha=0.4,
               color=C["kNN"], label=f"Noisy RSSI (σ = {config.SHADOW_SIGMA:g} dB)")
    d = np.linspace(config.MIN_DIST, samples["distance"].max(), 300)
    model = config.TX_POWER - 10 * config.PATH_LOSS_N * np.log10(d)
    ax.plot(d, model, color="black", lw=2,
            label=f"Model: {config.TX_POWER:g} - 10·{config.PATH_LOSS_N:g}·log10(d)")
    ax.set_xlabel("Distance to beacon (m)")
    ax.set_ylabel("RSSI (dBm)")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Log-distance path-loss model")
    return _save(fig, out_dir, "fig3_path_loss.png")


def fig4_noise_sensitivity(exp2, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for m, errs in exp2["mean_error"].items():
        ax.plot(exp2["sigmas"], errs, "o-", color=C[m], lw=2, label=m)
    ax.set_xlabel("Shadowing σ (dB)")
    ax.set_ylabel("Mean error (m)")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Exp 2: noise sensitivity")
    return _save(fig, out_dir, "fig4_noise_sensitivity.png")


def fig5_beacon_count(exp3, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = [str(c) for c in exp3["counts"]]
    bars = ax.bar(labels, exp3["mean_error"], color=C["kNN"])
    ax.bar_label(bars, fmt="%.2f")
    ax.set_xlabel("Number of beacons")
    ax.set_ylabel("kNN mean error (m)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Exp 3: beacon density")
    return _save(fig, out_dir, "fig5_beacon_count.png")
