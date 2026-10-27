"""Figures for the README, from the result CSVs.

usage: figures.py   (writes docs/figures/*.png)
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"


def breadth() -> None:
    rows = list(csv.DictReader(open(ROOT / "runs/a8_sc128/library.csv")))
    by = defaultdict(list)
    for r in rows:
        by[(r["target"], r["estimator"])].append(float(r["nmse_db"]))
    targets = sorted({r["target"] for r in rows})
    ks = (1, 2, 4, 8, 13)
    mean, worst = [], []
    for k in ks:
        gaps = [(np.mean(by[(t, f"pool_k{k}")]) - by[(t, "ceiling")][0],
                 max(by[(t, f"pool_k{k}")]) - by[(t, "ceiling")][0]) for t in targets]
        mean.append(np.mean([g[0] for g in gaps]))
        worst.append(np.mean([g[1] for g in gaps]))
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(ks, worst, "o-", label="worst of 8 random pools")
    ax.plot(ks, mean, "o-", label="average pool")
    ax.axhspan(0.47, 0.73, color="0.85", label="3GPP CDL + 50 site measurements")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlabel("number of simulated cities pooled (no site measurements)")
    ax.set_ylabel("dB worse than site-trained estimator")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.set_title("Pooling cities shrinks the worst case", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "breadth.png", dpi=150)


def prediction() -> None:
    rows = [r for r in csv.DictReader(open(ROOT / "runs/a8_sc32_track/predict_seed0.csv"))
            if r["snr_db"] == "10.0" and r["speed"] == "all" and r["n"] == "0"]
    methods = {"hold_last": "repeat last measurement", "wiener_rt3": "statistical (pooled cities)",
               "learned": "neural network (best of two)", "tone": "classical per-user (tone)"}
    v = defaultdict(lambda: defaultdict(list))
    for r in rows:
        m = "learned" if r["method"].startswith("learned") else r["method"]
        v[m][(r["target"], float(r["horizon_ms"]))].append(float(r["nmse_db"]))
    fig, ax = plt.subplots(figsize=(6, 4.4))
    for m, label in methods.items():
        cells = v[m]
        hs = sorted({h for _, h in cells})
        ys = [np.mean([min(cells[(t, h)]) for t, hh in cells if hh == h]) for h in hs]
        ax.plot(hs, ys, "o-", label=label)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 5, 10], ["0.5", "1", "2", "5", "10"])
    ax.set_xlabel("how far ahead (ms)")
    ax.set_ylabel("prediction error, NMSE (dB, lower is better)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.set_title("Predicting the channel: classical beats learned", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "prediction.png", dpi=150)


def measured() -> None:
    """Measured DICHASUS channels, full array x frequency grid, 10 dB, no site data."""
    def load(name):
        v = defaultdict(list)
        for r in csv.DictReader(open(ROOT / f"runs/a16_sc128_dichasus/{name}.csv")):
            if r["snr_db"] == "10.0" and r["variant"] == "spatial":
                v[(r["array"], r["estimator"])].append(float(r["nmse_db"]))
        return v
    main, rich = load("dichasus_test"), load("dichasus_rich")
    bars = [("LS interpolation", main, "ls_interp"), ("RT pool, specular (14 cities)", main, "pool_all"),
            ("RT pool, scattering (7 cities)", rich, "pool_all"), ("3GPP CDL", main, "cdl"),
            ("site-trained ceiling", main, "ceiling")]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    width = 0.16
    for i, (label, v, est) in enumerate(bars):
        ys = [np.mean(v[(a, est)]) for a in "ABC"]
        ax.bar(np.arange(3) + (i - 2) * width, ys, width, label=label)
    ax.set_xticks(range(3), ["array A", "array B", "array C"])
    ax.set_ylabel("NMSE (dB); taller bar = better")
    ax.invert_yaxis()
    ax.set_ylim(-9, -21)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
    ax.set_title("Measured channels: simulated cities vs 3GPP CDL", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "measured.png", dpi=150)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    breadth()
    prediction()
    measured()
