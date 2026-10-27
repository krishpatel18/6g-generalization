"""Post-hoc: give the learned predictors a fairer shot (docs/PLAN.md, "Fairer learned predictors").

usage: predict_fair.py <num_bs_antennas> <seed> [fronts]   (fronts: comma list, default wiener_w,tone_res)

Same tracks, noise draws and test sets as predict.py, N = 0 only. Arms, each a
zero-initialised CNN residual trained with the loss on the target times only,
weighted so every horizon counts equally relative to its front end's error:
  wiener_w_{ant,ad}  - on the rt3 Wiener front (as predict.py, new loss)
  tone_res_{ant,ad}  - on the tone front (noisy history + tone extrapolation)

Writes runs/<grid tag>/predict_fair_seed<seed>.csv (same columns as predict.py)
and prints each arm against tone and against the original learned arms.
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

from sixg_gen import channels, estimators, grid, linear

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loco import RUNS  # noqa: E402
from predict import (DOMAINS, PRETRAIN_STEPS, WIDTH, InDomain, evaluate, rt3,  # noqa: E402
                     targets_of, tone_track)

WEIGHT_TRACKS, WEIGHT_SNR_DB = 2000, 10.0


class ToneFront:
    """tone_track behind the same `estimate` interface as an LMMSE front end."""

    estimate = staticmethod(tone_track)


def horizon_weights(front, pool: np.ndarray) -> np.ndarray:
    """0 on the history; 1 / (front's NMSE) on each target; scaled to sum to NUM_SYMBOLS."""
    h = np.asarray(pool[:WEIGHT_TRACKS])
    est = front.estimate(h, np.full(len(h), WEIGHT_SNR_DB), np.random.default_rng(99))
    hist = len(grid.PILOT_SYMBOLS)
    w = np.zeros(grid.NUM_SYMBOLS)
    for t in range(hist, grid.NUM_SYMBOLS):
        w[t] = 1.0 / 10 ** (grid.nmse_db(est[:, :, t], h[:, :, t]) / 10)
    return w * grid.NUM_SYMBOLS / w.sum()


def main(seed: int, only: set[str]) -> None:
    runs = RUNS / grid.config_tag()
    rows = []
    for target in channels.RT_SCENES:
        print(f"== target {target}", flush=True)
        test = channels.generate(f"rt:{target}", 2000, 3)
        speed = channels.speeds(f"rt:{target}", 2000, 3)
        pool = rt3(target)
        fronts = {"wiener_w": linear.LMMSE(linear.Stats.fit(pool)), "tone_res": ToneFront()}
        for name, front in fronts.items():
            if name not in only:
                continue
            weights = horizon_weights(front, pool)
            print(f"  {name} weights (targets): " + " ".join(f"{x:.2f}" for x in weights[-5:]), flush=True)
            for domain, (to, back) in DOMAINS.items():
                t = time.time()
                torch.manual_seed(seed)
                wrapped = InDomain(front, to, back)
                model = estimators.train(estimators.Refiner(width=WIDTH), wrapped, to(pool),
                                         PRETRAIN_STEPS, 1e-3, seed, time_weight=weights)
                refine = lambda h, s, rng: back(estimators.refine(model, wrapped, to(h), s, rng))  # noqa: E731
                scores = evaluate(targets_of(refine), test, speed)
                rows.extend((target, f"{name}_{domain}", 0, "", *r[:3], round(r[3], 3)) for r in scores)
                s10 = {ms: v for snr, ms, sp, v in scores if snr == 10.0 and sp == "all"}
                print(f"  {name}_{domain:3s} ({time.time()-t:.0f}s) 10dB " +
                      "  ".join(f"{ms:g}ms:{v:6.2f}" for ms, v in s10.items()), flush=True)
        with open(runs / f"predict_fair_seed{seed}.csv", "w", newline="") as f:
            out = csv.writer(f)
            out.writerow(("target", "method", "n", "blend_w", "snr_db", "horizon_ms", "speed", "nmse_db"))
            out.writerows(rows)
    summary(rows, runs / "predict_seed0.csv")  # tone has no seed; originals ran with seed 0
    print("ALL DONE", flush=True)


def summary(rows, original: Path) -> None:
    """Each new arm minus tone, and minus the best original learned arm (10 dB, all speeds, N=0)."""
    old = {}
    for r in csv.DictReader(open(original)):
        if r["snr_db"] == "10.0" and r["speed"] == "all" and r["n"] == "0":
            old[(r["target"], r["method"], float(r["horizon_ms"]))] = float(r["nmse_db"])
    new = {(t, m, ms): v for t, m, _, _, snr, ms, sp, v in rows if snr == 10.0 and sp == "all"}
    print("\n10 dB, all speeds, dB, negative = new arm better.")
    print("Each cell: arm minus tone [arm minus best original learned arm]")
    for arm in sorted({m for _, m, _ in new}):
        for t in channels.RT_SCENES:
            cells = []
            for ms in (2.0, 5.0, 10.0):
                x = new[(t, arm, ms)]
                orig = min(old[(t, "learned_ant", ms)], old[(t, "learned_ad", ms)])  # seed-0 originals
                cells.append(f"{ms:g}ms:{x - old[(t, 'tone', ms)]:6.2f} [{x - orig:6.2f}]")
            print(f"  {arm:12s} {t:14s} " + "  ".join(cells))


if __name__ == "__main__":
    grid.NUM_ANTENNAS = int(sys.argv[1])
    grid.use_tracks()
    main(int(sys.argv[2]), set(sys.argv[3].split(",")) if len(sys.argv) > 3 else {"wiener_w", "tone_res"})
