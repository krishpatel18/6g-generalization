"""Leave-one-city-out, run 2: which pre-training source gives the best prior for an unseen city?

usage: loco.py <num_bs_antennas> <seed>

Arms (each 12k channels of pre-training data):
  cdl  - 3GPP CDL-A..E
  rt3  - the three other cities, 4k each      (broad ray-traced prior)
  rt1  - one other city, rotated per target    (narrow ray-traced prior)
Each arm is an LMMSE built from its statistics plus a CNN correction pre-trained
on top of it. For N target channels, the statistics are blended toward the N
samples (weight by cross-validation) and the CNN is fine-tuned on them.

Baselines: LS + interpolation; LMMSE from the N target samples alone; LMMSE
from 12k in-domain channels (the ceiling).

Gates, checked before reading the arm comparison (see docs/PLAN.md):
  a) every arm's CNN at N=0 is no worse than its own LMMSE;
  b) every fine-tuned arm is no worse than LMMSE from the same N target samples.

Writes runs/<grid tag>/loco_seed<seed>.csv:
  target, arm, model, n, blend_w, snr_db, nmse_db
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

from sixg_gen import channels, estimators, grid, linear

FINETUNE_N = (0, 50, 200, 1000)
EVAL_SNR_DB = (0.0, 10.0, 20.0)
PRETRAIN_STEPS, FINETUNE_STEPS = 3000, 300
RUNS = Path(__file__).resolve().parents[1] / "runs"
SINGLE_SOURCE = {"munich": "etoile", "etoile": "florence",
                 "florence": "san_francisco", "san_francisco": "munich"}


def evaluate(estimate, h_test: np.ndarray) -> dict[float, float]:
    out = {}
    for snr in EVAL_SNR_DB:
        rng = np.random.default_rng(1234)  # identical pilot noise for every method
        out[snr] = grid.nmse_db(estimate(h_test, np.full(len(h_test), snr), rng), h_test)
    return out


def source_channels(arm: str, target: str) -> np.ndarray:
    if arm == "cdl":
        return channels.generate("cdl", 12000, 1, mmap=True)
    if arm == "rt3":
        return np.concatenate([channels.generate(f"rt:{c}", 12000, 1, mmap=True)[:4000]
                               for c in channels.RT_SCENES if c != target])
    return channels.generate(f"rt:{SINGLE_SOURCE[target]}", 12000, 1, mmap=True)


def source_name(arm: str, target: str) -> str:
    return {"cdl": "cdl", "rt3": f"rt_not_{target}", "rt1": f"rt_only_{SINGLE_SOURCE[target]}"}[arm]


def main(seed: int) -> None:
    runs = RUNS / grid.config_tag()
    runs.mkdir(parents=True, exist_ok=True)
    rows = []

    def record(target, arm, model, n, w, scores):
        rows.append([(target, arm, model, n, w, snr, round(v, 3)) for snr, v in scores.items()])
        print(f"  {target:14s} {arm:6s} {model:9s} N={n:<5d} w={w:<4} " +
              "  ".join(f"{snr:>4.0f}dB:{v:7.2f}" for snr, v in scores.items()), flush=True)

    stats_cache: dict[str, linear.Stats] = {}

    def stats_and_cnn(arm: str, target: str) -> tuple[linear.Stats, dict]:
        name = source_name(arm, target)
        if name not in stats_cache:
            stats_cache.clear()  # covariances are large at many antennas; keep one
            stats_cache[name] = linear.Stats.fit(source_channels(arm, target))
        stats = stats_cache[name]
        path = runs / f"pre_{name}_s{seed}.pt"
        if not path.exists():
            t = time.time()
            torch.manual_seed(seed)
            h = np.asarray(source_channels(arm, target))
            model = estimators.train(estimators.Refiner(), linear.LMMSE(stats), h, PRETRAIN_STEPS, 1e-3, seed)
            torch.save(model.state_dict(), path)
            print(f"  pre-trained {name} in {time.time()-t:.0f}s", flush=True)
        return stats, torch.load(path)

    for target in channels.RT_SCENES:
        print(f"== target {target}", flush=True)
        pool = channels.generate(f"rt:{target}", 1000, 2)
        test = channels.generate(f"rt:{target}", 2000, 3)
        record(target, "-", "ls_interp", 0, "", evaluate(grid.ls_estimate, test))
        oracle = linear.LMMSE(linear.Stats.fit(channels.generate(f"rt:{target}", 12000, 1, mmap=True)))
        record(target, "-", "lmmse_in", 12000, "", evaluate(oracle.estimate, test))
        for n in FINETUNE_N[1:]:
            own = linear.LMMSE(linear.Stats.fit(pool[:n]))
            record(target, "-", "lmmse_own", n, "", evaluate(own.estimate, test))

        for arm in ("cdl", "rt3", "rt1"):
            stats, state = stats_and_cnn(arm, target)
            for n in FINETUNE_N:
                fit = pool[:n]
                adapted, w = linear.Stats.blend(stats, fit, seed)
                front = linear.LMMSE(adapted)
                record(target, arm, "lin", n, w, evaluate(front.estimate, test))
                model = estimators.Refiner()
                model.load_state_dict(state)
                if n:
                    model = estimators.train(model, front, fit, FINETUNE_STEPS, 1e-4, seed + n, batch=min(64, n))
                record(target, arm, "lin+cnn", n, w,
                       evaluate(lambda h, s, rng: estimators.refine(model, front, h, s, rng), test))

        with open(runs / f"loco_seed{seed}.csv", "w", newline="") as f:
            out = csv.writer(f)
            out.writerow(("target", "arm", "model", "n", "blend_w", "snr_db", "nmse_db"))
            for group in rows:
                out.writerows(group)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    grid.NUM_ANTENNAS = int(sys.argv[1])
    main(int(sys.argv[2]))
