"""Leave-one-city-out: does ray-traced pre-training beat stochastic pre-training on an unseen city?

For each held-out target city:
  - pre-train A on CDL (12k channels; trained once, reused for every fold)
  - pre-train B on the other three RT cities (4k each = 12k, same budget)
  - pre-train C on ONE other RT city (12k) -- separates breadth from RT realism
  - for N in FINETUNE_N: fine-tune A, B and C on N target channels; also train from
    scratch on N, and fit LMMSE on N
  - score everything on 2k held-out target channels (different base-station sites)
Writes runs/loco_seed<seed>.csv: target, method, n, snr_db, nmse_db.
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

from sixg_gen import channels, estimators, grid

FINETUNE_N = (0, 50, 200, 1000)
EVAL_SNR_DB = (0.0, 10.0, 20.0)
PRETRAIN_STEPS, FINETUNE_STEPS, SCRATCH_STEPS = 4000, 600, 3000
RUNS = Path(__file__).resolve().parents[1] / "runs"


def evaluate(estimate, h_test: np.ndarray) -> dict[float, float]:
    out = {}
    for snr in EVAL_SNR_DB:
        rng = np.random.default_rng(1234)  # identical pilot noise for every method
        out[snr] = grid.nmse_db(estimate(h_test, np.full(len(h_test), snr), rng), h_test)
    return out


def pretrained(name: str, h: np.ndarray, seed: int) -> dict:
    path = RUNS / f"pre_{name}_s{seed}.pt"
    if not path.exists():
        torch.manual_seed(seed)
        t = time.time()
        model = estimators.train(estimators.Refiner(), h, PRETRAIN_STEPS, 1e-3, seed)
        torch.save(model.state_dict(), path)
        print(f"  pre-trained {name} in {time.time()-t:.0f}s", flush=True)
    return torch.load(path)


def main(seed: int) -> None:
    RUNS.mkdir(exist_ok=True)
    rows = []

    def record(target, method, n, scores):
        for snr, v in scores.items():
            rows.append((target, method, n, snr, round(v, 3)))
        print(f"  {target:14s} {method:12s} N={n:<5d} " +
              "  ".join(f"{snr:>4.0f}dB:{v:7.2f}" for snr, v in scores.items()), flush=True)

    cdl_state = pretrained("cdl", channels.generate("cdl", 12000, 1), seed)
    for target in channels.RT_SCENES:
        print(f"== target {target}", flush=True)
        others = [c for c in channels.RT_SCENES if c != target]
        rt_state = pretrained(f"rt_not_{target}", np.concatenate(
            [channels.generate(f"rt:{c}", 12000, 1)[:4000] for c in others]), seed)
        single = others[0]
        rt1_state = pretrained(f"rt_only_{single}", channels.generate(f"rt:{single}", 12000, 1), seed)
        pool = channels.generate(f"rt:{target}", 1000, 2)
        test = channels.generate(f"rt:{target}", 2000, 3)

        ls = lambda h, s, rng: grid.ls_estimate(h, s, rng)
        record(target, "ls_interp", 0, evaluate(ls, test))
        for n in FINETUNE_N:
            fit = pool[:n]
            for name, state in (("ft_cdl", cdl_state), ("ft_rt3", rt_state), ("ft_rt1", rt1_state)):
                model = estimators.Refiner()
                model.load_state_dict(state)
                if n:
                    model = estimators.train(model, fit, FINETUNE_STEPS, 3e-4, seed + n, batch=min(64, n))
                record(target, name, n, evaluate(lambda h, s, rng: estimators.refine(model, h, s, rng), test))
            if n:
                torch.manual_seed(seed + n)
                scratch = estimators.train(estimators.Refiner(), fit, SCRATCH_STEPS, 1e-3, seed + n, batch=min(64, n))
                record(target, "scratch", n, evaluate(lambda h, s, rng: estimators.refine(scratch, h, s, rng), test))
                record(target, "lmmse", n, evaluate(estimators.LMMSE(fit).estimate, test))

        with open(RUNS / f"loco_seed{seed}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(("target", "method", "n", "snr_db", "nmse_db"))
            w.writerows(rows)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
