"""Does run 2b's RT-vs-CDL result hold on fresh base-station sites?

usage: fresh_sites.py <num_bs_antennas>

The `lin` rows behind the realism claim are deterministic given the data, so the
check is to redraw the data, not the seed. Two variants, both on new test sites
(seed 4, 2k channels per city):
  A - original pre-training draws (seed 1)
  B - independent pre-training draws (seed 5) for every source, CDL included
Rows: LMMSE from each arm's statistics at N=0, and blended with 50 target
channels (seed-2 pool, as in loco.py). NMSE in dB at 0/10/20 dB SNR.
Writes runs/<grid tag>/fresh_sites.csv.
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np

from sixg_gen import channels, grid, linear

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loco import RUNS, SINGLE_SOURCE, evaluate  # noqa: E402

TEST_SEED = 4
PRETRAIN_SEED = {"A": 1, "B": 5}


def main() -> None:
    cities = channels.RT_SCENES
    t0 = time.time()
    for seed in PRETRAIN_SEED.values():
        channels.generate("cdl", 12000, seed)
        for c in cities:
            channels.generate(f"rt:{c}", 12000, seed)
    for c in cities:
        channels.generate(f"rt:{c}", 2000, TEST_SEED)
    print(f"data ready in {time.time()-t0:.0f}s", flush=True)

    city_stats = {}

    def stats(source: str, seed: int, rows=slice(None)) -> linear.Stats:
        key = (source, seed, rows.stop)
        if key not in city_stats:
            city_stats[key] = linear.Stats.fit(channels.generate(source, 12000, seed, mmap=True)[rows])
        return city_stats[key]

    def mean(parts: list[linear.Stats]) -> linear.Stats:
        return linear.Stats(sum(p.r_hp for p in parts) / len(parts), sum(p.r_pp for p in parts) / len(parts))

    out = []
    for variant, seed in PRETRAIN_SEED.items():
        for target in cities:
            t = time.time()
            test = channels.generate(f"rt:{target}", 2000, TEST_SEED)
            pool = channels.generate(f"rt:{target}", 1000, 2)[:50]
            others = [c for c in cities if c != target]
            arms = {
                "cdl": stats("cdl", seed),
                # equal 4k from each of three cities -> average of their covariances
                "rt3": mean([stats(f"rt:{c}", seed, slice(4000)) for c in others]),
                "rt1": stats(f"rt:{SINGLE_SOURCE[target]}", seed),
                "in_domain": stats(f"rt:{target}", seed),
            }
            for arm, st in arms.items():
                for n in ((0,) if arm == "in_domain" else (0, 50)):
                    adapted, w = linear.Stats.blend(st, pool[:n], seed)
                    scores = evaluate(linear.LMMSE(adapted).estimate, test)
                    out += [(variant, target, arm, n, w, snr, round(v, 3)) for snr, v in scores.items()]
                    print(f"  {variant} {target:14s} {arm:9s} N={n:<3d} w={w:<4} " +
                          "  ".join(f"{s:>4.0f}dB:{v:7.2f}" for s, v in scores.items()), flush=True)
            print(f"  {variant} {target} done in {time.time()-t:.0f}s", flush=True)
        city_stats.clear()  # covariances are large; keep one variant's at a time

    path = RUNS / grid.config_tag() / "fresh_sites.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("variant", "target", "arm", "n", "blend_w", "snr_db", "nmse_db"))
        w.writerows(out)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    grid.NUM_ANTENNAS = int(sys.argv[1])
    main()
