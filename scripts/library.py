"""Leave-one-city-out over the whole scene library (LMMSE only, no training).

usage: library.py <num_bs_antennas>

For each target city (4 built-in + scripts/scene_library.txt), with no site data:
  - breadth curve: pool the statistics of k random other cities (equal weight),
    k in BREADTH_K, DRAWS draws each; zero-shot NMSE on the target's test sites
  - candidates: CDL, every other city alone, all other cities pooled
  - selection: the candidate a 50-channel site pool picks, and its regret
  - baselines: LMMSE from the 50 site channels alone; in-domain ceiling (12k)
NMSE in dB at 10 dB SNR. Writes runs/<grid tag>/library.csv (one row per
target x estimator) and prints a summary.
"""

import csv
import sys
from pathlib import Path

import numpy as np

from sixg_gen import channels, grid, linear

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loco import RUNS  # noqa: E402
from make_library import library  # noqa: E402
from select_prior import pool_score, test_nmse  # noqa: E402

BUILT_IN = ("munich", "etoile", "florence", "san_francisco")
BREADTH_K = (1, 2, 4, 8, 13)
DRAWS = 8


def pooled(parts: list[linear.Stats]) -> linear.Stats:
    return linear.Stats(sum(p.r_hp for p in parts) / len(parts), sum(p.r_pp for p in parts) / len(parts))


def main() -> None:
    cities = list(BUILT_IN) + library()
    stats = {c: linear.Stats.fit(channels.generate(f"rt:{c}", 12000, 1, mmap=True)) for c in cities}
    print(f"fitted {len(stats)} cities", flush=True)
    cdl = linear.Stats.fit(channels.generate("cdl", 12000, 1, mmap=True))
    rng = np.random.default_rng(0)
    rows = []
    for target in cities:
        others = [c for c in cities if c != target]
        test = channels.generate(f"rt:{target}", 2000, 4)
        pool = channels.generate(f"rt:{target}", 1000, 2)[:50]
        score = lambda s: test_nmse(linear.LMMSE(s), test)  # noqa: E731

        for k in BREADTH_K:
            draws = [others] if k >= len(others) else [list(rng.choice(others, k, replace=False)) for _ in range(DRAWS)]
            for i, subset in enumerate(draws):
                rows.append((target, f"pool_k{k}", i, "+".join(subset), round(score(pooled([stats[c] for c in subset])), 3)))

        cands = {"cdl": cdl, "all_others": pooled([stats[c] for c in others]), **{f"city_{c}": stats[c] for c in others}}
        on_test = {name: score(s) for name, s in cands.items()}
        for name, v in on_test.items():
            rows.append((target, "candidate", 0, name, round(v, 3)))
        on_pool = {name: pool_score(linear.LMMSE(s), pool) for name, s in cands.items()}
        picked = min(on_pool, key=on_pool.get)
        rows.append((target, "selected", 0, picked, round(on_test[picked], 3)))
        rows.append((target, "own_50", 0, "", round(score(linear.Stats.fit(pool)), 3)))
        rows.append((target, "ceiling", 0, "", round(score(stats[target]), 3)))
        best = min(on_test, key=on_test.get)
        print(f"{target:20s} cdl {on_test['cdl']:6.2f}  all_others {on_test['all_others']:6.2f}  "
              f"best {best} {on_test[best]:6.2f}  picked {picked} {on_test[picked]:6.2f}  "
              f"ceiling {rows[-1][-1]:6.2f}", flush=True)

    path = RUNS / grid.config_tag() / "library.csv"
    with open(path, "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(("target", "estimator", "draw", "detail", "nmse_db"))
        out.writerows(rows)
    summarize(rows, cities)
    print("ALL DONE", flush=True)


def summarize(rows, cities) -> None:
    by = {}
    for t, est, _, detail, v in rows:
        by.setdefault((t, est), []).append((detail, v))
    print("\nBreadth curve (gap to in-domain ceiling, dB; mean over targets):")
    print("   k   mean-draw   worst-draw")
    for k in BREADTH_K:
        gaps = [(np.mean([v for _, v in by[(t, f'pool_k{k}')]]) - by[(t, 'ceiling')][0][1],
                 max(v for _, v in by[(t, f'pool_k{k}')]) - by[(t, 'ceiling')][0][1]) for t in cities]
        print(f"  {k:2d}   {np.mean([g[0] for g in gaps]):8.2f}   {np.mean([g[1] for g in gaps]):9.2f}")
    sel = [by[(t, "selected")][0][1] - min(v for _, v in by[(t, "candidate")]) for t in cities]
    cdl = [by[(t, "candidate")][[d for d, _ in by[(t, "candidate")]].index("cdl")][1] - by[(t, "ceiling")][0][1] for t in cities]
    allo = [by[(t, "candidate")][[d for d, _ in by[(t, "candidate")]].index("all_others")][1] - by[(t, "ceiling")][0][1] for t in cities]
    own = [by[(t, "own_50")][0][1] - by[(t, "ceiling")][0][1] for t in cities]
    print(f"\nGap to ceiling, mean over {len(cities)} targets: CDL {np.mean(cdl):.2f} | all others pooled "
          f"{np.mean(allo):.2f} | 50 site samples alone {np.mean(own):.2f}")
    print(f"Selection regret vs best candidate: mean {np.mean(sel):.2f} dB, worst {max(sel):.2f} dB, "
          f"exact {sum(s == 0 for s in sel)}/{len(sel)}")


if __name__ == "__main__":
    grid.NUM_ANTENNAS = int(sys.argv[1])
    main()
