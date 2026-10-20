"""Post-hoc follow-ups to the library experiment (not pre-registered).

usage: library_followup.py <num_bs_antennas>

1. Fair few-shot baselines: CDL blended with 50 site channels, and the
   all-others RT pool blended with 50 site channels (5-fold CV blend).
2. Selection diagnostic: pick among the same 15 candidates by scoring the pool
   at 10 dB only (the SNR regret is judged at), with 50 and with 200 site channels.
NMSE at 10 dB SNR as gap to the in-domain ceiling. Prints a per-target table
and means.
"""

import sys
from pathlib import Path

import numpy as np

from sixg_gen import channels, grid, linear

sys.path.insert(0, str(Path(__file__).resolve().parent))
from library import BUILT_IN, pooled  # noqa: E402
from make_library import library  # noqa: E402
from select_prior import test_nmse  # noqa: E402


def pool_score_at(est: linear.LMMSE, pool: np.ndarray, snr: float) -> float:
    rng = np.random.default_rng(77)
    return grid.nmse_db(est.estimate(pool, np.full(len(pool), snr), rng), pool)


def main() -> None:
    cities = list(BUILT_IN) + library()
    stats = {c: linear.Stats.fit(channels.generate(f"rt:{c}", 12000, 1, mmap=True)) for c in cities}
    cdl = linear.Stats.fit(channels.generate("cdl", 12000, 1, mmap=True))
    cols = ("cdl+50", "pool+50", "sel10@50", "sel10@200", "best")
    print(f"{'target':20s}" + "".join(f"{c:>11s}" for c in cols), flush=True)
    table = []
    for target in cities:
        others = [c for c in cities if c != target]
        test = channels.generate(f"rt:{target}", 2000, 4)
        site = channels.generate(f"rt:{target}", 1000, 2)
        ceiling = test_nmse(linear.LMMSE(stats[target]), test)
        gap = lambda s: test_nmse(linear.LMMSE(s), test) - ceiling  # noqa: E731

        all_others = pooled([stats[c] for c in others])
        row = [gap(linear.Stats.blend(cdl, site[:50])[0]), gap(linear.Stats.blend(all_others, site[:50])[0])]
        cands = {"cdl": cdl, "all_others": all_others, **{c: stats[c] for c in others}}
        est = {k: linear.LMMSE(s) for k, s in cands.items()}
        on_test = {k: test_nmse(e, test) - ceiling for k, e in est.items()}
        for n in (50, 200):
            picked = min(est, key=lambda k: pool_score_at(est[k], site[:n], 10.0))
            row.append(on_test[picked])
        row.append(min(on_test.values()))
        table.append(row)
        print(f"{target:20s}" + "".join(f"{v:11.2f}" for v in row), flush=True)
    print(f"{'mean':20s}" + "".join(f"{v:11.2f}" for v in np.mean(table, axis=0)))
    print("(all-others pooled, no site data: see library.py; mean gap 0.59)")
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    grid.NUM_ANTENNAS = int(sys.argv[1])
    main()
