"""Can 50 site measurements pick the right pre-training prior?

usage: select_prior.py <num_bs_antennas>

For each city and pre-training variant (as in fresh_sites.py), score each arm's
zero-shot LMMSE on the 50-channel site pool alone (noisy pilots at 0/10/20 dB,
no test data), pick the best, and compare with the arm that actually wins on the
fresh test sites. Reports hit rate and, on a miss, what the wrong pick costs.
"""

import sys
from pathlib import Path

import numpy as np

from sixg_gen import channels, grid, linear

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fresh_sites import PRETRAIN_SEED, TEST_SEED  # noqa: E402
from loco import SINGLE_SOURCE  # noqa: E402

POOL_N = 50
SCORE_SNR_DB = (0.0, 10.0, 20.0)
REPORT_SNR_DB = 10.0


def pool_score(est: linear.LMMSE, pool: np.ndarray) -> float:
    """Mean linear-scale NMSE on the pool across SNRs (what a site could measure)."""
    total = 0.0
    for snr in SCORE_SNR_DB:
        rng = np.random.default_rng(77)
        total += 10 ** (grid.nmse_db(est.estimate(pool, np.full(len(pool), snr), rng), pool) / 10)
    return total / len(SCORE_SNR_DB)


def test_nmse(est: linear.LMMSE, test: np.ndarray) -> float:
    rng = np.random.default_rng(1234)
    return grid.nmse_db(est.estimate(test, np.full(len(test), REPORT_SNR_DB), rng), test)


def main() -> None:
    cities = channels.RT_SCENES
    hits, regrets = 0, []
    for variant, seed in PRETRAIN_SEED.items():
        fit = {}
        def stats(source, rows=None):
            if (source, rows) not in fit:
                fit[(source, rows)] = linear.Stats.fit(channels.generate(source, 12000, seed, mmap=True)[:rows])
            return fit[(source, rows)]
        for target in cities:
            others = [c for c in cities if c != target]
            parts = [stats(f"rt:{c}", 4000) for c in others]
            arms = {
                "cdl": stats("cdl"),
                "rt3": linear.Stats(sum(p.r_hp for p in parts) / 3, sum(p.r_pp for p in parts) / 3),
                "rt1": stats(f"rt:{SINGLE_SOURCE[target]}"),
            }
            pool = channels.generate(f"rt:{target}", 1000, 2)[:POOL_N]
            test = channels.generate(f"rt:{target}", 2000, TEST_SEED)
            est = {a: linear.LMMSE(s) for a, s in arms.items()}
            on_pool = {a: pool_score(e, pool) for a, e in est.items()}
            on_test = {a: test_nmse(e, test) for a, e in est.items()}
            picked, best = min(on_pool, key=on_pool.get), min(on_test, key=on_test.get)
            regret = on_test[picked] - on_test[best]
            hits += picked == best
            regrets.append(regret)
            print(f"{variant} {target:14s} pool picks {picked:4s} test best {best:4s} "
                  f"regret {regret:5.2f} dB | test @10dB " +
                  " ".join(f"{a}:{v:6.2f}" for a, v in on_test.items()), flush=True)
    print(f"hit rate {hits}/{len(regrets)}; mean regret {np.mean(regrets):.2f} dB, worst {max(regrets):.2f} dB")
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    grid.NUM_ANTENNAS = int(sys.argv[1])
    main()
