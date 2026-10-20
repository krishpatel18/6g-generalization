"""Treat a new city as an unseen site: which prior transfers, and can 50 measurements tell?

usage: new_city.py <num_bs_antennas> <city>     (city: a built-in scene or data/scenes/<city>)

Candidate priors (12k channels each, seed 1): CDL; all four built-in cities
(3k each); each built-in city alone. For the new city: zero-shot LMMSE from each
prior on 2k test channels (seed 4), the prior a 50-channel site pool (seed 2)
would pick, LMMSE from those 50 alone, and the in-domain ceiling (12k, seed 1).
"""

import sys
from pathlib import Path

import numpy as np

from sixg_gen import channels, grid, linear

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_prior import pool_score, test_nmse  # noqa: E402

BUILT_IN = ("munich", "etoile", "florence", "san_francisco")


def main(city: str) -> None:
    for src, n, seed in [(f"rt:{city}", 12000, 1), (f"rt:{city}", 1000, 2), (f"rt:{city}", 2000, 4)]:
        channels.generate(src, n, seed)
    fits = {c: linear.Stats.fit(channels.generate(f"rt:{c}", 12000, 1, mmap=True)) for c in BUILT_IN}
    quarter = [linear.Stats.fit(channels.generate(f"rt:{c}", 12000, 1, mmap=True)[:3000]) for c in BUILT_IN]
    priors = {"cdl": linear.Stats.fit(channels.generate("cdl", 12000, 1, mmap=True)),
              "rt_all4": linear.Stats(sum(q.r_hp for q in quarter) / 4, sum(q.r_pp for q in quarter) / 4),
              **{f"rt_{c}": s for c, s in fits.items()}}
    pool = channels.generate(f"rt:{city}", 1000, 2)[:50]
    test = channels.generate(f"rt:{city}", 2000, 4)

    est = {k: linear.LMMSE(s) for k, s in priors.items()}
    on_test = {k: test_nmse(e, test) for k, e in est.items()}
    on_pool = {k: pool_score(e, pool) for k, e in est.items()}
    print(f"{city}: NMSE dB at 10 dB SNR on new-city test sites, no site data")
    for k in sorted(on_test, key=on_test.get):
        print(f"  {k:18s} {on_test[k]:7.2f}")
    picked, best = min(on_pool, key=on_pool.get), min(on_test, key=on_test.get)
    print(f"  50-sample pool picks {picked} (test {on_test[picked]:.2f}); best is {best} ({on_test[best]:.2f})")
    print(f"  LMMSE from the 50 samples alone   {test_nmse(linear.LMMSE(linear.Stats.fit(pool)), test):7.2f}")
    adapted, w = linear.Stats.blend(priors[picked], pool)
    print(f"  picked prior blended with the 50  {test_nmse(linear.LMMSE(adapted), test):7.2f} (w={w})")
    ceiling = linear.LMMSE(linear.Stats.fit(channels.generate(f"rt:{city}", 12000, 1, mmap=True)))
    print(f"  in-domain ceiling (12k)           {test_nmse(ceiling, test):7.2f}")
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    grid.NUM_ANTENNAS = int(sys.argv[1])
    main(sys.argv[2])
