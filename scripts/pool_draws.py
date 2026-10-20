"""Post-hoc: do the +50 blend results hold for other 50-channel site pools?

usage: pool_draws.py <num_bs_antennas>
Re-scores CDL+50 and all-others-pool+50 with site pools [50:100], [100:150],
[150:200] of the seed-2 pool (the main results used [:50]). Gap to ceiling, 10 dB.
"""
import sys
from pathlib import Path
import numpy as np
from sixg_gen import channels, grid, linear
sys.path.insert(0, str(Path(__file__).resolve().parent))
from library import BUILT_IN, pooled  # noqa: E402
from make_library import library  # noqa: E402
from select_prior import test_nmse  # noqa: E402

DRAWS = ((50, 100), (100, 150), (150, 200))

if __name__ == "__main__":
    grid.NUM_ANTENNAS = int(sys.argv[1])
    cities = list(BUILT_IN) + library()
    stats = {c: linear.Stats.fit(channels.generate(f"rt:{c}", 12000, 1, mmap=True)) for c in cities}
    cdl = linear.Stats.fit(channels.generate("cdl", 12000, 1, mmap=True))
    edges = []
    for target in cities:
        test = channels.generate(f"rt:{target}", 2000, 4)
        site = channels.generate(f"rt:{target}", 1000, 2)
        ceiling = test_nmse(linear.LMMSE(stats[target]), test)
        allo = pooled([stats[c] for c in cities if c != target])
        row = []
        for a, b in DRAWS:
            c50 = test_nmse(linear.LMMSE(linear.Stats.blend(cdl, site[a:b])[0]), test) - ceiling
            p50 = test_nmse(linear.LMMSE(linear.Stats.blend(allo, site[a:b])[0]), test) - ceiling
            row.append((c50, p50))
        edges.append([c - p for c, p in row])
        print(f"{target:20s} " + "  ".join(f"cdl+50 {c:5.2f} pool+50 {p:5.2f}" for c, p in row), flush=True)
    e = np.array(edges)  # [city, draw], positive = pool better
    wins = (e > 0).sum(axis=0)
    keep = np.array([c != "rochester_brighton" for c in cities])
    print("pool+50 better than cdl+50 per draw: " + ", ".join(f"{w}/14" for w in wins))
    print("mean edge per draw (14 cities): " + ", ".join(f"{v:.2f}" for v in e.mean(axis=0)))
    print("mean edge per draw (13, no rochester): " + ", ".join(f"{v:.2f}" for v in e[keep].mean(axis=0)))
    print("ALL DONE", flush=True)
