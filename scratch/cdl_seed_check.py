"""Does seeding Sionna's own generator change CDL statistics? Compare batches with and without it."""
import numpy as np, torch, sionna.phy
from sixg_gen import channels, grid, linear
grid.NUM_ANTENNAS = 8  # original rooftop setting (a8_sc128)
orig = channels._cdl.__code__
def gen(n, seed, seeded):
    if not seeded:
        # emulate the old code path: no sionna seed (fresh process state is what the old caches saw)
        pass
    return channels._cdl(n, seed)
cached = np.asarray(channels.generate("cdl", 12000, 1, mmap=True)[:2000])
fresh = channels._cdl(2000, 1)
fresh = grid.normalize_power(fresh)
def div(h):  # how different are consecutive batches? spatial covariance per 64-batch, mean pairwise distance
    f = h[:, :, 2, 64]
    covs = [np.cov(f[i:i+64].T) for i in range(0, 1984, 64)]
    covs = [c / np.trace(c).real for c in covs]
    return np.mean([np.linalg.norm(covs[i] - covs[i+1]) for i in range(len(covs) - 1)])
print("batch-to-batch spatial covariance distance: cached", round(div(cached), 3), "| fresh seeded", round(div(fresh), 3))
test = channels.generate("rt:munich", 2000, 3)
for lab, h in (("cached CDL (unseeded era)", cached), ("fresh CDL (seeded)", fresh)):
    e = linear.LMMSE(linear.Stats.fit(h)).estimate(test, np.full(len(test), 10.0), np.random.default_rng(1234))
    print(f"{lab}: Munich test NMSE {grid.nmse_db(e, test):.2f} dB (2000-sample prior)")
