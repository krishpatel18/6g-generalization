"""Facade/DICHASUS mode: does seeding Sionna's generator change the CDL prior? Array A, 10 dB, full grid."""
import sys
import numpy as np, sionna.phy
sys.path.insert(0, "scripts")
from sixg_gen import channels, dichasus, grid, linear
from dichasus_test import CAL_FILES, TEST_FILES
grid.use_dichasus()
arr = dichasus.Array.named("A")
cal, _ = dichasus.fit_calibration(arr, dichasus.load(arr, CAL_FILES))
test = dichasus.measured(dichasus.load(arr, TEST_FILES), cal)
score = lambda h: grid.nmse_db(linear.LMMSE(linear.Stats.fit(h)).estimate(test, np.full(len(test), 10.0), np.random.default_rng(1234)), test)
cached = np.asarray(channels.generate("cdl", 12000, 1, mmap=True))
print("cached 12000:", round(score(cached), 2), "| cached first 4000:", round(score(cached[:4000]), 2))
seeded = grid.normalize_power(channels._cdl(4000, 1))
print("fresh seeded 4000:", round(score(seeded), 2))
cls = type(sionna.phy.config)
orig = cls.seed
cls.seed = property(orig.fget, lambda self, v: None)  # make the seed assignment a no-op (the old code path)
unseeded = grid.normalize_power(channels._cdl(4000, 1))
cls.seed = orig
print("fresh unseeded 4000:", round(score(unseeded), 2))
for lab, h in (("cached", cached[:4000]), ("seeded", seeded), ("unseeded", unseeded)):
    f = h[:, :, 0, 64]; c = f.T @ f.conj() / len(f); lam = np.linalg.eigvalsh(c)[::-1]
    print(f"  {lab}: top-3 spatial eig share {lam[:3].sum()/lam.sum():.3f}")
