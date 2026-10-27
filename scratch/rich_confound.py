"""Is the scattering variant's gain from scattering, or from its smaller city set / sample size?
Same 7 cities, 2000 each: specular-only (rt:) vs scattering (rtd:), array A."""
import sys
import numpy as np
sys.path.insert(0, "scripts")
from sixg_gen import channels, dichasus, grid, linear
from library import pooled
from dichasus_test import CAL_FILES, TEST_FILES, per_antenna
from dichasus_rich import CITIES
grid.use_dichasus()
arr = dichasus.Array.named("A")
cal, _ = dichasus.fit_calibration(arr, dichasus.load(arr, CAL_FILES))
test = dichasus.measured(dichasus.load(arr, TEST_FILES), cal)
for variant, f in (("spatial", lambda h: np.asarray(h)), ("per antenna", per_antenna)):
    t = f(test)
    for prefix, n in (("rt", 6000), ("rtd", 2000)):
        pool = pooled([linear.Stats.fit(f(channels.generate(f"{prefix}:{c}", n, 1, mmap=True)[:2000])) for c in CITIES])
        e = grid.nmse_db(linear.LMMSE(pool).estimate(t, np.full(len(t), 10.0), np.random.default_rng(1234)), t)
        print(f"{variant:12s} 7-city pool, {prefix:3s} (2000 each): {e:.2f} dB")
