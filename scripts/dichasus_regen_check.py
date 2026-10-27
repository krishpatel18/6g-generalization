"""Do the cached facade-placement channels match a fresh regeneration with the current code?

usage: dichasus_regen_check.py
For each of the 14 cities: regenerate the first 256 channels of rt:<city> n6000 s1 and compare
with the cache. Matches are expected up to Sionna RT's non-deterministic path solver (a few
percent of channels differ slightly). Prints per-city share of channels within 1e-3 relative error.
"""
import sys
from pathlib import Path

import numpy as np

from sixg_gen import channels, grid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from library import BUILT_IN  # noqa: E402
from make_library import library  # noqa: E402

grid.use_dichasus()
for city in list(BUILT_IN) + library():
    cached = channels.generate(f"rt:{city}", 6000, 1, mmap=True)
    fresh = grid.normalize_power(channels._raytraced(city, 256, 1)[0])
    ref = np.asarray(cached[:len(fresh)])
    err = np.sum(np.abs(fresh - ref) ** 2, axis=(1, 2, 3)) / np.sum(np.abs(ref) ** 2, axis=(1, 2, 3))
    print(f"{city:20s} within 1e-3: {np.mean(err < 1e-3):5.0%}  within 1e-1: {np.mean(err < 0.1):5.0%}  "
          f"median {np.median(err):.1e}", flush=True)
print("ALL DONE", flush=True)
