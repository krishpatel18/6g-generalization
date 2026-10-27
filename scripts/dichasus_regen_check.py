"""Do the cached facade-placement channels match a fresh regeneration with the current code?

usage: dichasus_regen_check.py [rt|rtd]
For each city: regenerate the first channels of rt:<city> n6000 s1 (all 14 cities) or
rtd:<city> n2000 s1 (the 7 scattering cities) and compare with the cache. Matches are expected up to Sionna RT's non-deterministic path solver (a few
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
rich = len(sys.argv) > 1 and sys.argv[1] == "rtd"
if rich:
    from dichasus_rich import CITIES, PER_CITY  # noqa: E402
cities, prefix, n, check = (CITIES, "rtd", PER_CITY, 64) if rich else (list(BUILT_IN) + library(), "rt", 6000, 256)
for city in cities:
    cached = channels.generate(f"{prefix}:{city}", n, 1, mmap=True)
    fresh = grid.normalize_power(channels._raytraced(city, check, 1, rich=rich)[0])
    ref = np.asarray(cached[:len(fresh)])
    err = np.sum(np.abs(fresh - ref) ** 2, axis=(1, 2, 3)) / np.sum(np.abs(ref) ** 2, axis=(1, 2, 3))
    print(f"{city:20s} within 1e-3: {np.mean(err < 1e-3):5.0%}  within 1e-1: {np.mean(err < 0.1):5.0%}  "
          f"median {np.median(err):.1e}", flush=True)
print("ALL DONE", flush=True)
