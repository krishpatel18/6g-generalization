"""Do the cached facade-placement channels match a fresh regeneration with the current code?

usage: dichasus_regen_check.py [rt|rtd]
For each city: regenerate the first channels of rt:<city> n6000 s1 (all 14 cities) or
rtd:<city> n2000 s1 (the 7 scattering cities) and compare with the cache. Most channels should match to 1e-3; the rest differ by solver non-determinism or by
users whose paths change between runs (not separated here). Prints per-city share of channels within 1e-3 relative error of their closest cached
channel (index shifts from dropped users tolerated).
"""
import sys
from pathlib import Path

import numpy as np

from sixg_gen import channels, grid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from library import BUILT_IN  # noqa: E402
from make_library import library  # noqa: E402

WINDOW = 32
grid.use_dichasus()
rich = len(sys.argv) > 1 and sys.argv[1] == "rtd"
if rich:
    from dichasus_rich import CITIES, PER_CITY  # noqa: E402
cities, prefix, n, check = (CITIES, "rtd", PER_CITY, 64) if rich else (list(BUILT_IN) + library(), "rt", 6000, 256)
for city in cities:
    cached = channels.generate(f"{prefix}:{city}", n, 1, mmap=True)
    fresh = grid.normalize_power(channels._raytraced(city, check, 1, rich=rich)[0])
    # Users with no paths are dropped, and a marginal user can flip between runs, shifting every
    # later index; so each fresh channel is matched to its closest cached one within +-WINDOW.
    ref = np.asarray(cached[:len(fresh) + WINDOW])
    err = np.array([min(np.sum(np.abs(f - r) ** 2) / np.sum(np.abs(r) ** 2)
                        for r in ref[max(0, i - WINDOW):i + WINDOW + 1]) for i, f in enumerate(fresh)])
    print(f"{city:20s} within 1e-3: {np.mean(err < 1e-3):5.0%}  within 1e-1: {np.mean(err < 0.1):5.0%}  "
          f"median {np.median(err):.1e}", flush=True)
print("ALL DONE", flush=True)
