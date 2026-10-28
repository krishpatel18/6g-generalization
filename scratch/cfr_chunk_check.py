"""Chunked cfr reproduces the cached channels (same draws), so existing caches stay valid."""
import numpy as np
from sixg_gen import channels, grid
grid.use_dichasus()
for src, kw in (("rtdm:munich", dict(rich=True, user_range_m=channels.MATCHED_USER_RANGE_M)), ("rt:munich", {})):
    cached = channels.generate(src, 2000 if src.startswith("rtdm") else 6000, 1, mmap=True)
    fresh = grid.normalize_power(channels._raytraced("munich", 64, 1, **kw)[0])
    ref = np.asarray(cached[:len(fresh) + 16])
    err = [min(np.sum(np.abs(f - r) ** 2) / np.sum(np.abs(r) ** 2) for r in ref[max(0, i - 16):i + 17]) for i, f in enumerate(fresh)]
    print(src, f"within 1e-3: {np.mean(np.array(err) < 1e-3):.0%}")
