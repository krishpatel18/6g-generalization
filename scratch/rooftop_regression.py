"""The refactored rooftop placement reproduces the original cached channels (same draws)."""
import numpy as np
from sixg_gen import channels, grid
grid.NUM_ANTENNAS = 8
cached = channels.generate("rt:munich", 12000, 1, mmap=True)
h, _ = channels._raytraced("munich", 200, 1)
h = grid.normalize_power(h)
err = np.sum(np.abs(h - cached[:len(h)]) ** 2, axis=(1, 2, 3)) / np.sum(np.abs(cached[:len(h)]) ** 2, axis=(1, 2, 3))
print(f"{len(h)} channels: identical to cache (err < 1e-6) for {np.mean(err < 1e-6):.0%}, median err {np.median(err):.1e}")
