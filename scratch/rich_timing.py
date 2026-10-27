import time, numpy as np
from sixg_gen import channels, grid
grid.use_dichasus()
for rich in (False, True):
    t = time.time(); h, _ = channels._raytraced("munich", 128, 5, rich=rich)
    taps = np.fft.ifft(np.fft.ifftshift(h[:, :, 0], axes=-1), axis=-1)
    pdp = np.mean(np.abs(taps / np.sqrt(np.mean(np.abs(h) ** 2, axis=(1, 2, 3), keepdims=True)[:, :, 0])) ** 2, axis=(0, 1)); pdp /= pdp.sum()
    print(f"rich={rich}: {time.time()-t:.0f}s for 128, taps0..4 {np.round(100*pdp[:5], 1)}%")
