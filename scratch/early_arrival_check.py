"""After shifting each datapoint to its strongest tap, how often does a tap within 10 dB of that peak
sit 21-100 taps (420 ns - 2 us) earlier, i.e. outside wideband.to_grid's first-arrival search?"""
import sys
from pathlib import Path
import numpy as np
from sixg_gen import dichasus, grid
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cdl_gap import ENVS

env = sys.argv[1]
dataset, names, (r, c, tilt), fit_files, site_files, test_files, tx_z = ENVS[env]
grid.use_dichasus(r, c, tilt)
for name in names or dichasus.array_names(dataset):
    arr = dichasus.Array.named(name, dataset)
    for f in fit_files + test_files:
        raw = dichasus.load(arr, [f], dataset, tx_z)
        h = dichasus._to_peak(raw.csi[raw.usable])
        taps = np.fft.ifft(np.fft.ifftshift(h, axes=-1), axis=-1)
        pdp = np.sum(np.abs(taps) ** 2, axis=(1, 2))
        early = pdp[:, -100:-20] >= 0.1 * pdp[:, :1]
        print(f"{env} {name} {f}: {len(pdp)} usable, early arrival beyond the search in {100 * early.any(1).mean():.2f}%", flush=True)
