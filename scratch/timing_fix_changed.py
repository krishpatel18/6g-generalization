"""Which street / lab datapoints does the timing fix change? (old chain vs new measured())"""
import sys
from pathlib import Path
import numpy as np
from sixg_gen import dichasus, grid, wideband
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cdl_gap import ENVS

env = sys.argv[1]
dataset, names, (r, c, tilt), fit_files, site_files, test_files, tx_z = ENVS[env]
grid.use_dichasus(r, c, tilt)
names = names or dichasus.array_names(dataset)
files = fit_files + (site_files or []) + test_files
for name in names:
    arr = dichasus.Array.named(name, dataset)
    cal, coh = dichasus.fit_calibration(arr, dichasus.load(arr, fit_files, dataset, tx_z))
    for f in files:
        raw = dichasus.load(arr, [f], dataset, tx_z)
        new = dichasus.measured(raw, cal)
        old = grid.normalize_power(wideband.to_grid(raw.csi[raw.usable] * cal[None]))
        d = np.max(np.abs(new - old), axis=(1, 2, 3))
        print(f"{env} {name} {f}: {len(d)} usable, changed {int((d > 1e-4).sum())}, max diff {d.max():.2e}", flush=True)
