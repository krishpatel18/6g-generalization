"""Post-hoc: does the calibrated twin at the exact test positions match the measured channel there?
Per North-array test datapoint: |<h_twin, h_meas>| / (|h_twin| |h_meas|) over antennas at each grid
subcarrier (1 = same spatial signature up to a common phase), and the same with the twin's antenna order
mirrored, as a check on orientation. Recomputes stage 2's North test set."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from hallway_twin import A_MAX_Y, B_MIN_Y, CEILING_FRACTION, MAX_A, MAX_TEST, measured_chunks  # noqa: E402

from sixg_gen import dichasus, grid, hallway  # noqa: E402

grid.use_dichasus(hallway.ROWS, hallway.COLS, 0.0, hallway.CARRIER_HZ)
rng = np.random.default_rng(0)
data = hallway.load()
keep, y = data["keep"], data["pos"][:, 1]
a_idx = np.sort(rng.choice(np.flatnonzero(keep & (y < A_MAX_Y)), MAX_A, replace=False))
b_idx = np.flatnonzero(keep & (y > B_MIN_Y))
b_test = b_idx[int(CEILING_FRACTION * len(b_idx)):]
exact = np.load(hallway.DIR / "twin_exact_grid.npy", mmap_mode="r")
for i, arr in enumerate(hallway.arrays()):
    ok = hallway.usable(data, arr, b_test)
    sel = np.sort(np.random.default_rng(1 + i).choice(np.flatnonzero(ok), min(MAX_TEST, ok.sum()), replace=False))
    sel = sel[:300]
    cal, coh = dichasus.fit_calibration(dichasus.Array.named(arr["name"], "dcxx"), hallway.raw(data, arr, a_idx))
    meas = measured_chunks(data, arr, b_test[sel], cal)[:, :, 0]          # [n, 32, 128]
    tw = np.asarray(exact[sel, i])[:, :, 0]                                  # [n, 32, 128]
    def sim(a, b):
        num = np.abs(np.sum(a.conj() * b, axis=1))
        return np.median(num / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12))
    o = np.arange(32).reshape(8, 4)
    rand = np.median([sim(meas, meas[np.random.default_rng(k).permutation(len(meas))]) for k in range(3)])
    print(f"{arr['name'][:5]} (coherence {coh:.2f}): twin vs measured, same position {sim(tw, meas):.3f}; "
          f"cols mirrored {sim(tw[:, o[::-1].ravel()], meas):.3f}; rows mirrored {sim(tw[:, o[:, ::-1].ravel()], meas):.3f}; "
          f"measured vs measured at random other positions {rand:.3f}", flush=True)
