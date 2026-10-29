"""Lab (0d5x) positions are 2D: calibration coherence vs assumed transmitter height, on the first
6000 usable fit datapoints (the amendment in docs/PLAN.md, 'Follow-on ... pre-registered'). Re-run
2026-10-29 to log the sweep the amendment quotes."""
import numpy as np
from sixg_gen import dichasus, grid

grid.use_dichasus(2, 8, 0.0)
for name in dichasus.array_names("0d5x"):
    arr = dichasus.Array.named(name, "0d5x")
    raw = dichasus.load(arr, ["0d54"], "0d5x", 0.0)
    keep = np.flatnonzero(raw.usable)[:6000]
    out = []
    for z in np.round(np.arange(-1.5, 1.01, 0.1), 2):
        pos = raw.pos[keep].copy()
        pos[:, 2] = z
        sub = dichasus.Raw(raw.csi[keep], pos, np.ones(len(keep), bool))
        out.append((z, dichasus.fit_calibration(arr, sub)[1]))
    best = max(out, key=lambda t: t[1])
    print(f"{name}: best z {best[0]:+.1f} m (coherence {best[1]:.3f}); "
          + " ".join(f"{z:+.1f}:{c:.3f}" for z, c in out), flush=True)
