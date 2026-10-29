"""Factory: where in the hall are the fit block, the 50-measurement site block (last 20% of cf02) and the
test recording (cf06)? Per array, usable datapoints only (same SNR / interpolation-age rule as load)."""
import numpy as np
from sixg_gen import dichasus

ds = "cf0x"
z = {f: np.load(dichasus._file(ds, f)) for f in ("cf02", "cf06")}
for name in dichasus.array_names(ds):
    arr = dichasus.Array.named(name, ds)
    sets = {}
    for f in ("cf02", "cf06"):
        snr = z[f]["snr"][:, arr.order].mean(1)
        ok = (snr >= dichasus.MIN_SNR_DB) & (z[f]["interp_age"] < dichasus.MAX_INTERP_AGE_S)
        pos = z[f]["pos"]
        if f == "cf02":
            n = len(pos); cut = int(0.8 * n)
            sets["fit"] = pos[:cut][ok[:cut]]; sets["site"] = pos[cut:][ok[cut:]]
        else:
            sets["test"] = pos[ok]
    loc = arr.elements.mean(0)
    print(f"\n{name} (array at x {loc[0]:.1f}, y {loc[1]:.1f})")
    for k, p in sets.items():
        d = np.linalg.norm(p[:, :2] - loc[:2], axis=1)
        print(f"  {k:5s} n {len(p):6d}  x {np.percentile(p[:,0],5):6.1f}..{np.percentile(p[:,0],95):6.1f}  "
              f"y {np.percentile(p[:,1],5):6.1f}..{np.percentile(p[:,1],95):6.1f}  dist to array median {np.median(d):5.1f} m")
    # share of test points within 2 m of some site point
    s, t = sets["site"][:, :2], sets["test"][::10, :2]
    near = np.array([np.min(np.linalg.norm(s - q, axis=1)) < 2.0 for q in t])
    print(f"  test points within 2 m of a site point: {100 * near.mean():.0f}%")
