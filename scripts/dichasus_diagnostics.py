"""Diagnostics behind the sim-to-real write-up (docs/PLAN.md, "Sim-to-real" sections).

usage: dichasus_diagnostics.py      (log: runs/dichasus_diagnostics.log)

1. Scattering control, array A, 10 dB: 7-city pools of specular channels with 256 users per
   base-station site (rt:, first 2000), specular with 32 per site (rt32:, as many sites as
   rtd:), and scattering + diffraction with 32 per site (rtd:).
2. How clean each source is: share of energy in the first delay tap (160 ns) and of spatial
   power in the top 3 eigenvectors, for every city, CDL and the measured test channels.
3. Calibration coherence per array, excluding the reference antenna.
4. Measured user geometry vs the simulated facade placement.
"""
import sys
from pathlib import Path

import numpy as np

from sixg_gen import channels, dichasus, grid, linear

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dichasus_rich import CITIES, PER_CITY  # noqa: E402
from dichasus_test import CAL_FILES, TEST_FILES, per_antenna  # noqa: E402
from library import BUILT_IN, pooled  # noqa: E402
from make_library import library  # noqa: E402


def cleanliness(h) -> tuple[float, float]:
    h = np.asarray(h)[:1000, :, 0]
    taps = np.fft.ifft(np.fft.ifftshift(h, axes=-1), axis=-1)
    pdp = np.mean(np.abs(taps) ** 2, axis=(0, 1))
    f = h[:, :, grid.NUM_SUBCARRIERS // 2]
    lam = np.linalg.eigvalsh(f.T @ f.conj() / len(f))[::-1]
    return 100 * pdp[0] / pdp.sum(), 100 * lam[:3].sum() / lam.sum()


def main() -> None:
    grid.use_dichasus()
    arrays = {}
    print("== calibration coherence (reference antenna excluded)")
    for name in "ABC":
        arr = dichasus.Array.named(name)
        cal_raw = dichasus.load(arr, CAL_FILES)
        cal, coherence = dichasus.fit_calibration(arr, cal_raw)
        test_raw = dichasus.load(arr, TEST_FILES)
        arrays[name] = (arr, test_raw, dichasus.measured(test_raw, cal))
        print(f"  array {name}: {coherence:.3f}", flush=True)

    print("== scattering control, array A, 10 dB (7 cities x 2000)")
    test = arrays["A"][2]
    for label, src, n in (("specular, 256 users/site (rt:)", "rt", 6000), ("specular, 32 users/site (rt32:)", "rt32", PER_CITY),
                          ("scattering+diffraction, 32/site (rtd:)", "rtd", PER_CITY)):
        sims = [channels.generate(f"{src}:{c}", n, 1, mmap=True)[:PER_CITY] for c in CITIES]
        cells = []
        for variant, f in (("full grid", np.asarray), ("per antenna", per_antenna)):
            pool = pooled([linear.Stats.fit(f(h)) for h in sims])
            t = f(test)
            cells.append(f"{variant} {grid.nmse_db(linear.LMMSE(pool).estimate(t, np.full(len(t), 10.0), np.random.default_rng(1234)), t):6.2f}")
        print(f"  {label:40s} " + " | ".join(cells), flush=True)

    print("== cleanliness: first-tap energy %, top-3 spatial eigenvalues %")
    for name in "ABC":
        print(f"  measured {name:13s} " + "{:5.1f} {:5.1f}".format(*cleanliness(arrays[name][2])))
    print("  cdl                    " + "{:5.1f} {:5.1f}".format(*cleanliness(channels.generate("cdl", 12000, 1, mmap=True))))
    for c in list(BUILT_IN) + library():
        line = "{:5.1f} {:5.1f}".format(*cleanliness(channels.generate(f"rt:{c}", 6000, 1, mmap=True)))
        if c in CITIES:
            line += "   rtd: {:5.1f} {:5.1f}".format(*cleanliness(channels.generate(f"rtd:{c}", PER_CITY, 1, mmap=True)))
        print(f"  rt:{c:20s} {line}", flush=True)

    print("== user geometry: horizontal distance (m) and angle below the array (deg), median [10th-90th pct]")
    for name in "ABC":
        arr, raw, _ = arrays[name]
        centre = arr.elements.mean(axis=0)
        d = raw.pos[raw.usable] - centre
        horiz = np.hypot(d[:, 0], d[:, 1])
        below = np.degrees(np.arctan2(-d[:, 2], horiz))
        print(f"  measured {name}: {np.median(horiz):5.1f} [{np.percentile(horiz, 10):.1f}-{np.percentile(horiz, 90):.1f}] m, "
              f"{np.median(below):5.1f} [{np.percentile(below, 10):.1f}-{np.percentile(below, 90):.1f}] deg")
    lo, hi = channels.FACADE_USER_RANGE_M
    r = np.sqrt(np.random.default_rng(0).uniform(lo**2, hi**2, 100000))
    below = np.degrees(np.arctan2(channels.FACADE_HEIGHT_M - channels.USER_HEIGHT_M, r))
    print(f"  simulated:  {np.median(r):5.1f} [{np.percentile(r, 10):.1f}-{np.percentile(r, 90):.1f}] m, "
          f"{np.median(below):5.1f} [{np.percentile(below, 10):.1f}-{np.percentile(below, 90):.1f}] deg (flat ground)")
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
