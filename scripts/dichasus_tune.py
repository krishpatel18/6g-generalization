"""Direction 1: tune the simulator's scattering from 50 site measurements (docs/PLAN.md,
"Direction 1"). Exploratory.

usage: dichasus_tune.py
For s in SCATTERING: a 7-city pool of "rtdmNN:" channels (users at the measured distances,
scattering coefficient s, diffraction on). For each of 4 draws of 50 site measurements, select
the s whose pool scores best on those 50; report the selected s on the test files, with and
without blending in the 50, against CDL and CDL + 50 on the same draws.
Writes runs/<grid tag>/dichasus_tune.csv: array, variant, estimator, scattering, draw, snr_db, nmse_db
"""

import csv
import sys
from pathlib import Path

import numpy as np

from sixg_gen import channels, dichasus, grid, linear

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dichasus_rich import CITIES, PER_CITY  # noqa: E402
from dichasus_test import CAL_FILES, SITE_DRAWS, SITE_FILES, SITE_N, TEST_FILES, nmse, per_antenna  # noqa: E402
from library import pooled  # noqa: E402
from loco import RUNS  # noqa: E402
from select_prior import pool_score  # noqa: E402

SCATTERING = (0.0, 0.15, 0.3, 0.5, 0.7)


def source(s: float, city: str) -> str:
    return f"rtdm:{city}" if s == 0.3 else f"rtdm{round(100 * s):02d}:{city}"  # 0.3 = the existing rtdm: sets


def run(array, variant, sims, site, test, rows, per_datapoint) -> None:
    def record(est, s, draw, scores):
        rows.extend((array, variant, est, s, draw, snr, round(v, 3)) for snr, v in scores.items())

    stats = {s: pooled([linear.Stats.fit(h) for h in hs]) for s, hs in sims["rt"].items()}
    cdl = linear.Stats.fit(sims["cdl"])
    record("cdl", "", 0, nmse(linear.LMMSE(cdl).estimate, test))
    on_test = {s: nmse(linear.LMMSE(st).estimate, test) for s, st in stats.items()}
    for s, scores in on_test.items():
        record("pool", s, 0, scores)
    for d in range(SITE_DRAWS):
        idx = np.sort(np.random.default_rng(100 + d).choice(len(site) // per_datapoint, SITE_N, replace=False))
        few = site[(idx[:, None] * per_datapoint + np.arange(per_datapoint)).ravel()]
        picked = min(stats, key=lambda s: pool_score(linear.LMMSE(stats[s]), few))
        record("tuned", picked, d, on_test[picked])
        record("tuned+50", picked, d, nmse(linear.LMMSE(linear.Stats.blend(stats[picked], few, d)[0]).estimate, test))
        record("cdl+50", "", d, nmse(linear.LMMSE(linear.Stats.blend(cdl, few, d)[0]).estimate, test))
    print(f"  {array} {variant}: done", flush=True)


def main() -> None:
    grid.use_dichasus()
    rt = {s: [channels.generate(source(s, c), PER_CITY, 1, mmap=True) for c in CITIES] for s in SCATTERING}
    cdl = channels.generate("cdl", 12000, 1, mmap=True)
    rows = []
    for name in "ABC":
        arr = dichasus.Array.named(name)
        cal, _ = dichasus.fit_calibration(arr, dichasus.load(arr, CAL_FILES))
        site = dichasus.measured(dichasus.load(arr, SITE_FILES), cal)
        test = dichasus.measured(dichasus.load(arr, TEST_FILES), cal)
        run(name, "spatial", {"rt": rt, "cdl": cdl}, site, test, rows, 1)
        run(name, "freq_only", {"rt": {s: [per_antenna(h) for h in hs] for s, hs in rt.items()}, "cdl": per_antenna(cdl)},
            per_antenna(site), per_antenna(test), rows, grid.NUM_ANTENNAS)
        out_dir = RUNS / grid.config_tag()
        with open(out_dir / "dichasus_tune.csv", "w", newline="") as f:
            out = csv.writer(f)
            out.writerow(("array", "variant", "estimator", "scattering", "draw", "snr_db", "nmse_db"))
            out.writerows(rows)
    summary(rows)
    print("ALL DONE", flush=True)


def summary(rows) -> None:
    v = {}
    for a, var, est, s, d, snr, x in rows:
        if snr == 10.0:
            v.setdefault((a, var, est), []).append((s, x))
    for var in ("spatial", "freq_only"):
        print(f"\n[{var}] NMSE dB at 10 dB SNR")
        for a in "ABC":
            pools = " ".join(f"s={s}:{x:.2f}" for s, x in v[(a, var, "pool")])
            mean = lambda est: np.mean([x for _, x in v[(a, var, est)]])  # noqa: E731
            picks = [s for s, _ in v[(a, var, "tuned")]]
            best = min(x for _, x in v[(a, var, "pool")])
            print(f"  array {a}: cdl {mean('cdl'):.2f} | pools {pools}")
            print(f"           picked s per draw {picks} | tuned {mean('tuned'):.2f} (regret {mean('tuned') - best:.2f}) "
                  f"| tuned+50 {mean('tuned+50'):.2f} | cdl+50 {mean('cdl+50'):.2f} | tuned - cdl {mean('tuned') - mean('cdl'):+.2f}")


if __name__ == "__main__":
    main()
