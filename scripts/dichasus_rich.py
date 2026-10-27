"""Exploratory: ray-traced priors with diffuse scattering and diffraction, on DICHASUS.

usage: dichasus_rich.py

Added after a 300-per-city dry run of dichasus_test.py had touched the test data
(docs/PLAN.md, "Scattering variant"). Same arrays, files, processing and claims as
dichasus_test.py, with the 4 built-in cities and 3 New York scenes re-simulated as
"rtd:" sources (scattering coefficient 0.3, Lambertian; edge diffraction), 2000 each.
Writes runs/<grid tag>/dichasus_rich.csv (same columns as dichasus_test.csv).
"""

import csv
import sys
from pathlib import Path

from sixg_gen import channels, dichasus, grid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dichasus_test import (CAL_FILES, SITE_FILES, TEST_FILES, per_antenna, run_variant,  # noqa: E402
                           summary)
from loco import RUNS  # noqa: E402

CITIES = ("munich", "etoile", "florence", "san_francisco", "wall_street", "midtown", "park_slope")
PER_CITY = 2000


def main() -> None:
    grid.use_dichasus()
    sims = {"cdl": channels.generate("cdl", 12000, 1, mmap=True),
            **{c: channels.generate(f"rtd:{c}", PER_CITY, 1, mmap=True) for c in CITIES}}
    rows, notes = [], []
    for name in "ABC":
        arr = dichasus.Array.named(name)
        cal_raw = dichasus.load(arr, CAL_FILES)
        cal, coherence = dichasus.fit_calibration(arr, cal_raw)
        fit = dichasus.measured(cal_raw, cal)
        site = dichasus.measured(dichasus.load(arr, SITE_FILES), cal)
        test = dichasus.measured(dichasus.load(arr, TEST_FILES), cal)
        notes.append(f"array {name}: coherence {coherence:.3f}; test {len(test)}")
        run_variant(name, "spatial", sims, fit, site, test, rows, breadth=(1, 2, 4))
        run_variant(name, "freq_only", {k: per_antenna(v) for k, v in sims.items()}, per_antenna(fit),
                    per_antenna(site), per_antenna(test), rows, grid.NUM_ANTENNAS, breadth=(1, 2, 4))
        with open(RUNS / grid.config_tag() / "dichasus_rich.csv", "w", newline="") as f:
            out = csv.writer(f)
            out.writerow(("array", "variant", "estimator", "detail", "draw", "snr_db", "nmse_db"))
            out.writerows(rows)
    summary(rows, notes, breadth=(1, 2, 4))
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
