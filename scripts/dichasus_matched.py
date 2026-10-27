"""Exploratory: ray-traced priors with scattering AND users placed at the measured site's
distances, on DICHASUS (docs/PLAN.md, "Matched-placement probe").

usage: dichasus_matched.py
Same arrays, files, processing and summary as dichasus_rich.py, with "rtdm:" sources.
Writes runs/<grid tag>/dichasus_matched.csv.
"""

import csv
import sys
from pathlib import Path

from sixg_gen import channels, dichasus, grid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dichasus_rich import CITIES, PER_CITY  # noqa: E402
from dichasus_test import CAL_FILES, SITE_FILES, TEST_FILES, per_antenna, run_variant, summary  # noqa: E402
from loco import RUNS  # noqa: E402


def main() -> None:
    grid.use_dichasus()
    sims = {"cdl": channels.generate("cdl", 12000, 1, mmap=True),
            **{c: channels.generate(f"rtdm:{c}", PER_CITY, 1, mmap=True) for c in CITIES}}
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
        with open(RUNS / grid.config_tag() / "dichasus_matched.csv", "w", newline="") as f:
            out = csv.writer(f)
            out.writerow(("array", "variant", "estimator", "detail", "draw", "snr_db", "nmse_db"))
            out.writerows(rows)
    summary(rows, notes, breadth=(1, 2, 4))
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
