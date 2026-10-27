"""The packaged pipeline reproduces the pilot (array A) and reports calibration coherence per array."""
import numpy as np
from sixg_gen import channels, dichasus, grid, linear
grid.use_dichasus()
for name in "ABC":
    arr = dichasus.Array.named(name)
    cal_raw = dichasus.load(arr, ["d010", "d012", "d014"])
    cal, coh = dichasus.fit_calibration(arr, cal_raw)
    print(f"array {name}: coherence {coh:.3f}, usable {cal_raw.usable.sum()}/{len(cal_raw.usable)}")
    if name != "A":
        continue
    fit = dichasus.measured(cal_raw, cal)
    test = dichasus.measured(dichasus.load(arr, ["d011", "d013"]), cal)
    cdl = channels.generate("cdl", 4000, 77)
    for label, est in (("cdl", linear.LMMSE(linear.Stats.fit(cdl))), ("own_fit", linear.LMMSE(linear.Stats.fit(fit)))):
        rng = np.random.default_rng(1234)
        print(f"  {label}: 10 dB NMSE {grid.nmse_db(est.estimate(test, np.full(len(test), 10.0), rng), test):.2f}")
