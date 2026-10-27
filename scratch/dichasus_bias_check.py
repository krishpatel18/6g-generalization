"""Bias or noise? Freq-only priors at 30 dB, and the measured tap profile by SNR."""
import sys
import numpy as np
sys.path.insert(0, "scripts")
from sixg_gen import channels, dichasus, grid, linear
from library import BUILT_IN, pooled
from make_library import library
from dichasus_test import per_antenna
grid.use_dichasus()
arr = dichasus.Array.named("A")
cal, _ = dichasus.fit_calibration(arr, dichasus.load(arr, ["d010", "d012", "d014"]))
test = per_antenna(dichasus.measured(dichasus.load(arr, ["d011", "d013"]), cal))
fit = per_antenna(dichasus.measured(dichasus.load(arr, ["d010", "d014"]), cal))
cities = list(BUILT_IN) + library()
st = {c: linear.Stats.fit(per_antenna(channels.generate(f"rt:{c}", 300, 1))) for c in cities}
est = {"ceiling": linear.Stats.fit(fit), "cdl": linear.Stats.fit(per_antenna(channels.generate("cdl", 12000, 1, mmap=True)[:3000])),
       "pool_all": pooled(list(st.values())), "munich": st["munich"]}
for snr in (10.0, 20.0, 30.0):
    print(f"{snr:.0f} dB: " + " | ".join(f"{k} {grid.nmse_db(linear.LMMSE(s).estimate(test, np.full(len(test), snr), np.random.default_rng(1)), test):6.2f}" for k, s in est.items()))
raw = dichasus.load(arr, ["d011", "d013"])
snr = np.load("data/dichasus/raw/dichasus-d011-complete.npz")["snr"]
z = [np.load(f"data/dichasus/raw/dichasus-{f}-complete.npz")["snr"][:, arr.order].mean(1) for f in ("d011", "d013")]
s = np.concatenate(z)[raw.usable]
h = dichasus.measured(raw, cal)
for lab, sel in (("lowest-SNR third", s <= np.percentile(s, 33)), ("highest-SNR third", s >= np.percentile(s, 67))):
    taps = np.fft.ifft(np.fft.ifftshift(h[sel][:, :, 0], axes=-1), axis=-1)
    pdp = np.mean(np.abs(taps) ** 2, axis=(0, 1)); pdp /= pdp.sum()
    print(f"{lab} (mean SNR {s[sel].mean():.1f} dB): taps0..4 {np.round(100*pdp[:5], 1)}%")
