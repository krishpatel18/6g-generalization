"""Follow-on: where does CDL fail? The CDL-prior gap across environments (docs/PLAN.md,
"Follow-on: where does CDL fail?").

usage: cdl_gap.py <street|lab|factory>

Per array: calibrate on the fit recording, then on the test recording score LMMSE from the
generic CDL prior, the site-trained ceiling, CDL + 50 site measurements (4 draws), each of
15 fixed CDL profiles (A-E x 30/100/300 ns) and the profile 50 site measurements pick; plus
channel statistics. Full grid and per antenna, 0 and 10 dB SNR.
Writes runs/cdl_gap_<env>.csv and prints a summary.
"""

import csv
import sys
from pathlib import Path

import numpy as np

from sixg_gen import channels, dichasus, grid, linear

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dichasus_test import SITE_DRAWS, SITE_N, nmse, per_antenna  # noqa: E402
from loco import RUNS  # noqa: E402
from select_prior import pool_score  # noqa: E402

PROFILES = [f"cdl-{m}-{ds}" for m in "ABCDE" for ds in (30, 100, 300)]
MIN_COHERENCE = 0.5
ENVS = {
    # dataset, array names, (rows, cols, downtilt), fit files, site files (or None: last 20% of fit), test files
    "street": ("dxxx", ["A", "B", "C"], (2, 8, grid.STREET_DOWNTILT_RAD),
               ["d006", "d007", "d008", "d009", "d010", "d014"], ["d012"], ["d011", "d013"]),
    "lab": ("0d5x", None, (2, 8, 0.0), ["0d54"], None, ["0d56"]),
    "factory": ("cf0x", None, (2, 4, 0.0), ["cf02"], None, ["cf06"]),
}
FIT_FRACTION = 0.8


def stats(h: np.ndarray) -> dict:
    """First-tap energy share (%), RMS delay spread (taps of 160 ns), top-3 spatial eigenvalue share (%)."""
    h = np.asarray(h)[:, :, 0]
    taps = np.fft.ifft(np.fft.ifftshift(h, axes=-1), axis=-1)
    pdp = np.mean(np.abs(taps) ** 2, axis=(0, 1))
    pdp = pdp / pdp.sum()
    k = np.arange(len(pdp))
    k = np.where(k < len(pdp) // 2, k, k - len(pdp))
    mean = np.sum(k * pdp)
    f = h[:, :, h.shape[-1] // 2]
    lam = np.linalg.eigvalsh(f.T @ f.conj() / len(f))[::-1]
    return {"first_tap_pct": 100 * pdp[0], "rms_delay_taps": float(np.sqrt(np.sum((k - mean) ** 2 * pdp))),
            "top3_spatial_pct": 100 * lam[:3].sum() / lam.sum()}


def run(env, array, variant, priors, fit, site, test, rows, per_datapoint) -> None:
    def record(est, detail, draw, scores):
        rows.extend((env, array, variant, est, detail, draw, snr, round(v, 3)) for snr, v in scores.items())

    stats_of = {k: linear.Stats.fit(h) for k, h in priors.items()}
    lmmse = lambda st: nmse(linear.LMMSE(st).estimate, test)  # noqa: E731
    record("ceiling", "", 0, lmmse(linear.Stats.fit(fit)))
    record("cdl", "", 0, lmmse(stats_of["cdl"]))
    for p in PROFILES:
        record("profile", p, 0, lmmse(stats_of[p]))
    n_site = len(site) // per_datapoint
    for d in range(SITE_DRAWS):
        idx = np.sort(np.random.default_rng(100 + d).choice(n_site, min(SITE_N, n_site), replace=False))
        few = site[(idx[:, None] * per_datapoint + np.arange(per_datapoint)).ravel()]
        record("cdl+50", "", d, lmmse(linear.Stats.blend(stats_of["cdl"], few, d)[0]))
        picked = min(PROFILES, key=lambda p: pool_score(linear.LMMSE(stats_of[p]), few))
        record("picked", picked, d, lmmse(stats_of[picked]))


def main(env: str) -> None:
    dataset, names, (rows_, cols_, tilt), fit_files, site_files, test_files = ENVS[env]
    grid.use_dichasus(rows_, cols_, tilt)
    names = names or dichasus.array_names(dataset)
    priors = {"cdl": channels.generate("cdl", 12000, 1, mmap=True),
              **{p: channels.generate(p, 4000, 1, mmap=True) for p in PROFILES}}
    rows, stat_rows = [], []
    for name in names:
        arr = dichasus.Array.named(name, dataset)
        fit_raw = dichasus.load(arr, fit_files, dataset)
        if site_files is None:
            fit_raw, site_raw = dichasus.part(fit_raw, 0, FIT_FRACTION), dichasus.part(fit_raw, FIT_FRACTION, 1)
        else:
            site_raw = dichasus.load(arr, site_files, dataset)
        cal, coherence = dichasus.fit_calibration(arr, fit_raw)
        fit, site = dichasus.measured(fit_raw, cal), dichasus.measured(site_raw, cal)
        test = dichasus.measured(dichasus.load(arr, test_files, dataset), cal)
        counts = f"fit {len(fit)}, site {len(site)}, test {len(test)}"
        if coherence < MIN_COHERENCE:
            print(f"  {env} {name}: coherence {coherence:.3f} < {MIN_COHERENCE}, excluded ({counts})", flush=True)
            continue
        st = stats(test)
        stat_rows.append((env, name, coherence, *st.values()))
        print(f"  {env} {name}: coherence {coherence:.3f}; {counts}; "
              + ", ".join(f"{k} {v:.1f}" for k, v in st.items()), flush=True)
        run(env, name, "spatial", priors, fit, site, test, rows, 1)
        run(env, name, "freq_only", {k: per_antenna(v) for k, v in priors.items()},
            per_antenna(fit), per_antenna(site), per_antenna(test), rows, grid.NUM_ANTENNAS)
    with open(RUNS / f"cdl_gap_{env}.csv", "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(("env", "array", "variant", "estimator", "detail", "draw", "snr_db", "nmse_db"))
        out.writerows(rows)
    with open(RUNS / f"cdl_gap_{env}_stats.csv", "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(("env", "array", "coherence", "first_tap_pct", "rms_delay_taps", "top3_spatial_pct"))
        out.writerows(stat_rows)
    summary(rows)
    print("ALL DONE", flush=True)


def summary(rows) -> None:
    v = {}
    for env, a, var, est, detail, d, snr, x in rows:
        if snr == 10.0:
            v.setdefault((a, var, est), []).append((detail, x))
    for var in ("spatial", "freq_only"):
        print(f"\n[{var}] 10 dB, dB above the site-trained ceiling (lower = closer)")
        for a in dict.fromkeys(a for a, vv, _ in v if vv == var):
            ceil = v[(a, var, "ceiling")][0][1]
            g = lambda est: np.mean([x for _, x in v[(a, var, est)]]) - ceil  # noqa: E731
            best = min(v[(a, var, "profile")], key=lambda t: t[1])
            picks = [p for p, _ in v[(a, var, "picked")]]
            print(f"  {a:28s} cdl {g('cdl'):5.2f} | cdl+50 {g('cdl+50'):5.2f} | best profile {best[0]} "
                  f"{best[1] - ceil:5.2f} | picked {g('picked'):5.2f} ({', '.join(picks)})")


if __name__ == "__main__":
    main(sys.argv[1])
