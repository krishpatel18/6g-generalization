"""Stage 2: a calibrated ray-traced twin vs CDL where nothing was measured (docs/PLAN.md,
"Trial: does a calibrated ray-traced twin beat CDL?").

usage: hallway_twin.py [max_evals]   (run twice: the first run calibrates if needed and
simulates the twin, saving its grid channels; the second scores)

Region A (y < 0, model frame) is measured: it calibrates the twin's materials and the
antenna phases and supplies the 50 site datapoints. Region B (y > 10) is not: its first
70% (recorded order) trains the ceiling, its last 30% is the test pool (per array, up to
1,500 datapoints usable for that array under the study's SNR rule).
Per array, LMMSE from: generic CDL; CDL + 50 from A; the calibrated twin and the ITU
twin simulated at 2,000 positions over B's footprint; the calibrated twin at the exact
test positions; the calibrated twin + 50 from A; CDL + 50 from B (reference).
The twin is calibrated on A datapoints usable for both arrays (the study's SNR rule).
Full grid and per antenna, 0 and 10 dB SNR. Writes runs/hallway_twin.csv and
runs/hallway_twin_notes.json (coherence, counts, descriptive channel statistics).
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

from sixg_gen import channels, dichasus, grid, hallway, linear, wideband

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdl_gap import stats as channel_stats  # noqa: E402
from dichasus_test import SITE_DRAWS, SITE_N, nmse, per_antenna  # noqa: E402
from hallway_calibrate import calibrate, materials, stats  # noqa: E402
from loco import RUNS  # noqa: E402

A_MAX_Y, B_MIN_Y = 0.0, 10.0  # model frame; both regions are corridor (the lobby is y 0..7.5)
B_FOOTPRINT = ((0.5, 1.8), (10.0, 25.0))
CEILING_FRACTION, MAX_TEST = 0.7, 1500
N_TWIN, N_FIT_TWIN, MAX_A = 12000, 256, 4000
MATERIALS = RUNS / "hallway_twin_materials.json"


def to_grid(h: np.ndarray) -> np.ndarray:
    """Simulated [n, 2, 32, 1024] -> grid channels [n, 2, 32, 1, 128], not normalised (a batch
    reducer for `Twin.simulate`)."""
    return np.stack([wideband.to_grid(h[:, a, :, None, :]) for a in range(h.shape[1])], axis=1)


def on_grid(h: np.ndarray, array: int) -> np.ndarray:
    """One array's grid channels [n, 32, 1, 128] at unit power, dropping positions the ray
    tracer finds no path to."""
    h = h[:, array]
    return grid.normalize_power(h[np.sum(np.abs(h) ** 2, axis=(1, 2, 3)) > 0])


def measured_chunks(data, arr, idx, cal, chunk: int = 1000) -> np.ndarray:
    """`dichasus.measured` over `idx` (sorted), reading the recording a chunk at a time."""
    idx = np.sort(idx)
    return np.concatenate([dichasus.measured(hallway.raw(data, arr, idx[j:j + chunk]), cal)
                           for j in range(0, len(idx), chunk)])


def main(max_evals: int = 250) -> None:
    grid.use_dichasus(hallway.ROWS, hallway.COLS, 0.0, hallway.CARRIER_HZ)
    rng = np.random.default_rng(0)
    data = hallway.load()
    keep, y = data["keep"], data["pos"][:, 1]
    a_idx = np.flatnonzero(keep & (y < A_MAX_Y))
    b_idx = np.flatnonzero(keep & (y > B_MIN_Y))
    a_idx = np.sort(rng.choice(a_idx, min(MAX_A, len(a_idx)), replace=False))
    cut = int(CEILING_FRACTION * len(b_idx))
    b_fit, b_test = b_idx[:cut], b_idx[cut:]  # each array tests up to MAX_TEST of b_test usable for it
    print(f"A {len(a_idx)} datapoints; B ceiling {len(b_fit)}, test pool {len(b_test)}", flush=True)

    # Twin materials, fitted on A only (stage 1's method)
    twin = None if MATERIALS.exists() else hallway.Twin()
    both = np.logical_and.reduce([hallway.usable(data, arr, a_idx) for arr in hallway.arrays()])
    fit_idx = np.sort(rng.choice(a_idx[both], min(N_FIT_TWIN, both.sum()), replace=False))
    print(f"twin fit: {len(fit_idx)} of {both.sum()} A datapoints usable for both arrays", flush=True)
    if MATERIALS.exists():  # calibration is the slow step (~45 min); reuse a finished one
        saved = json.loads(MATERIALS.read_text())
        assert saved["fit_idx"] == fit_idx.tolist(), "saved calibration used other fit positions"
        fitted, itu_loss, cal_loss, evals = saved["materials"], saved["itu_loss"], saved["cal_loss"], saved["evaluations"]
        print(f"reusing {MATERIALS}", flush=True)
    else:
        _, u, *_, itu_loss, cal_loss, evals = calibrate(twin, data["pos"][fit_idx],
                                                        stats(hallway.calibrated(data, fit_idx)), max_evals)
        fitted = {k: [float(v) for v in p] for k, p in materials(u).items()}
        MATERIALS.write_text(json.dumps(dict(materials=fitted, itu_loss=itu_loss, cal_loss=cal_loss,
                                             evaluations=evals, fit_idx=fit_idx.tolist())))
    print(f"twin calibrated on A: loss {itu_loss:.4f} -> {cal_loss:.4f} ({evals} evaluations); {json.dumps(fitted)}",
          flush=True)
    (lo_x, hi_x), (lo_y, hi_y) = B_FOOTPRINT
    z = float(np.median(data["pos"][keep, 2]))
    nb = np.c_[rng.uniform(lo_x, hi_x, N_TWIN), rng.uniform(lo_y, hi_y, N_TWIN), np.full(N_TWIN, z)]
    labels = ("twin_itu", "twin_cal", "twin_exact")
    if not all((hallway.DIR / f"{k}_grid.npy").exists() for k in labels):
        twin = twin or hallway.Twin()
        # Simulate once and stop: scoring runs in a fresh process without the ray tracer loaded,
        # which keeps its memory small (rerun the same command).
        for label, params in (("twin_itu", twin.itu), ("twin_cal", fitted)):
            twin.set_materials({k: tuple(v) for k, v in params.items()})
            np.save(hallway.DIR / f"{label}_grid.npy", twin.simulate(nb, reduce=to_grid))
        np.save(hallway.DIR / "twin_exact_grid.npy", twin.simulate(data["pos"][b_test], reduce=to_grid))
        print("twin channels simulated and saved; rerun to score", flush=True)
        return
    sims = {k: np.load(hallway.DIR / f"{k}_grid.npy", mmap_mode="r") for k in labels}
    print("twin channels loaded", flush=True)

    cdl = channels.generate("cdl", 12000, 1, mmap=True)
    rows, notes = [], {"materials": fitted, "itu_loss": itu_loss, "cal_loss": cal_loss}
    for i, arr in enumerate(hallway.arrays()):
        ok = hallway.usable(data, arr, b_test)
        sel = np.sort(np.random.default_rng(1 + i).choice(np.flatnonzero(ok), min(MAX_TEST, ok.sum()), replace=False))
        ra = hallway.raw(data, arr, a_idx)
        cal, coherence = dichasus.fit_calibration(dichasus.Array.named(arr["name"], "dcxx"), ra)
        site_a = dichasus.measured(ra, cal)
        del ra
        ceil_fit, test = (measured_chunks(data, arr, idx, cal) for idx in (b_fit, b_test[sel]))
        site_b = ceil_fit  # the reference "CDL + 50 from B" draws from B's ceiling part, not the test part
        priors = {"cdl": cdl, **{k: on_grid(v[sel] if k == "twin_exact" else v, i) for k, v in sims.items()}}
        notes[arr["name"]] = dict(coherence=coherence, site_a=len(site_a), ceiling=len(ceil_fit), test=len(test),
                                  stats={k: channel_stats(np.asarray(v[:2000])) for k, v in
                                         {**priors, "measured_test": test}.items()})
        print("    " + "; ".join(f"{k}: " + ", ".join(f"{m} {x:.1f}" for m, x in v.items())
                                 for k, v in notes[arr["name"]]["stats"].items()), flush=True)
        print(f"  {arr['name']}: coherence {coherence:.3f}; A {len(site_a)}, B ceiling {len(ceil_fit)}, "
              f"test {len(test)}", flush=True)
        for variant, f in (("spatial", lambda h: h), ("freq_only", per_antenna)):
            st = {k: linear.Stats.fit(f(v)) for k, v in priors.items()}
            t = f(test)

            def record(est, draw, scores):
                rows.extend((arr["name"], variant, est, draw, snr, round(v, 3)) for snr, v in scores.items())

            lmmse = lambda s: nmse(linear.LMMSE(s).estimate, t)  # noqa: E731
            record("ceiling", 0, lmmse(linear.Stats.fit(f(ceil_fit))))
            for k in ("cdl", "twin_itu", "twin_cal", "twin_exact"):
                record(k, 0, lmmse(st[k]))
            for d in range(SITE_DRAWS):
                for src, site in (("a", site_a), ("b", site_b)):
                    idx = np.sort(np.random.default_rng(100 + d).choice(len(site), SITE_N, replace=False))
                    few = f(site[idx])
                    record(f"cdl+50{src}", d, lmmse(linear.Stats.blend(st["cdl"], few, d)[0]))
                    if src == "a":
                        record("twin_cal+50a", d, lmmse(linear.Stats.blend(st["twin_cal"], few, d)[0]))
    with open(RUNS / "hallway_twin.csv", "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(("array", "variant", "estimator", "draw", "snr_db", "nmse_db"))
        out.writerows(rows)
    with open(RUNS / "hallway_twin_notes.json", "w") as f:
        json.dump(notes, f, indent=1, default=float)
    summary(rows)
    print("ALL DONE", flush=True)


def gaps(rows, snr=10.0):
    v = {}
    for a, var, est, d, s, x in rows:
        if s == snr:
            v.setdefault((a, var, est), []).append(x)
    return {k: np.mean(x) - np.mean(v[(k[0], k[1], "ceiling")]) for k, x in v.items()}


def summary(rows) -> None:
    for snr in (10.0, 0.0):
        g = gaps(rows, snr)
        arrays = list(dict.fromkeys(a for a, _, _ in g))
        ests = ["cdl", "cdl+50a", "twin_itu", "twin_cal", "twin_exact", "twin_cal+50a", "cdl+50b"]
        for var in ("spatial", "freq_only"):
            print(f"\n[{var}] {snr:.0f} dB, dB above the ceiling (lower = closer)")
            for a in arrays + ["mean"]:
                vals = [np.mean([g[(x, var, e)] for x in arrays]) if a == "mean" else g[(a, var, e)] for e in ests]
                print(f"  {a[:24]:24s} " + " | ".join(f"{e} {x:5.2f}" for e, x in zip(ests, vals)))
    g = gaps(rows)
    arrays = list(dict.fromkeys(a for a, _, _ in g))
    m = {e: np.mean([g[(a, "spatial", e)] for a in arrays]) for e in ("cdl", "cdl+50a", "twin_itu", "twin_cal")}
    print(f"\nT1 calibrated twin beats generic CDL by >= 0.5 dB: {m['cdl'] - m['twin_cal']:.2f} dB -> "
          f"{'met' if m['cdl'] - m['twin_cal'] >= 0.5 else 'not met'}")
    print(f"T2 calibrated beats ITU twin by >= 0.3 dB: {m['twin_itu'] - m['twin_cal']:.2f} dB -> "
          f"{'met' if m['twin_itu'] - m['twin_cal'] >= 0.3 else 'not met'}")
    print(f"T3 calibrated twin beats CDL + 50 from A: {m['cdl+50a'] - m['twin_cal']:.2f} dB -> "
          f"{'met' if m['cdl+50a'] - m['twin_cal'] > 0 else 'not met'}")


if __name__ == "__main__":
    main(*map(int, sys.argv[1:]))
