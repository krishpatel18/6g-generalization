"""Sim-to-real: do ray-traced city priors beat 3GPP CDL on measured channels? (DICHASUS)

usage: dichasus_test.py [cities_per_prior]     (default 6000)

Pre-registered in docs/PLAN.md, "Sim-to-real experiment". For each array (A primary;
B, C count if their calibration coherence >= 0.5), on measured test channels:
  ls_interp; LMMSE from CDL; from each ray-traced city alone; from pools of k random
  cities (8 draws) and of all 14; CDL / 14-city pool / nothing blended with 50 site
  measurements (4 draws); the site-trained ceiling.
Both on the full array x frequency grid ("spatial") and per antenna ("freq_only").
NMSE dB at 0 and 10 dB SNR. Writes runs/<grid tag>/dichasus_test.csv:
  array, variant, estimator, detail, draw, snr_db, nmse_db
"""

import csv
import sys
from pathlib import Path

import numpy as np

from sixg_gen import channels, dichasus, grid, linear

sys.path.insert(0, str(Path(__file__).resolve().parent))
from library import BUILT_IN, pooled  # noqa: E402
from loco import RUNS  # noqa: E402
from make_library import library  # noqa: E402

CAL_FILES = ["d006", "d007", "d008", "d009", "d010", "d014"]
SITE_FILES = ["d012"]
TEST_FILES = ["d011", "d013"]
EVAL_SNR_DB = (0.0, 10.0)
BREADTH_K = (1, 2, 4, 8)
DRAWS = 8
SITE_N, SITE_DRAWS = 50, 4
MIN_COHERENCE = 0.5


def per_antenna(h: np.ndarray) -> np.ndarray:
    """[n, A, T, F] -> [n * A, 1, T, F], each antenna's channel at unit power."""
    return grid.normalize_power(np.asarray(h).reshape(-1, 1, *h.shape[2:]))


def nmse(estimate, test: np.ndarray) -> dict[float, float]:
    out = {}
    for snr in EVAL_SNR_DB:
        rng = np.random.default_rng(1234)  # identical pilot noise for every method
        out[snr] = grid.nmse_db(estimate(test, np.full(len(test), snr), rng), test)
    return out


def run_variant(array, variant, sims, fit, site, test, rows, rows_per_datapoint=1, breadth=BREADTH_K) -> None:
    """`rows_per_datapoint`: samples per measured datapoint (16 in freq_only), so 50 site
    measurements are 50 datapoints in both variants."""
    def record(estimator, detail, draw, scores):
        rows.extend((array, variant, estimator, detail, draw, snr, round(v, 3)) for snr, v in scores.items())

    stats = {name: linear.Stats.fit(h) for name, h in sims.items()}
    cities = [c for c in stats if c != "cdl"]
    lmmse = lambda s: nmse(linear.LMMSE(s).estimate, test)  # noqa: E731
    record("ls_interp", "", 0, nmse(grid.ls_estimate, test))
    record("ceiling", "", 0, lmmse(linear.Stats.fit(fit)))
    record("cdl", "", 0, lmmse(stats["cdl"]))
    for c in cities:
        record("city", c, 0, lmmse(stats[c]))
    pool = pooled([stats[c] for c in cities])
    record("pool_all", "", 0, lmmse(pool))
    rng = np.random.default_rng(0)
    for k in breadth:
        for d in range(DRAWS):
            pick = list(rng.choice(cities, k, replace=False))
            record(f"pool_k{k}", "+".join(pick), d, lmmse(pooled([stats[c] for c in pick])))
    for d in range(SITE_DRAWS):
        idx = np.sort(np.random.default_rng(100 + d).choice(len(site) // rows_per_datapoint, SITE_N, replace=False))
        few = site[(idx[:, None] * rows_per_datapoint + np.arange(rows_per_datapoint)).ravel()]
        record("own_50", "", d, lmmse(linear.Stats.fit(few)))
        for name, prior in (("cdl+50", stats["cdl"]), ("pool_all+50", pool)):
            adapted, w = linear.Stats.blend(prior, few, d)
            record(name, f"w={w}", d, lmmse(adapted))
    print(f"  {array} {variant}: done", flush=True)


def main(per_city: int) -> None:
    grid.use_dichasus()
    cities = list(BUILT_IN) + library()
    sims = {"cdl": channels.generate("cdl", 12000, 1, mmap=True),
            **{c: channels.generate(f"rt:{c}", per_city, 1, mmap=True) for c in cities}}
    rows, notes = [], []
    for name in "ABC":
        arr = dichasus.Array.named(name)
        cal_raw = dichasus.load(arr, CAL_FILES)
        cal, coherence = dichasus.fit_calibration(arr, cal_raw)
        fit = dichasus.measured(cal_raw, cal)
        site = dichasus.measured(dichasus.load(arr, SITE_FILES), cal)
        test = dichasus.measured(dichasus.load(arr, TEST_FILES), cal)
        note = (f"array {name}: coherence {coherence:.3f} ({'counts' if coherence >= MIN_COHERENCE else 'excluded'}); "
                f"calibration/ceiling {len(fit)}, site pool {len(site)}, test {len(test)}")
        print(note, flush=True)
        notes.append(note)
        run_variant(name, "spatial", sims, fit, site, test, rows)
        run_variant(name, "freq_only", {k: per_antenna(v) for k, v in sims.items()},
                    per_antenna(fit), per_antenna(site), per_antenna(test), rows, grid.NUM_ANTENNAS)
        out_dir = RUNS / grid.config_tag()
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "" if per_city == 6000 else f"_n{per_city}"
        with open(out_dir / f"dichasus_test{suffix}.csv", "w", newline="") as f:
            out = csv.writer(f)
            out.writerow(("array", "variant", "estimator", "detail", "draw", "snr_db", "nmse_db"))
            out.writerows(rows)
    summary(rows, notes)
    print("ALL DONE", flush=True)


def summary(rows, notes, breadth=BREADTH_K) -> None:
    print("\n" + "\n".join(notes))
    v = {}
    for a, var, est, detail, draw, snr, x in rows:
        v.setdefault((a, var, est, snr), []).append((detail, x))
    for var in ("spatial", "freq_only"):
        print(f"\n[{var}] NMSE dB at 10 dB SNR (gap to ceiling in brackets)")
        for a in "ABC":
            if (a, var, "ceiling", 10.0) not in v:
                continue
            ceil = v[(a, var, "ceiling", 10.0)][0][1]
            g = lambda est: np.mean([x for _, x in v[(a, var, est, 10.0)]])  # noqa: E731
            worst = lambda est: max(x for _, x in v[(a, var, est, 10.0)])  # noqa: E731
            print(f"  array {a}: ls {g('ls_interp'):.2f} | ceiling {ceil:.2f} | cdl {g('cdl'):.2f} [{g('cdl')-ceil:+.2f}] | "
                  f"pool_all {g('pool_all'):.2f} [{g('pool_all')-ceil:+.2f}] | best city {min(x for _, x in v[(a, var, 'city', 10.0)]):.2f} | "
                  f"worst city {worst('city'):.2f}")
            print(f"           worst-draw gap k={breadth}: " + " ".join(f"{worst(f'pool_k{k}')-ceil:.2f}" for k in breadth) +
                  " | mean-draw gap: " + " ".join(f"{g(f'pool_k{k}')-ceil:.2f}" for k in breadth))
            print(f"           +50 (mean of {SITE_DRAWS} draws): own {g('own_50'):.2f} | cdl+50 {g('cdl+50'):.2f} | "
                  f"pool+50 {g('pool_all+50'):.2f}")
            p1 = g("pool_all") < g("cdl")
            p2 = (worst("pool_k1") - worst(f"pool_k{breadth[-1]}")) >= 1.0
            p3 = g("pool_all") - ceil <= 1.0
            print(f"           P1 pool beats CDL: {p1} | P2 insurance >= 1 dB: {p2} | P3 pool within 1 dB of ceiling: {p3}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6000)
