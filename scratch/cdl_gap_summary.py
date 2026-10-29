"""Summary of runs/cdl_gap_{street,lab,factory}.csv: gaps to the ceiling per array (mean over draws),
ranges, environment means, and the registered claims G1-G3 at 10 and 0 dB SNR."""
import csv
from collections import defaultdict
import numpy as np

ENVS = ("street", "lab", "factory")
for snr in (10.0, 0.0):
    print(f"\n######## {snr:.0f} dB SNR")
    env_mean = {}
    for env in ENVS:
        v = defaultdict(list)
        for r in csv.DictReader(open(f"runs/cdl_gap_{env}.csv")):
            if float(r["snr_db"]) == snr:
                v[(r["array"], r["variant"], r["estimator"], r["detail"] if r["estimator"] == "profile" else "")].append(float(r["nmse_db"]))
        arrays = list(dict.fromkeys(k[0] for k in v))
        for var in ("spatial", "freq_only"):
            g = defaultdict(dict)
            for a in arrays:
                ceil = v[(a, var, "ceiling", "")][0]
                for est in ("cdl", "cdl+50", "picked"):
                    g[est][a] = np.mean(v[(a, var, est, "")]) - ceil
                profs = {k[3]: x[0] - ceil for k, x in v.items() if k[0] == a and k[1] == var and k[2] == "profile"}
                g["oracle"][a] = min(profs.values())
            env_mean[(env, var)] = {e: np.mean(list(d.values())) for e, d in g.items()}
            row = " | ".join(f"{e} {min(d.values()):.2f}-{max(d.values()):.2f} (mean {np.mean(list(d.values())):.2f})" for e, d in g.items())
            print(f"{env:8s} {var:9s} {row}")
            print(" " * 19 + "per array: " + "; ".join(f"{a[-12:]}: " + ", ".join(f"{e} {g[e][a]:.2f}" for e in g) for a in arrays))
    for var in ("spatial", "freq_only"):
        f, s = env_mean[("factory", var)]["cdl"], env_mean[("street", var)]["cdl"]
        print(f"G1 [{var}] factory {f:.2f} > street {s:.2f}: {f > s}")
        print(f"G2 [{var}] all < 1 dB: {all(env_mean[(e, var)]['cdl'] < 1 for e in ENVS)}")
        for e in ENVS:
            m = env_mean[(e, var)]
            if m["cdl"] >= 1:
                closed = (m["cdl"] - m["picked"]) / m["cdl"]
                print(f"G3 [{var}] {e}: gap {m['cdl']:.2f} -> picked {m['picked']:.2f}, closed {100 * closed:.0f}% -> {'met' if closed >= 0.5 else 'not met'}")
            else:
                print(f"G3 [{var}] {e}: gap {m['cdl']:.2f} < 1 dB, not applicable")
