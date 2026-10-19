"""Check run 2's pre-registered gates and claims (docs/PLAN.md) against a results CSV.

usage: report.py runs/<grid tag>/loco_seed<seed>.csv [snr_db]
"""

import csv
import sys

CITIES = ("munich", "etoile", "florence", "san_francisco")
ARMS = ("cdl", "rt3", "rt1")


def main(path: str, snr: float) -> None:
    d = {}
    for r in csv.DictReader(open(path)):
        if float(r["snr_db"]) == snr:
            d[(r["target"], r["arm"], r["model"], int(r["n"]))] = float(r["nmse_db"])
    cities = [c for c in CITIES if (c, "-", "ls_interp", 0) in d]

    print(f"NMSE dB at {snr:.0f} dB SNR, lower is better\n")
    print(f"{'row':22s}" + "".join(f"{c[:9]:>10s}" for c in cities))
    for label, key in [("LS interp", ("-", "ls_interp", 0)), ("LMMSE in-domain 12k", ("-", "lmmse_in", 12000))] + \
            [(f"LMMSE own N={n}", ("-", "lmmse_own", n)) for n in (50, 200, 1000)] + \
            [(f"{a} {m} N={n}", (a, m, n)) for n in (0, 50, 200, 1000) for a in ARMS for m in ("lin", "lin+cnn")]:
        print(f"{label:22s}" + "".join(f"{d.get((c,) + key, float('nan')):10.2f}" for c in cities))

    print("\nGate a (lin+cnn at N=0 within 0.1 dB of its lin, or better):")
    worst = max(d[(c, a, "lin+cnn", 0)] - d[(c, a, "lin", 0)] for c in cities for a in ARMS)
    print(f"  worst excess {worst:+.2f} dB -> {'PASS' if worst <= 0.1 else 'FAIL'}")
    print("Gate b (every arm at N>=50 no worse than LMMSE from the same N target samples):")
    worst = max(d[(c, a, m, n)] - d[(c, "-", "lmmse_own", n)]
                for c in cities for a in ARMS for m in ("lin", "lin+cnn") for n in (50, 200, 1000))
    print(f"  worst excess {worst:+.2f} dB -> {'PASS' if worst <= 0 else 'FAIL'}")

    for model in ("lin", "lin+cnn"):
        wins = [c for c in cities if d[(c, "rt3", model, 0)] < d[(c, "cdl", model, 0)]]
        print(f"\nRT realism ({model}): rt3 beats cdl at N=0 in {len(wins)}/{len(cities)} ({', '.join(wins)})"
              f" -> {'SUPPORTED' if len(wins) >= 3 else 'FALSIFIED'}")
        for n in (0, 50):
            gap = sum(d[(c, "rt1", model, n)] - d[(c, "rt3", model, n)] for c in cities) / len(cities)
            print(f"Breadth ({model}, N={n}): rt3 better than rt1 by {gap:+.2f} dB on average"
                  f" -> {'SUPPORTED' if gap >= 0.5 else 'no evidence'}")
    best = max(d[(c, a, "lin", n)] - d[(c, a, "lin+cnn", n)] for c in cities for a in ARMS for n in (0, 50, 200, 1000))
    print(f"\nBeyond covariance: largest lin+cnn gain over lin {best:+.2f} dB"
          f" -> {'SUPPORTED' if best >= 0.5 else 'no evidence'}")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 10.0)
