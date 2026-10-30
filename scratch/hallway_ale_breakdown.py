"""Where do stage 1's power errors (ALE) come from? Per array, signed bias, lobby vs corridor, tail.
Re-simulates stage 1's test positions with the ITU and calibrated materials from runs/hallway_calibrate.json."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from hallway_calibrate import stats  # noqa: E402

from sixg_gen import hallway  # noqa: E402

res = json.load(open("runs/hallway_calibrate.json"))
rng = np.random.default_rng(0)
data = hallway.load()
usable = np.flatnonzero(data["keep"])[:10000]
rng.choice(usable[:5000], res["n_fit"], replace=False)  # same draws as the script
test_idx = np.sort(rng.choice(usable[5100:], res["n_test"], replace=False))
p_m, d_m = stats(hallway.calibrated(data, test_idx))
pos = data["pos"][test_idx]
lobby = pos[:, 0] < 0.5
twin = hallway.Twin()
for label in ("itu", "calibrated"):
    twin.set_materials({k: tuple(v) for k, v in res[label]["materials"].items()})
    p_rt, d_rt = stats(twin.simulate(pos))
    err = 10 * np.log10(p_rt.mean(-1) + 1e-30) - 10 * np.log10(res[label]["alpha"] * p_m.mean(-1))  # [n, 2] signed
    print(f"{label}: signed dB error mean per array {np.round(err.mean(0), 2)}, |err| median per array "
          f"{np.round(np.median(np.abs(err), 0), 2)}; share |err| > 15 dB {np.mean(np.abs(err) > 15):.3f}")
    for name, m in (("corridor", ~lobby), ("lobby", lobby)):
        e = np.abs(err[m])
        print(f"   {name} ({m.sum()} positions): ALE mean {e.mean():.2f}, median {np.median(e):.2f}")
    worst = np.argsort(-np.abs(err).max(1))[:5]
    print("   worst positions (x, y), errors per array:", [(tuple(np.round(pos[w, :2], 1)), tuple(np.round(err[w], 1))) for w in worst])
    print("   per-array bias removed: ALE mean", np.round(np.abs(err - err.mean(0)).mean(), 2))
