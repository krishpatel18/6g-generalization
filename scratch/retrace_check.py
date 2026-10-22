"""Does re-tracing along a trajectory differ from Doppler-extrapolating fixed paths?

If not, channel evolution in this simulator is a sum of complex exponentials and a
linear predictor is near-ideal: no room for learned prediction.
"""
import sys
import mitsuba as mi
import numpy as np

mi.set_variant("metal_ad_mono_polarized")
import sionna.rt as rt

sys.path.insert(0, "src")
from sixg_gen import grid

grid.NUM_ANTENNAS = 8
scene_name = sys.argv[1] if len(sys.argv) > 1 else "munich"
rng = np.random.default_rng(11)
scene = rt.load_scene(getattr(rt.scene, scene_name))
scene.frequency = grid.CARRIER_HZ
iso = dict(num_rows=1, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="iso", polarization="V")
scene.tx_array = rt.PlanarArray(num_cols=8, **iso)
scene.rx_array = rt.PlanarArray(num_cols=1, **iso)
bb = scene.mi_scene.bbox()
lo, hi = np.array(bb.min)[:2], np.array(bb.max)[:2]
centre, half = (lo + hi) / 2, 0.3 * (hi - lo)
tx_xy = centre + rng.uniform(-1, 1, 2) * half
scene.add(rt.Transmitter("tx", position=[*map(float, tx_xy), 30.0]))
N = 1024
pos, vel = [], []
for i in range(N):
    r, phi = 250 * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
    hd, sp = rng.uniform(0, 2 * np.pi), rng.uniform(0, 30)
    p = np.array([tx_xy[0] + r * np.cos(phi), tx_xy[1] + r * np.sin(phi), 1.5])
    v = np.array([sp * np.cos(hd), sp * np.sin(hd), 0.0])
    pos.append(p); vel.append(v)
    scene.add(rt.Receiver(f"rx{i}", position=[*map(float, p)], velocity=[*map(float, v)]))
pos, vel = np.array(pos), np.array(vel)
speed = np.linalg.norm(vel, axis=1)
solver = rt.PathSolver(deterministic=bool(int(__import__("os").environ.get("DET", "0"))))
freqs = mi.Float(grid.subcarrier_offsets_hz())


def trace():
    return solver(scene, max_depth=3, samples_per_src=int(__import__("os").environ.get("SPS", 10**5)), synthetic_array=True, diffuse_reflection=False)


def cfr(paths, dt=None):
    kw = dict(frequencies=freqs, normalize_delays=False, out_type="numpy")
    if dt is None:
        h = paths.cfr(sampling_frequency=1.0, num_time_steps=1, **kw)[..., 0, :]
    else:
        h = paths.cfr(sampling_frequency=1.0 / dt, num_time_steps=2, **kw)[..., 1, :]
    return h.reshape(N, 8, grid.NUM_SUBCARRIERS)


p0 = trace()
h0 = cfr(p0)
ok = np.mean(np.abs(h0) ** 2, axis=(1, 2)) > 0
bins = [(0, 10), (10, 20), (20, 30)]
print(f"{scene_name}: {ok.sum()}/{N} receivers with paths", flush=True)
print("dt_ms  " + "  ".join(f"v{a}-{b}:dop/hold" for a, b in bins) + "   (NMSE dB vs re-traced; dop = Doppler-extrapolated fixed paths)")
for dt in [float(x) for x in __import__("os").environ.get("DTS", "0,1e-3,2e-3,5e-3,10e-3,20e-3,50e-3,100e-3").split(",")]:
    for i in range(N):
        scene.receivers[f"rx{i}"].position = [*map(float, pos[i] + vel[i] * dt)]
    h_true = cfr(trace())
    h_dop = cfr(p0, dt) if dt > 0 else h0
    m = ok & (np.mean(np.abs(h_true) ** 2, axis=(1, 2)) > 0)
    def nmse(a, sel):
        e = np.sum(np.abs(a[sel] - h_true[sel]) ** 2, axis=(1, 2)) / np.sum(np.abs(h_true[sel]) ** 2, axis=(1, 2))
        return 10 * np.log10(np.mean(e))
    cells = []
    e_dop = np.sum(np.abs(h_dop - h_true) ** 2, axis=(1, 2)) / np.maximum(np.sum(np.abs(h_true) ** 2, axis=(1, 2)), 1e-30)
    for a, b in bins:
        sel = m & (speed >= a) & (speed < b)
        broken = np.mean(e_dop[sel] > 0.01)
        cells.append(f"{nmse(h_dop, sel):6.1f}/{nmse(h0, sel):5.1f} {100*broken:3.0f}%")
        counts = sel.sum()
    print(f"{dt*1e3:5.0f}  " + "  ".join(f"{c:>21s}" for c in cells) + f"   n={[int((m & (speed>=a) & (speed<b)).sum()) for a,b in bins]}", flush=True)
print("ALL DONE", flush=True)
