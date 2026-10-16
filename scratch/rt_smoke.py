import time, sys
import mitsuba as mi
mi.set_variant(sys.argv[1])
import sionna.rt as rt
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver
t=time.time()
scene = load_scene(rt.scene.munich)
scene.frequency = 3.5e9
scene.tx_array = PlanarArray(num_rows=1, num_cols=4, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="tr38901", polarization="V")
scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="iso", polarization="V")
scene.add(Transmitter("tx", position=[8.5, 21, 27]))
import numpy as np
rng=np.random.default_rng(0)
for i in range(64):
    scene.add(Receiver(f"rx{i}", position=[float(rng.uniform(-150,150)), float(rng.uniform(-150,150)), 1.5], velocity=[10.,0.,0.]))
print("load", round(time.time()-t,2), "s", flush=True)
t=time.time()
paths = PathSolver()(scene, max_depth=3, samples_per_src=10**5)
a, tau = paths.cir(normalize_delays=True, out_type="numpy")
print("paths", round(time.time()-t,2), "s", a.shape, tau.shape, flush=True)
t=time.time()
h = paths.cfr(frequencies=np.linspace(-1,1,72)*36*30e3/2+0, normalize_delays=True, out_type="numpy")
print("cfr", round(time.time()-t,2), "s", h.shape, "nonzero rx", int((np.abs(h).sum(axis=tuple(range(1,h.ndim)))>0).sum()))
