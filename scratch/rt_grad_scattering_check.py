import sys, time
import mitsuba as mi
mi.set_variant("metal_ad_mono_polarized")
import drjit as dr, sionna.rt as rt
diffuse = sys.argv[1] == "diffuse"; mode = sys.argv[2]
scene = rt.load_scene(rt.scene.simple_street_canyon)
scene.frequency = 3.5e9
scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
scene.rx_array = scene.tx_array
scene.add(rt.Transmitter("tx", position=[-30, 0, 20])); scene.add(rt.Receiver("rx", position=[40, 2, 1.5]))
mat = scene.radio_materials["concrete"]; [setattr(m, "scattering_coefficient", 0.3) for m in scene.radio_materials.values()]; solver = rt.PathSolver(); solver.loop_mode = "evaluated"
def loss(c):
    mat.scattering_coefficient = c
    p = solver(scene, max_depth=3, diffuse_reflection=diffuse, samples_per_src=10**6, seed=1)
    a_re, a_im = p.a
    return dr.sum(a_re.array**2 + a_im.array**2)
c = mi.Float(0.3); dr.enable_grad(c); t0 = time.time()
L = loss(c)
if mode == "reverse": dr.backward(L); g = dr.grad(c)[0]
else: dr.forward(c); g = dr.grad(L)[0]
e = 5e-2; fd = (loss(mi.Float(0.3 + e))[0] - loss(mi.Float(0.3 - e))[0]) / (2 * e)
print(f"diffuse={diffuse} {mode}: grad {g:.4e}  finite-diff {fd:.4e}  ({time.time()-t0:.1f} s)")
