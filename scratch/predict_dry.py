import sys, time, numpy as np, torch
sys.path.insert(0, "scripts")
from sixg_gen import channels, estimators, grid, linear
grid.NUM_ANTENNAS = 8; grid.use_tracks(); grid.use_tracks()
assert grid.NUM_SUBCARRIERS == 32
import predict
h = channels.generate("rt:munich", 300, 99); sp = channels.speeds("rt:munich", 300, 99)
site, test = h[:100], h[100:]
rt = linear.Stats.fit(site); rows = []
def rec(t, m, n, w, scores): rows.extend((t, m, n, w, *r[:3], r[3]) for r in scores)
rec("munich", "tone", 0, "", predict.evaluate(predict.tone, test, sp[100:]))
for name in ("cdl", "rt3"):
    for n in (0, 50):
        a, w = linear.Stats.blend(rt, site[:n], 0)
        rec("munich", f"wiener_{name}", n, w, predict.evaluate(predict.targets_of(linear.LMMSE(a).estimate), test, sp[100:]))
rec("munich", "wiener_own", 50, "", predict.evaluate(predict.targets_of(linear.LMMSE(linear.Stats.fit(site[:50])).estimate), test, sp[100:]))
for dom, (to, back) in predict.DOMAINS.items():
    front = predict.InDomain(linear.LMMSE(rt), to, back)
    # identity check: transform round trip
    assert np.allclose(back(to(test)), test, atol=1e-5)
    t = time.time()
    m = estimators.train(estimators.Refiner(width=48), front, to(site), 40, 1e-3, 0)
    print(dom, f"{(time.time()-t)/40*1000:.0f} ms/step", flush=True)
    for n in (0, 50):
        ref = lambda hh, s, rng: back(estimators.refine(m, front, to(hh), s, rng))
        rec("munich", f"learned_{dom}", n, 0, predict.evaluate(predict.targets_of(ref), test, sp[100:]))
    # untrained-equivalence: zero-init refiner == its front
    z = estimators.Refiner(width=48)
    a1 = predict.targets_of(lambda hh, s, rng: back(estimators.refine(z, front, to(hh), s, rng)))(test, np.full(len(test), 10.), np.random.default_rng(1))
    a2 = predict.targets_of(linear.LMMSE(rt).estimate)(test, np.full(len(test), 10.), np.random.default_rng(1))
    print(dom, "zero-init == front:", np.allclose(a1, a2, atol=1e-4))
channels.RT_SCENES = ("munich",)
predict.gate(rows)
