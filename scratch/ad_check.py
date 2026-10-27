import sys, numpy as np
sys.path.insert(0, "scripts")
from sixg_gen import channels, estimators, grid
grid.NUM_ANTENNAS = 8; grid.use_tracks()
import predict, predict_fair
h = channels.generate("rt:munich", 300, 99); small = h[200:]
args = lambda: (np.full(len(small), 10.0), np.random.default_rng(5))
front = predict_fair.ToneFront(); to, back = predict.DOMAINS["ad"]
wrapped = predict.InDomain(front, to, back)
out = back(estimators.refine(estimators.Refiner(width=48), wrapped, to(small), *args()))
ref = front.estimate(small, *args())
d = np.abs(out - ref); print("max diff", d.max(), "where", np.unravel_index(d.argmax(), d.shape), "max |ref|", np.abs(ref).max())
print("per-time max diff", d.max(axis=(0, 1, 3)))
w = predict.InDomain(front, to, back).estimate(to(small), *args())
print("wrapped-only roundtrip max diff", np.abs(back(w) - ref).max())
