"""Dry run for predict_fair.py on small cached data (plus a check that tone is unchanged)."""
import sys, time, numpy as np
sys.path.insert(0, "scripts")
from sixg_gen import channels, estimators, grid, linear
grid.NUM_ANTENNAS = 8; grid.use_tracks()
import predict, predict_fair

# 1. tone unchanged by the refactor: Munich test, 10 dB, 2 ms was -7.48 in predict_seed0.csv
test = channels.generate("rt:munich", 2000, 3); sp = channels.speeds("rt:munich", 2000, 3)
r = {(snr, ms, s): v for snr, ms, s, v in predict.evaluate(predict.tone, test, sp)}
print("tone munich 10dB 2ms:", round(r[(10.0, 2.0, "all")], 2), "(expect -7.48)")

h = channels.generate("rt:munich", 300, 99)
pool, small = h[:200], h[200:]
rng_args = lambda: (np.full(len(small), 10.0), np.random.default_rng(5))
a = predict.tone_track(small, *rng_args())[:, :, 8:]
b = predict.tone(small, *rng_args())
print("tone_track targets == tone:", np.allclose(a, b))
for name, front in (("wiener_w", linear.LMMSE(linear.Stats.fit(pool))), ("tone_res", predict_fair.ToneFront())):
    w = predict_fair.horizon_weights(front, pool)
    print(name, "weights", np.round(w, 2))
    for dom, (to, back) in predict.DOMAINS.items():
        wrapped = predict.InDomain(front, to, back)
        z = estimators.Refiner(width=48)
        out = back(estimators.refine(z, wrapped, to(small), *rng_args()))[:, :, 8:]
        ref = front.estimate(small, *rng_args())[:, :, 8:]
        print(f"  {dom}: zero-init == front:", np.allclose(out, ref, atol=1e-4), end="")
        t = time.time()
        m = estimators.train(estimators.Refiner(width=48), wrapped, to(pool), 20, 1e-3, 0, time_weight=w)
        print(f"  {(time.time()-t)/20*1000:.0f} ms/step")
print("DRY DONE")
