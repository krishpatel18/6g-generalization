import time, numpy as np
from sixg_gen import channels, grid, linear
grid.NUM_ANTENNAS = 8
grid.use_tracks()
print(grid.config_tag(), grid.TRACK_TIMES, grid.subcarrier_offsets_hz()[:3])
for src in ("cdl", "rt:munich"):
    t = time.time(); h = channels.generate(src, 300, 99); print(src, h.shape, f"{time.time()-t:.0f}s", channels.speeds(src, 300, 99)[:4])
    # correlation of mid-slot targets with last history point, per horizon
    last = h[:, :, 7]
    for k, hz in enumerate(grid.HORIZON_SLOTS):
        tgt = h[:, :, grid.target_index(hz)]
        print(f"  hold-last noiseless NMSE h={hz:2d}: {grid.nmse_db(last, tgt):6.2f}")
    s = linear.Stats.fit(h); e = linear.LMMSE(s)
    est = e.estimate(h, np.full(len(h), 10.0), np.random.default_rng(0))
    for hz in grid.HORIZON_SLOTS:
        i = grid.target_index(hz); print(f"  in-sample LMMSE h={hz:2d}: {grid.nmse_db(est[:, :, i], h[:, :, i]):6.2f}")
