import sys, numpy as np
sys.path.insert(0, "scripts")
from sixg_gen import channels, grid
grid.NUM_ANTENNAS = 8; grid.use_tracks()
import predict
h = channels.generate("rt:munich", 300, 99); sp = channels.speeds("rt:munich", 300, 99)
for snr in (10.0, 40.0):
    for name, f in (("hold", predict.hold_last), ("tone", predict.tone)):
        est = f(h, np.full(len(h), snr), np.random.default_rng(0)); tr = h[:, :, 8:]
        print(snr, name, " ".join(f"{grid.nmse_db(est[:, :, k], tr[:, :, k]):6.2f}" for k in range(5)))
