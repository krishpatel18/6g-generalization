"""Probe a grid variant cheaply: small sets in a scratch cache, LMMSE gap only.
usage: probe.py <num_subcarriers> <num_antennas>"""
import sys, time
from pathlib import Path
import numpy as np
from sixg_gen import channels, estimators, grid
grid.NUM_SUBCARRIERS = int(sys.argv[1]); grid.NUM_ANTENNAS = int(sys.argv[2])
grid.PILOT_SUBCARRIERS = np.arange(0, grid.NUM_SUBCARRIERS, 2)
channels.DATA_DIR = Path(__file__).parent / "probe"
cities = channels.RT_SCENES
t0 = time.time()
fit = {c: channels.generate(f"rt:{c}", 2000, 11) for c in cities}
test = {c: channels.generate(f"rt:{c}", 1000, 13) for c in cities}
cdl = channels.generate("cdl", 2000, 11)
print(f"generated in {time.time()-t0:.0f}s", flush=True)
for snr in (10., 20.):
    g_cdl, g_rt3 = [], []
    for c in cities:
        t = test[c]
        score = lambda f: grid.nmse_db(estimators.LMMSE(f).estimate(t, np.full(len(t), snr), np.random.default_rng(1234)), t)
        o = score(fit[c])
        g_cdl.append(score(cdl) - o)
        g_rt3.append(score(np.concatenate([fit[x][:667] for x in cities if x != c])) - o)
    print(f"{grid.config_tag()} @{snr:.0f}dB  CDL-fit gap {' '.join(f'{g:4.1f}' for g in g_cdl)} | 3-city gap {' '.join(f'{g:4.1f}' for g in g_rt3)}  (mu/et/fl/sf)", flush=True)
