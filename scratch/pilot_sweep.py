"""How big is the CDL-vs-in-domain LMMSE gap as pilots get sparser? (no new data needed)"""
import numpy as np
from sixg_gen import channels, estimators, grid
cities = channels.RT_SCENES
full = {c: channels.generate(f"rt:{c}", 12000, 1) for c in cities}
cdl = channels.generate("cdl", 12000, 1)
tests = {c: channels.generate(f"rt:{c}", 2000, 3) for c in cities}
for step, syms in [(2, (2, 11)), (4, (2, 11)), (6, (2, 11)), (4, (2,)), (8, (2,)), (12, (2,))]:
    grid.PILOT_SUBCARRIERS = np.arange(0, grid.NUM_SUBCARRIERS, step)
    grid.PILOT_SYMBOLS = syms
    gaps, rt3gaps = [], []
    for c in cities:
        t = tests[c]
        score = lambda fit: grid.nmse_db(estimators.LMMSE(fit).estimate(t, np.full(len(t), 10.), np.random.default_rng(1234)), t)
        oracle, from_cdl = score(full[c]), score(cdl)
        from_rt3 = score(np.concatenate([full[o][:4000] for o in cities if o != c]))
        gaps.append(from_cdl - oracle); rt3gaps.append(from_rt3 - oracle)
    print(f"pilots every {step:2d} sc, symbols {str(syms):8s} ({len(grid.PILOT_SUBCARRIERS)*len(syms):3d} pilots): "
          f"CDL-fit gap {' '.join(f'{g:4.1f}' for g in gaps)} | 3-city-fit gap {' '.join(f'{g:4.1f}' for g in rt3gaps)}  (dB @10dB, mu/et/fl/sf)", flush=True)
