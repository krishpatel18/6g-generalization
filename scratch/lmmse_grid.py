"""Linear twin of the pre-training arms: LMMSE fitted on each source, scored on each city."""
import numpy as np
from sixg_gen import channels, estimators, grid
cities = channels.RT_SCENES
full = {c: channels.generate(f"rt:{c}", 12000, 1) for c in cities}
cdl = channels.generate("cdl", 12000, 1)
print(f"{'fit on':22s}" + "".join(f"{c:>22s}" for c in cities))
def row(name, fit_for):
    cells = []
    for c in cities:
        test = channels.generate(f"rt:{c}", 2000, 3)
        est = estimators.LMMSE(fit_for(c))
        cells.append("/".join(f"{grid.nmse_db(est.estimate(test, np.full(len(test), s), np.random.default_rng(1234)), test):.1f}" for s in (0., 10., 20.)))
    print(f"{name:22s}" + "".join(f"{x:>22s}" for x in cells), flush=True)
row("cdl 12k", lambda c: cdl)
row("3 other cities 12k", lambda c: np.concatenate([full[o][:4000] for o in cities if o != c]))
row("1 other city 12k", lambda c: full[[o for o in cities if o != c][0]])
row("in-domain 12k (oracle)", lambda c: full[c])
print("cells: NMSE dB at 0/10/20 dB SNR")
