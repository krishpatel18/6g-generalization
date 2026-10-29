"""(2026-10-29, before the timing fix; log runs/timing_offset_check.log) Where is the first arrival before alignment, and does the alignment window cut it? factory vs street vs lab."""
import sys
import numpy as np
from sixg_gen import dichasus, grid, wideband

sys.path.insert(0, "/Users/krishpatel/projects/6g-generalization/scripts")
from cdl_gap import ENVS, stats

env = sys.argv[1]
dataset, names, (r, c, tilt), fit_files, site_files, test_files, tx_z = ENVS[env]
grid.use_dichasus(r, c, tilt)
names = names or dichasus.array_names(dataset)
N = 3000
for name in names:
    arr = dichasus.Array.named(name, dataset)
    raw = dichasus.load(arr, test_files[:1], dataset, tx_z)
    fit = dichasus.load(arr, fit_files[:1], dataset, tx_z)
    cal, coh = dichasus.fit_calibration(arr, fit)
    idx = np.flatnonzero(raw.usable)
    idx = idx[np.linspace(0, len(idx) - 1, min(N, len(idx))).astype(int)]
    h = raw.csi[idx] * cal[None]
    w = h.shape[-1]
    taps = np.fft.ifft(np.fft.ifftshift(h, axes=-1), axis=-1)
    pdp = np.sum(np.abs(taps) ** 2, axis=(1, 2))  # [n, w]
    k = np.fft.fftfreq(w) * w
    peak = k[np.argmax(pdp, axis=1)]
    thr = pdp >= pdp.max(axis=1, keepdims=True) * 0.1
    # earliest above threshold, searching whole circle in signed order
    order = np.argsort(k)
    firsts = k[order][np.argmax(thr[:, order], axis=1)]
    q = lambda x: np.percentile(x, [5, 50, 95]).round(1)
    # energy shares by absolute delay (signed taps of 20 ns) averaged
    mp = (pdp / pdp.sum(1, keepdims=True)).mean(0)
    def share(lo, hi):
        return 100 * mp[(k >= lo) & (k < hi)].sum()
    print(f"\n== {env} {name} coh {coh:.3f}, n {len(idx)}")
    print(f"  peak tap (20 ns) pct5/50/95 {q(peak)}; earliest>-10dB (global) {q(firsts)}")
    print(f"  energy share by signed tap: <-20 {share(-512,-20):.1f}%  [-20,0) {share(-20,0):.1f}%  [0,20) {share(0,20):.1f}%  [20,100) {share(20,100):.1f}%  >=100 {share(100,513):.1f}%")
    # aligned PDP: how much energy kept by window; tail shape
    for st in (20, 100):
        wideband.SEARCH_TAPS = st
        g = wideband.to_grid(h)
        s = stats(grid.normalize_power(g))
        print(f"  SEARCH_TAPS={st}: " + ", ".join(f"{a} {b:.1f}" for a, b in s.items()))
    wideband.SEARCH_TAPS = 20
    # aligned full-band PDP decay (20 ns taps) relative to first arrival, mean over points
    near = np.arange(-20, 21)
    p = pdp[:, near % w]
    first = near[np.argmax(p >= p.max(1, keepdims=True) * 0.1, axis=1)]
    al = np.stack([np.roll(pdp[i], -first[i]) for i in range(len(first))])
    al = al / al.sum(1, keepdims=True)
    m = al.mean(0)
    kept = 100 * m[np.arange(-10, 100) % w].sum()
    print(f"  aligned: energy kept in window {kept:.1f}%; mean PDP dB at taps -10,-5,0,5,10,20,40,60,80,99,150,300: "
          + " ".join(f"{10*np.log10(m[t % w]+1e-12):.1f}" for t in (-10,-5,0,5,10,20,40,60,80,99,150,300)))
