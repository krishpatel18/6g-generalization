"""Is the channel phase-coherent from one snapshot to the next (needed for prediction)?"""
import sys, numpy as np
f = sys.argv[1]
z = np.load(f, mmap_mode="r")
print("csi shape", z["csi"].shape)
t = np.asarray(z["time"], float)
o = np.argsort(t)[:3000]; o.sort()
o = o[np.argsort(t[o])]
h = np.asarray(z["csi"][np.sort(o)]).astype(np.complex64)
t = t[np.sort(o)]; h = h[np.argsort(t)]; t = np.sort(t)
h = h.reshape(len(h), -1)
dt = np.diff(t); step = np.median(dt)
runs = np.split(np.arange(len(t)), np.where(dt > 1.5 * step)[0] + 1)
L = np.array([len(r) for r in runs])
print("step ms %.2f; contiguous runs: n %d, median len %d, share of samples in runs >=40: %.2f" % (step*1e3, len(L), np.median(L), L[L >= 40].sum() / L.sum()))
for lag in (1, 2, 4, 8):
    a, b = [], []
    for r in runs:
        if len(r) > lag:
            x, y = h[r[:-lag]], h[r[lag:]]
            c = np.sum(np.conj(x) * y, axis=1) / np.sqrt(np.sum(abs(x)**2, 1) * np.sum(abs(y)**2, 1))
            a.append(c)
    c = np.concatenate(a)
    # copy-last NMSE raw, and after the best common phase rotation
    print("lag %d (%.0f ms): |corr| median %.3f; phase of corr: std %.2f rad, |mean resultant| %.2f; copy-last NMSE raw %.1f dB, common-phase-fixed %.1f dB"
          % (lag, lag*step*1e3, np.median(abs(c)), np.std(np.angle(c)), abs(np.mean(c/abs(c))),
             10*np.log10(np.mean(2 - 2*c.real)), 10*np.log10(np.mean(2 - 2*abs(c)))))
