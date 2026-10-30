import glob, numpy as np, sys
for f in sorted(glob.glob(sys.argv[1] + "/**/*.npz", recursive=True)):
    z = np.load(f, mmap_mode="r")
    t = np.asarray(z["time"], float); p = np.asarray(z["pos"], float)
    if len(t) < 3: continue
    dt = np.diff(t); o = np.argsort(t); ts = t[o]; d = np.diff(ts)
    v = np.linalg.norm(np.diff(p[o][:, :2], axis=0), axis=1) / np.maximum(d, 1e-9)
    print(f.split("/")[-1], "n", len(t), "span s %.1f" % (ts[-1]-ts[0]),
          "dt ms p5/50/95", np.round(np.percentile(d, [5, 50, 95]) * 1e3, 2),
          "speed m/s median %.2f" % np.median(v[d > 0]), "frac dt<5ms %.2f" % np.mean(d < 5e-3))
