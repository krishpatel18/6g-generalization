"""Why are the ray-traced priors far off on DICHASUS? Compare delay profiles and covariance spread."""
import sys
import numpy as np
from sixg_gen import channels, dichasus, grid
grid.use_dichasus()
arr = dichasus.Array.named("A")
raw = dichasus.load(arr, ["d011", "d013"])
cal, _ = dichasus.fit_calibration(arr, dichasus.load(arr, ["d010", "d012", "d014"]))
srcs = {"measured": dichasus.measured(raw, cal), "cdl": channels.generate("cdl", 12000, 1, mmap=True)[:2000]}
for c in ("munich", "florence", "wall_street", "rochester_brighton"):
    srcs[c] = channels.generate(f"rt:{c}", 300, 1)
for name, h in srcs.items():
    h = np.asarray(h)
    taps = np.fft.ifft(np.fft.ifftshift(h[:, :, 0], axes=-1), axis=-1)  # 128 taps of 160 ns
    pdp = np.mean(np.abs(taps) ** 2, axis=(0, 1)); pdp /= pdp.sum()
    f = h[:, :, 0].reshape(len(h), -1)
    lam = np.linalg.eigvalsh(f.T @ f.conj() / len(f))[::-1]; lam /= lam.sum()
    fc = np.mean(h[:, 0, 0, :, None] * h[:, 0, 0, None, :].conj(), axis=0)
    corr_df = np.abs(np.mean([fc[k, k + 8] for k in range(120)])) / np.mean(np.abs(np.diag(fc)))
    print(f"{name:20s} taps0..4 {np.round(100*pdp[:5]).astype(int)}% last-taps {100*pdp[-3:].sum():.0f}% | "
          f"top eig {100*lam[0]:.0f}% top-10 {100*lam[:10].sum():.0f}% | freq corr @8sc {corr_df:.2f} | nan {np.isnan(h).any()}")
