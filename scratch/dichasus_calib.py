"""Estimate per-antenna phase offsets of one DICHASUS array from line-of-sight geometry.

For each candidate element layout (8-axis horizontal or vertical, both signs), fit one
constant phase offset per antenna to the measured inter-antenna phases, using the known
transmitter positions, and report how coherent the fit is (1 = perfect).
"""
import itertools, json, sys
import numpy as np

LAMBDA = 299792458 / 1.272e9
spec = json.load(open("data/dichasus/raw/spec.json"))
files = sys.argv[2:]
arr = spec["antennas"]["ABC".index(sys.argv[1])]
asg = np.array(arr["assignments"]).ravel()
loc, bore = np.array(arr["location"]), np.array(arr["direction"])
h_ax = np.cross([0, 0, 1], bore); h_ax /= np.linalg.norm(h_ax)
v_ax = np.cross(bore, h_ax); v_ax /= np.linalg.norm(v_ax)
csi, pos, snr = [], [], []
for f in files:
    z = np.load(f)
    csi.append(z["csi"][:, asg]); pos.append(z["pos"]); snr.append(z["snr"][:, asg])
csi, pos, snr = np.concatenate(csi), np.concatenate(pos), np.concatenate(snr)
keep = snr.mean(1) > np.percentile(snr.mean(1), 50)
csi, pos = csi[keep], pos[keep]
r = np.sum(csi * csi[:, :1].conj(), axis=2)  # [n, 16]: phase relative to element 0, summed over subcarriers
r /= np.abs(r) + 1e-12
for horiz8, s8, s2 in itertools.product([True, False], [1, -1], [1, -1]):
    ax8, ax2 = (h_ax, v_ax) if horiz8 else (v_ax, h_ax)
    elem = np.array([loc + s2 * (i // 8 - 0.5) * 0.118 * ax2 + s8 * (i % 8 - 3.5) * 0.118 * ax8 for i in range(16)])
    d = np.linalg.norm(pos[:, None, :] - elem[None], axis=2)  # [n, 16]
    model = np.exp(-2j * np.pi * (d - d[:, :1]) / LAMBDA)
    off = np.angle(np.sum(r * model.conj(), axis=0))
    resid = r * model.conj() * np.exp(-1j * off)
    coh = np.abs(resid.mean(axis=1)).mean()
    print(f"8-axis {'horizontal' if horiz8 else 'vertical  '} signs ({s8:+d},{s2:+d}): coherence {coh:.3f}  "
          f"offsets deg {np.round(np.degrees(off[:8])).astype(int)}")
# baseline: no geometry, offsets only
off = np.angle(r.mean(0)); print("offsets only, no geometry: coherence", round(float(np.abs((r * np.exp(-1j * off)).mean(1)).mean()), 3))
