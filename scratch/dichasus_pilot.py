"""Pilot: does processed DICHASUS array-A data have covariance structure a prior could exploit?

Calibrate on d010/d012/d014 (per-antenna phase + delay from line-of-sight geometry and
tachymeter positions), process every source with one chain (timing alignment -> delay
window -> 128-subcarrier chunk -> unit power), then compare on d011/d013 at 0 and 10 dB:
LS interpolation, LMMSE from a CDL prior (1.272 GHz, 2x8, 38.901 pattern), LMMSE fitted
on the calibration files themselves (pilot "ceiling", undersampled).
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
from sixg_gen import grid, linear  # noqa: E402

RAW = "data/dichasus/raw"
FULL = 1024
SCS = 50e6 / FULL
CARRIER = 1.272e9
LAMBDA = 299792458 / CARRIER
WINDOW = (-10, 100)          # delay taps kept around the first arrival (20 ns taps)
CHUNK = slice(448, 576)      # 128 subcarriers around the carrier (fftshifted order)
TILT = np.arcsin(0.349)

spec = json.load(open(f"{RAW}/spec.json"))
arr = spec["antennas"][0]
asg = np.array(arr["assignments"])
order = [asg[r][c] for c in range(8) for r in range(2)]  # Sionna order: column-major, row 0 on top
loc, bore = np.array(arr["location"]), np.array(arr["direction"])
h_ax = np.cross([0, 0, 1], bore); h_ax /= np.linalg.norm(h_ax)
v_ax = np.cross(bore, h_ax); v_ax /= np.linalg.norm(v_ax)
elem = np.array([loc + (c - 3.5) * 0.118 * h_ax + (0.5 - r) * 0.118 * v_ax for c in range(8) for r in range(2)])
freqs = (np.arange(FULL) - FULL // 2) * SCS  # fftshifted order


def load(names):
    out = [np.load(f"{RAW}/dichasus-{n}-complete.npz") for n in names]
    csi = np.fft.fftshift(np.concatenate([z["csi"][:, order] for z in out]), axes=2)
    pos = np.concatenate([z["pos"] for z in out])
    snr = np.concatenate([z["snr"][:, order] for z in out])
    age = np.concatenate([z["interp_age"] for z in out])
    return csi, pos, snr, age


def fit_calibration(csi, pos):
    """Per antenna: phase offset + delay (relative to antenna 0) + gain."""
    d = np.linalg.norm(pos[:, None] - elem[None], axis=2)
    model = np.exp(-2j * np.pi * (d - d[:, :1]) / LAMBDA)
    r = csi * csi[:, :1].conj()
    r /= np.abs(r) + 1e-12
    ph = np.unwrap(np.angle(np.sum(r * model.conj()[..., None], axis=0)), axis=1)  # [A, FULL]
    lin = np.array([np.polyfit(freqs, p, 1) for p in ph])  # slope (rad/Hz), intercept
    corr = np.exp(-1j * (lin[:, :1] * freqs[None] + lin[:, 1:]))
    gain = np.sqrt(np.mean(np.abs(csi) ** 2, axis=(0, 2)))
    return corr / (gain / gain.mean())[:, None]


def to_grid(h):
    """The one chain for every source: [n, A, FULL] fftshifted -> [n, A, 1, 128], unit power."""
    taps = np.fft.ifft(np.fft.ifftshift(h, axes=2), axis=2)  # natural order -> delay taps
    pdp = np.sum(np.abs(taps) ** 2, axis=1)
    peak = pdp.max(axis=1, keepdims=True)
    t0 = np.empty(len(h), dtype=int)
    for i in range(len(h)):
        cand = np.arange(-20, 21) % FULL
        p = pdp[i, cand] / peak[i]
        first = np.flatnonzero(p > 0.1)
        t0[i] = (cand[first[0]] if len(first) else np.argmax(pdp[i]))
    t0 = np.where(t0 > FULL // 2, t0 - FULL, t0)
    taps = np.stack([np.roll(taps[i], -t0[i], axis=1) for i in range(len(h))])
    keep = np.zeros(FULL, bool)
    keep[np.arange(WINDOW[0], WINDOW[1]) % FULL] = True
    taps[:, :, ~keep] = 0
    g = np.fft.fftshift(np.fft.fft(taps, axis=2), axes=2)[:, :, CHUNK]
    return grid.normalize_power(g[:, :, None, :])


def cdl(n, seed):
    from sionna.phy.channel import cir_to_ofdm_channel
    from sionna.phy.channel.tr38901 import CDL, AntennaArray
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ant = dict(polarization="single", polarization_type="V", antenna_pattern="38.901", carrier_frequency=CARRIER)
    f = torch.tensor(freqs, dtype=torch.float32)
    out = []
    for start in range(0, n, 64):
        b = min(64, n - start)
        ori = torch.zeros(b, 3)
        ori[:, 0] = torch.from_numpy(rng.uniform(-np.radians(75), np.radians(75), b))
        ori[:, 1] = TILT
        model = CDL("ABCDE"[rng.integers(5)], float(rng.uniform(30e-9, 300e-9)), CARRIER,
                    ut_array=AntennaArray(num_rows=1, num_cols=1, **{**ant, "antenna_pattern": "omni"}),
                    bs_array=AntennaArray(num_rows=2, num_cols=8, **ant), direction="uplink",
                    bs_orientation=ori, min_speed=0.0, max_speed=0.0, device="cpu")
        a, tau = model(b, 1, 1.0)
        h = cir_to_ofdm_channel(f, a, tau).reshape(b, 16, FULL).numpy()
        out.append(to_grid(h))
    return np.concatenate(out)


def ls(h, snr_db, rng):
    p, nv = linear.noisy_pilots(h, snr_db, rng)
    p = p.reshape(len(h), 16, 64)
    est = np.empty(h.shape[:2] + (128,), np.complex64)
    est[:, :, 0::2] = p
    est[:, :, 1:-1:2] = (p[:, :, :-1] + p[:, :, 1:]) / 2
    est[:, :, -1] = p[:, :, -1]
    return est[:, :, None, :]


def main():
    grid.NUM_ANTENNAS, grid.NUM_SYMBOLS, grid.NUM_SUBCARRIERS = 16, 1, 128
    grid.PILOT_SYMBOLS, grid.PILOT_SUBCARRIERS = (0,), np.arange(0, 128, 2)
    cal_csi, cal_pos, cal_snr, cal_age = load(["d010", "d012", "d014"])
    corr = fit_calibration(cal_csi[cal_age < 0.1], cal_pos[cal_age < 0.1])
    test_csi, _, test_snr, test_age = load(["d011", "d013"])
    good = lambda snr, age: (snr.mean(1) >= 10) & (age < 0.1)  # noqa: E731
    fit = to_grid(cal_csi[good(cal_snr, cal_age)] * corr)
    test = to_grid(test_csi[good(test_snr, test_age)] * corr)
    raw_test = to_grid(test_csi[good(test_snr, test_age)])  # no calibration, for comparison
    print(f"fit {len(fit)}  test {len(test)}  (of {len(cal_csi)} / {len(test_csi)})")
    sim = cdl(4000, 1)
    ests = {"ls_interp": ls,
            "lmmse_cdl": linear.LMMSE(linear.Stats.fit(sim)).estimate,
            "lmmse_own_fit": linear.LMMSE(linear.Stats.fit(fit)).estimate}
    for snr in (0.0, 10.0):
        line = []
        for name, est in ests.items():
            rng = np.random.default_rng(1234)
            line.append(f"{name} {grid.nmse_db(est(test, np.full(len(test), snr), rng), test):6.2f}")
        rng = np.random.default_rng(1234)
        line.append(f"own_fit on uncalibrated {grid.nmse_db(ests['lmmse_own_fit'](raw_test, np.full(len(raw_test), snr), rng), raw_test):6.2f}")
        print(f"{snr:4.0f} dB: " + " | ".join(line))
    # how much of the measured power the delay window keeps (noise vs signal check)
    taps = np.fft.ifft(np.fft.ifftshift(test_csi[:200] * corr, axes=2), axis=2)
    print("power share in first 100 taps after alignment ~", round(float(np.mean(
        np.sum(np.abs(taps[:, :, :100]) ** 2, axis=(1, 2)) / np.sum(np.abs(taps) ** 2, axis=(1, 2)))), 3))


if __name__ == "__main__":
    main()
