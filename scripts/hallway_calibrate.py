"""Stage 1: calibrate the hallway twin's materials to dc01 and check held-out error.

usage: hallway_calibrate.py [n_fit] [n_test] [max_evals]

Reproduces the "Learned Materials" experiment of Hoydis et al. (arXiv 2311.18558,
Table II) in Sionna 2.2: relative permittivity and conductivity of the three
objects (floor, walls, ceiling), no scattering, fitted to measured per-antenna
total power and RMS delay spread (symmetric relative errors, as in the paper),
with one power scale (the measurements have no absolute reference). The paper
used gradient descent; with six parameters a derivative-free search (Nelder-Mead
on log-parameters) reaches the same kind of optimum without the noisy gradients.

Positions: the first 10,000 usable dc01 datapoints in recorded order (the paper's
set); fit on the first 5,000, test on the last 4,900 (the paper's shuffle buffer
of 256 keeps its split close to this). Reported per position and array, as in the
paper: ALE, the absolute dB error of the array-mean power, and RAE, the relative
error of the array-mean delay spread. Writes runs/hallway_calibrate.json.
"""

import json
import sys
import time

import numpy as np
from scipy.optimize import minimize

from sixg_gen import hallway

TAPS = (-8, 80)  # delay window in taps of 20 ns around delay 0, as in the paper
RUNS = "runs/hallway_calibrate.json"


def power(h):
    return np.sum(np.abs(h) ** 2, axis=-1)


def delay_spread(h):
    """RMS delay spread (ns) of each [..., 1024] response over the TAPS window."""
    taps = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(h, axes=-1), axis=-1), axes=-1)
    w = np.abs(taps[..., 512 + TAPS[0]:512 + TAPS[1]]) ** 2
    d = np.arange(*TAPS) * 1e9 / hallway.BANDWIDTH_HZ
    mean = np.sum(w * d, -1) / np.sum(w, -1)
    return np.sqrt(np.sum(w * (d - mean[..., None]) ** 2, -1) / np.sum(w, -1))


def stats(h):
    return power(h), delay_spread(h)


def scale(p_rt, p_meas):
    """Least-squares alpha for p_rt ~ alpha * p_meas."""
    return float(np.sum(p_rt * p_meas) / np.sum(p_meas ** 2))


def smape(a, b):
    return float(np.mean(np.abs(a - b) / (a + b + 1e-30)))


def errors(sim, meas, alpha):
    (p_rt, d_rt), (p_m, d_m) = sim, meas
    pr, pm = p_rt.mean(-1), alpha * p_m.mean(-1)
    ale = np.abs(10 * np.log10(pr + 1e-30) - 10 * np.log10(pm))
    rae = np.abs(d_rt.mean(-1) - d_m.mean(-1)) / d_m.mean(-1)
    return ale.ravel(), rae.ravel()


def summary(ale, rae):
    return dict(ale_mean=float(ale.mean()), ale_std=float(ale.std()), ale_median=float(np.median(ale)),
                rae_mean=float(rae.mean()), rae_std=float(rae.std()), rae_median=float(np.median(rae)))


def materials(u):
    """Log-parameters -> object: (relative permittivity, conductivity), clipped to the paper's
    range (relative permittivity - 1 in [1e-3, 200], conductivity in [1e-3, 1e6] S/m)."""
    return {name: (1 + np.exp(np.clip(u[2 * i], np.log(1e-3), np.log(200))),
                   np.exp(np.clip(u[2 * i + 1], np.log(1e-3), np.log(1e6))))
            for i, name in enumerate(hallway.OBJECTS)}


def calibrate(twin, pos, meas, max_evals: int = 250):
    """Fit the three objects' materials so simulated per-antenna power and delay spread at `pos`
    match `meas` (= stats(measured)). Returns (ITU log-params, fitted log-params, ITU power
    scale, fitted power scale, ITU loss, fitted loss, evaluations); leaves the twin at the fit."""
    def loss(u):
        twin.set_materials(materials(u))
        p_rt, d_rt = stats(twin.simulate(pos))
        alpha = scale(p_rt, meas[0])
        return smape(p_rt, alpha * meas[0]) + smape(d_rt, meas[1]), alpha

    u0 = np.ravel([[np.log(eps - 1), np.log(sigma)] for eps, sigma in (twin.itu[n] for n in hallway.OBJECTS)])
    history = []

    def objective(u):
        value, _ = loss(u)
        history.append(value)
        if len(history) % 10 == 1:
            print(f"  eval {len(history)}: loss {value:.4f} (best {min(history):.4f})", flush=True)
        return value

    itu_loss, itu_alpha = loss(u0)
    print(f"ITU materials: fit loss {itu_loss:.4f}", flush=True)
    res = minimize(objective, u0, method="Nelder-Mead", options=dict(maxfev=max_evals, xatol=1e-2, fatol=1e-4))
    cal_loss, cal_alpha = loss(res.x)
    return u0, res.x, itu_alpha, cal_alpha, itu_loss, cal_loss, len(history)


def main(n_fit: int = 256, n_test: int = 1000, max_evals: int = 250) -> None:
    rng = np.random.default_rng(0)
    data = hallway.load()
    usable = np.flatnonzero(data["keep"])[:10000]
    fit_idx = np.sort(rng.choice(usable[:5000], n_fit, replace=False))
    test_idx = np.sort(rng.choice(usable[5100:], n_test, replace=False))
    meas_fit, meas_test = stats(hallway.calibrated(data, fit_idx)), stats(hallway.calibrated(data, test_idx))
    twin = hallway.Twin()
    pos_fit, pos_test = data["pos"][fit_idx], data["pos"][test_idx]
    t0 = time.time()
    u0, res_x, itu_alpha, cal_alpha, itu_loss, cal_loss, evals = calibrate(twin, pos_fit, meas_fit, max_evals)
    print(f"ITU fit loss {itu_loss:.4f}; calibrated fit loss {cal_loss:.4f} after {evals} evaluations "
          f"({(time.time() - t0) / 60:.0f} min)", flush=True)
    out = dict(n_fit=n_fit, n_test=n_test, evaluations=evals, paper_table2=dict(
        itu=dict(ale_mean=4.93, ale_std=5.26, rae_mean=0.54, rae_std=0.23),
        learned=dict(ale_mean=2.16, ale_std=1.69, rae_mean=0.24, rae_std=0.17)))
    for label, u, alpha in [("itu", u0, itu_alpha), ("calibrated", res_x, cal_alpha)]:
        twin.set_materials(materials(u))
        out[label] = dict(materials={k: [float(v) for v in p] for k, p in materials(u).items()}, alpha=alpha,
                          fit=summary(*errors(stats(twin.simulate(pos_fit)), meas_fit, alpha)),
                          test=summary(*errors(stats(twin.simulate(pos_test)), meas_test, alpha)))
        print(label, json.dumps(out[label], indent=1), flush=True)
    with open(RUNS, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main(*map(int, sys.argv[1:]))
