"""Channel prediction, leave-one-city-out: does a learned predictor beat the classical ones?

usage: predict.py <num_bs_antennas> <seed>

Tracks (grid.use_tracks): 8 noisy history observations over 4 slots, targets
1-20 slots ahead. Every method sees the same noisy history. Methods:
  hold_last  - last noisy observation, repeated
  tone       - per user, no training data: one Doppler tone per angle-delay bin
  wiener     - space-time LMMSE from a covariance prior (cdl, rt3 = other three
               cities pooled, own = the N site tracks alone, ceiling = own 12k),
               blended with N site tracks
  learned    - CNN residual on the rt3 Wiener (pre-trained on rt3 tracks,
               fine-tuned on the N site tracks)
Pre-registration and gate: docs/PLAN.md, "Prediction experiment".

Writes runs/<grid tag>/predict_seed<seed>.csv:
  target, method, n, blend_w, snr_db, horizon_ms, speed, nmse_db
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

from sixg_gen import channels, estimators, grid, linear

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loco import RUNS  # noqa: E402

EVAL_SNR_DB = (0.0, 10.0, 20.0)
SITE_N = (0, 50)
PRETRAIN_STEPS, FINETUNE_STEPS = 4000, 300
WIDTH = 48
SPEED_BINS = ((0, 10), (10, 20), (20, 30))
MAX_DOPPLER_HZ = 400.0


def history_times_s() -> np.ndarray:
    return np.array(grid.TRACK_TIMES[:len(grid.PILOT_SYMBOLS)]) * grid.SYMBOL_PERIOD_S


def target_times_s() -> np.ndarray:
    return np.array(grid.TRACK_TIMES[len(grid.PILOT_SYMBOLS):]) * grid.SYMBOL_PERIOD_S


def noisy_history(h: np.ndarray, snr_db: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """The same noise draw `linear.LMMSE.estimate` makes, shaped [n, A, history, F]."""
    p, noise_var = linear.noisy_pilots(h, snr_db, rng)
    return p.reshape(len(h), grid.NUM_ANTENNAS, len(grid.PILOT_SYMBOLS), grid.NUM_SUBCARRIERS), noise_var


def hold_last(h, snr_db, rng):
    x, _ = noisy_history(h, snr_db, rng)
    return np.repeat(x[:, :, -1:], len(grid.HORIZON_SLOTS), axis=2)


def tone(h, snr_db, rng, chunk: int = 200):
    """Angle-delay bins each modelled as one Doppler tone, fitted to the history.

    Unitary transforms over antennas and subcarriers keep the noise white. Per
    bin: frequency = periodogram peak on a 5 Hz grid within +-MAX_DOPPLER_HZ,
    amplitude = projection on that tone, shrunk by its estimated SNR (Wiener
    factor), then extrapolated to the target times.
    """
    x, noise_var = noisy_history(h, snr_db, rng)
    t_hist, t_tgt = history_times_s(), target_times_s()
    freqs = np.arange(-MAX_DOPPLER_HZ, MAX_DOPPLER_HZ + 1, 5.0)
    steer = np.exp(-2j * np.pi * np.outer(t_hist, freqs)) / len(t_hist)  # [T, f]
    out = np.empty((len(h), grid.NUM_ANTENNAS, len(t_tgt), grid.NUM_SUBCARRIERS), dtype=np.complex64)
    for i in range(0, len(h), chunk):
        y = np.fft.ifft(np.fft.fft(x[i:i + chunk], axis=1, norm="ortho"), axis=3, norm="ortho")
        y = np.moveaxis(y, 2, -1)  # [n, A, F, T]
        proj = y @ steer  # [n, A, F, f]: amplitude of each candidate tone
        best = np.argmax(np.abs(proj), axis=-1)
        c = np.take_along_axis(proj, best[..., None], -1)[..., 0]
        f = freqs[best]
        noise_c = noise_var[i:i + chunk, None, None] / len(t_hist)
        c = c * np.clip(1 - noise_c / np.maximum(np.abs(c) ** 2, 1e-12), 0, 1)
        pred = c[..., None] * np.exp(2j * np.pi * f[..., None] * t_tgt)  # [n, A, F, targets]
        pred = np.moveaxis(pred, -1, 2)
        out[i:i + chunk] = np.fft.fft(np.fft.ifft(pred, axis=1, norm="ortho"), axis=3, norm="ortho")
    return out


def targets_of(estimate):
    """Wrap a full-track estimator so it returns the target times only."""
    return lambda h, s, rng: estimate(h, s, rng)[:, :, len(grid.PILOT_SYMBOLS):]


def evaluate(estimate, test: np.ndarray, speed: np.ndarray) -> list[tuple]:
    rows = []
    truth = test[:, :, len(grid.PILOT_SYMBOLS):]
    for snr in EVAL_SNR_DB:
        rng = np.random.default_rng(1234)  # identical history noise for every method
        est = estimate(test, np.full(len(test), snr), rng)
        for k, hz in enumerate(grid.HORIZON_SLOTS):
            ms = hz * 0.5
            rows.append((snr, ms, "all", grid.nmse_db(est[:, :, k], truth[:, :, k])))
            for lo, hi in SPEED_BINS:
                sel = (speed >= lo) & (speed < hi)
                rows.append((snr, ms, f"{lo}-{hi}", grid.nmse_db(est[sel][:, :, k], truth[sel][:, :, k])))
    return rows


def rt3(target: str) -> np.ndarray:
    return np.concatenate([channels.generate(f"rt:{c}", 12000, 1, mmap=True)[:4000]
                           for c in channels.RT_SCENES if c != target])


def main(seed: int) -> None:
    runs = RUNS / grid.config_tag()
    runs.mkdir(parents=True, exist_ok=True)
    rows = []

    def record(target, method, n, w, scores):
        rows.extend((target, method, n, w, *r[:3], round(r[3], 3)) for r in scores)
        s10 = {ms: v for snr, ms, sp, v in scores if snr == 10.0 and sp == "all"}
        print(f"  {target:14s} {method:14s} N={n:<3d} w={w:<4} 10dB " +
              "  ".join(f"{ms:g}ms:{v:6.2f}" for ms, v in s10.items()), flush=True)

    cdl = linear.Stats.fit(channels.generate("cdl", 12000, 1, mmap=True))
    for target in channels.RT_SCENES:
        print(f"== target {target}", flush=True)
        site = channels.generate(f"rt:{target}", 1000, 2)
        test = channels.generate(f"rt:{target}", 2000, 3)
        speed = channels.speeds(f"rt:{target}", 2000, 3)
        record(target, "hold_last", 0, "", evaluate(hold_last, test, speed))
        record(target, "tone", 0, "", evaluate(tone, test, speed))
        ceiling = linear.LMMSE(linear.Stats.fit(channels.generate(f"rt:{target}", 12000, 1, mmap=True)))
        record(target, "wiener_ceiling", 12000, "", evaluate(targets_of(ceiling.estimate), test, speed))
        own = linear.LMMSE(linear.Stats.fit(site[:50]))
        record(target, "wiener_own", 50, "", evaluate(targets_of(own.estimate), test, speed))

        pool = rt3(target)
        rt = linear.Stats.fit(pool)
        for name, prior in (("cdl", cdl), ("rt3", rt)):
            for n in SITE_N:
                adapted, w = linear.Stats.blend(prior, site[:n], seed)
                record(target, f"wiener_{name}", n, w, evaluate(targets_of(linear.LMMSE(adapted).estimate), test, speed))

        path = runs / f"pre_track_rt_not_{target}_s{seed}.pt"
        if not path.exists():
            t = time.time()
            torch.manual_seed(seed)
            model = estimators.train(estimators.Refiner(width=WIDTH), linear.LMMSE(rt), np.asarray(pool),
                                     PRETRAIN_STEPS, 1e-3, seed)
            torch.save(model.state_dict(), path)
            print(f"  pre-trained in {time.time()-t:.0f}s", flush=True)
        state = torch.load(path)
        for n in SITE_N:
            adapted, w = linear.Stats.blend(rt, site[:n], seed)
            front = linear.LMMSE(adapted)
            model = estimators.Refiner(width=WIDTH)
            model.load_state_dict(state)
            if n:
                model = estimators.train(model, front, site[:n], FINETUNE_STEPS, 1e-4, seed + n, batch=min(64, n))
            refine = lambda h, s, rng: estimators.refine(model, front, h, s, rng)  # noqa: E731
            record(target, "learned", n, w, evaluate(targets_of(refine), test, speed))

        with open(runs / f"predict_seed{seed}.csv", "w", newline="") as f:
            out = csv.writer(f)
            out.writerow(("target", "method", "n", "blend_w", "snr_db", "horizon_ms", "speed", "nmse_db"))
            out.writerows(rows)
    gate(rows)
    print("ALL DONE", flush=True)


def gate(rows) -> None:
    """Pre-registered gate: learned vs the best non-learned method at the same N, 10 dB, >= 2 ms."""
    v = {(t, m, n, ms): x for t, m, n, _, snr, ms, sp, x in rows if snr == 10.0 and sp == "all"}
    print("\nGate (10 dB, all speeds): learned minus best non-learned, dB (negative = learned better)")
    for n in SITE_N:
        for t in channels.RT_SCENES:
            cells = []
            for ms in (2.0, 5.0, 10.0):
                rivals = [v[(t, "tone", 0, ms)], v[(t, "wiener_cdl", n, ms)], v[(t, "wiener_rt3", n, ms)]]
                if n:
                    rivals.append(v[(t, "wiener_own", n, ms)])
                cells.append(v[(t, "learned", n, ms)] - min(rivals))
            print(f"  N={n:<3d} {t:14s} " + "  ".join(f"{ms:g}ms:{c:6.2f}" for ms, c in zip((2, 5, 10), cells)))


if __name__ == "__main__":
    grid.NUM_ANTENNAS = int(sys.argv[1])
    grid.use_tracks()
    main(int(sys.argv[2]))
