"""Linear MMSE channel estimation from second-order statistics, and adapting them.

`Stats.fit(h)` summarizes channels by the two covariances LMMSE needs (full grid
vs pilots, pilots vs pilots). `Stats.blend(source, target)` adapts a source's
statistics to a few target channels by covariance shrinkage, choosing the blend
weight by two-fold cross-validation on the target channels. `LMMSE(stats)`
estimates the full grid from noisy pilots at any per-sample SNR.

This is the linear twin of "pre-train, then fine-tune": the source covariance is
the prior, the blend is the fine-tuning.
"""

from dataclasses import dataclass

import numpy as np

from . import grid

BLEND_WEIGHTS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
CV_SNR_DB = (0.0, 10.0, 20.0)


def pilots(h: np.ndarray) -> np.ndarray:
    """Pilot entries of each channel, flattened: [n, antennas * pilot symbols * pilot subcarriers]."""
    return h[..., list(grid.PILOT_SYMBOLS), :][..., grid.PILOT_SUBCARRIERS].reshape(len(h), -1)


def noisy_pilots(h: np.ndarray, snr_db: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    p = pilots(h)
    noise_var = 10.0 ** (-np.asarray(snr_db, dtype=np.float64) / 10.0)
    noise = (rng.standard_normal(p.shape) + 1j * rng.standard_normal(p.shape)) * np.sqrt(noise_var / 2)[:, None]
    return (p + noise).astype(np.complex64), noise_var


@dataclass
class Stats:
    r_hp: np.ndarray  # E[h p^H], [grid, pilots]
    r_pp: np.ndarray  # E[p p^H], [pilots, pilots]

    @classmethod
    def fit(cls, h: np.ndarray, chunk: int = 1000) -> "Stats":
        """Sample covariances, accumulated in chunks so large memory-mapped sets stay on disk."""
        r_hp = r_pp = 0
        for i in range(0, len(h), chunk):
            x = np.asarray(h[i:i + chunk]).reshape(-1, h[0].size).astype(np.complex128)
            p = pilots(x.reshape((-1,) + h.shape[1:]))
            r_hp = r_hp + x.T @ p.conj()
            r_pp = r_pp + p.T @ p.conj()
        return cls(r_hp / len(h), r_pp / len(h))

    def mix(self, other: "Stats", w: float) -> "Stats":
        return Stats((1 - w) * self.r_hp + w * other.r_hp, (1 - w) * self.r_pp + w * other.r_pp)

    @classmethod
    def blend(cls, source: "Stats", target: np.ndarray, seed: int = 0) -> tuple["Stats", float]:
        """Shrink `source` toward the sample statistics of `target`; weight by 2-fold CV."""
        if len(target) == 0:
            return source, 0.0
        if len(target) < 4:
            return source.mix(cls.fit(target), 1.0), 1.0
        idx = np.random.default_rng(seed).permutation(len(target))
        halves = (idx[: len(idx) // 2], idx[len(idx) // 2:])
        fits = [cls.fit(target[h]) for h in halves]
        best_w, best_err = 0.0, np.inf
        for w in BLEND_WEIGHTS:
            err = 0.0
            for k in range(2):
                held = target[halves[1 - k]]
                est = LMMSE(source.mix(fits[k], w))
                for snr in CV_SNR_DB:
                    rng = np.random.default_rng(1000 + k)
                    err += 10 ** (grid.nmse_db(est.estimate(held, np.full(len(held), snr), rng), held) / 10)
            if err < best_err:
                best_w, best_err = w, err
        return source.mix(cls.fit(target), best_w), best_w


class LMMSE:
    """h_hat = R_hp (R_pp + s^2 I)^-1 p, via one eigendecomposition so any per-sample SNR is cheap.

    Diagonal loading (a small fraction of the mean pilot power) keeps
    small-sample covariances usable.
    """

    def __init__(self, stats: Stats, loading: float = 1e-3):
        lam, u = np.linalg.eigh(stats.r_pp)
        self.lam = np.clip(lam, 0.0, None) + loading * float(np.mean(np.clip(lam, 0.0, None)))
        self.u_conj = u.conj().astype(np.complex64)
        self.b = (stats.r_hp @ u).astype(np.complex64)  # [grid, pilots]

    def from_pilots(self, p: np.ndarray, noise_var: np.ndarray) -> np.ndarray:
        y = (p @ self.u_conj) / (self.lam[None, :] + noise_var[:, None]).astype(np.float32)
        return y @ self.b.T

    def estimate(self, h: np.ndarray, snr_db: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        p, noise_var = noisy_pilots(h, snr_db, rng)
        return self.from_pilots(p, noise_var).reshape(h.shape)
