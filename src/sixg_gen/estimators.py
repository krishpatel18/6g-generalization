"""Channel estimators: a small learned refiner and a sample-covariance LMMSE baseline.

Every estimator maps (true channels, per-sample SNR, noise rng) to estimates, so
`evaluate` can score them identically. Noise is drawn inside, from the rng, so
two estimators given equal seeds see identical pilots.
"""

import numpy as np
import torch
from torch import nn

from . import grid

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TRAIN_SNR_DB = (-5.0, 25.0)


class Refiner(nn.Module):
    """Residual CNN that cleans up the interpolated LS grid (ChannelNet-style).

    Input/output: [batch, 2 (re/im), symbols, subcarriers] plus an SNR plane on input.
    """

    def __init__(self, width: int = 48, depth: int = 6):
        super().__init__()
        layers = [nn.Conv2d(3, width, 3, padding=1), nn.GELU()]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(width, width, 3, padding=1), nn.GELU()]
        layers += [nn.Conv2d(width, 2, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, ls: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        snr_plane = (snr_db / 20.0)[:, None, None, None].expand(-1, 1, *ls.shape[2:])
        return ls + self.net(torch.cat([ls, snr_plane], dim=1))


def _to_planes(h: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.stack([h.real, h.imag], axis=1).astype(np.float32))


def _from_planes(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().numpy()
    return (x[:, 0] + 1j * x[:, 1]).astype(np.complex64)


def train(model: Refiner, h: np.ndarray, steps: int, lr: float, seed: int, batch: int = 128) -> Refiner:
    """Train on channels `h` with fresh pilots noise and SNR drawn every step."""
    rng = np.random.default_rng(seed)
    model.to(DEVICE).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        idx = rng.integers(len(h), size=min(batch, len(h)))
        snr = rng.uniform(*TRAIN_SNR_DB, size=len(idx))
        ls = grid.ls_estimate(h[idx], snr, rng)
        pred = model(_to_planes(ls).to(DEVICE), torch.tensor(snr, dtype=torch.float32, device=DEVICE))
        loss = torch.mean((pred - _to_planes(h[idx]).to(DEVICE)) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return model


@torch.no_grad()
def refine(model: Refiner, h: np.ndarray, snr_db: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    model.to(DEVICE).eval()
    ls = grid.ls_estimate(h, snr_db, rng)
    out = [model(_to_planes(ls[i:i + 512]).to(DEVICE),
                 torch.tensor(snr_db[i:i + 512], dtype=torch.float32, device=DEVICE))
           for i in range(0, len(h), 512)]
    return _from_planes(torch.cat(out))


class LMMSE:
    """Linear MMSE from pilots to the full grid, with covariances estimated from samples.

    The classical answer to "I have N channels from this site": no learning
    beyond second-order statistics. Diagonal loading keeps small-N covariances usable.
    """

    def __init__(self, h_fit: np.ndarray, loading: float = 1e-3):
        x = h_fit.reshape(len(h_fit), -1)
        p = self._pilots(h_fit)
        self.r_hp = x.T @ p.conj() / len(x)
        self.r_pp = p.T @ p.conj() / len(p)
        self.loading = loading * np.real(np.trace(self.r_pp)) / len(self.r_pp)

    @staticmethod
    def _pilots(h: np.ndarray) -> np.ndarray:
        return h[:, list(grid.PILOT_SYMBOLS)][:, :, grid.PILOT_SUBCARRIERS].reshape(len(h), -1)

    def estimate(self, h: np.ndarray, snr_db: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        p = self._pilots(h)
        noise_var = 10.0 ** (-np.asarray(snr_db) / 10.0)
        p = p + (rng.standard_normal(p.shape) + 1j * rng.standard_normal(p.shape)) * np.sqrt(noise_var / 2)[:, None]
        out = np.empty((len(h), h.shape[1] * h.shape[2]), dtype=np.complex64)
        eye = np.eye(len(self.r_pp))
        for nv in np.unique(noise_var):
            rows = noise_var == nv
            w = self.r_hp @ np.linalg.inv(self.r_pp + (nv + self.loading) * eye)
            out[rows] = p[rows] @ w.T
        return out.reshape(h.shape)
