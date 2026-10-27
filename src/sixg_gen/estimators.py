"""A learned correction on top of LMMSE: does anything beyond second-order statistics help?

`Refiner` is a CNN that takes the LMMSE estimate (and SNR) and predicts a
residual. Its last layer starts at zero, so an untrained Refiner *is* its LMMSE
front end; training can only move it away from that if the data supports it.
The front end is passed in, not owned: pre-training uses the source's LMMSE,
fine-tuning uses the adapted one (see `linear`).

Every estimate draws pilot noise from the caller's rng, so methods given equal
seeds see identical pilots.
"""

import numpy as np
import torch
from torch import nn

from . import grid, linear

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TRAIN_SNR_DB = (-5.0, 25.0)


class Refiner(nn.Module):
    """Input: [batch, 2 x antennas (re, im), symbols, subcarriers] plus an SNR plane."""

    def __init__(self, width: int = 32, depth: int = 6):
        super().__init__()
        planes = 2 * grid.NUM_ANTENNAS
        layers = [nn.Conv2d(planes + 1, width, 3, padding=1), nn.GELU()]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(width, width, 3, padding=1), nn.GELU()]
        last = nn.Conv2d(width, planes, 3, padding=1)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.net = nn.Sequential(*layers, last)

    def forward(self, lin: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        snr_plane = (snr_db / 20.0)[:, None, None, None].expand(-1, 1, *lin.shape[2:])
        return lin + self.net(torch.cat([lin, snr_plane], dim=1))


def _to_planes(h: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.concatenate([h.real, h.imag], axis=1).astype(np.float32))


def _from_planes(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().numpy()
    a = x.shape[1] // 2
    return (x[:, :a] + 1j * x[:, a:]).astype(np.complex64)


def train(model: Refiner, front: linear.LMMSE, h: np.ndarray, steps: int, lr: float, seed: int,
          batch: int = 128, time_weight: np.ndarray | None = None) -> Refiner:
    """Train on channels `h` with fresh pilot noise and SNR drawn every step.

    `time_weight` (one value per symbol) weights the squared error along the
    time axis; by default every grid point counts equally.
    """
    rng = np.random.default_rng(seed)
    w = 1.0 if time_weight is None else torch.tensor(time_weight, dtype=torch.float32, device=DEVICE)[:, None]
    model.to(DEVICE).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        idx = np.sort(rng.choice(len(h), size=min(batch, len(h)), replace=False))
        target = np.asarray(h[idx])
        snr = rng.uniform(*TRAIN_SNR_DB, size=len(idx))
        lin = front.estimate(target, snr, rng)
        pred = model(_to_planes(lin).to(DEVICE), torch.tensor(snr, dtype=torch.float32, device=DEVICE))
        loss = torch.mean(w * (pred - _to_planes(target).to(DEVICE)) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return model


@torch.no_grad()
def refine(model: Refiner, front: linear.LMMSE, h: np.ndarray, snr_db: np.ndarray,
           rng: np.random.Generator) -> np.ndarray:
    model.to(DEVICE).eval()
    lin = front.estimate(h, snr_db, rng)
    out = [model(_to_planes(lin[i:i + 256]).to(DEVICE),
                 torch.tensor(snr_db[i:i + 256], dtype=torch.float32, device=DEVICE))
           for i in range(0, len(h), 256)]
    return _from_planes(torch.cat(out))
