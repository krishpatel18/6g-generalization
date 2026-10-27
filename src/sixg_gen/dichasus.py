"""Measured channels from the DICHASUS outdoor dataset, on the experiment grid.

`measured(load(array, files), cal)` returns one array's channels from converted
files (`scripts/dichasus_convert.py`), calibrated, reduced to the grid by the same
`wideband.to_grid` chain the simulated channels use, at unit power, keeping
datapoints with usable SNR and a fresh tachymeter position.

The receivers carry constant per-antenna phase and sampling-time offsets (the
dataset authors' calibration tutorial; offset files are not shipped for this
dataset). `fit_calibration` estimates them, per antenna, as a phase plus a
delay relative to antenna 0, from the line-of-sight phase the tachymeter
positions predict, and reports how well that model fits (coherence, 1 = exact).

Antennas are reordered to Sionna's planar-array order (column-major, top row
first), with the 8-element axis horizontal, as the data fits best.
"""

import json
from dataclasses import dataclass

import numpy as np

from . import grid, wideband
from .channels import DATA_DIR

RAW = DATA_DIR / "dichasus" / "raw"
SPACING_M = 0.118
MIN_SNR_DB = 10.0
MAX_INTERP_AGE_S = 0.1


def _spec() -> dict:
    return json.load(open(RAW / "spec.json"))


@dataclass
class Array:
    order: list[int]      # dataset antenna index of each Sionna-order element
    elements: np.ndarray  # [16, 3] positions in the dataset's frame

    @classmethod
    def named(cls, name: str) -> "Array":
        a = next(x for x in _spec()["antennas"] if x["name"].endswith(name))
        asg, loc, bore = np.array(a["assignments"]), np.array(a["location"]), np.array(a["direction"])
        horiz = np.cross([0, 0, 1], bore)
        horiz /= np.linalg.norm(horiz)
        up = np.cross(bore, horiz)
        up /= np.linalg.norm(up)
        cols, rows = asg.shape[1], asg.shape[0]
        order = [int(asg[r][c]) for c in range(cols) for r in range(rows)]
        elements = np.array([loc + (c - (cols - 1) / 2) * SPACING_M * horiz + ((rows - 1) / 2 - r) * SPACING_M * up
                             for c in range(cols) for r in range(rows)])
        return cls(order, elements)


@dataclass
class Raw:
    csi: np.ndarray   # [n, 16, 1, W], ascending frequency
    pos: np.ndarray   # [n, 3]
    usable: np.ndarray  # [n] bool


def load(array: Array, files: list[str]) -> Raw:
    parts = [np.load(RAW / f"dichasus-{f}-complete.npz") for f in files]
    csi = np.concatenate([z["csi"][:, array.order] for z in parts])
    snr = np.concatenate([z["snr"][:, array.order] for z in parts])
    age = np.concatenate([z["interp_age"] for z in parts])
    usable = (snr.mean(axis=1) >= MIN_SNR_DB) & (age < MAX_INTERP_AGE_S)
    return Raw(np.fft.fftshift(csi, axes=-1)[:, :, None, :], np.concatenate([z["pos"] for z in parts]), usable)


def _freqs(w: int) -> np.ndarray:
    return (np.arange(w) - w // 2) * grid.SUBCARRIER_SPACING_HZ


def fit_calibration(array: Array, raw: Raw) -> tuple[np.ndarray, float]:
    """Correction [16, 1, W] to multiply channels by, and the line-of-sight coherence after it."""
    csi, pos = raw.csi[raw.usable][:, :, 0], raw.pos[raw.usable]
    w = csi.shape[-1]
    wavelength = 299792458 / grid.CARRIER_HZ
    d = np.linalg.norm(pos[:, None] - array.elements[None], axis=2)
    los = np.exp(-2j * np.pi * (d - d[:, :1]) / wavelength)  # [n, 16] expected phase vs antenna 0
    rel = csi * csi[:, :1].conj()
    rel /= np.abs(rel) + 1e-12
    phase = np.unwrap(np.angle(np.sum(rel * los.conj()[..., None], axis=0)), axis=1)  # [16, W]
    f = _freqs(w)
    fit = np.array([np.polyfit(f, p, 1) for p in phase])
    corr = np.exp(-1j * (fit[:, :1] * f[None] + fit[:, 1:]))
    gain = np.sqrt(np.mean(np.abs(csi) ** 2, axis=(0, 2)))
    corr = corr / (gain / gain.mean())[:, None]
    resid = np.sum(rel * np.exp(-1j * (fit[:, :1] * f[None] + fit[:, 1:]))[None], axis=2) * los.conj()
    unit = resid[:, 1:] / (np.abs(resid[:, 1:]) + 1e-12)  # antenna 0 is the reference: always 1
    coherence = float(np.mean(np.abs(np.mean(unit, axis=1))))
    return corr[:, None, :].astype(np.complex64), coherence


def measured(raw: Raw, cal: np.ndarray) -> np.ndarray:
    """Usable datapoints, calibrated, on the grid, at unit power: [n, 16, 1, NUM_SUBCARRIERS]."""
    return grid.normalize_power(wideband.to_grid(raw.csi[raw.usable] * cal[None]))
