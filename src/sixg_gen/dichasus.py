"""Measured channels from DICHASUS datasets, on the experiment grid.

`measured(load(array, files), cal)` returns one array's channels from converted
files (`scripts/dichasus_convert.py`), calibrated, reduced to the grid by the same
`wideband.to_grid` chain the simulated channels use, at unit power, keeping
datapoints with usable SNR and a fresh position fix. Datasets: "dxxx" (outdoor
street, converted files in data/dichasus/raw), and others under
data/dichasus/<name>/ (e.g. "0d5x" lab room, "cf0x" factory), each with its
spec.json.

The receivers carry constant per-antenna phase and sampling-time offsets (the
dataset authors' calibration tutorial; offset files are not shipped for this
dataset). `fit_calibration` estimates them, per antenna, as a phase plus a
delay relative to antenna 0, from the line-of-sight phase the tachymeter
positions predict, and reports how well that model fits (coherence, 1 = exact).

Recordings can also carry a bulk timing offset common to all antennas of a
datapoint. In the street and lab sets it stays within a few taps of zero; in the
factory set (cf0x) it drifts over the whole 1024-tap delay range between
datapoints, beyond the first-arrival search of `wideband.to_grid`. `measured`
therefore first shifts each datapoint so its strongest tap sits at delay zero
(`scripts/check_timing.py` checks the result does not depend on the offset).

Antennas are reordered to Sionna's planar-array order (column-major, top row
first), with the 8-element axis horizontal, as the data fits best.
"""

import json
from dataclasses import dataclass

import numpy as np

from . import grid, wideband
from .channels import DATA_DIR

RAW = DATA_DIR / "dichasus" / "raw"
SPACING_M = 0.118  # used when a spec gives no element spacing
MIN_SNR_DB = 10.0
MAX_INTERP_AGE_S = 0.1
CHUNK = 2000  # datapoints per block when removing the timing offset


def _dir(dataset: str):
    return RAW if dataset == "dxxx" else DATA_DIR / "dichasus" / dataset


def _file(dataset: str, name: str):
    return _dir(dataset) / (f"dichasus-{name}-complete.npz" if dataset == "dxxx" else f"dichasus-{name}.npz")


def _spec(dataset: str = "dxxx") -> dict:
    return json.load(open(_dir(dataset) / "spec.json"))


def array_names(dataset: str = "dxxx") -> list[str]:
    return [a["name"] for a in _spec(dataset)["antennas"]]


@dataclass
class Array:
    order: list[int]      # dataset antenna index of each Sionna-order element
    elements: np.ndarray  # [16, 3] positions in the dataset's frame

    @classmethod
    def named(cls, name: str, dataset: str = "dxxx") -> "Array":
        a = next(x for x in _spec(dataset)["antennas"] if x["name"] == name or x["name"].endswith(" " + name))
        spacing = a.get("spacingX", SPACING_M)
        asg, loc, bore = np.array(a["assignments"]), np.array(a["location"]), np.array(a["direction"])
        horiz = np.cross([0, 0, 1], bore)
        horiz /= np.linalg.norm(horiz)
        up = np.cross(bore, horiz)
        up /= np.linalg.norm(up)
        cols, rows = asg.shape[1], asg.shape[0]
        order = [int(asg[r][c]) for c in range(cols) for r in range(rows)]
        elements = np.array([loc + (c - (cols - 1) / 2) * spacing * horiz + ((rows - 1) / 2 - r) * spacing * up
                             for c in range(cols) for r in range(rows)])
        return cls(order, elements)


@dataclass
class Raw:
    csi: np.ndarray   # [n, 16, 1, W], ascending frequency
    pos: np.ndarray   # [n, 3]
    usable: np.ndarray  # [n] bool


def load(array: Array, files: list[str], dataset: str = "dxxx", tx_height_m: float = 0.0) -> Raw:
    """`tx_height_m`: transmitter height for datasets whose positions are 2D (lidar)."""
    parts = [np.load(_file(dataset, f)) for f in files]
    csi = np.concatenate([z["csi"][:, array.order] for z in parts])
    snr = np.concatenate([z["snr"][:, array.order] for z in parts])
    age = np.concatenate([z["interp_age"] for z in parts])
    usable = (snr.mean(axis=1) >= MIN_SNR_DB) & (age < MAX_INTERP_AGE_S)
    pos = np.concatenate([z["pos"] for z in parts])
    if pos.shape[1] == 2:
        pos = np.c_[pos, np.full(len(pos), tx_height_m)]
    return Raw(np.fft.fftshift(csi, axes=-1)[:, :, None, :], pos, usable)


def part(raw: Raw, start: float, stop: float) -> Raw:
    """Datapoints from fraction `start` to `stop` of the recording, in recorded order."""
    n = len(raw.usable)
    sl = slice(int(start * n), int(stop * n))
    return Raw(raw.csi[sl], raw.pos[sl], raw.usable[sl])


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


def _to_peak(h: np.ndarray) -> np.ndarray:
    """Shift each datapoint (circularly, whole taps) so its strongest tap is at delay 0."""
    taps = np.fft.ifft(np.fft.ifftshift(h, axes=-1), axis=-1)
    peak = np.argmax(np.sum(np.abs(taps) ** 2, axis=(1, 2)), axis=1)
    rows = (np.arange(h.shape[-1])[None] + peak[:, None]) % h.shape[-1]
    taps = np.take_along_axis(taps, rows[:, None, None, :], axis=-1)
    return np.fft.fftshift(np.fft.fft(taps, axis=-1), axes=-1)


def measured(raw: Raw, cal: np.ndarray) -> np.ndarray:
    """Usable datapoints, calibrated, on the grid, at unit power: [n, 16, 1, NUM_SUBCARRIERS]."""
    h = raw.csi[raw.usable]
    return np.concatenate([grid.normalize_power(wideband.to_grid(_to_peak(h[i:i + CHUNK] * cal[None])))
                           for i in range(0, max(len(h), 1), CHUNK)])
