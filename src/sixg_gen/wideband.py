"""Wideband channels -> the experiment grid, by one chain for every source.

Measured channels (DICHASUS) carry an unknown timing offset and measurement noise;
simulated ones don't. To compare them fairly, every source goes through the same
steps: align the first arrival to delay 0, keep a delay window around it, then take
the grid's subcarriers around the carrier. Input frequencies ascend (fftshifted).
"""

import numpy as np

from . import grid

WINDOW_TAPS = (-10, 100)   # kept around the first arrival; 20 ns taps at 50 MHz
FIRST_ARRIVAL_DB = -10.0   # first tap within this of the strongest counts as the arrival
SEARCH_TAPS = 20           # arrival searched within +-SEARCH_TAPS of delay 0


def to_grid(h: np.ndarray) -> np.ndarray:
    """[n, A, T, W] over the whole band -> [n, A, T, NUM_SUBCARRIERS] around the carrier."""
    w = h.shape[-1]
    taps = np.fft.ifft(np.fft.ifftshift(h, axes=-1), axis=-1)
    pdp = np.sum(np.abs(taps) ** 2, axis=(1, 2))  # [n, W]
    near = np.arange(-SEARCH_TAPS, SEARCH_TAPS + 1)
    p = pdp[:, near % w]
    above = p >= p.max(axis=1, keepdims=True) * 10 ** (FIRST_ARRIVAL_DB / 10)
    first = near[np.argmax(above, axis=1)]  # earliest tap above the threshold
    k = np.fft.fftfreq(w) * w  # tap-index frequency of each natural-order bin
    spec = np.fft.fft(taps, axis=-1) * np.exp(2j * np.pi * k * first[:, None, None, None] / w)
    taps = np.fft.ifft(spec, axis=-1)
    keep = np.zeros(w, bool)
    keep[np.arange(*WINDOW_TAPS) % w] = True
    taps[..., ~keep] = 0
    full = np.fft.fftshift(np.fft.fft(taps, axis=-1), axes=-1)
    lo = w // 2 - grid.NUM_SUBCARRIERS // 2
    return full[..., lo:lo + grid.NUM_SUBCARRIERS].astype(np.complex64)
