"""The OFDM resource grid every experiment shares, and what a receiver sees of it.

One slot of 14 OFDM symbols x 128 subcarriers at 30 kHz (5G NR numerology 1,
~3.84 MHz) on a 3.5 GHz carrier, SISO. Pilots follow DMRS type 1 with one
additional position: every other subcarrier on symbols 2 and 11.

Channels are complex arrays shaped [n, NUM_SYMBOLS, NUM_SUBCARRIERS], normalized
so each sample has unit mean power; SNR is therefore per-sample.
"""

import numpy as np

CARRIER_HZ = 3.5e9
SUBCARRIER_SPACING_HZ = 30e3
NUM_SUBCARRIERS = 128
NUM_SYMBOLS = 14
# Symbol period including the normal cyclic prefix (144/2048 of the useful symbol, ~7%).
SYMBOL_PERIOD_S = (1 + 144 / 2048) / SUBCARRIER_SPACING_HZ

PILOT_SYMBOLS = (2, 11)
PILOT_SUBCARRIERS = np.arange(0, NUM_SUBCARRIERS, 2)


def subcarrier_offsets_hz() -> np.ndarray:
    """Baseband frequency of each subcarrier, centred on the carrier."""
    return (np.arange(NUM_SUBCARRIERS) - NUM_SUBCARRIERS // 2) * SUBCARRIER_SPACING_HZ


def normalize_power(h: np.ndarray) -> np.ndarray:
    """Scale each sample to unit mean power."""
    p = np.mean(np.abs(h) ** 2, axis=(1, 2), keepdims=True)
    return (h / np.sqrt(p)).astype(np.complex64)


def ls_estimate(h: np.ndarray, snr_db: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Noisy least-squares channel at the pilots, linearly interpolated to the full grid.

    With unit-modulus pilots, LS at a pilot is the true channel plus noise of
    variance 1/SNR. Interpolation is linear across subcarriers, then linear
    across symbols (held flat outside the two pilot symbols).
    """
    n = h.shape[0]
    noise_var = 10.0 ** (-np.asarray(snr_db, dtype=np.float64) / 10.0)
    noise_var = np.broadcast_to(noise_var, (n,))[:, None, None]
    sym = list(PILOT_SYMBOLS)
    pilots = h[:, sym][:, :, PILOT_SUBCARRIERS]
    noise = (rng.standard_normal(pilots.shape) + 1j * rng.standard_normal(pilots.shape)) * np.sqrt(noise_var / 2)
    pilots = pilots + noise

    k = np.arange(NUM_SUBCARRIERS)
    freq_interp = np.empty((n, len(sym), NUM_SUBCARRIERS), dtype=np.complex128)
    for i in range(n):
        for j in range(len(sym)):
            freq_interp[i, j] = (
                np.interp(k, PILOT_SUBCARRIERS, pilots[i, j].real)
                + 1j * np.interp(k, PILOT_SUBCARRIERS, pilots[i, j].imag)
            )

    t = np.arange(NUM_SYMBOLS)
    w = np.clip((t - sym[0]) / (sym[1] - sym[0]), 0.0, 1.0)[None, :, None]
    return ((1 - w) * freq_interp[:, :1] + w * freq_interp[:, 1:]).astype(np.complex64)


def nmse_db(h_hat: np.ndarray, h: np.ndarray) -> float:
    """Normalized mean squared error over a set of samples, in dB."""
    err = np.sum(np.abs(h_hat - h) ** 2)
    return float(10 * np.log10(err / np.sum(np.abs(h) ** 2)))
