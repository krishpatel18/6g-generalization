"""The OFDM resource grid every experiment shares, and what a receiver sees of it.

One slot of 14 OFDM symbols x 128 subcarriers at 30 kHz (5G NR numerology 1,
~3.84 MHz) on a 3.5 GHz carrier, between a single-antenna user and a base
station with NUM_ANTENNAS (half-wavelength ULA). Pilots follow DMRS type 1 with
one additional position: every other subcarrier on symbols 2 and 11.

Channels are complex arrays shaped [n, NUM_ANTENNAS, NUM_SYMBOLS, NUM_SUBCARRIERS],
normalized so each sample has unit mean power; SNR is therefore per-sample.
"""

import numpy as np

CARRIER_HZ = 3.5e9
SUBCARRIER_SPACING_HZ = 30e3
NUM_SUBCARRIERS = 128
NUM_SYMBOLS = 14
NUM_ANTENNAS = 1
# Symbol period including the normal cyclic prefix (144/2048 of the useful symbol, ~7%).
SYMBOL_PERIOD_S = (1 + 144 / 2048) / SUBCARRIER_SPACING_HZ

PILOT_SYMBOLS = (2, 11)
PILOT_SUBCARRIERS = np.arange(0, NUM_SUBCARRIERS, 2)

# Deployment the simulated channels mimic (see `use_dichasus`): array rows (the
# antennas form a ARRAY_ROWS x NUM_ANTENNAS/ARRAY_ROWS planar array), element
# pattern, downtilt, and where base stations and users are placed.
ARRAY_ROWS = 1
ANTENNA_PATTERN = "iso"       # or "tr38901" (directional, 3GPP 38.901)
DOWNTILT_RAD = 0.0
PLACEMENT = "rooftop"         # or "facade"
# Wideband mode: channels are generated on WIDEBAND_SUBCARRIERS and reduced to
# the grid by `wideband.to_grid`, the same chain measured channels go through.
WIDEBAND_SUBCARRIERS: int | None = None


# Track mode (channel prediction): instead of one slot, each sample is the
# channel at TRACK_TIMES (symbol indices from the start of slot 0) on every 4th
# subcarrier. The first 8 times are the pilot symbols of slots 0-3 (the observed
# history); the rest are mid-slot symbols HORIZON_SLOTS slots after slot 3.
# `use_tracks()` re-points the grid at these axes, so pilots (= history), LMMSE,
# blending and the Refiner work on tracks unchanged.
TRACK_TIMES: tuple[int, ...] | None = None
HISTORY_SLOTS = 4
HORIZON_SLOTS = (1, 2, 4, 10, 20)
TRACK_SUBCARRIER_STEP = 4


def use_tracks() -> None:
    global TRACK_TIMES, NUM_SYMBOLS, NUM_SUBCARRIERS, PILOT_SYMBOLS, PILOT_SUBCARRIERS
    if TRACK_TIMES:
        return
    history = [14 * k + s for k in range(HISTORY_SLOTS) for s in PILOT_SYMBOLS]
    targets = [14 * (HISTORY_SLOTS - 1 + h) + 7 for h in HORIZON_SLOTS]
    TRACK_TIMES = tuple(history + targets)
    NUM_SYMBOLS = len(TRACK_TIMES)
    NUM_SUBCARRIERS //= TRACK_SUBCARRIER_STEP
    PILOT_SYMBOLS = tuple(range(len(history)))
    PILOT_SUBCARRIERS = np.arange(NUM_SUBCARRIERS)


def use_dichasus() -> None:
    """The DICHASUS outdoor setup: 1.272 GHz, 2 x 8 array on a facade tilted ~20 deg down,
    50 MHz / 1024 subcarriers reduced to one snapshot of 128 subcarriers (6.25 MHz)."""
    global CARRIER_HZ, SUBCARRIER_SPACING_HZ, SYMBOL_PERIOD_S, NUM_ANTENNAS, NUM_SYMBOLS, NUM_SUBCARRIERS, \
        PILOT_SYMBOLS, PILOT_SUBCARRIERS, ARRAY_ROWS, ANTENNA_PATTERN, DOWNTILT_RAD, PLACEMENT, \
        WIDEBAND_SUBCARRIERS
    if WIDEBAND_SUBCARRIERS:
        return
    CARRIER_HZ = 1.272e9
    WIDEBAND_SUBCARRIERS = 1024
    SUBCARRIER_SPACING_HZ = 50e6 / WIDEBAND_SUBCARRIERS
    SYMBOL_PERIOD_S = (1 + 144 / 2048) / SUBCARRIER_SPACING_HZ  # unused: one snapshot, static users
    NUM_ANTENNAS, ARRAY_ROWS = 16, 2
    NUM_SYMBOLS, NUM_SUBCARRIERS = 1, 128
    PILOT_SYMBOLS, PILOT_SUBCARRIERS = (0,), np.arange(0, NUM_SUBCARRIERS, 2)
    ANTENNA_PATTERN, DOWNTILT_RAD, PLACEMENT = "tr38901", float(np.arcsin(0.349)), "facade"


def target_index(horizon_slots: int) -> int:
    """Time index in a track of the target `horizon_slots` after the last observed slot."""
    return len(PILOT_SYMBOLS) + HORIZON_SLOTS.index(horizon_slots)


def config_tag() -> str:
    """Names the settings that change a channel array, for cache keys."""
    return (f"a{NUM_ANTENNAS}_sc{NUM_SUBCARRIERS}" + ("_track" if TRACK_TIMES else "")
            + ("_dichasus" if WIDEBAND_SUBCARRIERS else ""))


def subcarrier_offsets_hz() -> np.ndarray:
    """Baseband frequency of each generated subcarrier, centred on the carrier, ascending
    (in wideband mode: the whole band, before `wideband.to_grid`)."""
    if WIDEBAND_SUBCARRIERS:
        return (np.arange(WIDEBAND_SUBCARRIERS) - WIDEBAND_SUBCARRIERS // 2) * SUBCARRIER_SPACING_HZ
    step = TRACK_SUBCARRIER_STEP if TRACK_TIMES else 1
    n = NUM_SUBCARRIERS * step
    return (np.arange(0, n, step) - n // 2) * SUBCARRIER_SPACING_HZ


def normalize_power(h: np.ndarray) -> np.ndarray:
    """Scale each sample to unit mean power."""
    p = np.mean(np.abs(h) ** 2, axis=tuple(range(1, h.ndim)), keepdims=True)
    return (h / np.sqrt(p)).astype(np.complex64)


def ls_estimate(h: np.ndarray, snr_db: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Noisy least-squares channel at the pilots, linearly interpolated to the full grid.

    With unit-modulus pilots, LS at a pilot is the true channel plus noise of
    variance 1/SNR. Interpolation is linear across subcarriers, then linear
    across symbols (held flat outside the two pilot symbols).
    """
    n = h.shape[0]
    noise_var = 10.0 ** (-np.asarray(snr_db, dtype=np.float64) / 10.0)
    noise_var = np.broadcast_to(noise_var, (n,)).reshape((n,) + (1,) * (h.ndim - 1))
    sym = list(PILOT_SYMBOLS)
    pilots = h[..., sym, :][..., PILOT_SUBCARRIERS]
    noise = (rng.standard_normal(pilots.shape) + 1j * rng.standard_normal(pilots.shape)) * np.sqrt(noise_var / 2)
    pilots = pilots + noise

    # Pilots sit on even subcarriers: odd ones take the mean of their neighbours,
    # and the last (odd) subcarrier holds the final pilot.
    freq_interp = np.empty(pilots.shape[:-1] + (NUM_SUBCARRIERS,), dtype=np.complex128)
    freq_interp[..., PILOT_SUBCARRIERS] = pilots
    freq_interp[..., 1:-1:2] = (pilots[..., :-1] + pilots[..., 1:]) / 2
    freq_interp[..., -1] = pilots[..., -1]

    if len(sym) == 1:  # one pilot symbol: held across the slot
        return np.broadcast_to(freq_interp, freq_interp.shape[:-2] + (NUM_SYMBOLS, NUM_SUBCARRIERS)).astype(np.complex64)
    t = np.arange(NUM_SYMBOLS)
    w = np.clip((t - sym[0]) / (sym[1] - sym[0]), 0.0, 1.0)[:, None]
    return ((1 - w) * freq_interp[..., :1, :] + w * freq_interp[..., 1:, :]).astype(np.complex64)


def nmse_db(h_hat: np.ndarray, h: np.ndarray) -> float:
    """Normalized mean squared error over a set of samples, in dB."""
    err = np.sum(np.abs(h_hat - h) ** 2)
    return float(10 * np.log10(err / np.sum(np.abs(h) ** 2)))
