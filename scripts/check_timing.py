"""Self-check: measured channels must not depend on the recording's bulk timing offset.

usage: check_timing.py   (exits non-zero on failure)

The factory recordings (cf0x) carry a timing offset that drifts across the whole
1024-tap delay range between datapoints; the street and lab recordings sit within
a few taps of zero. `dichasus.measured` must give the same grid channels whatever
the offset, and the same channels as the unshifted chain at zero offset.
Synthetic multipath channels are shifted by random whole-tap delays (a timing
offset common to all antennas), including channels whose first path is weaker
than one up to 19 taps later. Known limit: a path within 10 dB of the strongest
that arrives more than 20 taps (400 ns) before it is cut (none in the data:
scratch/early_arrival_check.py).
"""

import numpy as np

from sixg_gen import dichasus, grid, wideband


def channels(rng, n=64, antennas=16, w=1024, weak_first=False) -> np.ndarray:
    """[n, A, 1, w] ascending frequency: a first path, a second one 2-19 taps later, a tail. The first path
    is the strongest, or with `weak_first` 6 dB weaker than the second (still within the chain's search)."""
    taps = np.zeros((n, antennas, 1, w), complex)
    for i in range(n):
        first = rng.integers(-3, 3)
        powers = (0.25, 1.0) if weak_first else (1.0, 0.25)
        for k, power in [(first, powers[0]), (first + rng.integers(2, 20), powers[1])] + [(first + d, 0.01) for d in range(8, 60, 4)]:
            taps[i, :, 0, k % w] += np.sqrt(power) * np.exp(2j * np.pi * rng.random(antennas))
    return np.fft.fftshift(np.fft.fft(taps, axis=-1), axes=-1)


def delayed(h: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Delay each datapoint by a whole number of taps (circular), the same for every antenna."""
    taps = np.fft.ifft(np.fft.ifftshift(h, axes=-1), axis=-1)
    taps = np.stack([np.roll(t, o, axis=-1) for t, o in zip(taps, offsets)])
    return np.fft.fftshift(np.fft.fft(taps, axis=-1), axes=-1)


def main() -> None:
    grid.use_dichasus()
    rng = np.random.default_rng(0)
    for weak_first in (False, True):
        h = channels(rng, weak_first=weak_first)
        n, a, _, w = h.shape
        cal = np.ones((a, 1, w), np.complex64)
        raw = lambda csi: dichasus.Raw(csi, np.zeros((n, 3)), np.ones(n, bool))  # noqa: E731
        reference = grid.normalize_power(wideband.to_grid(h))  # the chain without the shift, at zero offset
        for name, offsets in [("zero", np.zeros(n, int)), ("small (street-like)", rng.integers(-8, 9, n)),
                              ("whole range (factory-like)", rng.integers(-512, 512, n))]:
            out = dichasus.measured(raw(delayed(h, offsets)), cal)
            err = np.max(np.abs(out - reference))
            label = f"{'weaker' if weak_first else 'strongest'} first path, {name} offsets"
            print(f"{label}: max abs difference {err:.2e}")
            assert err < 1e-4, f"measured channels depend on the timing offset ({label})"
    print("OK")


if __name__ == "__main__":
    main()
