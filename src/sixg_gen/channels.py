"""Where channels come from: two sources behind one interface.

`generate(source, n, seed)` returns `n` power-normalized channel grids shaped
[n, NUM_ANTENNAS, NUM_SYMBOLS, NUM_SUBCARRIERS] (see `grid`). A source is either

- ``"cdl"``: 3GPP TR 38.901 CDL-A..E, drawn uniformly, with random delay spread
  and user speed (Sionna PHY), or
- ``"rt:<scene>"``: ray-traced paths in one of Sionna RT's built-in city scenes,
  with a rooftop base station and moving street-level users.

Both use the same carrier, bandwidth, single isotropic antennas, speed range
and slot timing, so a model trained on one can be evaluated on the other.
Results are cached under ``data/`` keyed by grid settings, source, n and seed.
"""

from pathlib import Path

import numpy as np

from . import grid

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MAX_SPEED_MPS = 30.0
RT_SCENES = ("munich", "etoile", "florence", "san_francisco")


def generate(source: str, n: int, seed: int) -> np.ndarray:
    path = DATA_DIR / grid.config_tag() / f"{source.replace(':', '_')}_n{n}_s{seed}.npy"
    if path.exists():
        return np.load(path)
    if source == "cdl":
        h = _cdl(n, seed)
    elif source.startswith("rt:"):
        h = _raytraced(source[3:], n, seed)
    else:
        raise ValueError(f"unknown channel source {source!r}")
    h = grid.normalize_power(h)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, h)
    return h


def _cdl(n: int, seed: int) -> np.ndarray:
    import torch
    from sionna.phy.channel import cir_to_ofdm_channel
    from sionna.phy.channel.tr38901 import CDL, AntennaArray

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ant = dict(num_rows=1, polarization="single", polarization_type="V",
               antenna_pattern="omni", carrier_frequency=grid.CARRIER_HZ)
    freqs = torch.tensor(grid.subcarrier_offsets_hz(), dtype=torch.float32)
    out, batch = [], 64
    for start in range(0, n, batch):
        b = min(batch, n - start)
        model = "ABCDE"[rng.integers(5)]
        delay_spread = float(rng.uniform(30e-9, 300e-9))
        cdl = CDL(model, delay_spread, grid.CARRIER_HZ, ut_array=AntennaArray(num_cols=1, **ant),
                  bs_array=AntennaArray(num_cols=grid.NUM_ANTENNAS, **ant), direction="uplink",
                  min_speed=0.0, max_speed=MAX_SPEED_MPS, device="cpu")
        a, tau = cdl(b, grid.NUM_SYMBOLS, 1.0 / grid.SYMBOL_PERIOD_S)
        h = cir_to_ofdm_channel(freqs, a, tau)  # [b, rx, rx_ant (BS), tx, tx_ant, sym, sc]
        out.append(h.reshape(b, grid.NUM_ANTENNAS, grid.NUM_SYMBOLS, grid.NUM_SUBCARRIERS).numpy())
    return np.concatenate(out)


def _raytraced(scene_name: str, n: int, seed: int) -> np.ndarray:
    import mitsuba as mi

    if mi.variant() is None:
        mi.set_variant("metal_ad_mono_polarized")
    import sionna.rt as rt

    rng = np.random.default_rng(seed)
    scene = rt.load_scene(getattr(rt.scene, scene_name))
    scene.frequency = grid.CARRIER_HZ
    iso = dict(num_rows=1, vertical_spacing=0.5, horizontal_spacing=0.5,
               pattern="iso", polarization="V")
    scene.tx_array = rt.PlanarArray(num_cols=grid.NUM_ANTENNAS, **iso)  # base station
    scene.rx_array = rt.PlanarArray(num_cols=1, **iso)
    bb = scene.mi_scene.bbox()
    lo, hi = np.array(bb.min)[:2], np.array(bb.max)[:2]
    centre, half = (lo + hi) / 2, 0.3 * (hi - lo)
    solver = rt.PathSolver()
    rx_per_tx, out, have = 256, [], 0
    while have < n:
        for name in list(scene.transmitters) + list(scene.receivers):
            scene.remove(name)
        tx_xy = centre + rng.uniform(-1, 1, 2) * half
        scene.add(rt.Transmitter("tx", position=[*map(float, tx_xy), float(rng.uniform(25, 40))]))
        for i in range(rx_per_tx):
            r, phi = 250 * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
            heading, speed = rng.uniform(0, 2 * np.pi), rng.uniform(0, MAX_SPEED_MPS)
            scene.add(rt.Receiver(
                f"rx{i}",
                position=[float(tx_xy[0] + r * np.cos(phi)), float(tx_xy[1] + r * np.sin(phi)), 1.5],
                velocity=[float(speed * np.cos(heading)), float(speed * np.sin(heading)), 0.0]))
        paths = solver(scene, max_depth=3, samples_per_src=10**5, synthetic_array=True,
                       diffuse_reflection=False)
        h = paths.cfr(frequencies=mi.Float(grid.subcarrier_offsets_hz()),
                      sampling_frequency=1.0 / grid.SYMBOL_PERIOD_S,
                      num_time_steps=grid.NUM_SYMBOLS, normalize_delays=True, out_type="numpy")
        # [rx, rx_ant=1, tx=1, tx_ant (BS), time, freq]
        h = h.reshape(rx_per_tx, grid.NUM_ANTENNAS, grid.NUM_SYMBOLS, grid.NUM_SUBCARRIERS)
        h = h[np.mean(np.abs(h) ** 2, axis=(1, 2, 3)) > 0]
        out.append(h)
        have += len(h)
        print(f"  {scene_name}: {have}/{n} channels", flush=True)
    return np.concatenate(out)[:n]
