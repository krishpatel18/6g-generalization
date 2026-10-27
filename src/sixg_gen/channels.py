"""Where channels come from: two sources behind one interface.

`generate(source, n, seed)` returns `n` power-normalized channel grids shaped
[n, NUM_ANTENNAS, NUM_SYMBOLS, NUM_SUBCARRIERS] (see `grid`). A source is either

- ``"cdl"``: 3GPP TR 38.901 CDL-A..E, drawn uniformly, with random delay spread,
  user speed and base-station heading (Sionna PHY), or
- ``"rt:<scene>"`` / ``"rtd:<scene>"``: ray-traced paths (specular only / also diffuse
  scattering and diffraction) in a city scene, with a rooftop base
  station and moving street-level users. ``<scene>`` is one of Sionna RT's
  built-in scenes, or a directory ``data/scenes/<scene>/`` holding a
  ``scene.xml`` (e.g. generated from OpenStreetMap with geo2sigmap).

Both use the same carrier, bandwidth, single isotropic antennas, speed range
and slot timing, so a model trained on one can be evaluated on the other.
Results are cached under ``data/`` keyed by grid settings, source, n and seed.
"""

from pathlib import Path

import numpy as np

from . import grid, wideband

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SCENES_DIR = DATA_DIR / "scenes"
MAX_SPEED_MPS = 30.0
FACADE_HEIGHT_M = 10.0
FACADE_USER_RANGE_M = (10.0, 65.0)
FACADE_SECTOR_RAD = np.radians(75)
USER_HEIGHT_M = 1.5
RICH_SCATTERING = 0.3
RT_SCENES = ("munich", "etoile", "florence", "san_francisco")


def generate(source: str, n: int, seed: int, mmap: bool = False) -> np.ndarray:
    """`mmap=True` leaves a cached set on disk and pages it in as it is sliced."""
    path = DATA_DIR / grid.config_tag() / f"{source.replace(':', '_')}_n{n}_s{seed}.npy"
    if path.exists():
        return np.load(path, mmap_mode="r" if mmap else None)
    if source == "cdl":
        h, speed = _cdl(n, seed), np.full(n, np.nan)
    elif source.startswith("rt:"):
        h, speed = _raytraced(source[3:], n, seed)
    elif source.startswith("rtd:"):
        h, speed = _raytraced(source[4:], n, seed, rich=True)
    else:
        raise ValueError(f"unknown channel source {source!r}")
    h = grid.normalize_power(h)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, h)
    np.save(_speed_path(path), speed.astype(np.float32))
    return h


def speeds(source: str, n: int, seed: int) -> np.ndarray:
    """User speed (m/s) of each channel in a cached set (NaN for CDL; sets cached before
    speeds were recorded have none)."""
    path = DATA_DIR / grid.config_tag() / f"{source.replace(':', '_')}_n{n}_s{seed}.npy"
    return np.load(_speed_path(path))


def _speed_path(path: Path) -> Path:
    return path.with_name(path.stem + "_speed.npy")


def _num_steps() -> int:
    return max(grid.TRACK_TIMES) + 1 if grid.TRACK_TIMES else grid.NUM_SYMBOLS


def _sample_times(h: np.ndarray) -> np.ndarray:
    """Keep the track's time points (track mode); one slot is kept whole. In wideband
    mode, reduce the band to the grid the same way measured channels are reduced."""
    h = h[:, :, list(grid.TRACK_TIMES)] if grid.TRACK_TIMES else h
    return wideband.to_grid(h) if grid.WIDEBAND_SUBCARRIERS else h


def _num_freqs() -> int:
    return len(grid.subcarrier_offsets_hz())


def _cdl_pattern() -> str:
    return {"iso": "omni", "tr38901": "38.901"}[grid.ANTENNA_PATTERN]


def _cdl(n: int, seed: int) -> np.ndarray:
    import torch
    from sionna.phy.channel import cir_to_ofdm_channel
    from sionna.phy.channel.tr38901 import CDL, AntennaArray

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ant = dict(polarization="single", polarization_type="V", carrier_frequency=grid.CARRIER_HZ)
    freqs = torch.tensor(grid.subcarrier_offsets_hz(), dtype=torch.float32)
    out, batch = [], 64
    for start in range(0, n, batch):
        b = min(batch, n - start)
        model = "ABCDE"[rng.integers(5)]
        delay_spread = float(rng.uniform(30e-9, 300e-9))
        # CDL clusters have fixed angles; a random base-station heading per sample
        # spreads users around the array, as the ray-traced placement does.
        # On a facade, users are only in front: headings within the placement sector.
        heading = torch.zeros(b, 3)
        if grid.PLACEMENT == "facade":
            heading[:, 0] = torch.from_numpy(rng.uniform(-FACADE_SECTOR_RAD, FACADE_SECTOR_RAD, b))
        else:
            heading[:, 0] = torch.from_numpy(rng.uniform(0, 2 * np.pi, b))
        heading[:, 1] = grid.DOWNTILT_RAD
        cdl = CDL(model, delay_spread, grid.CARRIER_HZ,
                  ut_array=AntennaArray(num_rows=1, num_cols=1, antenna_pattern="omni", **ant),
                  bs_array=AntennaArray(num_rows=grid.ARRAY_ROWS, num_cols=grid.NUM_ANTENNAS // grid.ARRAY_ROWS,
                                        antenna_pattern=_cdl_pattern(), **ant), direction="uplink",
                  bs_orientation=heading, min_speed=0.0, max_speed=MAX_SPEED_MPS, device="cpu")
        a, tau = cdl(b, _num_steps(), 1.0 / grid.SYMBOL_PERIOD_S)
        h = cir_to_ofdm_channel(freqs, a, tau)  # [b, rx, rx_ant (BS), tx, tx_ant, sym, sc]
        h = h.reshape(b, grid.NUM_ANTENNAS, _num_steps(), _num_freqs()).numpy()
        out.append(_sample_times(h))
    return np.concatenate(out)


def _raytraced(scene_name: str, n: int, seed: int, rich: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """`rich`: also diffuse scattering (every material's scattering coefficient set to
    RICH_SCATTERING, Lambertian) and edge diffraction; otherwise specular paths only."""
    import mitsuba as mi

    if mi.variant() is None:
        mi.set_variant("metal_ad_mono_polarized")
    import sionna.rt as rt

    rng = np.random.default_rng(seed)
    external = SCENES_DIR / scene_name / "scene.xml"
    scene = rt.load_scene(str(external) if external.exists() else getattr(rt.scene, scene_name))
    scene.frequency = grid.CARRIER_HZ
    if rich:
        for material in scene.radio_materials.values():
            material.scattering_coefficient = RICH_SCATTERING
    half_wave = dict(vertical_spacing=0.5, horizontal_spacing=0.5, polarization="V")
    scene.tx_array = rt.PlanarArray(num_rows=grid.ARRAY_ROWS, num_cols=grid.NUM_ANTENNAS // grid.ARRAY_ROWS,
                                    pattern=grid.ANTENNA_PATTERN, **half_wave)  # base station
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", **half_wave)
    bb = scene.mi_scene.bbox()
    lo, hi = np.array(bb.min)[:2], np.array(bb.max)[:2]
    centre, half = (lo + hi) / 2, 0.3 * (hi - lo)
    solver = rt.PathSolver()
    # Diffuse scattering multiplies the path count; smaller batches keep cfr under Dr.Jit's 2^32 limit.
    rx_per_tx, out, speed_out, have = (32 if rich else 256), [], [], 0
    while have < n:
        for name in list(scene.transmitters) + list(scene.receivers):
            scene.remove(name)
        if grid.PLACEMENT == "facade":
            tx, users = _facade(scene, rng, centre, half, rx_per_tx)
        else:
            tx, users = _rooftop(rng, centre, half, rx_per_tx)
        scene.add(rt.Transmitter("tx", **tx))
        batch_speed = []
        for i, (pos, vel) in enumerate(users):
            batch_speed.append(float(np.linalg.norm(vel)))
            scene.add(rt.Receiver(f"rx{i}", position=[*map(float, pos)], velocity=[*map(float, vel)]))
        paths = solver(scene, max_depth=3, samples_per_src=10**5, synthetic_array=True,
                       diffuse_reflection=rich, diffraction=rich)
        h = paths.cfr(frequencies=mi.Float(grid.subcarrier_offsets_hz().astype(np.float32)),
                      sampling_frequency=1.0 / grid.SYMBOL_PERIOD_S,
                      num_time_steps=_num_steps(), normalize_delays=True, out_type="numpy")
        # [rx, rx_ant=1, tx=1, tx_ant (BS), time, freq]
        h = _sample_times(h.reshape(rx_per_tx, grid.NUM_ANTENNAS, _num_steps(), _num_freqs()))
        has_paths = np.mean(np.abs(h) ** 2, axis=(1, 2, 3)) > 0
        h = h[has_paths]
        out.append(h)
        speed_out.append(np.array(batch_speed)[has_paths])
        have += len(h)
        print(f"  {scene_name}: {have}/{n} channels", flush=True)
    return np.concatenate(out)[:n], np.concatenate(speed_out)[:n]


def _rooftop(rng, centre, half, count):
    """Base station 25-40 m up in the central area; users anywhere within 250 m, moving."""
    tx_xy = centre + rng.uniform(-1, 1, 2) * half
    tx = dict(position=[*map(float, tx_xy), float(rng.uniform(25, 40))])
    users = []
    for _ in range(count):
        r, phi = 250 * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
        heading, speed = rng.uniform(0, 2 * np.pi), rng.uniform(0, MAX_SPEED_MPS)
        users.append(((tx_xy[0] + r * np.cos(phi), tx_xy[1] + r * np.sin(phi), USER_HEIGHT_M),
                      (speed * np.cos(heading), speed * np.sin(heading), 0.0)))
    return tx, users


def _facade(scene, rng, centre, half, count):
    """Base station FACADE_HEIGHT_M above the local ground and above any roof there, random
    heading, downtilted; static users USER_HEIGHT_M above open ground in front of it
    (FACADE_USER_RANGE_M, within FACADE_SECTOR_RAD). Heights follow terrain."""
    while True:
        tx_xy = centre + rng.uniform(-1, 1, 2) * half
        top, ground = _heights(scene, tx_xy[None])
        if top[0] < ground[0] + FACADE_HEIGHT_M:
            break
    yaw = rng.uniform(0, 2 * np.pi)
    tx = dict(position=[*map(float, tx_xy), float(ground[0] + FACADE_HEIGHT_M)],
              orientation=[float(yaw), grid.DOWNTILT_RAD, 0.0])
    lo, hi = FACADE_USER_RANGE_M
    users = []
    while len(users) < count:
        r = np.sqrt(rng.uniform(lo**2, hi**2, 4 * count))
        az = yaw + rng.uniform(-FACADE_SECTOR_RAD, FACADE_SECTOR_RAD, 4 * count)
        xy = tx_xy + np.stack([r * np.cos(az), r * np.sin(az)], axis=1)
        top, ground = _heights(scene, xy)
        outdoors = np.isfinite(ground) & (top < ground + 0.5)
        users += [((x, y, g + USER_HEIGHT_M), (0.0, 0.0, 0.0)) for (x, y), g in zip(xy[outdoors], ground[outdoors])]
    return tx, users[:count]


_GROUND_SCENES: dict[int, object] = {}


def _heights(scene, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each (x, y): height of the first surface seen from above, and of the ground there."""
    import mitsuba as mi

    key = id(scene)
    if key not in _GROUND_SCENES:
        _GROUND_SCENES.clear()
        _GROUND_SCENES[key] = mi.load_dict({"type": "scene", "ground": _ground_object(scene).mi_mesh})
    return _first_hit(scene.mi_scene, xy), _first_hit(_GROUND_SCENES[key], xy)


def _ground_object(scene):
    """The scene's ground or terrain: named so, else the flattest of the widest objects."""
    objs = list(scene.objects.values())
    named = [o for o in objs if any(k in o.name.lower() for k in ("terrain", "ground", "plane"))]
    if named:
        return named[0]

    def extent(o):
        b = o.mi_mesh.bbox()
        lo, hi = np.array(b.min), np.array(b.max)
        return (hi - lo)[0] * (hi - lo)[1], (hi - lo)[2]

    widest = max(extent(o)[0] for o in objs)
    return min((o for o in objs if extent(o)[0] > 0.5 * widest), key=lambda o: extent(o)[1])


def _first_hit(mi_scene, xy: np.ndarray) -> np.ndarray:
    import mitsuba as mi

    n = len(xy)
    origin = mi.Point3f(mi.Float(xy[:, 0].astype(np.float32)), mi.Float(xy[:, 1].astype(np.float32)),
                        mi.Float(np.full(n, 1000.0, np.float32)))
    hit = mi_scene.ray_intersect(mi.Ray3f(origin, mi.Vector3f(0, 0, -1)))
    return np.where(np.array(hit.is_valid()), np.array(hit.p.z), -np.inf)
