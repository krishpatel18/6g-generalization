"""The DICHASUS dc01 institute hallway: measured channels and a ray-traced twin.

Two 4 x 8 arrays (DICHASUS `dcxx`, 3.438 GHz, 1024 subcarriers over 50 MHz) at
opposite ends of a hallway observe a robot-carried dipole. The hallway's
simplified 3D model (walls, floor, ceiling; one material each) is the one
published with Hoydis et al., "Learning Radio Environments by Differentiable
Ray Tracing" (arXiv 2311.18558); it is read from that repository's clone and
not copied here (NVIDIA source-code license).

Everything is expressed in the model's frame: origin at the surveyed point "AU",
rotated about z so the point "NWU" lies on +y (as in that paper). `load`
returns measured channels per array in Sionna planar-array order; `simulate`
ray-traces the same positions with given material parameters. Simulated
channels are [n, 2 arrays, 32 antennas, 1024 subcarriers] at the arrays,
ascending frequency, the arrays treated as transmitters (reciprocity).
"""

import csv
import json
from pathlib import Path

import numpy as np

from . import dichasus
from .channels import DATA_DIR

DIR = DATA_DIR / "dichasus" / "dcxx"
SCENE = Path.home() / "projects/ext/diff-rt-calibration/scenes/inue_simple/inue_simple.xml"
CARRIER_HZ = 3.438e9
BANDWIDTH_HZ = 50e6
SUBCARRIERS = 1024
ROWS, COLS = 4, 8
OBJECTS = ("floor", "walls", "ceiling")
Y_RANGE_M = (-15.0, 25.0)  # positions kept, as in the paper (too close to an array otherwise)


def _frame() -> tuple[np.ndarray, np.ndarray]:
    """Offset and rotation from the dataset frame to the model frame."""
    poi = {}
    with open(DIR / "coordinates.csv") as f:
        for row in csv.DictReader(f):
            if row["South"] != "noLoS":
                poi[row["Name"]] = np.array([float(row["West"]), float(row["South"]), float(row["Height"])])
    centre = poi["AU"]
    nwu = poi["NWU"] - centre
    a = np.pi / 2 - np.arctan2(nwu[1], nwu[0])
    rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return centre, rot


def to_model(p: np.ndarray) -> np.ndarray:
    centre, rot = _frame()
    return (np.asarray(p, float) - centre) @ rot.T


def arrays() -> list[dict]:
    """Per array: Sionna-order dataset antenna indices, centre and boresight in the model frame."""
    _, rot = _frame()
    out = []
    for name in dichasus.array_names("dcxx"):
        spec = next(a for a in dichasus._spec("dcxx")["antennas"] if a["name"] == name)
        arr = dichasus.Array.named(name, "dcxx")
        out.append(dict(name=name, order=arr.order, centre=to_model(spec["location"]),
                        boresight=np.asarray(spec["direction"], float) @ rot.T))
    return out


def load(offsets: str = "reftx-offsets-dichasus-dc01.json") -> dict:
    """Measured dc01: csi [n, 64, 1024] (memory-mapped, dataset antenna order, FFT order),
    positions [n, 3] in the model frame (and the dataset frame), and the mask of positions in
    Y_RANGE_M with a fresh position fix."""
    meta = np.load(DIR / "dichasus-dc01.npz")
    csi = np.load(DIR / "dichasus-dc01.csi.npy", mmap_mode="r")
    pos = to_model(meta["pos"])
    keep = (pos[:, 1] > Y_RANGE_M[0]) & (pos[:, 1] < Y_RANGE_M[1]) & (meta["interp_age"] < dichasus.MAX_INTERP_AGE_S)
    with open(DIR / offsets) as f:
        off = json.load(f)
    return dict(csi=csi, pos=pos, raw_pos=meta["pos"], keep=keep, snr=meta["snr"], age=meta["interp_age"],
                sto=np.array(off["sto"]), cpo=np.array(off["cpo"]))


def usable(data: dict, array: dict, idx: np.ndarray) -> np.ndarray:
    """The study's usable rule (mean SNR over the array's antennas, fresh position fix) for `idx`."""
    snr = data["snr"][idx][:, array["order"]]
    return (snr.mean(axis=1) >= dichasus.MIN_SNR_DB) & (data["age"][idx] < dichasus.MAX_INTERP_AGE_S)


def raw(data: dict, array: dict, idx: np.ndarray) -> "dichasus.Raw":
    """Datapoints `idx` of one array (an entry of `arrays()`) as the study's `dichasus.Raw`
    (dataset frame, uncalibrated, the study's usable rule), for `dichasus.fit_calibration`
    and `dichasus.measured`."""
    idx = np.sort(idx)
    csi = data["csi"][np.ix_(idx, array["order"])]
    return dichasus.Raw(np.fft.fftshift(csi, axes=-1)[:, :, None, :], data["raw_pos"][idx], usable(data, array, idx))


def calibrated(data: dict, idx: np.ndarray) -> np.ndarray:
    """Datapoints `idx` as [len(idx), 2, 32, 1024], ascending frequency, with the published
    per-antenna sampling-time and phase offsets applied (DICHASUS calibration)."""
    h = np.asarray(data["csi"][np.sort(idx)])  # [m, 64, 1024], FFT order
    h = h[np.argsort(np.argsort(idx))]
    k = np.arange(SUBCARRIERS)
    h = h * np.exp(1j * (2 * np.pi * data["sto"][:, None] * k[None] / SUBCARRIERS + data["cpo"][:, None]))[None]
    h = np.fft.fftshift(h, axes=-1)
    return np.stack([h[:, a["order"]] for a in arrays()], axis=1).astype(np.complex64)


class Twin:
    """The hallway model in Sionna RT, with settable per-object materials."""

    def __init__(self, max_depth: int = 5, diffraction: bool = True, gradients: bool = False):
        import mitsuba as mi
        if mi.variant() is None:
            mi.set_variant("metal_ad_mono_polarized")
        import sionna.rt as rt
        self.rt, self.mi = rt, mi
        self.scene = rt.load_scene(str(SCENE))
        self.scene.frequency = CARRIER_HZ
        half = dict(vertical_spacing=0.5, horizontal_spacing=0.5, polarization="V")
        self.scene.tx_array = rt.PlanarArray(num_rows=ROWS, num_cols=COLS, pattern="tr38901", **half)
        self.scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="dipole", **half)
        for i, a in enumerate(arrays()):
            b = a["boresight"]
            yaw, pitch = np.arctan2(b[1], b[0]), -np.arcsin(b[2] / np.linalg.norm(b))
            self.scene.add(rt.Transmitter(f"tx{i}", position=[*map(float, a["centre"])],
                                          orientation=[float(yaw), float(pitch), 0.0]))
        self.materials = {}
        for name in OBJECTS:
            obj = self.scene.get(name)
            itu = obj.radio_material
            mat = rt.RadioMaterial(f"{name}-cal", relative_permittivity=float(itu.relative_permittivity[0]),
                                   conductivity=float(itu.conductivity[0]))
            obj.radio_material = mat
            self.materials[name] = mat
        self.itu = {n: (float(m.relative_permittivity[0]), float(m.conductivity[0])) for n, m in self.materials.items()}
        self.solver = rt.PathSolver()
        if gradients:
            self.solver.loop_mode = "evaluated"
        self.kw = dict(max_depth=max_depth, los=True, specular_reflection=True, diffuse_reflection=False,
                       refraction=False, diffraction=diffraction, synthetic_array=True)
        self.freqs = (np.arange(SUBCARRIERS) - SUBCARRIERS // 2) * BANDWIDTH_HZ / SUBCARRIERS

    def set_materials(self, params: dict) -> None:
        """params: object -> (relative permittivity, conductivity S/m); floats or Dr.Jit Floats."""
        for name, (eps, sigma) in params.items():
            self.materials[name].relative_permittivity = eps
            self.materials[name].conductivity = sigma

    def paths(self, pos: np.ndarray):
        for name in list(self.scene.receivers):
            self.scene.remove(name)
        for i, p in enumerate(pos):
            self.scene.add(self.rt.Receiver(f"rx{i}", position=[*map(float, p)]))
        return self.solver(self.scene, **self.kw)

    def cfr(self, paths):
        """Dr.Jit (real, imag) tensors [rx, 1, 2, 32, 1, 1024] (differentiable)."""
        return paths.cfr(frequencies=self.mi.Float(self.freqs.astype(np.float32)), normalize_delays=False,
                         out_type="drjit")

    def simulate(self, pos: np.ndarray, batch: int = 32, reduce=None) -> np.ndarray:
        """[n, 2, 32, 1024] complex64 at the current materials, or `reduce` applied to each
        batch of that (so large sets never hold the full 1024-subcarrier responses)."""
        out = []
        for i in range(0, len(pos), batch):
            h = self.cfr(self.paths(pos[i:i + batch]))
            h = np.array(h[0]) + 1j * np.array(h[1]) if isinstance(h, tuple) else np.array(h)
            h = h.reshape(len(pos[i:i + batch]), 2, ROWS * COLS, SUBCARRIERS).astype(np.complex64)
            out.append(reduce(h) if reduce else h)
        return np.concatenate(out)
