"""Convert DICHASUS .tfrecords to .npz (run with a Python that has TensorFlow).

usage: <tf-python> -I scripts/dichasus_convert.py <file.tfrecords> [...]
Writes <file>.npz next to each input: csi [n, 48, 1024] complex64, pos [n, 3],
snr [n, A], time [n], interp_age [n] (positions from the tachymeter or the lidar,
whichever the dataset has). Antenna order is DICHASUS's index order;
spec.json maps each array's 2 x 8 elements to these indices.
"""
import sys

import numpy as np
import tensorflow as tf

def features(path: str) -> tuple[dict, str]:
    """Feature spec for a file, and the position source it uses ("tachy" or "lidar")."""
    first = next(iter(tf.data.TFRecordDataset(path)))
    keys = tf.train.Example.FromString(first.numpy()).features.feature.keys()
    source = "tachy" if "pos-tachy" in keys else "lidar"
    return ({**{k: tf.io.FixedLenFeature([], tf.string) for k in ("csi", f"pos-{source}", "snr")},
             "time": tf.io.FixedLenFeature([], tf.float32),
             f"gt-interp-age-{source}": tf.io.FixedLenFeature([], tf.float32)}, source)


def tensor(raw, *dtypes):
    for dtype in dtypes:
        try:
            return tf.io.parse_tensor(raw, dtype).numpy()
        except tf.errors.InvalidArgumentError:
            continue
    raise ValueError("unexpected tensor type")


for path in sys.argv[1:]:
    spec, source = features(path)
    csi, pos, snr, time, age = [], [], [], [], []
    for rec in tf.data.TFRecordDataset(path):
        ex = tf.io.parse_single_example(rec, spec)
        c = tf.io.parse_tensor(ex["csi"], tf.float32).numpy()
        csi.append((c[..., 0] + 1j * c[..., 1]).astype(np.complex64))
        pos.append(tensor(ex[f"pos-{source}"], tf.float64, tf.float32).astype(np.float64))
        snr.append(tf.io.parse_tensor(ex["snr"], tf.float32).numpy())
        time.append(ex["time"].numpy())
        age.append(ex[f"gt-interp-age-{source}"].numpy())
    out = path.rsplit(".", 1)[0] + ".npz"
    np.savez(out, csi=np.stack(csi), pos=np.stack(pos), snr=np.stack(snr), time=np.array(time),
             interp_age=np.array(age))
    print(f"{out}: {len(csi)} datapoints", flush=True)
