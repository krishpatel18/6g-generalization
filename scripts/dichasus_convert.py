"""Convert DICHASUS .tfrecords to .npz (run with a Python that has TensorFlow).

usage: <tf-python> -I scripts/dichasus_convert.py <file.tfrecords> [...]
Writes <file>.npz next to each input: csi [n, 48, 1024] complex64, pos [n, 3],
snr [n, 48], time [n], interp_age [n]. Antenna order is DICHASUS's index order;
spec.json maps each array's 2 x 8 elements to these indices.
"""
import sys

import numpy as np
import tensorflow as tf

FEATURES = {**{k: tf.io.FixedLenFeature([], tf.string) for k in ("csi", "pos-tachy", "snr")},
            "time": tf.io.FixedLenFeature([], tf.float32),
            "gt-interp-age-tachy": tf.io.FixedLenFeature([], tf.float32)}

for path in sys.argv[1:]:
    csi, pos, snr, time, age = [], [], [], [], []
    for rec in tf.data.TFRecordDataset(path):
        ex = tf.io.parse_single_example(rec, FEATURES)
        c = tf.io.parse_tensor(ex["csi"], tf.float32).numpy()
        csi.append((c[..., 0] + 1j * c[..., 1]).astype(np.complex64))
        pos.append(tf.io.parse_tensor(ex["pos-tachy"], tf.float64).numpy())
        snr.append(tf.io.parse_tensor(ex["snr"], tf.float32).numpy())
        time.append(ex["time"].numpy())
        age.append(ex["gt-interp-age-tachy"].numpy())
    out = path.rsplit(".", 1)[0] + ".npz"
    np.savez(out, csi=np.stack(csi), pos=np.stack(pos), snr=np.stack(snr), time=np.array(time),
             interp_age=np.array(age))
    print(f"{out}: {len(csi)} datapoints", flush=True)
