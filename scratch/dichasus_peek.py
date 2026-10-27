import sys, numpy as np, tensorflow as tf
feat = {k: tf.io.FixedLenFeature([], tf.string) for k in ("csi", "pos-tachy", "snr", "cfo")}
feat.update({"time": tf.io.FixedLenFeature([], tf.float32), "gt-interp-age-tachy": tf.io.FixedLenFeature([], tf.float32)})
ds = tf.data.TFRecordDataset(sys.argv[1])
n = 0
for rec in ds:
    ex = tf.io.parse_single_example(rec, feat)
    if n == 0:
        csi = tf.io.parse_tensor(ex["csi"], tf.float32).numpy()
        pos = tf.io.parse_tensor(ex["pos-tachy"], tf.float64).numpy()
        snr = tf.io.parse_tensor(ex["snr"], tf.float32).numpy()
        cfo = tf.io.parse_tensor(ex["cfo"], tf.float32).numpy()
        print("csi", csi.shape, csi.dtype, "pos", pos, "snr", snr.shape, np.round(snr.ravel()[:6], 1), "cfo", cfo.shape)
        t0 = ex["time"].numpy()
    t = ex["time"].numpy()
    n += 1
print("records", n, "duration s", t - t0, "mean spacing ms", 1000 * (t - t0) / max(n - 1, 1))
