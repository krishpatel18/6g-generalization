import numpy as np, mitsuba as mi
mi.set_variant("metal_ad_mono_polarized")
import sionna.rt as rt
from sixg_gen import channels
for name in ("san_francisco", "munich", "etoile", "florence", "data/scenes/rochester_brighton/scene.xml", "data/scenes/midtown/scene.xml"):
    scene = rt.load_scene(getattr(rt.scene, name) if "/" not in name else name)
    g = channels._ground_object(scene)
    bb = scene.mi_scene.bbox(); lo, hi = np.array(bb.min), np.array(bb.max)
    xy = lo[:2] + np.random.default_rng(0).uniform(0.2, 0.8, (2000, 2)) * (hi[:2] - lo[:2])
    top, ground = channels._first_hit(scene.mi_scene, xy), channels._first_hit(channels._ground_scene(scene), xy)
    out = np.isfinite(ground) & (top < ground + 0.5)
    print(f"{name}: ground={g.name!r}, outdoors {out.mean():.2f}, ground z range {np.nanmin(np.where(np.isfinite(ground), ground, np.nan)):.1f}..{np.nanmax(np.where(np.isfinite(ground), ground, np.nan)):.1f}")
