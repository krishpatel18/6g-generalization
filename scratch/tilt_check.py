"""Which sign of the downtilt points the facade array at the street? Mean received power of the same users."""
import numpy as np
from sixg_gen import channels, grid
grid.use_dichasus()
for tilt in (grid.DOWNTILT_RAD, -grid.DOWNTILT_RAD):
    grid.DOWNTILT_RAD = tilt
    h, _ = channels._raytraced("munich", 256, 7)
    print(f"tilt {np.degrees(tilt):+.0f} deg: mean power {10*np.log10(np.mean(np.abs(h)**2)):.2f} dB, n={len(h)}")
