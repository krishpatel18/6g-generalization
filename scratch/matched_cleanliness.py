"""Are the matched-placement (closer users) channels cleaner than the farther-placement ones?"""
import sys
sys.path.insert(0, "scripts")
from sixg_gen import channels, grid
from dichasus_diagnostics import cleanliness
from dichasus_rich import CITIES, PER_CITY
grid.use_dichasus()
print("first-tap energy %, top-3 spatial eigenvalues %: rtd (users 10-65 m) vs rtdm (users 12-30 m)")
for c in CITIES:
    a = cleanliness(channels.generate(f"rtd:{c}", PER_CITY, 1, mmap=True))
    b = cleanliness(channels.generate(f"rtdm:{c}", PER_CITY, 1, mmap=True))
    print(f"  {c:16s} rtd {a[0]:5.1f} {a[1]:5.1f}   rtdm {b[0]:5.1f} {b[1]:5.1f}")
