"""Generate the channel sets for every scene in scripts/scene_library.txt.

usage: make_library.py <num_bs_antennas>
Per city: 12k pre-training (seed 1), 1k site pool (seed 2), 2k test sites (seed 4).
"""
import sys
import time
from pathlib import Path
from sixg_gen import channels, grid

LIBRARY = Path(__file__).with_name("scene_library.txt")


def library() -> list[str]:
    return [line.split()[0] for line in LIBRARY.read_text().splitlines() if line.strip() and not line.startswith("#")]


if __name__ == "__main__":
    grid.NUM_ANTENNAS = int(sys.argv[1])
    start = time.time()
    for city in library():
        for n, seed in ((12000, 1), (1000, 2), (2000, 4)):
            t = time.time()
            channels.generate(f"rt:{city}", n, seed)
            print(f"[{city}] n={n} seed={seed} in {time.time()-t:.0f}s (total {(time.time()-start)/60:.1f} min)", flush=True)
    print("ALL DONE", flush=True)
