"""Generate and cache every channel set the leave-one-city-out experiment needs."""
import time
from sixg_gen import channels

SETS = [("cdl", 12000, 1)] + [
    (f"rt:{city}", n, seed)
    for city in channels.RT_SCENES
    for n, seed in ((4000, 1), (1000, 2), (2000, 3))
]

if __name__ == "__main__":
    start = time.time()
    for i, (source, n, seed) in enumerate(SETS, 1):
        t = time.time()
        channels.generate(source, n, seed)
        print(f"[{i}/{len(SETS)}] done {source} n={n} seed={seed} in {time.time()-t:.0f}s "
              f"(total {(time.time()-start)/60:.1f} min)", flush=True)
    print("ALL DONE", flush=True)
