"""Run loco.py end to end with tiny step counts on single-antenna data, into scratch/."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import loco
from sixg_gen import channels, grid
loco.RUNS = Path(__file__).parent / "smoke_runs"
loco.PRETRAIN_STEPS, loco.FINETUNE_STEPS = 30, 10
channels.RT_SCENES = ("munich", "etoile")
loco.SINGLE_SOURCE = {"munich": "etoile", "etoile": "munich"}
grid.NUM_ANTENNAS = 1
t = time.time(); loco.main(0); print(f"smoke ok in {time.time()-t:.0f}s")
