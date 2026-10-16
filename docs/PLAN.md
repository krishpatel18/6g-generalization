# Physics-aware pre-training for generalizable 6G channel estimation

Corrected version of the FLIGHT plan drafted in the Gemini notebook ("NVIDIA 6G Dev
Program"), checked against primary sources on 2026-10-16. Facts are tagged
**[verified]** (primary source linked) or **[unverified]**.

## The question

Learned channel estimators trained on 3GPP stochastic models (CDL/UMa) degrade
when deployed at a real site. Hypothesis: pre-training on many **ray-traced**
city scenes (Sionna RT) gives a better prior than pre-training on stochastic
models, measured by error on an **unseen city** after fine-tuning on few samples
from it.

Status of novelty: two-phase "pre-train on simulation, fine-tune on site" is
established; the recent literature recommends stochastic pre-training plus RT
evaluation. Using *broad multi-scene RT* as the pre-training corpus is named as
future work rather than done — the contribution is the controlled comparison.
**[unverified — from the notebook's sources; needs a proper related-work pass]**

## Pilot experiment (runs on a laptop, no cloud)

Leave-one-city-out over Sionna RT's four built-in real-city scenes (Munich,
Paris-Étoile, Florence, San Francisco):

| Arm | Pre-training data (12k channels each) |
|---|---|
| A — stochastic | CDL-A..E, random delay spread 30–300 ns |
| B — ray-traced, broad | the three other cities, 4k each |
| C — ray-traced, narrow | one other city, 12k |

B vs C isolates **breadth** (the novel claim) from **RT realism** (already
reported in the literature: RT→RT transfer beats stochastic→RT).

Each arm is fine-tuned on N ∈ {0, 50, 200, 1000} channels from the held-out city
and tested on 2k channels from different base-station sites in that city.
Controls: same estimator, same budget, same grid; plus training from scratch on N,
and a classical LMMSE fitted on the same N samples.

Setup: 3.5 GHz, 30 kHz SCS, 128 subcarriers × 14 symbols, SISO, DMRS-like pilots
(every other subcarrier, symbols 2 and 11), users at 0–30 m/s. Code in `src/sixg_gen`,
experiment in `scripts/loco.py`.

**Caveats, written before results.**
- *Sim-to-sim.* Test channels come from the same ray tracer, antenna setup,
  placement rule and speed range as arms B and C, so they have a home advantage
  over arm A. A→target vs B→target measures "RT prior vs CDL prior for an RT
  target", not for real hardware. B vs C is clean, since both share the generator.
- *Refraction on.* Sionna's default solver lets users placed inside buildings
  receive through-wall paths, and per-sample normalization lifts them to unit
  power. They act like indoor users; kept, not filtered.
- *Single seed, 4 folds.* Differences under ~0.5 dB are not findings until a
  second seed is run.

Results: see "Results" below.

## Corrections to the notebook plan

1. **Real-time target.** An NR slot at 30 kHz SCS is 0.5 ms and a frame is 10 ms;
   "under 20 ms per frame" cannot keep up. Target well under 0.5 ms per slot for
   channel estimation; report anything slower as offline. **[verified]** 3GPP TS 38.211
   numerology; the PFM paper reports 2.81 ms with a 24M-param model
   (https://arxiv.org/html/2512.15562v2).
2. **"~10 dB gain".** In the sources this is RT-similar-scene pre-training vs
   stochastic pre-training in low-data regimes, not a gain over LS/MMSE. The
   "152.4%" figure (U-BiTRCE) is an NMSE improvement over plain LS for IRS cascaded
   channels at 18 ms/frame — not a relevant benchmark here. **[verified]**
   https://ieeexplore.ieee.org/document/11202473/
3. **Two cities can't test "broad".** Wall Street → Houston is a one-pair
   transfer. Broad pre-training needs a scene library; the pilot uses 4 built-in
   cities, and a real study should use 20+ (scripted generation, see 6).
4. **"New York 80% LoS"** comes from a DeepMIMO-generated New York dataset
   (1151 samples) in https://arxiv.org/html/2602.09685v1 — not Wall Street, and not
   a property of urban canyons. **[verified]**
5. **High-frequency trading framing.** HFT links use fibre and microwave, not
   cellular access. Keep Wall Street as a hard urban-canyon test scene; drop the
   HFT claim or treat it as colour only. **[unverified judgment]**
6. **Scene building.** Sionna's documented OSM path is Blender + Blosm +
   Mitsuba-Blender add-on (https://nvlabs.github.io/sionna/rt/tutorials/Introduction.html);
   Blender is not installed here. geo2sigmap
   (https://github.com/functions-lab/geo2sigmap) claims a pure-Python pipeline
   that loads in Sionna — worth trying first for a 20+ scene library.
   **[verified docs; geo2sigmap untested]**
7. **Sionna is PyTorch now.** Sionna 2.x depends on torch, not TensorFlow, so
   "GCP because TensorFlow" no longer holds. **[verified]** PyPI `sionna` 2.2.0.
8. **Apple GPU works.** Sionna RT 2.2 / Mitsuba 3.9 has a Metal backend; ray
   tracing 256 users in a city scene takes seconds on an M4, so the pilot needs
   no cloud. **[verified locally]**

## TimesFM

- Latest is TimesFM 3.0 (`google/timesfm-3.0-pytorch`, ~0.3B params, 15,360
  context, PyTorch + MLX). **Weights are under a non-commercial licence**; code is
  Apache-2.0. **[verified]** https://huggingface.co/google/timesfm-3.0-pytorch
- The pilot-reduction paper used **TimesFM 1.0** (200M), splitting the channel
  into univariate sequences per antenna/subcarrier/re-im and fine-tuning; AdaLN is
  in its pilot-processing ViT, not in TimesFM. 2P pilots matched 4P LMMSE.
  Code not found. **[verified]** https://arxiv.org/html/2512.15562v2
- Fit with this project: TimesFM predicts the *next* channel from history
  (multi-slot), while the pilot experiment estimates the *current* slot. It is a
  second stage, not the starting point.

## Compute

- GCP free trial **cannot use GPUs**; you must upgrade to a paid account (credits
  carry over, still expire 90 days from sign-up), and new accounts often need a GPU
  quota request (reports of 48 h+). **[verified]**
  https://docs.cloud.google.com/free/docs/free-cloud-features
- Rough on-demand prices (us-central1, third-party tracker): A100 40GB ≈ $3.67/h,
  L4 ≈ $0.71/h → $300 ≈ 80 A100-hours or ≈ 420 L4-hours. **[unverified]**
- The pilot runs on the M4. Cloud is needed only for a large scene library or
  TimesFM fine-tuning.

## Open decisions (Krish)

See the end of the session summary; recorded here once decided.

## Results

_Pending — pilot run in progress._
