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

### Run 1 (2026-10-19): SISO, 3.84 MHz — the setting can't measure the question

`runs/loco_seed0.csv`, seed 0. NMSE in dB at 10 dB SNR, lower is better:

| method | N | Munich | Étoile | Florence | San Francisco |
|---|---|---|---|---|---|
| LS + interpolation | 0 | −12.2 | −12.2 | −12.2 | −12.2 |
| CDL-pretrained, zero-shot | 0 | −18.7 | −18.5 | −18.7 | −19.2 |
| 3-city RT, zero-shot | 0 | −19.1 | −18.3 | −19.3 | −20.1 |
| 1-city RT, zero-shot | 0 | −19.1 | −18.5 | −19.4 | −20.0 |
| CDL-pretrained, fine-tuned | 200 | −19.1 | −18.6 | −19.2 | −20.3 |
| 3-city RT, fine-tuned | 200 | −19.1 | −18.6 | −19.3 | −20.4 |
| from scratch | 200 | −19.1 | −18.5 | −19.3 | −20.3 |
| LMMSE fitted on target | 50 | −19.9 | −19.6 | −22.4 | −22.9 |
| LMMSE fitted on target | 200 | −21.7 | −20.4 | −22.7 | −23.9 |

- Every learned arm lands within ~1 dB of every other, including training from
  scratch on 50 samples. The CNN learns a generic denoiser; the prior doesn't matter.
- A linear LMMSE fitted on 50 target samples beats all of them by 1–3 dB: the
  CNN (13×13 receptive field, 13 of 128 subcarriers) doesn't reach the linear optimum.

### Diagnosis with LMMSE as the linear twin of each arm

Fit LMMSE on each pre-training source, test on each city, compare with LMMSE fitted
on the city itself (12k). Extra error vs in-domain, dB at 10 dB SNR (Mu/Ét/Fl/SF):

| setting | fit on CDL | fit on 3 other cities |
|---|---|---|
| SISO, 128 sc (run 1) | 0.8 / 0.3 / 1.5 / 2.8 | 0.1 / 1.9 / 0.3 / 1.4 |
| SISO, pilots every 4–12 sc, 1 symbol | ≤1.3 | ≤0.4 |
| SISO, 512 sc (15.4 MHz) | 0.6 / 0.6 / 1.3 / 3.1 | 0.2 / 2.9 / 0.4 / 2.2 |
| **8 BS antennas**, 128 sc | 2.0 / 1.5 / 2.7 / 4.2 | 0.6 / 2.2 / 0.4 / 1.1 |
| **16 BS antennas**, 128 sc | 2.2 / 1.7 / 3.2 / 4.8 | 0.7 / 2.1 / 0.2 / 0.8 |

(Probes use 2k fit / 1k test channels; they reproduce the 12k numbers within ~0.2 dB.)

- Single-antenna channels barely differ between sources; pilot density and
  bandwidth don't change that. **Antennas do**: spatial structure is where a city's
  geometry shows, and the CDL-vs-RT gap grows with array size.
- With 16 antennas, other-city RT statistics are 1.5–4 dB closer to the target than
  CDL statistics in Munich, Florence and San Francisco — consistent with the thesis.
- **Étoile is the exception everywhere**: other-city RT transfers *worse* than CDL
  (up to 5 dB at 20 dB SNR). Its open star-shaped plaza doesn't resemble the street
  grids of the other three. A "broad RT prior" claim has to survive sites like this.

Next: rerun with a multi-antenna base station and an estimator that at least
matches in-domain LMMSE.

### Run 2 — pre-registered before launch (2026-10-19)

Code: `scripts/loco.py` (run 1's version is in git at `ef199f4`).

- **Setting:** 8-antenna base station (half-wavelength ULA), otherwise as run 1.
  16 antennas adds only 0.2–0.6 dB of CDL gap for 4x the cost; deferred.
- **Estimator per arm:** LMMSE from the arm's 12k pre-training channels, plus a
  CNN correction pre-trained on top of it (last layer zero-initialised, so it
  starts as exactly that LMMSE). Fine-tuning on N target channels = blend the
  covariances toward the N samples (weight by 2-fold CV) + 300 low-rate CNN steps.
- **Arms:** `cdl`; `rt3` (other three cities, 4k each); `rt1` (one other city,
  12k, **rotated**: Munich←Étoile, Étoile←Florence, Florence←SF, SF←Munich, so
  the outlier Étoile is the single source once only).
- **Baselines:** LS + interpolation; LMMSE from the N target samples alone;
  LMMSE from 12k in-domain channels (ceiling).
- **Gates — the arm comparison is not read unless both hold:**
  a) every arm's `lin+cnn` at N=0 is within 0.1 dB of (or better than) its `lin`;
  b) every arm at N ≥ 50 is no worse than LMMSE from the same N target samples.
- **Claims and what falsifies them** (NMSE at 10 dB SNR, seed 0; confirm with seed 1):
  - *RT realism*: `rt3` beats `cdl` at N=0 in ≥ 3 of 4 cities. Falsified otherwise.
  - *Breadth*: `rt3` beats `rt1` by ≥ 0.5 dB averaged over the 4 cities at N=0
    and N=50. Smaller or negative → no evidence that breadth matters here.
  - *Beyond covariance*: `lin+cnn` beats `lin` by ≥ 0.5 dB in some arm. Otherwise
    the prior's value is all second-order statistics, and the LMMSE-from-source
    grid is the whole story.
- **Not a deployable estimator:** dense 14336×1024 LMMSE plus a CNN is a
  measuring instrument, not something that meets the <0.5 ms slot budget.
- **Side result to carry forward (pre-registered):** the LMMSE-from-source gap is itself a cheap
  "which pre-training source resembles this site?" metric — an open question the
  notebook listed. If Étoile stays an outlier, the project's contribution may be
  "when does an RT prior transfer, and can you tell in advance" rather than
  "RT beats CDL".

### Run 2 — results (seed 0, 2026-10-19)

`runs/a8_sc128/loco_seed0.csv`; checked with `scripts/report.py`. NMSE dB at 10 dB SNR:

| row | Munich | Étoile | Florence | San Francisco |
|---|---|---|---|---|
| LMMSE in-domain 12k (ceiling) | −22.3 | −20.8 | −23.2 | −24.7 |
| LMMSE own N=50 | −16.8 | −15.3 | −16.6 | −21.2 |
| LMMSE own N=200 | −19.3 | −17.7 | −20.6 | −23.1 |
| LMMSE own N=1000 | −21.3 | −20.1 | −22.7 | −24.1 |
| cdl, N=0 | −20.2 | **−19.0** | −20.3 | −20.3 |
| rt3, N=0 | **−21.9** | −18.7 | **−22.7** | −23.2 |
| rt1, N=0 | −21.9 | −18.6 | −22.2 | **−23.3** |
| cdl, N=50 | −21.6 | **−20.0** | −22.1 | −23.9 |
| rt3, N=50 | −21.9 | −19.1 | **−22.7** | **−24.2** |
| rt1, N=50 | **−22.0** | −19.1 | −22.2 | −24.2 |

(`lin` rows; `lin+cnn` is within ±0.3 dB except Munich-rt1, see below.)

- **Gate a: pass** (worst +0.03 dB at 10 dB, +0.05 dB at 0/20 dB).
- **Gate b: fails by 0.01–0.03 dB in one cell** — San Francisco, cdl, N=200, where
  the cross-validated blend chose weight 1.0 (pure target) and so tied
  LMMSE-own; the CNN then added noise. The failure is the blend rule's weight
  choice, not the prior: the same arm scored −23.9 dB at N=50 with weight 0.9.
  Strictly per pre-registration the N ≥ 50 comparison is not clean; N=0
  comparisons don't depend on gate b.
- **RT realism: supported** — rt3 beats cdl at N=0 in 3/4 cities at 10 and 20 dB
  (Munich +1.8, Florence +2.4, SF +2.8, Étoile −0.3 dB at 10 dB) and 4/4 at 0 dB.
  With **zero** site data the 3-city RT prior beats LMMSE from 200 site samples in
  all four cities, and roughly matches 1,000 in Munich and Florence.
- **Breadth: no evidence** — rt3 vs rt1 averages +0.15 dB at 10 dB, +0.48 dB at
  20 dB, ≈0 with the CNN. One other city is as good a prior as three here.
- **Beyond covariance: weak** — clears 0.5 dB only for Munich with the
  Étoile-only prior (+0.5 to +0.66 dB at every N); ≤ 0.3 dB everywhere else.
  One arm × one city; treat as unconfirmed.
- **Étoile again**: the CDL prior is the best prior for Étoile at every N ≥ 50;
  RT priors are 0.3–1 dB worse. Note the asymmetry: Étoile-only is an excellent
  prior *for Munich*, but other cities are poor priors *for Étoile*.
- **Blend rule weakness**: 2-fold CV on N samples is noisy (e.g. it chose pure
  target at SF N=200, worse than its own N=50 result). A better adaptation rule
  is a cheap improvement before seed 1.
