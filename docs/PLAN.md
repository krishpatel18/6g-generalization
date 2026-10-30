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
   HFT claim or treat it as colour only. **[unverified judgment]** → dropped
   (see Decisions).
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

## Decisions (2026-10-19)

- **Solo project.** The notebook's four roles (lead, ML research, systems
  integration, data engineering) collapse to one person; scope and timeline
  below are sized for that.
- **Compute: this Mac until it's not enough.** No GCP yet. Revisit when the
  scene library grows past ~20 cities or TimesFM fine-tuning starts; that needs
  a paid upgrade and a GPU quota request (2+ days).
- **Code: private GitHub repo** (`krishpatel18/6g-generalization`); data and
  runs stay local (regenerable from the scripts).
- **Framing: high-frequency trading dropped.** Wall Street stays as a candidate
  hard urban-canyon scene, with no trading claim.

## Solo plan (replaces the 8-month team timeline)

1. **Now — settle the pilot** on Sionna's 4 built-in cities: run 2b (fair CDL),
   seed 1, and a shrinkage-regularized "own N samples" baseline.
2. **Scene library** — build 10–20 more cities with geo2sigmap or Blender+Blosm
   (Wall Street and Houston among them), chosen to vary layout (grid streets,
   open plazas like Étoile, high-rise canyons). This is what can actually test
   *breadth*; four cities can't.
3. **Similarity metric** — turn the LMMSE-from-source gap into a predictor of
   which pre-training source suits a new site, and test whether it predicts
   transfer (Étoile is the motivating case).
4. **Sequence stage** — channel prediction across slots, where TimesFM (or a
   small causal Transformer) belongs; check the non-commercial weight licence
   is acceptable first.
5. **Write-up** for the NVIDIA 6G Developer Program.

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
- **Side result to carry forward:** the LMMSE-from-source gap is itself a cheap
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
  Munich, Étoile and Florence (ties in SF), and roughly matches 1,000 in Munich
  and Florence. Caveat: the own-N LMMSE baseline uses only light diagonal loading
  for 1,024 pilot dimensions; a shrinkage-regularized baseline would be stronger.
- **Caveats on this result:** simulation-to-simulation (test channels come from the
  same ray tracer as the RT arms); seed 0 only; the CDL arm's angular diversity is
  under re-check (see below).
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

### Run 2 correction (2026-10-19): the CDL arm had fixed arrival angles

Sionna's CDL defaults to a fixed base-station orientation, and CDL clusters have
fixed angles, so at 8 antennas the 12k CDL channels spanned only five angular
signatures: the top spatial eigenvalue held 48% of the energy vs ~24% for the RT
cities. That made the CDL arm an unfairly narrow prior, which biases the "RT
realism" comparison above in RT's favour. **The run-2 seed-0 numbers above are
superseded** (kept in `runs/a8_sc128/loco_seed0_fixedcdl.csv`).

Fix: a uniform random base-station heading per CDL sample (top eigenvalue now
25%, matching Munich's 24%). Also changed before the rerun: the blend weight is
chosen by 5-fold CV over a finer grid (0.85/0.9/0.95 added), since 2-fold CV
picked pure-target weights at SF N=200. Gates and claims are unchanged.

### Run 2b — results with the fair CDL arm (seed 0, 2026-10-19)

`runs/a8_sc128/loco_seed0.csv`; `scripts/report.py`. NMSE dB at 10 dB SNR (`lin`; `lin+cnn` in brackets where it differs by ≥ 0.2 dB):

| row | Munich | Étoile | Florence | San Francisco |
|---|---|---|---|---|
| LMMSE own N=50 / 200 / 1000 | −16.8 / −19.3 / −21.3 | −15.3 / −17.7 / −20.1 | −16.6 / −20.6 / −22.7 | −21.2 / −23.1 / −24.1 |
| cdl, N=0 | −21.4 (−21.6) | **−20.6** (−20.8) | −21.5 (−21.7) | −21.6 (−22.1) |
| rt3, N=0 | **−21.9** | −18.7 | **−22.7** (−23.0) | −23.2 |
| rt1, N=0 | −21.9 (−22.5) | −18.6 | −22.2 (−22.4) | **−23.3** (−23.5) |
| cdl, N=50 | −21.9 (−22.1) | **−20.6** | −22.2 (−22.5) | −23.8 (−24.1) |
| rt3, N=50 | −21.8 | −19.1 | **−22.7** (−22.9) | **−24.4** |
| rt1, N=50 | **−22.0** (−22.4) | −19.1 | −22.2 (−22.4) | −24.3 |
| cdl, N=1000 | −22.1 (−22.3) | **−20.6** (−20.8) | −22.9 (−23.3) | −24.1 (−24.4) |
| rt3, N=1000 | −22.0 | −20.2 | −22.8 (−23.1) | −24.1 |
| rt1, N=1000 | −22.1 (−22.6) | −20.2 | −22.8 (−23.2) | −24.1 |

- **Gates:** a passes everywhere; b passes at 0 and 10 dB, misses by 0.02 dB at 20 dB.
- **RT realism, zero-shot: supported but smaller than run 2 suggested** — rt3 beats
  the fair CDL prior by +0.5 (Munich), +1.2 (Florence), +1.5 dB (SF), and loses by
  1.8 dB on Étoile. Most of run 2's apparent advantage in Munich came from the
  CDL flaw.
- **The advantage fades with site data.** At N=50 rt3 leads only in Florence and
  SF (~0.5 dB); by N=1000 all arms are within ~0.2 dB, except Étoile (0.4 dB). An RT prior is worth most
  when a new site has (almost) no measurements — which is the deployment case
  that matters, but a narrower claim than "RT pre-training generalizes better".
- **Breadth: no evidence** (rt3 vs rt1 +0.15 dB at 10 dB, +0.48 at 20 dB, ≈0 with CNN).
- **Beyond covariance: small and fairly consistent on CDL** (+0.2–0.45 dB at
  every N), up to +0.65 dB for Munich-rt1. Above 0.5 dB in only one cell at 10 dB.
- **Étoile**: the fair CDL prior is best there at every N.
- **Blend limitation:** the CV blend can leave an arm worse than its own zero-shot
  (Munich rt3: −21.9 at N=0, −21.8 at N=50, −21.7 at N=200). Gate b still holds.
- **What "seed" varies** (clarification, not a change to the pre-registration):
  in `loco.py` the seed sets CNN init, sample order and CV folds only. Data and
  test noise are fixed, so the `lin N=0` rows behind the realism claim cannot
  change with seed. They are confirmed instead on fresh base-station sites
  (new test and pre-training draws), below.
- Simulation-to-simulation.

### Fresh-site check (2026-10-20): the realism result holds; breadth is inside site noise

`scripts/fresh_sites.py`, `runs/a8_sc128/fresh_sites.csv`. LMMSE (`lin`) rows only,
on 2k channels from **new base-station sites** per city (seed 4). Variant A keeps
the original pre-training draws; variant B redraws every pre-training set,
CDL included (seed 5).

rt3 advantage over CDL at 10 dB SNR (dB, positive = RT prior better), N=0 / N=50:

| | Munich | Étoile | Florence | San Francisco |
|---|---|---|---|---|
| run 2b (original test sites) | +0.5 / −0.1 | −1.8 / −1.5 | +1.2 / +0.5 | +1.5 / +0.6 |
| A: new sites, same pre-training | +0.3 / −0.3 | −1.2 / −1.3 | +1.2 / +0.5 | +1.5 / +0.6 |
| B: new sites, new pre-training | +0.4 / −0.2 | −1.3 / −1.3 | +1.1 / +0.4 | +1.3 / +0.6 |

- **RT realism holds** on fresh sites and fresh pre-training draws: same sign in
  every city, magnitudes within ~0.3–0.6 dB. Florence and SF gain ~1.1–1.5 dB with
  no site data; Munich gains little (+0.3–0.5); Étoile loses 1.2–1.8 dB.
- **Breadth (rt3 − rt1, N=0): +0.43 (A) and +0.52 dB (B) on average**, vs +0.15 on
  the original sites — almost all from Florence (+0.6 → +1.5/+1.9 dB with the
  same pre-training, new test sites). Per-city breadth gaps move by ~1 dB between
  site draws, so four cities cannot resolve a ~0.5 dB breadth effect.
  **Inconclusive**, not "no effect". Testing breadth needs more cities (solo
  plan step 2), not more seeds.
- The gap from the zero-shot RT prior to the in-domain ceiling is 0.3–0.5 dB in
  Munich/Florence and ~1.5–1.8 dB in Étoile/SF: room a better prior could still
  take.

### Prior selection from 50 site measurements (2026-10-20)

`scripts/select_prior.py`, `runs/select_prior_a8.log`. For each city and
pre-training variant (fresh test sites, as above), score each prior's zero-shot
LMMSE on the 50-channel site pool only (noisy pilots at 0/10/20 dB, no test
data), pick the best, and compare with the test-set winner at 10 dB.

| | Munich | Étoile | Florence | San Francisco |
|---|---|---|---|---|
| A: picked / test best | rt3 / rt3 | **cdl / cdl** | rt3 / rt3 | rt3 / rt1 (−0.15 dB) |
| B: picked / test best | rt3 / rt3 | **cdl / cdl** | rt3 / rt3 | rt3 / rt1 (−0.25 dB) |

- **6/8 exact; mean regret 0.05 dB, worst 0.25 dB.** Both misses choose between
  two RT priors that are within 0.3 dB of each other.
- **It never picks the wrong family:** CDL for Étoile, an RT prior everywhere
  else. Compared with always using the 3-city RT prior, selection recovers
  ~1.2 dB on Étoile and costs nothing elsewhere.
- This reframes the project: *maintain a few candidate priors (stochastic and
  ray-traced) and let a handful of site measurements choose* beats any single
  prior, and makes the Étoile failure a non-issue. It answers the notebook's
  open "which source resembles this site?" question with a working, cheap rule.
- Caveats: 4 cities × 2 draws; 3 candidate priors; simulation-to-simulation.
  The obvious next test is a larger candidate library (more cities), which is
  also what breadth needs.

### Scene library: geo2sigmap blocked on OSM data access (2026-10-20)

geo2sigmap (functions-lab, updated 2026-10-11) installs on macOS after
`brew install libusb` (needed by open3d). Generating a Wall Street box
(−74.0125, 40.7040, −74.0060, 40.7085) built the ground but failed fetching
buildings: the public Overpass API refused the request, stating it is for
browser/human use and asking automated tools to run their own instance. Not
worked around. Legitimate paths: download an OSM extract (e.g. Geofabrik's
New York file), cut the box locally, and point geo2sigmap's building fetch at the
file; or run a local Overpass instance. Its scene XML uses the older material
naming (`mat-itu_concrete`, `radio-material` bsdf); loading in Sionna 2.2 is
still untested.

### Wall Street as a fifth, unseen city (2026-10-20)

Scene: `scripts/build_scene.sh wall_street -74.0125 40.7040 -74.0060 40.7085 <new-york extract>`
(geo2sigmap at a pinned commit, buildings read from a Geofabrik extract cut with
osmium; Overpass not used). Sionna 2.2 loads it and maps the legacy material
names to ITU concrete / metal / wet ground. Results: `scripts/new_city.py 8 wall_street`.

NMSE at 10 dB SNR on Wall Street test sites with no Wall Street data:
Munich prior −22.0 · all-4-cities −21.7 · Étoile −21.5 · Florence −21.4 ·
CDL −21.4 · San Francisco −17.9. LMMSE from 50 site samples alone −15.5;
in-domain ceiling (12k) −22.1.

- The best prior (Munich) is within 0.05 dB of the in-domain ceiling, and the
  50-sample pool picked it.
- A single mismatched city is risky: the SF prior costs ~4 dB here (and SF alone
  was also the weakest single source for Florence, ~1.5 dB below rt3). The
  pooled 4-city prior is never best but never bad. **Hypothesis:** breadth acts
  as insurance against a bad single source rather than raising the average —
  testable with a larger library (worst-case, not mean).
- Blending the 50 samples into the picked prior (w=0.25) was 0.1 dB worse than
  the prior alone — the known blend weakness.

### Library experiment — pre-registered (2026-10-20, before results)

14 cities (4 built-in + 10 New York areas in `scripts/scene_library.txt`, cut
from a Geofabrik extract; `rochester_brighton` is nearly empty — an open-area
case). Each city in turn is the unseen target; LMMSE only, no training;
`scripts/library.py 8`. NMSE at 10 dB SNR, reported as gap to the target's
in-domain ceiling.

- **Breadth-as-insurance:** pool k random other cities (k = 1, 2, 4, 8, 13; 8
  draws). Supported if the *worst* draw's gap shrinks by ≥ 1 dB more than the
  *mean* draw's gap between k=1 and k=13. If both shrink alike, breadth raises
  the average; if neither shrinks, breadth doesn't matter at this scale.
- **Selection:** a 50-channel site pool picks among CDL, each other city, and
  all-others pooled. Supported if mean regret vs the best candidate is ≤ 0.3 dB
  and it beats always-use-all-others-pooled on average.
- **RT vs CDL at scale:** all-others-pooled beats CDL on average over the 14
  targets. Falsified otherwise.

### Library experiment — results (2026-10-20)

`runs/a8_sc128/library.csv`, `runs/library_a8.log`. 14 targets, LMMSE, no site
data unless stated, NMSE at 10 dB SNR, gap to each target's in-domain ceiling.

Breadth curve (pool of k random other cities, 8 draws; mean over targets):

| k | mean draw | worst draw |
|---|---|---|
| 1 | 1.59 | 3.99 |
| 2 | 0.84 | 1.70 |
| 4 | 0.71 | 1.26 |
| 8 | 0.60 | 0.82 |
| 13 (all) | 0.59 | — (one draw) |

- **Breadth-as-insurance: supported.** From k=1 to k=8 the worst draw's gap
  shrinks 3.2 dB vs 1.0 dB for the mean. A single wrong source city can cost up
  to 7.0 dB; the worst 8-city pool costs at most 2.7 dB. Breadth also helps the
  average (1.6 → 0.6 dB), with most of the gain by k≈4.
- **Selection among 15 candidates: not supported.** Mean regret vs the best
  candidate 0.48 dB (bar: ≤ 0.3), worst 2.0 dB (Albany picked Bayside), exact in
  5/14; on average it ties always-use-all-others-pooled (−0.01 dB, better in
  7/14). With 3 candidates (4-city study) the same 50 samples chose well; with
  15, they overfit the pool. Selection needs fewer, more distinct candidates or
  more site samples.
- **RT vs CDL at scale: supported, modestly.** All-others-pooled beats CDL in
  11/14 targets, by 0.29 dB on average; it loses by at most 0.4 dB (Midtown,
  Albany, Cornell). The pooled RT prior sits 0.59 dB from the in-domain ceiling
  on average; LMMSE from 50 site samples alone is 6.5 dB away.
- Some priors land at or slightly beyond the "ceiling" (e.g. Wall Street,
  Étoile): the ceiling is itself a 12k-sample estimate, so treat gaps under
  ~0.1 dB as zero.

- **Insurance premium** (added after results): the best single source city per
  target sits 0.13 dB from the ceiling on average, the all-others pool 0.59 dB,
  so pooling costs ~0.45 dB relative to an oracle choice — about what 50-sample
  selection loses (regret 0.48). Worst single-source cases (6.3–7.2 dB) are all
  `rochester_brighton`, the nearly empty open-area scene, as the only source for
  dense downtowns. Worst-case shrinking faster than the mean with k is partly a
  generic property of averaging covariances; the informative numbers are the
  premium and the size of the worst case.

**What the evidence supports so far:** a ray-traced prior pooled from many
cities, with **no site measurements**, sits ~0.6 dB from in-domain performance
— roughly what CDL achieves once blended with 50 site measurements (run 2b,
four built-in cities: 0.3–1.0 dB, mean ≈ 0.6). Its value is saving those
measurements and removing the risk of a badly matched single source, at a
~0.45 dB premium over the (unknowable in advance) best single city. It is
modestly better than CDL with no measurements (0.29 dB). "LMMSE from 50 site
samples alone" (6.5 dB) is a naive baseline, not the fair comparison. Choosing a
prior from site measurements helped with 3 candidates, not with 15 (see the
diagnostic below). All of this is simulation-to-simulation, LMMSE, one array
geometry (8-antenna ULA), 3.5 GHz.

### Library follow-up (post-hoc, 2026-10-20)

`scripts/library_followup.py 8`, `runs/library_followup_a8.log`. Gap to the
in-domain ceiling at 10 dB SNR, mean over 14 targets:

| method | site channels | mean gap |
|---|---|---|
| CDL blended with site data | 50 | 0.70 |
| all-others RT pool, no adaptation | 0 | 0.59 |
| **all-others RT pool blended with site data** | 50 | **0.33** |
| select one of 15 priors, pool scored at 10 dB | 50 | 0.71 |
| select one of 15 priors, pool scored at 10 dB | 200 | 0.44 |
| best single prior (oracle) | — | 0.10 |

- **Pool + blend is the best practical recipe:** 0.33 dB from the ceiling with 50
  site channels; better than CDL + 50 in 12/14 targets (worse in Midtown and
  Cornell by ≤ 0.15 dB), by 0.37 dB on average. The RT pool with *no* site data
  is already slightly better than CDL with 50 (0.59 vs 0.70).
- **Hard selection stays fragile.** Scoring the pool at the evaluation SNR does
  not fix it (0.71 dB), so the earlier failure wasn't an SNR mismatch; 200 site
  channels help (0.44) but still lose to pool + blend. Choosing one source
  throws away the pool's insurance; blending keeps it.
- CDL + 50 fails badly on the open-area scene (`rochester_brighton`, 3.1 dB)
  where the RT pool is fine (0.2).

- **The 14-city means are carried by one outlier.** Without `rochester_brighton`
  (13 cities): CDL + 50 → 0.52 dB, pool + 50 → 0.33, pool with no site data →
  0.46; medians over 14: 0.29 / 0.24 / 0.32. The robust statistics are the 12/14
  count and a ~0.2 dB edge on built-up sites.
- The blend rule (5-fold CV over a fixed weight grid) was settled after run 2,
  before the library existed; it is a generic method, not fitted to these results.

### Pool-draw check (post-hoc, 2026-10-20)

`scripts/pool_draws.py 8`, `runs/pool_draws_a8.log`. The +50 rows above used
one 50-channel site pool per target ([:50] of the seed-2 draw). Re-scored on
three other pools ([50:100], [100:150], [150:200]); same priors, tests and
ceiling. Gap to ceiling at 10 dB, mean over 14 targets:

| site pool | CDL + 50 | pool + 50 | pool + 50 better | mean edge (13, no Rochester) |
|---|---|---|---|---|
| [:50] (follow-up above) | 0.70 | 0.33 | 12/14 | 0.19 |
| [50:100] | 0.61 | 0.48 | 11/14 | 0.11 |
| [100:150] | 0.73 | 0.62 | 10/14 | 0.11 |
| [150:200] | 0.47 | 0.34 | 10/14 | 0.11 |

- **The direction holds, the size shrinks.** Pool + 50 is ahead on average in
  every draw, but by ~0.1 dB on built-up sites, not ~0.2, and it wins 10–12 of
  14 targets, not reliably 12. The losses move around (Midtown, Albany, Cornell,
  Buffalo), so on dense sites the two are comparable.
- **The open-terrain gap was largely the first draw.** On `rochester_brighton`
  CDL + 50 trails by 0.43 and 0.49 dB in two draws and ties in the third (both
  3.26: the blend fell back to the site data alone), against 2.9 dB in the
  first draw.
- The gap to the ceiling with pool + 50 ranges 0.33–0.62 dB across draws, so
  "within ~0.3 dB" was the favourable draw.

**Current one-line result:** for a new site, a ray-traced prior pooled from
many cities and blended with ~50 site measurements gets within ~0.3–0.6 dB of a
site-trained estimator. Against the 3GPP-CDL equivalent with the same 50
measurements it is comparable on built-up sites (ahead by ~0.1 dB on average,
in 10–12 of 14 cities) and ahead by up to a few dB on open terrain, depending on
which 50 measurements are taken. Its clearer value is with no site data and as
insurance: it avoids the multi-dB failures of any single mismatched source.
(Simulation-to-simulation, LMMSE, 8-antenna ULA, 3.5 GHz; four 50-channel
site-pool draws.)

## Direction (b): multi-slot channel prediction

### Re-trace check (diagnostic, 2026-10-20)

`scratch/retrace_check.py <city>` (1024 users, one base station, 8-antenna ULA).
Question: does re-tracing a moving user at each time step differ from rotating
the phases of its t=0 paths (Doppler on a fixed path set, which is how both
sources generate time evolution now)? Percent of users whose fixed-path channel
is off by more than −20 dB NMSE from the re-traced one (mean dB is driven by a
few outliers):

| time ahead | Munich 0–10 / 10–20 / 20–30 m/s | Florence 0–10 / 10–20 / 20–30 m/s |
|---|---|---|
| 0 ms (same positions) | 1 / 1 / 0 % | 4 / 0 / 2 % |
| 5 ms | 1 / 4 / 7 % | 8 / 4 / 3 % |
| 10 ms | 3 / 8 / 13 % | 6 / 2 / 3 % |
| 20 ms | 5 / 8 / 27 % | 12 / 9 / 34 % |
| 50 ms | 11 / 55 / 84 % | 13 / 59 / 81 % |

- **The solver disagrees with itself on 1–4% of users at identical positions**
  (repeated at 10× rays: 1–4%). At 256 users per solve it matched exactly; at
  1024 it doesn't. Re-traced trajectories would carry this flicker as fake
  path changes, and regenerating cached channels is not bit-reproducible — the
  saved `.npy` files are the record.
- **Up to ~10 ms, re-tracing is within that noise floor** except the fastest
  users (Munich 13%). Real path-set changes show at ≥ 20 ms.
- **Decision:** prediction uses fixed-path Doppler evolution, horizons ≤ 10 ms;
  no re-tracing. Consequence, stated up front: on this simulator the channel
  over ≤ 10 ms is a finite sum of complex exponentials, so an ideal linear
  predictor with long, clean history is near-perfect. **The only room for
  learning is with short, noisy history.** That is the version of (b) to run.
- Population caveat: 70–84% of placed users get no paths (max depth 3, no
  diffraction) and are dropped, so users skew toward few-path, line-of-sight-like
  links, which favours linear predictors.

### Prediction experiment — pre-registered (2026-10-20, before results)

`scripts/make_tracks.py 8`, `scripts/predict.py 8 0`. Same four built-in cities,
leave-one-city-out, fixed-path Doppler evolution (re-trace check above).

- **Track:** history = the two pilot symbols of 4 consecutive slots (8 noisy
  observations over 1.4 ms), 8 antennas × every 4th subcarrier (32); targets =
  the mid-slot symbol 1, 2, 4, 10 and 20 slots after the last observed slot
  (0.5, 1, 2, 5, 10 ms). Each track normalized to unit power once.
- **Methods**, all given identical noisy history (SNR 0/10/20 dB):
  - *hold-last*: the last noisy observation, repeated (naive reference);
  - *tone*: classical, per user, no training data — angle-delay transform,
    one Doppler tone per bin fitted to the history (periodogram peak within
    ±400 Hz), noise-shrunk, extrapolated;
  - *Wiener*: space-time LMMSE from covariance priors — CDL, rt3 (other three
    cities pooled), each blended with N = 50 site tracks; own 12k as ceiling;
  - *learned*: CNN residual on the rt3 Wiener output (the run-2 Refiner,
    width 48), pre-trained on rt3 tracks (N = 0), fine-tuned on 50 site tracks
    (N = 50).
- **Expectation from the smoke test:** a pooled covariance predicts little past
  ~1 ms (Doppler phases differ per user), so per-user methods (tone, learned)
  have room at longer horizons.
- **Primary gate:** at 10 dB, horizon ≥ 2 ms, learned beats the best
  non-learned method with the same site data (max of tone and Wiener at that N)
  by ≥ 0.5 dB in ≥ 3/4 cities → "learning has room here". Falsified if learned
  is within 0.3 dB of, or worse than, the best non-learned method at every
  horizon ≥ 2 ms in every city. In between: inconclusive.
- **Secondary:** per speed bin (0–10, 10–20, 20–30 m/s) at 10 dB; RT vs CDL
  prior for prediction (does the (a) result carry over?).
- **Caveats written now:** one seed for CNN init; the CNN is one learned model,
  so a fail is "this model", not "learning"; users skew few-path (most placed
  users get no paths); simulation-to-simulation; the tone baseline is my design,
  not a published method, and fixed before results.
- **Amendment (2026-10-20, before any result):** two learned arms instead of
  one, gate passes if either does. *learned_ant* is the CNN as described;
  *learned_ad* is the same CNN on the angle × time × delay view (unitary
  transform of antennas and subcarriers, the domain the tone baseline uses).
  Reason: with 3×3 kernels the antenna-domain CNN cannot form a delay
  transform across 32 subcarriers, so a loss to tone would be uninformative.
  Both are reported. The gate printout also shows each learned arm against its
  own rt3 Wiener front.

### Prediction experiment — results (seed 0, 2026-10-20)

`scripts/predict.py 8 0`, `runs/a8_sc32_track/predict_seed0.csv`,
`runs/predict_a8.log`. NMSE dB at 10 dB SNR, all speeds, N = 0:

| city | method | 0.5 ms | 1 ms | 2 ms | 5 ms | 10 ms |
|---|---|---|---|---|---|---|
| Munich | hold-last | −7.1 | −2.3 | +1.9 | +2.7 | +3.0 |
| | Wiener rt3 | −15.1 | −8.6 | −1.7 | −0.1 | 0.0 |
| | learned (ant / ad) | −15.3 / −15.3 | −9.8 / −9.8 | −4.9 / −4.9 | −2.4 / −1.8 | −0.4 / −0.1 |
| | **tone** | −12.8 | **−10.2** | **−7.5** | **−4.1** | **−2.6** |
| Étoile | Wiener rt3 / learned (best arm per cell) / **tone** | −14.2 / −14.5 / −14.5 | −8.4 / −9.9 / **−12.1** | −1.6 / −6.1 / **−9.6** | −0.1 / −2.9 / **−6.1** | 0.0 / −0.7 / **−4.1** |
| Florence | Wiener rt3 / learned best / **tone** | −15.6 / −15.9 / −13.1 | −8.6 / −10.0 / **−10.7** | −1.6 / −4.9 / **−8.4** | −0.1 / −2.1 / **−4.9** | 0.0 / −0.3 / **−3.1** |
| San Francisco | Wiener rt3 / learned best / **tone** | −16.2 / −16.8 / **−19.5** | −9.1 / −11.6 / **−18.9** | −1.8 / −7.9 / **−17.8** | −0.1 / −4.3 / **−14.5** | 0.0 / −1.5 / **−10.7** |

- **Primary gate: falsified.** At every horizon ≥ 2 ms in every city, both
  learned arms are worse than the classical per-user tone predictor — by
  1.8–4.7 dB in Munich/Étoile/Florence and ~9–12 dB in San Francisco
  (N = 0; N = 50 is worse still).
- **The learned models did learn something:** 2–6 dB better than their own
  Wiener front at 2 ms, i.e. they pick up per-user extrapolation that no
  pooled covariance can. They just recover much less of it than fitting one
  Doppler tone per angle-delay bin does. The angle-delay arm is better than
  the antenna arm in Étoile and SF at 2 ms (−6.1 vs −4.9, −7.9 vs −5.7), about
  equal elsewhere.
- **Fine-tuning on 50 site tracks hurts** the learned arms (0.5–1.3 dB worse at
  5 ms) and the CV blend chose w = 0 for every Wiener prior: 50 tracks are not
  enough to move a 2048-dimensional space-time covariance.
- **Where learning is competitive:** only at 0.5 ms (one slot ahead), where the
  Wiener/learned methods match or beat tone in Munich, Florence and for fast
  users (Munich 20–30 m/s: tone −10.8, learned −13.9). Tone is weakest for
  fast users and strongest for slow ones.
- **RT vs CDL prior (secondary):** rt3 Wiener beats CDL Wiener by 0.2–1.5 dB
  at 0.5 ms and 0.7–1.1 dB at 1 ms; nothing past 2 ms, where every covariance
  predictor is near 0 dB. The (a) result carries over only to the shortest
  horizon.
- **Why tone is so strong here, stated plainly:** the simulator's motion is a
  fixed set of paths rotating in phase (re-trace check), which is exactly
  tone's model. San Francisco users are likely few-path, so one tone per bin is
  nearly exact there. On real channels, where paths appear and vanish, tone
  would degrade; this simulator cannot show by how much without the
  re-tracing whose noise floor ruled it out.
- Caveats as pre-registered: one CNN seed; one learned architecture (two
  domains); simulation-to-simulation; tone designed for this experiment (not a published method) and fixed before results.
- **Post-hoc, a handicap of this learned model:** it was trained with MSE over
  the whole 13-point track, in linear scale. The 8 history points and the
  far targets (near 0 dB) dominate that loss; the 0.5–2 ms targets, where
  prediction is possible, contribute little gradient. A loss weighted toward
  the predictable horizons might do better. Not rerun.

## Next step (2026-10-22)

Goal: portfolio project. First, a related-work check on both findings (is
"pooled ray-traced priors as insurance" known? is "classical per-user Doppler
fitting beats learned prediction on fixed-path simulators" known?). Then choose
between:

1. **Publish what exists.** Turn `docs/RESULT.md` into a short public write-up:
   simulated cities are a good starting prior; learned models did not beat
   classical methods here. Cost: a day or two, no money. Risk: modest,
   simulation-only.
2. **Test on measured channels.** Check the findings against a public dataset
   of measured channels (sim-to-real), where learning may also have more room.
   Cost: a few days; maybe a cloud GPU later. Risk: no dataset fits this setup.

Related-work findings and the dataset survey go below, then the comparison.

### Re-trace check, deterministic solver (2026-10-22)

The related-work search found that Sionna RT's path solver is documented as
non-deterministic by default (hash collisions with many receivers; GPU
floating point) and has `PathSolver(deterministic=True)` since 2.1.0. Our
2.2.0 runs used the default. Re-run with the flag
(`DET=1 scratch/retrace_check.py <city>`, `runs/retrace_check_det_*.log`):
identical positions now agree to −150 dB (0% of users), and the fixed-path vs
re-traced comparison is unchanged — 0–4% of users differ by more than −20 dB
up to 10 ms (Munich 20–30 m/s: 12%), 3–31% at 20 ms, 54–83% at 50 ms for
moving users. So: the 1–4% self-disagreement is known, avoidable behaviour,
not a finding; the decision to use fixed-path evolution up to 10 ms stands.
Cached channel sets were generated without the flag: each sample is still a
valid trace, but regeneration is not bit-reproducible.

### Related work (2026-10-22; search by subagent, key sources spot-checked)

Finding A (pooled RT covariance prior):
- Single-site RT/digital-twin covariance or subspace priors exist
  (arXiv 2607.09334; 2501.02758), as does an uncalibrated digital twin as a
  prior for channel statistics from a few measurements (2411.13360).
- Leave-one-city-out pre-training over 10 DeepMIMO cities exists for CSI
  feedback (2605.16141), without worst-case analysis.
- Morais et al. (2512.12449, abstract checked): stochastic-only evaluation
  can over- or under-estimate ML performance relative to ray tracing.
- **Not found:** an equal-weight multi-city RT covariance as a
  zero-measurement LMMSE prior; its rough equivalence to CDL + 50-sample
  shrinkage; the worst-case-vs-number-of-cities curve; selection overfitting
  as candidates grow. Novelty: medium, and it is in the empirical results,
  not the method (pooling + shrinkage is standard). Robust/minimax MMSE under
  covariance uncertainty is the likely theoretical ancestor of "pooling as
  insurance" and must be cited (not yet located).

Finding B (classical vs learned prediction):
- On QuaDRiGa/CDL, published comparisons have deep models beating Prony
  angle-delay prediction (WiFo, 2412.08908; LLM4CP); one study finds a Kalman
  filter better than neural networks up to ~4 ms on 3GPP TDL without noise
  (2308.14020; full-text claim from the subagent, abstract checked only).
- Morais et al. re-trace per position and use only sample-and-hold as baseline.
- **Not found:** a classical per-user predictor beating learned models on
  ray-traced channels. Novelty: medium-high, with the obvious objection that
  Sionna's fixed-path evolution is the classical model's own assumption; the
  deterministic re-trace check is the answer up to ~10 ms at 3.5 GHz.

### Measured-channel datasets (2026-10-22; subagent survey, not downloaded)

No dataset fits both questions. Best candidates:
- **DICHASUS outdoor** (Univ. Stuttgart, CC BY 4.0): 1.27 GHz, 50 MHz,
  three 2×8 arrays, one campus, mostly line-of-sight, ~70 ms snapshot spacing
  (inferred; the data turned out to have ~48 ms). Usable for a one-campus
  version of A; too coarse for B.
- **Argos channel survey** (Rice, research and commercial use, no
  redistribution): 2.4/5 GHz, 20 MHz, 96 antennas, 8 users, **2 ms
  resounding** over 20–120 s. The only verified fit for B. Download host
  refused connections on 2026-10-22; files 7.5–30 GB each.
- Others: CAEZ (ETH + NVIDIA testbed, 3.45 GHz, 20 ms spacing); Niigata
  (3 urban BS sites but 8 antennas, 0.5 s spacing); DICHASUS indoor with a
  3D model used in NVIDIA's Sionna calibration paper (2311.18558).

### Comparison of the two options

| | 1. Publish what exists | 2. Measured channels |
|---|---|---|
| What it adds | Two pre-registered results, one likely-new (B), one modestly new (A) | Answers "does any of this hold on real channels?" — the main caveat of both |
| Fit | Ready now | A: one campus, 1.27 GHz, weak test. B: Argos is a good fit if its host is back |
| Cost | 1–2 days, no money | B on Argos: a few days, 30+ GB download, laptop is enough |
| Risk | Reviewers call A obvious; B's simulator objection | Host unavailable; 20 MHz/2.4 GHz differs from the sim setup |

### Fairer learned predictors — post-hoc check, written before running (2026-10-27)

Not pre-registered with the prediction experiment: added after its results,
to address its two stated handicaps. `scripts/predict_fair.py 8 0`, N = 0
only (fine-tuning on 50 tracks hurt before), same tracks, same noise draws,
one seed.

- **Weighted loss:** train only on the 5 target times, each weighted by
  1 / (its front end's NMSE at that horizon on the training pool, 10 dB), so
  every horizon counts equally in relative terms; history points get 0.
- **Arms:** `wiener_w_{ant,ad}` — the original CNN on the rt3 Wiener front,
  weighted loss; `tone_res_{ant,ad}` — the same CNN as a residual on the tone
  predictor (front = tone's fit at the history times plus its extrapolation),
  weighted loss. Zero-initialised last layer, so each arm starts exactly at its
  front end.
- **Bar for changing finding 3** ("a classical method beat the neural
  networks"): any arm beats tone by ≥ 0.5 dB at 10 dB SNR, at some horizon
  ≥ 2 ms, in ≥ 3 of 4 cities → restate as "learning helps on top of the
  classical method" (tone_res) or "the training loss was the handicap"
  (wiener_w). Otherwise finding 3 stands, strengthened.
- **Amendment before running:** the tone front = the raw noisy history
  (same draw) + tone's extrapolation at the targets, so the residual can see
  what the tone fit removed. Weights are rescaled to sum to 13 (the uniform
  loss's total) and computed per city and front on 2000 pool tracks at 10 dB.
  Four arms × three horizons × four cities against a 0.5 dB bar is the most
  favourable reading for learning; a pass needs a second seed before any
  change to the README.

## Sim-to-real: finding 1 on measured channels (DICHASUS)

### Data and processing (2026-10-27)

DICHASUS `dxxx-complete` (Univ. Stuttgart, DaRUS doi:10.18419/DARUS-3153,
CC BY 4.0): three 2×8 arrays (λ/2 at 1.272 GHz, 50 MHz, 1024 subcarriers) on
one building facade, ~10 m up and tilted ~20° down; a hand-pulled transmitter
on the street in front, tachymeter positions, a snapshot every ~48 ms
(`scratch/dichasus_peek.py`, not logged). Checked on the data, not assumed:

- **Subcarrier order is FFT order**; after `fftshift`, each antenna's phase
  offset is a straight line across the band (fit residual 3–5°; scratch
  check, not logged). The authors'
  calibration tutorial uses the same fftshift and describes constant
  per-receiver carrier phase and sampling-time offsets; offset files are not
  shipped for this dataset.
- **Calibration:** per antenna, a phase offset + delay (relative to antenna 0)
  fitted to the line-of-sight phase predicted from the tachymeter positions,
  plus a gain. The 8-element axis is horizontal: that layout fits far better
  than the alternatives (array A coherence 0.73 vs ≤ 0.32; B 0.46, C 0.37 on
  the first five files).
- **One processing chain for every source** (measured, CDL, ray-traced), all
  generated on the full 1024-subcarrier band: align the first arrival (first
  delay tap within 10 dB of the peak) to 0, keep taps −10…+99 (−200 ns…2 µs),
  take the 128 subcarriers around the carrier (6.25 MHz), normalise to unit
  power. On usable datapoints, > 99.5% of measured energy lies within taps
  −10…+30 (scratch check, not logged; the pilot log's "0.608 in the first
  100 taps" is a different measure: all test datapoints, noise included, no
  first-arrival shift).
- **Pilot** (array A, 639 calibration-file datapoints, 620 test datapoints,
  10 dB; `scratch/dichasus_pilot.py`, rerun into `runs/dichasus_pilot.log`):
  LS interpolation −11.2 dB, CDL prior −17.9, LMMSE fitted on the
  calibration files −19.6. The data has structure a prior can exploit.
  Layout and offset fits: `runs/dichasus_calib_layout.log`.

### Sim-to-real experiment — pre-registered (2026-10-27, before results)

`scripts/dichasus_test.py`. Grid: 16 antennas (2×8) × 128 subcarriers, one
snapshot, pilots on every other subcarrier; LMMSE only; NMSE at 0 and 10 dB.

- **Target data.** Array A primary. B and C count toward a claim only if their
  calibration coherence on the calibration files is ≥ 0.5. Datapoints kept if
  mean per-antenna SNR ≥ 10 dB and tachymeter interpolation age < 0.1 s.
  Calibration and ceiling: d006–d010, d014. Site measurements: 50 random
  datapoints from d012, 4 draws. Test: d011 + d013. The three sets are disjoint
  files.
- **Priors.** CDL (A–E, delay spread 30–300 ns as before, array heading
  uniform within ±75°, 20° downtilt, 38.901 element pattern). Ray-traced: the
  14 scenes re-simulated at 1.272 GHz with the 2×8 array and a facade-like
  placement — base station 10 m up, 20° downtilt, 38.901 pattern, random
  heading; users 1.5 m up, 10–65 m away within ±75° of boresight; positions
  inside buildings rejected; 6000 channels per city. Pools: all 14 (equal
  weight), and k random cities (k = 1, 2, 4, 8; 8 draws).
- **Ceiling:** LMMSE fitted on the calibration files (site-trained).
- **Claims, 10 dB, array A:**
  - P1 — the 14-city RT pool beats CDL with no site data. Falsified if CDL is
    better.
  - P2 — insurance: the worst of 8 single-city priors is ≥ 1 dB further from
    the ceiling than the worst 8-city pool. Falsified if not.
  - P3 — the 14-city pool is within 1 dB of the ceiling. Reported either way.
  - P4 — pool + 50 vs CDL + 50 (4 draws): reported as in simulation.
- **Secondary, robust to residual phase miscalibration:** the same ladder on
  each antenna alone (frequency only, 1×128).
- **Can't be tested if:** the ceiling is within 3 dB of LS interpolation, or
  calibration coherence on array A is < 0.5.
- **Parked:** a ray-traced model of the campus itself (geo-registration of the
  local frame is uncertain; a different hypothesis).

### Fairer learned predictors — results (seed 0, 2026-10-27)

`runs/predict_fair_a8.log`, `runs/a8_sc32_track/predict_fair_seed0.csv`.
Arm minus tone at 10 dB (negative = arm better), horizons 2 / 5 / 10 ms:

| arm | Munich | Étoile | Florence | San Francisco |
|---|---|---|---|---|
| tone_res_ad | −0.36 / −0.45 / **−0.65** | −0.26 / −0.36 / **−0.52** | −0.32 / −0.42 / **−0.58** | **−0.56** / +0.78 / +1.16 |
| tone_res_ant | −0.37 / −0.34 / **−0.55** | −0.50 / −0.32 / −0.38 | −0.37 / −0.39 / **−0.55** | +0.38 / +0.86 / +0.93 |
| wiener_w_ad | +1.8 / +2.7 / +2.5 | +3.1 / +4.6 / +4.1 | +2.5 / +3.6 / +3.0 | +9.5 / +12.6 / +10.5 |
| wiener_w_ant | +3.0 / +2.9 / +2.5 | +4.4 / +4.8 / +4.1 | +3.9 / +3.6 / +3.1 | +11.9 / +12.3 / +10.5 |

- **Bar met, narrowly, by the tone residual:** tone_res_ad beats tone by
  ≥ 0.5 dB in 4/4 cities (0.52–0.65 dB; at 10 ms in three, 2 ms in San
  Francisco); tone_res_ant in 2/4 (Étoile −0.499, just under the bar). It is ahead of
  tone by 0.3–0.6 dB at every horizon ≥ 2 ms in Munich, Étoile and Florence,
  and worse than tone by 0.8–1.2 dB at 5–10 ms in San Francisco, where tone is
  already very good.
- **The loss was not the main handicap:** the weighted loss on the Wiener
  front gains up to ~1 dB at 2 ms in the angle-delay arm and loses elsewhere;
  still ~2–13 dB behind tone (1.8–12.6).
- As written beforehand: a second seed of the tone-residual arms runs before
  any change to the README (`scripts/predict_fair.py 8 1 tone_res`).
- **Seed 1 (`runs/predict_fair_seed1_a8.log`) reproduces seed 0 within
  ~0.2 dB (largest difference 0.21, San Francisco 2 ms):** tone_res_ad
  −0.61 / −0.52 / −0.60 dB at 10 ms (Munich, Étoile, Florence), −0.61 at 2 ms
  in San Francisco (4/4); tone_res_ant 3/4 (Florence −0.501). In San
  Francisco tone_res_ant also costs 0.38–0.59 dB at 2 ms (seeds 0 / 1). Finding 3 is restated: the classical tone predictor beats
  learned models trained alone by 2–13 dB, but a learned correction on top of
  it adds a small, repeatable ~0.3–0.6 dB at 2–10 ms (cells 0.26–0.65) —
  except where tone is already near-exact (San Francisco, where it costs
  0.8–1.2 dB at 5–10 ms). The horizon weights were computed on the first 2000
  pool tracks, i.e. one of the three source cities.

- **Implementation note (sim-to-real, before results):** "positions inside
  buildings rejected" is implemented relative to each scene's ground mesh, so
  heights follow terrain (San Francisco's ground spans 2–79 m over the central
  area sampled, 0–104 m overall; scratch checks, not logged); a spot counts
  as outdoors if the first surface above it is within 0.5 m of the ground.

### Scattering variant — exploratory (2026-10-27, after the dry run, before the full run's results)

A 300-per-city dry run of the registered test (which touched the test data)
showed the ray-traced priors far behind CDL. Diagnosis on that run (an
uncommitted run whose hilly-scene caches were later found suspect; figures
not logged): the
ray-traced channels are cleaner than the measured ones — 95–100% of energy in
the first 160 ns tap for most cities vs 85% measured (75% for the weaker,
farther datapoints, 94% for the strongest; measurement noise explains ~1%).
The registered experiment inherits the original study's solver settings:
`max_depth=3`, specular reflection and refraction only, no diffuse scattering,
no diffraction, and every material's scattering coefficient is 0 by default.
It runs unchanged. In addition, exploratory and labelled so: `rtd:` sources
(scattering coefficient 0.3 on all materials, Lambertian, plus Sionna's wedge
diffraction; its separate edge-diffraction option stays off) for the 4 built-in cities and wall_street, midtown, park_slope,
2000 channels each, through the same test (`scripts/dichasus_rich.py`).
A first look on Munich suggests scattering adds little energy (first-tap share
90% → 89%; scratch check, not logged), so a large change is not expected.

### Sim-to-real experiment — results (2026-10-27, corrected after review)

`scripts/dichasus_test.py`, `runs/dichasus_test.log`,
`runs/a16_sc128_dichasus/dichasus_test.csv`. **Rerun after a review found a
bug:** the facade placement cached each scene's ground mesh by `id(scene)`,
which Python can reuse for a later scene, handing it an earlier scene's
ground. A first regeneration check flagged park_slope and bayside
(`runs/dichasus_regen_check.log`: 4–28% matching), but that check compared
channels index by index, and users dropped for having no paths shift every
later index — a later review showed the same mismatch with the fixed code
(reviewer's check, not logged). So
whether the bug actually fired is unproven. The bug was fixed, both caches
regenerated, and everything rerun; reported numbers moved by ≤ 0.02 dB. A
shift-tolerant check (`runs/dichasus_regen_check_rt_v2.log`,
`..._rtd_v2.log`) regenerates the first base-station site of each cache (256
channels for `rt:`, 64 for `rtd:`) with the fixed code: 75–100% of channels
match their cached counterpart to 1e-3 relative error, against 0–3.5% when
cached channels are matched to *other* cached channels (a reviewer's null
check, not logged), so the caches are reproduced; why the rest differ
(solver non-determinism or users whose paths change) is not shown. A
per-scene ground mix-up would affect every site, so the first site suffices
to catch it. The `_settings.json` sidecars were written after generation from
the current code, so they record, rather than prove, the settings; what backs
the caches is these regeneration checks (ray-traced) and
`runs/cdl_seed_check_dichasus.log` (CDL). All three arrays pass the coherence bar (A 0.80, B 0.58,
C 0.62 with the reference antenna excluded; the registered check used the
metric including it: 0.81 / 0.59 / 0.63, in `runs/dichasus_test.log` at
commit 605663a). Test sets 620 / 754 / 391
datapoints. NMSE at 10 dB SNR:

| array, grid | LS | ceiling | CDL | 14-city RT pool | best / worst single city | CDL + 50 | pool + 50 |
|---|---|---|---|---|---|---|---|
| A, array × freq | −11.2 | −20.5 | **−17.9** | −14.7 | −15.3 / −10.2 | **−18.8** | −17.2 |
| B, array × freq | −11.2 | −20.2 | **−17.5** | −13.4 | −13.9 / −8.9 | **−18.6** | −16.9 |
| C, array × freq | −11.2 | −20.4 | **−17.9** | −14.4 | −14.8 / −9.7 | **−19.0** | −17.8 |
| A, per antenna | −11.2 | −20.0 | **−19.3** | −18.0 | −19.2 / −10.9 | −19.9 | −19.9 |
| B, per antenna | −11.2 | −20.1 | −19.3 | −18.6 | **−19.5** / −10.9 | −20.0 | −20.0 |
| C, per antenna | −11.2 | −20.2 | −19.3 | −18.8 | **−19.7** / −11.9 | −20.2 | −20.2 |

- **P1 falsified on every array:** with no site data the pooled ray-traced
  prior is worse than CDL — by 3.2–4.1 dB on the full array × frequency grid,
  0.5–1.3 dB per antenna. On the full grid no single city beats CDL either
  (best city 2.7–3.5 dB behind); per antenna the best of 14 cities, picked
  after the fact, edges CDL on B and C (0.2 / 0.4 dB).
- **P2 (insurance), as registered** (worst of 8 one-city draws vs worst of 8
  eight-city pools): holds on A and C (8.4 → 6.1, 7.5 → 6.4 dB from the
  ceiling) and per antenna on all three (6.1 → 2.5, 5.2 → 1.8, 4.8 → 1.7);
  fails narrowly on B's full grid (8.2 → 7.3, 0.97 dB) — 5 of 6 checks.
  Pooling protects against the worst city, but around a level below CDL.
- **P3 falsified:** the pool is 5.8–6.8 dB from the ceiling on the full grid
  (1.4–2.0 per antenna), vs ~0.6 in simulation.
- **P4:** with 50 site measurements, CDL + 50 beats pool + 50 by 1.2–1.7 dB
  on the full grid; per antenna both reach the ceiling.
- **How blind this was:** the pilot, committed with the pre-registration,
  had already scored CDL on the same array-A test data (−17.9), and a
  300-per-city dry run touched the test data before the full run (its cached
  channels for hilly scenes were also affected by the bug above). The
  registered script and claims were not changed afterwards.
- **Reading:** the simulation result did not transfer; the caveat written
  before run 1 ("RT prior vs CDL prior for an RT target") applies. The
  simulated channels are cleaner in delay than the measured street (first
  160 ns tap: 89–100% of energy across the 14 cities vs 79–84% measured, CDL
  79%; `runs/dichasus_diagnostics.log`, 1000 channels spread across each set).
  The simulated channels are not more spatially concentrated (top-3 spatial
  eigenvalues 39–46% simulated, 40–51% measured).
  Every ray-traced candidate is biased the same way, so pooling cannot insure
  against it.
- **Placement is not matched to the real users** (fixed before results, not
  tuned): simulated users sit at a median 46 m and ~10° below the array, the
  measured ones at ~18–20 m and 24–26° below (simulated figures from the
  placement rule on flat ground, not from generated positions). Per-array
  downtilts in spec.json are 20.4 / 21.0 / 24.1°; 20.4° was used for all. Either could add to the gap; a
  matched-placement rerun is listed as a follow-up.
- **Smaller asymmetries, bounded:** measured channels sit ~10–20 ns later
  after alignment than simulated ones (a fractional timing offset the
  first-arrival rule can't remove); adding a random 0–20 ns delay to the
  simulated channels moved CDL by −0.26 and the pool by +0.21 dB on array A
  (a reviewer's scratch check, not logged), so at most ~0.5 dB of the
  CDL-vs-pool gap. Observed and unexplained (reviewer's check, not logged):
  the correlation between the two antenna rows is 0.40–0.48 in magnitude
  measured vs ~0.9 simulated (multipath, or calibration error that varies
  across frequency).
- **CDL reproducibility:** the cached CDL set was generated before Sionna's
  own generator was seeded (fixed after review), so it can't be regenerated
  bitwise. Statistically it is equivalent: on array A, 4000 channels score
  −17.9 cached, −17.92 regenerated seeded, −17.95 regenerated unseeded, with
  the same spatial spread (`runs/cdl_seed_check_dichasus.log`). (A reviewer's
  −18.26 for a fresh set — not logged — came from skipping the per-channel
  power normalisation the pipeline applies.) Seeding also leaves the original
  study's CDL prior unchanged (Munich, 2000 channels: −21.58 vs −21.57;
  `scratch/cdl_seed_check.py`, not logged).
- Limits: one site (one street, one building, three arrays on its facade),
  1.272 GHz, mostly line-of-sight; LMMSE only.

### Scattering variant — results (exploratory, 2026-10-27, corrected after review)

`scripts/dichasus_rich.py`, `runs/dichasus_rich.log`,
`runs/a16_sc128_dichasus/dichasus_rich.csv`, control and diagnostics in
`runs/dichasus_diagnostics.log`. 7 cities × 2000 `rtd:` channels (scattering
0.3, diffraction, 32 users per base-station site). Rerun after the review fix
(park_slope regenerated; the shift-tolerant check in
`runs/dichasus_regen_check_rtd_v2.log` reproduces all seven). NMSE at 10 dB:

| array | grid | CDL | rtd pool | CDL + 50 | rtd pool + 50 |
|---|---|---|---|---|---|
| A | array × freq | −17.9 | −17.6 | −18.8 | −18.5 |
| B | array × freq | −17.5 | −17.0 | −18.6 | −18.1 |
| C | array × freq | −17.9 | −17.5 | −19.0 | −19.0 |
| A / B / C | per antenna | −19.3 / −19.3 / −19.3 | −19.2 / −19.6 / −19.7 | −19.9 / −20.0 / −20.2 | same |

- **With scattering the gap mostly closes:** the ray-traced pool goes from
  3.2–4.1 dB behind CDL (full grid, specular only) to 0.3–0.5 dB behind, and
  per antenna from 0.5–1.3 dB behind to −0.0 / +0.3 / +0.3 dB (A slightly
  behind, B and C slightly ahead). With 50 site measurements: 0.1–0.5 dB
  behind on the full grid, level per antenna.
- **Control (array A):** the same 7 cities, 2000 each, specular only, score
  −14.6 (256 users per site) and −14.8 (32 per site, as many sites as rtd)
  on the full grid, vs −17.6 with scattering; per antenna −18.3 / −18.4 vs
  −19.2. So ~0.2 dB comes from the extra base-station sites and ~2.8 dB from
  scattering and diffraction.
- **What scattering changes:** delay spread — first-tap share 89–95% → 84–94%
  across the 7 cities at the same number of sites (`rt32:` → `rtd:`). Spatial
  spread barely changes (top-3 eigenvalues 39–43% → 39–42%); an earlier
  version of this line credited scattering with narrowing it, which came from
  scoring only the first 1000 channels (~4 base-station sites for `rt:`).
- **My first look was wrong:** the Munich-only first-tap check before the run
  (90% → 89%, not logged) suggested little change.
- **Insurance:** on the full grid, pooling 4 cities still shrinks the worst
  case (A 5.5 → 3.3, B 6.3 → 3.6, C 5.6 → 3.3 dB from the ceiling); per
  antenna the drop is under 1 dB (1.8 → 1.1, 1.4 → 0.8, 1.2 → 0.7). (Only 7
  cities, so k = 4, not the registered 8.)
- **Still not a win:** with these solver settings, ray-traced cities roughly
  match CDL on measured channels; they do not beat it. Exploratory: chosen
  after the dry run, one scattering value (0.3, not tuned), 7 cities, one
  site.

## Where next (2026-10-27)

### Matched-placement probe — written before running (exploratory)

The simulated users in the sim-to-real test sat at a median 46 m and ~10°
below the array; the measured ones at ~18–20 m and 24–26°. Probe: `rtdm:`
sources = scattering + diffraction (as `rtd:`) with users 12–30 m from the
base station (median ~23 m, ~20° below on flat ground), same 7 cities × 2000,
same test (`scripts/dichasus_matched.py`). Bar, array A, full grid, 10 dB, no
site data: the matched pool beats CDL by ≥ 0.5 dB → "a site-matched simulator
helps"; within ±0.5 dB → "level"; worse by > 0.5 dB → "geometry was not the
missing piece". Arrays B and C and per antenna reported alongside. Exploratory:
one site, decided after seeing the earlier results.

### Matched-placement probe — results (exploratory, 2026-10-28)

`scripts/dichasus_matched.py`, `runs/dichasus_matched.log`,
`runs/a16_sc128_dichasus/dichasus_matched.csv`; cleanliness in
`runs/dichasus_matched_cleanliness.log`. (The first launch hung on
wall_street: with users only 12–30 m away, a base station facing a wall had no
outdoor spots in range and the placement loop never redrew it; fixed in
96f6bcd, which leaves every earlier set unchanged.) NMSE at 10 dB, no site
data:

| array | grid | CDL | rtd pool (users 10–65 m) | rtdm pool (users 12–30 m) |
|---|---|---|---|---|
| A / B / C | full | −17.9 / −17.5 / −17.9 | −17.6 / −17.0 / −17.5 | −15.0 / −14.4 / −15.0 |
| A / B / C | per antenna | −19.3 / −19.3 / −19.3 | −19.2 / −19.6 / −19.7 | −17.0 / −17.9 / −18.2 |

- **Bar: worse by > 0.5 dB on array A (−2.9 dB vs CDL) → "geometry was not
  the missing piece."** Matching the users' distance made the ray-traced
  prior worse on every array and grid, by 2.4–2.6 dB (full grid) and
  1.5–2.2 dB (per antenna) against the farther
  placement.
- **Why:** closer simulated users are more line-of-sight dominated — first-tap
  share 89–97% vs 85–94% for the farther placement, against 79–84% for the
  real users at those distances. The measured street has delay spread the
  scenes do not produce even with scattering on; plausible sources are objects
  the OpenStreetMap-built scenes lack (vehicles, trees, street furniture,
  facade detail). Inferred, not tested.
- So the 0.3–0.5 dB match of the scattering variant came partly from placing
  users farther than the real ones, which added multipath the scene was
  otherwise missing.

### Directions (2026-10-28)

1. **Tune the simulator from site measurements, then simulate other cities**
   (no 3D model of the target site). The gap is set by global simulator
   settings and missing scene content, not by which cities are pooled. Cheap
   first test: grid over the scattering coefficient {0, 0.15, 0.3, 0.5, 0.7}
   (and optionally a random-clutter model), pick by the 50 site measurements,
   evaluate on the test files; ~1 h of laptop time per 4 values at 7 cities.
   Closest prior work tunes one fully modelled site (Hoydis et al.,
   arXiv:2311.18558; mmDiff, arXiv:2605.26406, not read in full).
2. **More measured sites.** One street can't settle a question about unseen
   sites; no public outdoor dataset with several sites and a planar array was
   found (survey 2026-10-22).
3. **Prediction on measured channels.** Needs ms-spaced moving-user
   measurements: Argos (2 ms) is offline; DICHASUS (~48 ms) and CAEZ (20 ms)
   are too coarse for fast users.
4. **Publish the write-up** (README + docs/paper/paper.pdf) — the repo
   visibility is Krish's call.

### Direction 1: tune the simulator from site measurements — written before running (2026-10-28, exploratory)

Users at the measured distances (12–30 m, as `rtdm:`), scattering + diffraction
on, the same 7 cities × 2000, at five diffuse-scattering coefficients
s ∈ {0, 0.15, 0.3, 0.5, 0.7} (s = 0 keeps diffraction only). For each of the
4 draws of 50 site measurements (the same draws as before), pick the s whose
pooled prior scores best on those 50 (LMMSE NMSE on their own noisy pilots,
0/10/20 dB), then score that prior on the test files.
`scripts/dichasus_tune.py`.

- **Q1 (is there a good setting?):** best s on the test files vs CDL,
  array A, full grid, 10 dB, no site data.
- **Q2 (can 50 measurements find it?):** regret of the selected s vs the best
  s, on the test files.
- **Q3 (the recipe):** selected-s pool, no blend, vs CDL zero-shot; and
  selected-s pool + 50 (blend) vs CDL + 50.
- **Bar (array A, full grid, 10 dB):** the tuned pool (selected s, no blend)
  beats CDL zero-shot by ≥ 0.5 dB → "tuning the simulator is promising";
  within ±0.5 dB → "level"; worse by > 0.5 dB → "a scattering knob is not
  enough". Note the tuned pool uses 50 site measurements (for selection) and
  CDL zero-shot uses none, so the fair comparison for the recipe is also
  reported against CDL + 50. B, C and per antenna alongside.

### Direction 1: related work (2026-10-28; subagent search, abstracts spot-checked)

- **Per-site calibration is the norm, and calibrated values absorb geometry
  errors.** Hoydis et al. (arXiv:2311.18558) learn permittivity, scattering
  coefficient and cross-polarisation by gradient descent against DICHASUS
  *indoor* data (dc01, 3.438 GHz); no transfer to other scenes; per-location
  "neural materials" generalise poorly (full-text claims via the subagent).
  mmDiff (arXiv:2605.26406) calibrates from sparse mmWave measurements and
  generalises to unseen poses in the same scene. Bakirtzis et al.
  (arXiv:2509.19337; abstract checked: differentiable ray tracers "struggle to
  generalize from real-world data at a large scale"); per the subagent's
  full-text reading, per-scene material values did not transfer across 13
  UK areas.
- **Plain ray tracing underpredicts street multipath; clutter is the usual
  fix.** Diffuse scatter ~20% of received power at 28 GHz with separate
  models for buildings, vehicles, foliage (Gentile et al., IEEE TVT 2020; not
  verified); plain RT overpredicts LoS street power by > 20 dB at 300 m until
  clutter scatter is added (Chizhik et al., ~2022; not verified); tree trunks,
  lamp posts and vehicles as simple shapes improve agreement (Göktepe et al.,
  2016); persistent scattering points in Sionna RT match measured delay
  spread better at 5.9 GHz (arXiv:2610.20387; not verified).
- **Standard hybrid:** 3GPP TR 38.901 §8 map-based hybrid model adds random
  clusters (and an optional random-small-object scattering term) to
  deterministic paths; 3GPP notes it is not calibrated.
- **Looks open (medium confidence):** whether a few *global* settings
  (scattering level, clutter density) tuned at one site transfer to other
  cities; calibrating without a site 3D model by matching statistics (first-tap
  share, delay spread, covariance eigen-spread) rather than paths.
- **Next levers if the scattering knob is not enough:** (a) random clutter in
  the scenes (cylinders for poles/trunks, boxes for vehicles along OSM streets,
  density as the one knob); (b) a §8-style hybrid that adds random clusters
  with power and count as knobs; (c) tune by matching statistics rather than
  estimator score.

### Direction 1 — results (exploratory, 2026-10-28)

`scripts/dichasus_tune.py`, `runs/dichasus_tune.log`,
`runs/a16_sc128_dichasus/dichasus_tune.csv`. (Two crashes on the way: the
frequency response hit Dr.Jit's 2^32-element limit with many diffuse paths at
short range; now computed in 64-subcarrier chunks, outputs unchanged —
`scratch/cfr_chunk_check.py`.) NMSE at 10 dB, users at the measured
distances, 7 cities:

| array, grid | CDL | s = 0 | 0.15 | 0.3 | 0.5 | 0.7 | tuned (picked by 50) | tuned + 50 | CDL + 50 |
|---|---|---|---|---|---|---|---|---|---|
| A, full | −17.9 | −14.5 | −14.8 | −15.0 | −15.5 | −16.0 | −16.0 (s 0.7 ×4) | −17.6 | −18.8 |
| B, full | −17.5 | −13.5 | −14.0 | −14.4 | −14.9 | −15.5 | −15.5 (0.7 ×4) | −17.2 | −18.6 |
| C, full | −17.9 | −14.3 | −14.8 | −15.0 | −15.5 | −16.0 | −16.0 (0.7 ×4) | −18.2 | −19.0 |
| A / B / C, per antenna | −19.3 | −17.1 / −18.0 / −18.2 | | | | −17.0 / −17.8 / −18.2 | −17.0 / −17.9 / −18.2 | −19.9 / −20.0 / −20.2 | same |

- **Bar: worse than CDL by > 0.5 dB (array A −1.9 dB) → "a scattering knob
  is not enough."** More diffuse scattering helps steadily on the full grid
  (+1.5 dB from 0 to 0.7) but leaves the pool 1.9–2.0 dB behind CDL on every
  array; with 50 measurements blended in, still 0.8–1.4 dB behind CDL + 50.
- **Selection works but sits at the edge of the grid:** 50 measurements
  picked s = 0.7 in every draw, which is also the best on the test files
  (regret 0). The optimum may lie above 0.7 (the coefficient's maximum is 1);
  extrapolating the trend (~+0.5 dB per 0.2) would not close a ~2 dB gap.
  Not tested.
- **Per antenna (frequency only), scattering does nothing** (within 0.2 dB
  across s; 1.2–2.3 dB behind CDL): what the street has and the scenes lack
  is delay spread at short range, which diffuse scattering off the modelled
  buildings does not add. That points to the levers in the related work —
  objects the map-built scenes lack (vehicles, poles, trees), or 3GPP-style
  random clusters on top of the ray-traced paths.

## Wrap-up (2026-10-28)

Project stopped here by decision. Matching CDL would not be useful (CDL is
free and needs no maps, simulation or tuning), and the one measured site is
not one where CDL fails. The paper (`docs/paper/paper.pdf`) and a plain
version (`docs/PLAIN.md`) are the final write-ups.

## Follow-on proposal: where does CDL fail? (2026-10-28, not started)

**Question.** A ray-traced or site-specific prior is only worth its cost at
sites the 3GPP models describe badly. How large is the CDL-prior gap (CDL
LMMSE vs a site-trained LMMSE) across environments, and which measurable
channel properties predict it?

**Related work (subagent search; abstracts spot-checked where noted).**
- 3GPP parameters miss in specific places: InF (factory) K-factors 5–10 dB
  higher than measured at 10–12 GHz (Runtti et al., arXiv:2503.10046); TDL
  "fails to capture the spatial propagation characteristics" of a commercial
  deployment (arXiv:2603.01843, abstract only); Rel-19 added near-field,
  spatial non-stationarity and variable cluster counts (arXiv:2507.19266).
- Receiver-level evidence that 3GPP training is often nearly enough:
  estimator trained on synthetic data performs "comparable" to others on
  measured data (Baur et al., WSA 2024, arXiv:2312.03450; abstract checked;
  full-text details — Nokia Stuttgart, QuaDRiGa UMi, +6.3% from fine-tuning —
  via the subagent); fine-tuning a TDL-trained receiver on measured macro-cell
  data gains 1.85 dB (arXiv:2410.19565); randomised 38.901 training
  generalises over the air (arXiv:2408.04182). Transfer between stochastic
  and ray-traced data fails both ways for prediction and compression (Morais
  et al., arXiv:2512.12449).
- No measured comparison of MIMO rank/conditioning against 38.901 was found.

**Data (public, same sounder family, so the environment is the main
variable).** DICHASUS: outdoor street (`dxxx`, done); indoor lab room
(`0d5x`, DARUS-2218, 43 GB, 7 files); factory with distributed 4×8 arrays,
LoS + NLoS (`cf0x`, DARUS-2854, 47 GB, 6 files of 3.6–15.6 GB). Also KU
Leuven ultra-dense indoor (64 antennas, URA LoS vs NLoS, same room — a
single-factor contrast; IEEE DataPort, login needed).

**Design (cheapest first).**
1. Run the existing pipeline unchanged (calibration, one processing chain,
   CDL prior, site-trained ceiling, 50-measurement blend) on one or two files
   each of `0d5x` and `cf0x`; report the CDL-prior gap per environment and
   LoS/NLoS. Mostly reuse; ~15–30 GB download. Watch: indoor sets may ship
   reference-transmitter offsets (use them instead of the line-of-sight fit),
   and distributed arrays need per-array processing.
2. Sweep CDL profile (A–E) and delay spread per site: does a better-chosen
   CDL close the gap? If yes, "pick the CDL profile from 50 measurements" is
   a cheap, useful recipe.
3. Per site, measure delay spread, first-tap share, angular spread, K-factor
   and covariance eigen-spread; relate them to the CDL gap. Output: a rule
   for when a site-specific prior would pay off.

**Stop rule written now.** If the CDL gap stays under ~1 dB everywhere
(including NLoS factory), conclude that 3GPP priors are good enough for
LMMSE estimation at these sites and stop.

**Correction (2026-10-28):** the prediction history spans pilot symbols 2 to
53 (slots 0–3), i.e. ~1.8 ms, not the 1.4 ms written in the pre-registration
above; the experiment itself is unchanged. Write-ups corrected.

## Follow-on: where does CDL fail? — pre-registered (2026-10-28, before results)

`scripts/cdl_gap.py`. Same estimator, grid and processing chain as the
measured-channel test (LMMSE, one snapshot, 128 subcarriers around 1.272 GHz,
pilots on every other subcarrier, `wideband.to_grid`, unit power), on three
environments recorded with the same sounder:

| environment | dataset | arrays | fit + ceiling + site | test |
|---|---|---|---|---|
| street, outdoor, mostly LoS | `dxxx` (done) | A, B, C: 2×8, 20° downtilt | d006–d010, d014; site d012 | d011, d013 |
| lab room, indoor, mostly LoS | `0d5x` (DARUS-2218) | 2 arrays, 2×8, no tilt | `0d54` (first 80% in time; site = last 20%) | `0d56` |
| factory, indoor, LoS + NLoS | `cf0x` (DARUS-2854) | 4 arrays, 2×4, no tilt | `cf02` (first 80%; site = last 20%) | `cf06` |

- **Calibration:** per array, the line-of-sight phase + delay fit as before;
  an array counts only if its coherence on the fit file is ≥ 0.5 (in the
  factory, NLoS points lower it).
- **Priors:** generic CDL (A–E, delay spread 30–300 ns, heading within ±75°,
  38.901 element pattern), re-simulated per array geometry (2×8 or 2×4) and
  tilt (20° street, 0° indoors); 12k channels. Profile sweep: each of CDL-A..E
  at delay spreads 30 / 100 / 300 ns (15 priors, 4000 channels each).
- **Measures, 10 dB SNR, full grid and per antenna:** CDL gap = generic-CDL
  NMSE minus site-trained ceiling; CDL + 50 gap; best single profile on the
  test file (oracle) and the profile picked by 50 site measurements (4 draws).
  Channel statistics per array: first-tap energy share, RMS delay spread,
  top-3 spatial eigenvalue share.
- **Claims:**
  - G1: the generic-CDL gap is larger in the factory than on the street
    (full grid, mean over arrays). Falsified if not.
  - G2 (stop rule, as written before): if the generic-CDL gap is < 1 dB in
    every environment (full grid), conclude 3GPP priors are good enough here.
  - G3: a profile picked from 50 site measurements closes at least half of
    the generic-CDL gap, in any environment where that gap is ≥ 1 dB.
  - Reported alongside, no bar: how the gap relates to the channel
    statistics.
- Caveats now: different hardware layouts and arrays per environment; one
  recording for test per environment; the street numbers come from the earlier
  run (same code path for CDL and ceiling).
- **Amendment before results (lab):** `0d5x` positions are 2D (lidar). The
  transmitter height in the dataset frame was set to −0.5 m from calibration
  coherence alone on the first 6000 fit datapoints (near wall: best −0.6 to
  −0.8, 0.76; near window: best −0.4, 0.83; flat within ~0.02 over ±0.3 m),
  i.e. before any estimation result.

### Follow-on — results (2026-10-28)

> **Superseded for the factory (2026-10-29).** The factory row, the factory
> parts of G1–G3, the "both directions" pattern, the post-hoc longer-delay
> check and the conclusion below came from a processing error: the factory
> recording's timing offset fell outside the first-arrival search. See
> "Follow-on — correction: factory timing offset" below. Street and lab
> numbers stand (some rounding corrected there).

`scripts/cdl_gap.py <env>`, `runs/cdl_gap_{street,lab,factory}.log/.csv`,
per-array statistics in `runs/cdl_gap_*_stats.csv`. dB above each array's
site-trained ceiling at 10 dB SNR (lower = closer):

| environment (arrays) | first-tap share | RMS delay | generic CDL, full / per antenna | CDL + 50, full / per antenna | picked profile, full / per antenna |
|---|---|---|---|---|---|
| street (A, B, C; 2×8) | 79–84% | 1.9–2.4 taps | 2.6–2.7 / 0.8–0.9 | 1.4–1.7 / 0.0–0.2 | 4.8–5.6 / 0.3–0.6 (B-100, C-100) |
| lab room (2 arrays; 2×8) | 95–96% | 1.3–1.4 taps | 4.7 / 4.3 | 1.5–2.5 / 0.0 | 2.4–2.5 / 0.4 (A-30, B-30) |
| factory (A–D; 2×4) | 11–12% | 4.4–4.5 taps | 10.2–11.8 / 9.9–10.1 | 4.9–6.3 / 0.1–0.8 | 11.7–12.9 / 7.1–7.4 (B-300, C-300) |

(taps are 160 ns; CDL's own first-tap share is ~79%.) All arrays passed the
coherence bar (lab 0.71 / 0.81, factory 0.85–0.91).

- **G1 supported:** the generic-CDL gap is about 4× larger in the factory
  (10–12 dB) than on the street (2.6–2.7 dB).
- **G2 (stop rule) not triggered:** the gap is ≥ 1 dB in every environment
  on the full grid, and in the lab and factory per antenna too.
- **G3 mixed:** a profile picked from 50 site measurements closed at least half
  of the gap only in the lab per antenna (4.3 → 0.4 dB) and on the lab's
  near-window array (4.7 → 2.4); not on the near-wall array (47%), not on the
  street (a single profile is worse than the generic mixture on the full grid),
  and not in the factory (picked profiles are worse on the full grid; per
  antenna 10 → 7 dB). The 50 measurements picked the test-best profile in 18
  of 24 draws.
- **Pattern:** CDL fails in *both* directions — the lab is far cleaner than
  CDL (95% first tap), the factory far richer (12%, ~700 ns RMS delay, beyond
  the swept 300 ns) — and the gap grows with the distance from CDL's own
  statistics. Blending in 50 measurements (CDL + 50) is the robust fix:
  per antenna it reaches the ceiling everywhere (≤ 0.8 dB); on the full grid
  it still leaves 1.4–6.3 dB, largest in the factory.
- Caveats: different arrays per environment (2×4 in the factory); one test
  recording each; the factory's first-tap share depends on the alignment rule
  (first tap within 10 dB of the peak) in very dense multipath.
- **Post-hoc (after the factory result): longer delay spreads. Withdrawn 2026-10-29 (fitted the corrupted factory channels).** Adding
  CDL-A..E at 600 / 1000 / 2000 ns (`scripts/cdl_gap.py factory
  600,1000,2000`, `runs/cdl_gap_factory_long.log`): the 50 measurements pick
  CDL-B at 1000 ns in every draw, also the best on the test file. It improves
  on the 300 ns profiles but leaves large gaps: 9.3–10.4 dB on the full grid
  (vs 10.2–11.8 generic), ~6 dB per antenna (vs ~10). So the factory's misfit
  is in the shape of its multipath (clusters, angles), not only its length;
  blending in site measurements remains the fix (per antenna ≤ 0.8 dB).

**Follow-on conclusion (superseded 2026-10-29; see the correction below).** CDL as a starting prior is close on an open street
(2.6 dB full grid, 0.8 per antenna), poor in a small clean room (4.7 / 4.3)
and far off in a dense factory hall (10–12 / ~10), failing in both
directions. Choosing a CDL profile from 50 site measurements fixes the clean
room per antenna, but not the factory or the full array anywhere; blending the
50 measurements into the prior is what works everywhere per antenna, and
roughly halves the full-grid gap. So the sites where a better prior would pay
off are the ones least like CDL's typical outdoor channel — dense,
reflective indoor spaces — which is also where map-built ray tracing is
weakest (no machinery, shelving or clutter in the maps).

### Follow-on — correction: factory timing offset (2026-10-29)

**What was wrong.** A review of the follow-on (three independent reviewers, and
a separate check, `scratch/timing_offset_check.py`, `runs/timing_offset_check.log`)
found that the factory recordings (`cf0x`) carry a timing offset, common to all
antennas of a datapoint, that drifts across the whole 1024-tap delay range
between datapoints (about 200 taps per second, a reviewer's estimate from
consecutive datapoints; only ~4% of the energy lies within ±20 taps of zero,
consistent with a near-uniform peak position). `wideband.to_grid` searches the
first arrival only within ±20 taps (±400 ns) of zero and keeps −200 ns to
2 µs around it, so for most factory datapoints the kept window held ~10% of
the energy, mostly leakage and noise, which the unit-power normalisation then
scaled up. The calibration coherence could not show it: it compares antennas
with each other, which cancels a delay they share. Street (strongest tap −7 to
+1 taps, 5th–95th percentile) and lab (−1 to +2) sit inside the search.

**Fix.** `dichasus.measured` first shifts each datapoint, circularly by whole
taps, so its strongest tap sits at delay zero, then runs the unchanged chain;
the chain's own rule then finds the first arrival up to 400 ns before that
peak. `scripts/check_timing.py` checks that the grid channels do not depend on
the offset (it failed before the fix for offsets across the whole range and
passes after). Simulated channels have no such offset; cached channel sets are
unaffected. Effect elsewhere (`scratch/timing_fix_changed.py`,
`runs/timing_fix_rerun.log`): no street datapoint (all nine recordings × arrays A–C) and no lab datapoint changed by more than 1e-6, and the street and lab result tables are identical. A tap within 10 dB of the
strongest more than 20 taps before it, still outside the search, occurs in
0.00% of factory datapoints (`scratch/early_arrival_check.py`).

**Rerun** (`runs/timing_fix_rerun.log`, `runs/cdl_gap_{street,lab,factory}.csv`,
summary `scratch/cdl_gap_summary.py`). dB above each array's site-trained
ceiling at 10 dB SNR, ranges over arrays, full grid / per antenna; street and
lab are unchanged from the first run:

| environment (arrays) | first-tap share | generic CDL | CDL + 50 | picked profile | best profile on test (oracle) |
|---|---|---|---|---|---|
| street (A, B, C; 2×8) | 79–84% | 2.5–2.7 / 0.8–0.9 | 1.4–1.7 / 0.0–0.1 | 4.8–5.6 / 0.3–0.6 | 4.8–5.6 / 0.3–0.6 |
| lab room (2; 2×8) | 95–96% | 4.7 / 4.3 | 1.5–2.5 / 0.0 | 2.3–2.5 / 0.4 | 2.3–2.5 / 0.4 |
| factory (A–D; 2×4) | 81–94% | 2.1–2.8 / 0.7–1.4 | 1.0–2.5 / 0.2–1.1 | 3.5–6.8 / 0.5–1.9 | 3.3–4.5 / 0.3–0.5 |

Claims, re-evaluated on the registered basis (full grid, means over arrays):

- **G1 falsified:** the factory gap (2.44 dB) is not larger than the street's
  (2.62); the same at 0 dB SNR (2.19 vs 2.37). Per antenna it is (1.09 vs
  0.83), but that is not the registered basis.
- **G2 not triggered:** the full-grid gap is at least 1 dB in every
  environment (smallest: factory array B, 2.08).
- **G3 not met in any environment on the full grid:** the picked profile
  closed −98% of the gap on the street, 49% in the lab (near-window 50.3%,
  near-wall 47%) and −107% in the factory (negative: worse than the generic
  mixture). At 0 dB SNR the lab passes narrowly on the full grid (53%); street
  and factory still fail. Per antenna, a secondary reading, it was met only in the lab (90%);
  the factory per-antenna gap (1.09) went to 1.06. On the street the mixture
  beat every single profile on the full grid, so a single picked profile
  could not win there by construction; the same holds in the factory (best
  profile 3.3–4.5 dB vs generic 2.1–2.8). The 50 picked the test-best profile
  in 24/24 draws on the street, 14/16 in the lab and 2/32 in the factory
  (10 dB, both variants).
- The post-hoc longer-delay check (CDL at 600 / 1000 / 2000 ns) is withdrawn:
  it fitted the corrupted factory channels. Its files are removed
  (`cdl_gap_factory_long.*`, in git history).

**Site-block caveat (found on the rerun; cause not established).** Indoors the 50 site measurements
come from one contiguous block of the fit recording, not from across the
environment. On factory arrays A and C that block sits at a median 4.1–4.4 m
from the array, the test recording at 10.7–12.2 m
(`scratch/factory_site_positions.py`, `runs/factory_site_positions.log`). There
the 50 picked a line-of-sight profile (CDL-E) and blending them in helped
little per antenna (A: 0.84 → 0.69 dB) or hurt (C: 0.71 → 1.10). On arrays B
and D (block 10.5–12.8 m, test ~5 m), a mismatch the other way, blending
worked (1.4 → 0.2), so distance alone does not explain it; the per-antenna
picks on A and C were mostly CDL-E (one draw on C picked D-30). At 0 dB SNR
blending hurt per antenna in the factory on average (0.43 → 0.59 dB; C
0.12 → 1.27) and slightly on street array B (0.13 → 0.19).

**Other corrections from the review.**
- The lab recording `0d54` is stored in reverse time order: "first 80% in
  time" above should read "first 80% of the stored datapoints" (the latest
  ~80% in time); the site block is the earliest ~20%. Both are contiguous and
  disjoint; no number changes.
- Rounding: street generic CDL, full grid, is 2.5–2.7 (array C 2.548), not
  2.6–2.7; lab picked profile, full grid, is 2.3–2.5 (near window 2.349).
- The "RMS delay" statistic of `cdl_gap.py` has a leakage and noise floor and
  does not track the true delay spread (CDL-A at 30 ns gives 1.5 taps, at
  100 ns 1.1); the top-3 spatial share is not comparable between 8- and
  16-antenna arrays. Only the first-tap share is used from here on.
- CDL's own first-tap share is ~78% (78.3–78.6% over all 12k channels of the
  generic prior in each of the three array geometries; 78.5% in
  `runs/dichasus_diagnostics.log`), not 79%.
- The generic prior draws delay spreads of 30–300 ns (`CDL_DELAY_SPREAD_S`);
  TR 38.901 (§7.7.3) scales CDL to a scenario's delay spread, with examples
  from 10 to 1000 ns. Part of the lab gap is that mismatch (per antenna; about half on the full grid): a 30 ns profile
  closes it per antenna (4.3 → 0.4 dB).
- The "near wall −0.6 to −0.8 / near window −0.4" transmitter-height sweep
  quoted in the lab amendment had no log; re-run and logged
  (`scratch/lab_tx_height.py`, `runs/timing_fix_rerun.log`): near wall best
  −0.7 / −0.6 m (0.758), near window −0.4 m (0.831), within 0.02 of the best
  over about ±0.3 m: it reproduces the amendment.
- Ratios of dB gaps ("about 4× larger") are dropped; the numbers are given
  instead.
- The proposal's "distributed 4×8 arrays" for `cf0x` is wrong: four 2×4 arrays.

**Corrected conclusion.** At three measured sites, the generic 3GPP CDL prior
sits 2.1–4.7 dB from a site-trained LMMSE on the full array × frequency grid
and 0.7–4.3 dB per antenna. The factory hall (2.1–2.8 / 0.7–1.4) is no worse
than the open street (2.5–2.7 / 0.8–0.9); the outlier is the small lab room
(4.7 / 4.3), whose delay spread sits at the low end of the generic mixture's 30–300 ns,
and a short-delay profile fixes it per antenna. So the pre-registered
expectation that CDL fails in dense indoor spaces is not supported here, and
nothing in these three recordings marks out a kind of site where a
site-specific prior would clearly pay off. Blending 50 site measurements into
the generic prior was the step that helped most, though not on every factory
array, nor per antenna in the factory at 0 dB SNR (at 10 dB SNR: within 0.2 dB of the ceiling per antenna
on the street and in the lab, ≤1.1 dB in the factory, where the indoor site
block may have been unrepresentative for two arrays; on the full grid it
leaves 1.0–2.5 dB). A single CDL profile picked from those measurements is worse than
the generic mixture on the full grid everywhere except the lab. This does not
test a ray-traced prior indoors.

## Check: can the measured data test channel prediction? (2026-10-30)

Question: is the prediction result (finding 3, simulation only) testable on
the DICHASUS recordings? Scripts `scratch/prediction_time_check.py`,
`scratch/prediction_coherence_check.py`; log `runs/prediction_data_check.log`.

- **Sampling is fine in wavelength terms.** Snapshots come every ~48 ms. The
  transmitter moves at ~1.1 m/s on the street and ~0.3 m/s indoors, which is
  0.22 λ (street) and ~0.06 λ (indoor) per snapshot at 1.272 GHz. The
  simulation's 0.5–10 ms horizons at up to 30 m/s and 3.5 GHz span up to
  ~3.5 λ, so a few snapshots ahead covers a comparable distance.
- **The recordings are not phase-coherent from one snapshot to the next.**
  The phase of the correlation between consecutive snapshots is uniform
  (std 1.8 rad, mean resultant ≤ 0.09, street d006 and lab 0d56). In the
  factory (cf02) the drifting timing offset decorrelates consecutive raw
  snapshots completely (|corr| 0.015). Copying the last snapshot gives
  +3 dB NMSE as-is; after removing the best common phase, −3.0 dB at one step
  on the street and −9.7 dB in the lab.
- **Contiguous runs are short.** A gap above 1.5 steps ends a run. On the
  street, 45% of d006's samples sit in runs of ≥40 snapshots (~1.9 s); in the
  lab, none do (median run 13). In the factory, 93% do.

**Conclusion.** The full prediction claim needs absolute phase over time,
which these recordings don't carry, so it cannot be tested here. Only
prediction up to a common phase and delay per snapshot is possible. That is
a different and weaker question, and the tone method's line-of-sight Doppler
is exactly what the common phase removes. Not pursued.
