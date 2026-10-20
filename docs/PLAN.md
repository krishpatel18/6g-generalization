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

**Current one-line result:** for a new site, a ray-traced prior pooled from
many cities and blended with a few dozen site measurements gets within ~0.3 dB
of a site-trained estimator, beats the 3GPP-CDL equivalent by ~0.4 dB, and
avoids the multi-dB failures of any single mismatched source.
(Simulation-to-simulation, LMMSE, 8-antenna ULA, 3.5 GHz, one seed of site draws.)
