# Ray-traced city priors for channel estimation: what the pilot found

Readable summary of the pilot. The lab notebook, with every pre-registration
and raw result in order, is `docs/PLAN.md`.

## The answer, in two parts

**On measured channels (one site), the simulation result did not hold.** With
no site data, the pooled ray-traced prior was 3.2–4.1 dB worse than 3GPP CDL
on the full array (0.5–1.3 dB per antenna); an exploratory rerun with diffuse
scattering and diffraction in the ray tracer brought it within 0.3–0.5 dB of
CDL — a match, not a win. See "Sim-to-real" below.

**In simulation (ray-traced test sites),** for a new site with **no channel
measurements**, a covariance prior pooled from
many ray-traced cities sits about 0.6 dB from an estimator trained on that site,
which is about what the standard 3GPP CDL prior reaches once it is blended with 50 site
measurements (0.47–0.73 dB across four draws of those 50). Pooling many
cities also removes the multi-dB failures you get from relying on one
mismatched city. Once 50 site measurements are available, the pooled prior and
CDL are comparable on built-up sites (pooled ahead by ~0.1 dB on average). In
everyday terms: **a pooled ray-traced prior is insurance that saves the first
~50 site measurements; it is not an accuracy win once you have them.**

Limits of the simulation part: simulation-to-simulation (the test channels come
from the same ray tracer as the priors); a linear estimator (LMMSE, built from channel
covariance); one array (8-antenna ULA); 3.5 GHz; 14 cities, 10 of them in New
York State. "Gap to ceiling" means NMSE at 10 dB SNR minus that of an LMMSE
fitted on 12k channels from the target itself. The ceiling is itself an
estimate, so gaps under ~0.1 dB are treated as zero.

## The question as posed, and what changed

The starting hypothesis was that a *learned* channel estimator pre-trained on
many ray-traced (RT) cities generalizes to an unseen city better than one
pre-trained on 3GPP stochastic models (CDL). The learning part did not hold up:
a CNN refiner on top of LMMSE added ~0.2–0.45 dB on the CDL arm,
at most 0.65 dB in any cell, and often nothing on the RT arms (one CNN seed only). The linear estimator did almost all of the
work, so the result is about **which channel statistics to start from**, not
about learning.

Two fixes were needed along the way: the single-antenna setup could not tell the
priors apart (moved to 8 antennas), and the first CDL arm had fixed arrival
angles that made it look worse than it is (fixed with a random base-station
heading per sample; the earlier run is kept but superseded).

## What held on every repeat

| finding | numbers | checked on | registered |
|---|---|---|---|
| RT prior beats CDL with no site data, except Paris-Étoile | +0.3–0.5 (Munich), +1.1–1.2 (Florence), +1.3–1.5 dB (San Francisco); −1.2 to −1.8 dB (Étoile) | original sites, new base-station sites, new pre-training draws | pre-registered |
| Pooling cities is insurance | gap to ceiling, worst of 8 random pools: 3.99 dB with 1 city → 0.82 dB with 8; mean 1.59 → 0.60 | 14 targets × 8 draws | pre-registered |
| Pooled RT prior beats CDL, no site data | better in 11/14 targets, by 0.29 dB on average; worst loss 0.4 dB | 14 targets | pre-registered |
| Pooled RT, no site data ≈ CDL + 50 site measurements | 0.59 vs 0.47–0.73 dB gap to ceiling | four 50-measurement draws | post-hoc |

Pooling has a price: the best single source city per target (which you can't
know in advance) is 0.13 dB from the ceiling on average, the pool 0.59, so the
insurance costs ~0.45 dB. The worst single-source cases (6–7 dB) all use the
nearly empty open-area scene (`rochester_brighton`) as the only source for a
dense downtown.

## What did not hold

- **Breadth at 4 cities.** Three cities vs one city differed by +0.15 to +0.5
  dB, but per-city results move by ~1 dB between site draws: inconclusive. The
  14-city library is what showed the insurance effect.
- **Choosing a prior from 50 site measurements.** With 3 candidates it chose
  well (6/8 exact, mean regret 0.05 dB, always CDL for Étoile). With 15 it
  overfit: mean regret 0.48 dB, worst 2.0 dB (pre-registered bar ≤ 0.3: not
  met). Scoring at the evaluation SNR didn't help (0.71 dB); 200 measurements
  helped (0.44) but still lost to pooling and blending. *(Post-hoc diagnostics.)*
- **A clear edge over CDL once 50 measurements are blended in.** Pool + 50 is
  0.33–0.62 dB from the ceiling depending on which 50 measurements are taken; it
  beats CDL + 50 in 10–12 of 14 targets, by ~0.1 dB on built-up sites. On the
  open-area scene the edge was 2.9 dB in one draw and 0–0.5 dB in the other
  three. *(Post-hoc.)*

## Recommendation

On the one measured site tested, start from CDL, not from ray-traced cities
at default solver settings, then blend in site measurements as they arrive
(5-fold cross-validated shrinkage). In one exploratory rerun, switching on
diffuse scattering and diffraction in the ray tracer brought ray-traced priors
level with CDL there, not ahead. In simulation, pooling many cities is still the safer choice than any
single city, and picking one best source city from a few measurements
overfits when there are many candidates.

Novelty against the literature is **unverified** and needs a proper
related-work pass before any claim is made.

## Follow-up: predicting the channel ahead (falsified)

The same question for prediction (0.5–10 ms ahead from 1.4 ms of noisy
history): does a learned predictor beat classical ones? **No, on this
simulator.** A simple classical method (one Doppler tone per angle-delay bin,
fitted per user, no training data) beat both learned models at every horizon
of 2 ms and more in every city, by 1.8–4.7 dB, and ~9–12 dB in San Francisco.
The learned models did beat covariance-based prediction by 2–6 dB at 2 ms, and
learning is competitive only one slot (0.5 ms) ahead, mainly for fast users.
The ray-traced prior beats CDL only at the shortest horizons (0.2–1.5 dB at
0.5 ms). Caveat that matters most: the simulator moves users by rotating the
phases of a fixed set of paths, which is exactly the tone model; on real
channels it would do worse, by an amount this simulator cannot measure.
Pre-registered before results (with one amendment, also before results); see
`docs/PLAN.md`, "Prediction experiment".

Post-hoc follow-up (2026-10-27): a CNN trained to correct the tone
prediction adds ~0.3–0.6 dB at 2–10 ms in Munich, Étoile and Florence (cells
0.26–0.65; two seeds agree within ~0.2 dB) and costs 0.8–1.2 dB at 5–10 ms in
San Francisco.
Re-weighting the original CNNs' loss toward the predictable horizons does not
help (still ~2–13 dB behind tone). See `docs/PLAN.md`, "Fairer learned
predictors".

## Sim-to-real: the pooled-prior result on measured channels (falsified)

On the DICHASUS outdoor measurements (Univ. Stuttgart; one building, three
2×8 arrays, a transmitter on the street, 1.272 GHz), with all 14 cities
re-simulated in the same setup and every source through one processing
chain, the pooled ray-traced prior is **worse than 3GPP CDL** with no site
data: by 3.2–4.1 dB on the full array × frequency grid and 0.5–1.3 dB per
antenna, on all three arrays. With 50 site measurements CDL still leads by
1.2–1.7 dB (full grid); per antenna both reach the site-trained level.
Pooling still shrinks the worst single-city case (5 of 6 checks), but around a
level below CDL. The simulated channels are cleaner in delay than the measured
street (first 160 ns tap: 89–100% of energy across the 14 cities vs 79–84%
measured). The simulated users were also placed farther away and at a
shallower angle than the real ones (median 46 m / 10° below the array by the
placement rule on flat ground, vs 18–20 m / 24–26° measured). Pre-registered, though not blind: a pilot had already
scored CDL on the test data, and a dry run preceded the full run. See
`docs/PLAN.md`, "Sim-to-real".

Exploratory follow-up (added after a dry run): with diffuse scattering
(coefficient 0.3) and diffraction on, 7 cities at 2000 channels, the pooled
ray-traced prior comes within 0.3–0.5 dB of CDL on the full grid and ties or
slightly beats it per antenna (−0.0 to +0.3 dB). On array A, the same 7 cities
specular-only score 2.8 dB (full grid) and 0.9 dB (per antenna) worse than
with scattering at the same number of base-station sites; the extra sites
alone account for ~0.2 dB. This suggests the solver settings drove most of
the gap; it is one site and one scattering value. A second exploratory probe
placed the simulated users at the real users' distances (12–30 m instead of
10–65 m): the gap reopened to 2.9 dB behind CDL on array A, because closer
simulated users are more line-of-sight dominated (first-tap share 89–97%)
than real ones at the same distance (79–84%). See `docs/PLAN.md`,
"Matched-placement probe".

## Reproduce

All scripts take the base-station antenna count as their first argument (`8`).
Channels are cached under `data/a8_sc128/`; seed 1 = 12k pre-training, seed 2 =
1k site pool, seed 4 = 2k test (fresh sites), seed 5 = redrawn pre-training.

| result | command | output |
|---|---|---|
| leave-one-city-out, 4 cities | `scripts/make_data.py 8`, `scripts/loco.py 8 0`, then `scripts/report.py runs/a8_sc128/loco_seed0.csv` | `runs/a8_sc128/loco_seed0.csv` |
| fresh sites | `scripts/fresh_sites.py 8` | `runs/a8_sc128/fresh_sites.csv` |
| prior selection, 3 candidates | `scripts/select_prior.py 8` | `runs/select_prior_a8.log` |
| New York scenes | `scripts/build_library.sh <new-york-latest.osm.pbf>` (Geofabrik), then `scripts/make_library.py 8` | `data/scenes/` |
| library, 14 cities | `scripts/library.py 8` | `runs/a8_sc128/library.csv`, `runs/library_a8.log` |
| follow-up (+50 blends, selection) | `scripts/library_followup.py 8` | `runs/library_followup_a8.log` |
| other site-pool draws | `scripts/pool_draws.py 8` | `runs/pool_draws_a8.log` |
| prediction | `scripts/make_tracks.py 8`, then `scripts/predict.py 8 0`; follow-up `scripts/predict_fair.py 8 0` | `runs/a8_sc32_track/predict_seed0.csv`, `predict_fair_seed*.csv` |
| measured channels | `scripts/dichasus_convert.py` (TensorFlow) on DaRUS files, then `scripts/dichasus_test.py`; exploratory `scripts/dichasus_rich.py`, `scripts/dichasus_diagnostics.py` | `runs/a16_sc128_dichasus/dichasus_test.csv`, `dichasus_rich.csv`, `runs/dichasus_diagnostics.log` |

Stack: Sionna 2.2 / Sionna RT 2.2 on Apple M4 (Metal), scenes from geo2sigmap
(pinned commit, buildings read from a local OpenStreetMap extract).
