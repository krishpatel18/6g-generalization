# Do simulated cities help a 5G base station at a site it has never seen?

A base station has to keep estimating the radio channel — how its signal
bounces, fades and shifts on the way to each phone — and estimators are usually
tuned on 3GPP's generic stochastic channel models, which don't resemble any
particular street. Ray tracers like NVIDIA Sionna RT can simulate real city
layouts instead. This project asks whether that helps at a **new** site, and
whether **learned** models are what make the difference.

Everything is simulation: 14 ray-traced city scenes, an 8-antenna base station
at 3.5 GHz, 5G NR-like OFDM. The three main experiments (4-city, 14-city and
prediction) had their expected outcome and pass/fail bar written down before
they ran; follow-up checks are labelled post-hoc in the
[lab notebook](docs/PLAN.md).

## Findings

**1. Pooling many simulated cities is insurance.** With no measurements at a
new site, a linear (LMMSE) estimator built from channel statistics pooled over
many ray-traced cities comes within about 0.6 dB of one trained on that site —
roughly what the 3GPP CDL models reach only after 50 site measurements. Any
single city can be a bad match (up to ~7 dB worse); pooling more cities shrinks
the worst case much faster than the average.

![Gap to a site-trained estimator vs number of pooled cities](docs/figures/breadth.png)

Once 50 site measurements exist, the pooled prior and CDL end up comparable
(pooled ahead by ~0.1 dB on built-up sites). Picking the single best-matching
city from 50 measurements worked with 3 candidates but overfit with 15.

**2. A neural network added little.** A CNN refining the linear estimate gained
at most 0.65 dB. The statistics did the work.

**3. Predicting the channel ahead: a classical method beat the neural
networks.** Predicting 0.5–10 ms ahead from 1.4 ms of noisy history, a simple
per-user method — fit one Doppler tone per angle-delay bin, extrapolate — beat
both CNN predictors at every horizon from 2 ms in all four cities, by 1.8–12
dB with no site data. The pre-registered test for "learning has room here" failed.

![Prediction error vs horizon](docs/figures/prediction.png)

The obvious objection, stated up front: Sionna moves users by rotating the
phases of a fixed set of paths, which is exactly the tone model. A separate
check (re-tracing each user at its moved position) showed this matches the full
simulation up to ~10 ms for nearly all users at 3.5 GHz — but on real channels,
the classical method would degrade by an amount simulation can't measure.

## How it was done

- **Scenes:** Sionna RT's four built-in cities plus ten New York areas built
  from OpenStreetMap with [geo2sigmap](https://github.com/functions-lab/geo2sigmap),
  reading buildings from an offline [Geofabrik](https://download.geofabrik.de)
  extract (the public Overpass API is for interactive use only).
- **Test design:** leave-one-city-out — each city in turn is the unseen site,
  with test users at base-station positions the priors never saw. Results were
  rechecked on freshly drawn sites, training sets and site samples.
- **Estimators:** LMMSE from covariance priors (3GPP CDL, single cities,
  pooled cities), blended with site data by cross-validated shrinkage; a
  residual CNN on top; for prediction, the same plus the per-user tone method
  and CNNs in both the antenna and angle-delay domains.
- **Compute:** one laptop (Apple M4, Sionna on Metal). No cloud.

## Limits

- Simulation-to-simulation only: test channels come from the same ray tracer as
  the priors. Nothing here is tested on measured channels.
- One linear estimator family, one array (8-antenna ULA), one band (3.5 GHz);
  10 of 14 cities from one US state.
- Most randomly placed users receive no ray-traced paths and are dropped, so
  the remaining users skew toward simpler, line-of-sight-like links.
- The CNNs used one random seed. The prediction CNNs were trained on the whole
  track, which under-weights the near horizons where prediction is possible.
- Sionna RT's path solver is non-deterministic by default; the prediction
  re-trace check was rerun with `deterministic=True` and its conclusion held.

## Related work

Closest found: site-specific ray-traced or digital-twin priors for a channel
covariance, subspace or channel statistics
([2607.09334](https://arxiv.org/abs/2607.09334),
[2501.02758](https://arxiv.org/abs/2501.02758),
[2411.13360](https://arxiv.org/abs/2411.13360)); pre-training across source
cities and adapting to a target site, for CSI feedback
([2605.16141](https://arxiv.org/abs/2605.16141));
stochastic vs ray-traced data for ML, including prediction
([2512.12449](https://arxiv.org/abs/2512.12449)); learned prediction beating a
classical Prony angle-delay baseline on 3GPP-compliant QuaDRiGa channels
([2412.08908](https://arxiv.org/abs/2412.08908)); neural network predictors
compared on the 3GPP tapped-delay-line model
([2308.14020](https://arxiv.org/abs/2308.14020)). A search did not find the
pooled-city worst-case result or a classical predictor beating learned ones on
ray-traced channels; that is a search, not a full review.

## Reproduce

Python ≥ 3.12 with [uv](https://docs.astral.sh/uv/): `uv sync`. Every script
takes the base-station antenna count (`8`). Channel sets are generated on first
use and cached under `data/` (~34 GB for everything); raw logs and result
tables are in `runs/`.

| result | command |
|---|---|
| 4-city leave-one-out | `scripts/make_data.py 8`, `scripts/loco.py 8 0`, `scripts/report.py runs/a8_sc128/loco_seed0.csv` |
| fresh-site recheck | `scripts/fresh_sites.py 8` |
| New York scenes | `scripts/build_library.sh <new-york-latest.osm.pbf>` (needs `osmium-tool`, `libusb`), then `scripts/make_library.py 8` |
| 14-city library | `scripts/library.py 8`, `scripts/library_followup.py 8`, `scripts/pool_draws.py 8` |
| prediction | `scripts/make_tracks.py 8`, `scripts/predict.py 8 0` |
| figures | `scripts/figures.py` |

A results summary with every number is in [docs/RESULT.md](docs/RESULT.md).

Built with [Sionna](https://github.com/NVlabs/sionna). Map data © OpenStreetMap
contributors.
