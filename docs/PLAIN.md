# The project in plain words

*A readable companion to the paper (`docs/paper/paper.pdf`). No background
needed.*

## The problem

A cell tower has to keep working out how its signal reaches each phone: what it
bounces off, how it fades, how it arrives at each of the tower's antennas.
It does this from short known signals the phone sends (pilots), and it does
much better if it already knows roughly what the "radio landscape" at that
spot looks like.

At a brand-new site it doesn't know that yet. The industry's usual answer is a
set of generic textbook models (3GPP "CDL" models), which describe typical
channels but no particular street. As the tower collects real measurements, it
adjusts.

The idea we tested: computer simulations of real city layouts (ray tracing)
might give a new tower a better starting point than the textbook models. And,
as the original pitch had it, maybe AI (neural networks) is what makes the
difference.

## How we tested it

- Built 14 simulated cities: four that come with NVIDIA's Sionna ray tracer and
  ten New York areas made from free OpenStreetMap data.
- Each city took a turn playing "the new site", with the starting point built
  only from the other cities.
- Before each main experiment, we wrote down what result would count as a
  success or a failure, so we couldn't move the goalposts afterwards.
- Then we checked the main finding on real radio measurements from the
  University of Stuttgart (three antenna panels on a building, a transmitter
  pulled along the street in front).

How to read the numbers: results are in decibels (dB) of error. Smaller error
is better, and **every 3 dB is a factor of two** in error. So "1 dB better" is
a modest gain; "3 dB worse" means twice the error.

## What we found

**1. In simulation, mixing many cities is good insurance.** With no
measurements at the new site, a starting point averaged from many simulated
cities came within about 0.6 dB of the best possible, as good as the textbook
models after 50 real measurements. Relying on a single city could go badly
wrong; mixing many protected against that.

**2. AI added little.** A neural network on top of the standard method gained
at most 0.65 dB. For predicting how the channel will change a few milliseconds
ahead, where AI should have the most room, a simple classical method beat the
neural networks by about 2–12 dB. A network that *corrects* the classical method's
guess added a small, repeatable 0.3–0.6 dB.

**3. On real measurements, the result flipped.** At the Stuttgart site, the
simulated-cities starting point was 3–4 dB *worse* than the free textbook
models — two to two-and-a-half times the error.

**4. Why: the simulated streets are too clean.** Real streets bounce signals
off cars, trees, lamp posts, and rough walls, creating many weak echoes. The
map-based simulations have only the buildings, so their channels look tidier
than reality. We tried three fixes:

| what we tried | result vs textbook models |
|---|---|
| turn on rough-surface scattering in the simulator | almost level (0.3–0.5 dB behind) — but only because the simulated users stood too far away |
| put simulated users at the real distances | 2.9 dB behind again |
| let 50 real measurements choose the best scattering setting | still ~2 dB behind |

So the missing ingredient is stuff the maps don't contain, not a setting we
can tune.

## What it means

- For a new tower, the free textbook models are still the better default than
  simulated cities, at least at this kind of site.
- Even if a better simulator only *matched* the textbook models, nothing would
  be gained: the textbook models are free and need no maps or simulation.
  Simulation is only worth it where it clearly beats them.
- The honest contribution is the reversal and its explanation: a result that
  looked good in simulation didn't survive real data, and we found out why.

## How careful this was

- Every main experiment had its pass/fail rule written down first; extra
  checks afterwards are labelled as such.
- Five independent AI reviewers checked the code and every number against the
  logs, three times over. They caught real mistakes (one bug in a cache,
  several over-stated sentences), all fixed and disclosed.
- Limits: one real site, one frequency, a mostly open street. The real-data
  test wasn't fully "blind" (a small pilot had already looked at the test
  data). A few small figures come from quick checks that weren't saved as
  logs; the notebook labels them.

## Where you could go next

The question this project leaves is: **where do the textbook models fail?**
Simulation (or anything site-specific) only pays off at sites they describe
badly. Two things from the literature make this a good next question:

- Other measurement studies also found the textbook models are often nearly
  enough. One from Nokia measurements in Stuttgart found an estimator trained
  only on synthetic 3GPP-style data was close to the best on real data.
- Where they do miss is specific: factories (the textbook models assume a
  much stronger direct signal than measured), and spatial detail in some
  deployments.

The same Stuttgart group has published measurements from an indoor lab and a
factory (with blocked line of sight), recorded with the same equipment. Our
whole pipeline could be rerun on them almost unchanged, to measure how far the
textbook models fall short in each environment, and whether simply picking
the best-fitting textbook profile from 50 measurements closes the gap.

Rough cost: 15–30 GB of downloads, a day or two of work, laptop only. If the
textbook models are within ~1 dB everywhere, the honest conclusion is "they're
good enough here" and that's the end of it. The full plan is in
`docs/PLAN.md` under "Follow-on proposal".

## Follow-on result: where do the textbook models fall short? (done, corrected)

We reran the whole pipeline on two more real sites recorded with the same
equipment in Stuttgart, and measured how far the free textbook models are
from the best possible at each (gaps on the full set of antennas and
frequencies; smaller is better):

| site | textbook models' gap |
|---|---|
| open street | 2.5–2.7 dB |
| factory hall | 2.1–2.8 dB |
| small lab room | 4.7 dB |

- **The factory is no worse than the street.** We had expected the textbook
  models to do much worse in a dense indoor space. They don't, here.
- **The lab room is the outlier**, and for a fixable reason: almost all of its
  signal arrives at once, faster than the range of echo lengths our version
  of the textbook models used. The textbook itself allows shorter settings,
  and picking a short one fixes most of the lab gap for each antenna on
  its own (about half of it across the whole panel).
- **What helps most is mixing a few real measurements into the starting
  point.** The exception: in the factory, on two of the four antenna panels,
  mixing them in helped little or made it slightly worse. On those two, the
  50 measurements came from close to the panel while the test users were
  farther away, but on the other two the mismatch ran the other way and it
  still worked, so we don't know the cause. In the factory, when the signal was
  very noisy, it made the per-antenna result slightly worse on average.

**So:** in these three places, nothing marks out a kind of site where a
custom starting point would clearly pay off. Finding one would take more
sites, recorded more uniformly.

**A correction.** A first version of this result said the factory was 10–12 dB
off ("about 10× the error"). That was our mistake: the factory recordings'
timing drifts, so the signal sat in a different place in each measurement, and
our processing looked for it in a fixed spot and mostly kept noise. Reviewers
caught it; we fixed the processing, checked that the street and lab results
did not change, and reran the factory. The numbers above are the corrected
ones.

## Last trial: tuning the simulator to real measurements (done, negative)

NVIDIA's ray tracer can run backwards: given real measurements, it adjusts the
building materials in a 3D model until the simulation matches them. We did
that on one end of a measured hallway in Stuttgart and tested the tuned model
at the other end, where nothing was measured.

- **The tuning mostly worked on what it was tuned for:** echo timing matched
  as well as in NVIDIA's own paper, and signal strength did in the corridor
  (not in the lobby, where the 3D model is too simple).
- **Judged one antenna at a time, the tuned model was the best starting point**
  we had without local measurements, by a small margin.
- **Across all 32 antennas together it lost** to the textbook models by
  4.5 dB. The simple 3D model doesn't reproduce how the signal differs from
  antenna to antenna, and that pattern is what matters most.
- **Fifty measurements from the tested end itself** got almost all the way
  to the best possible.

**So the project ends here.** In every test on real data, simulating the site
didn't beat the free textbook models. Measuring the site did.

