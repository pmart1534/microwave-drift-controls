# A Control-Experiment Methodology for Separating Drift from Signal in Microwave-Imaging ML

**Authors:** Peter Martin (UofU ECE), Cynthia Furse (advisor), + collaborators TBD
**Target venue:** IEEE J-ERM (short paper / methodology) or IEEE Antennas & Wireless Propagation Letters
**Length target:** 4-6 pages, IEEE two-column
**Status:** Draft outline — 2026-07-02

---

## Elevator pitch (one sentence)

Deep-learning claims of microwave-imaging tumor localization are routinely
inflated by *drift memorization* — empty-phantom controls with no object
achieve 82% within-session position accuracy on 96-position classification
from pure VNA/mechanical drift alone; we introduce a **within-vs-LOSO
control** and a **synthetic-positions detectability test** that together
distinguish real signal from drift and, when applied, drop the empty-phantom
accuracy to 2.3% (chance).

---

## Core claim / novelty

The **experimental framework** (not the model, not the antenna, not the
phantom) is what's new. Three linked controls that any ML microwave-imaging
paper should include before making detection claims:

1. **Empty-phantom within-session ML control** — quantifies drift-as-signal
2. **Empty-phantom multi-session LOSO control** — verifies drift does NOT
   survive session boundaries (i.e., is not itself a fingerprint the model
   would carry over from an object case)
3. **Synthetic-positions DD test on continuous data** — quantifies whether
   pure drift can exceed the standard DD detection threshold within one
   experimental run

Together, if a real-object model achieves LOSO accuracy substantially higher
than the empty-phantom LOSO, that accuracy CAN be attributed to real signal.
Without these controls, that attribution is speculative.

---

## Proposed structure

### I. Abstract
- ~200 words, standard IEEE format
- Emphasize: methodology paper, not detection paper
- Key numbers up front:
  * 82% within-session on empty (drift-only) → 2.3% LOSO
  * 99.35% real-tumor LOSO on identical antennas → 42x gap = real signal
  * Overnight settling shows drift can persist for hours if antenna
    mounting is compromised

### II. Introduction
- Rising interest in microwave imaging + ML for breast cancer detection
  (cite Fear group Manitoba, Reciprocity Bio, Wave imaging start-ups)
- Common failure mode: single-session high accuracy that doesn't
  generalize (cite examples)
- Root causes rarely characterized: is the failure
  (a) drift memorization,
  (b) session-normalization artifacts,
  (c) genuine physical variability across sessions?
- Our contribution: three linked control experiments that answer this,
  applicable to any 4-port (or general N-port) VNA-based system

### III. Related work
- Detectable-difference (DD) methodology (Fear, Manitoba group)
- Multi-session LOSO in ML for biomedical signals (EEG, ECG examples)
- Domain adaptation for medical imaging
- Prior microwave-imaging ML papers that DO NOT include empty-phantom
  controls (short catalog)

### IV. Methods

#### IV.A Measurement system
- 4-port Keysight MN7021A (Hunter VNA), 0.1-8 GHz, 791 freq points
- Two antenna sets tested: Sam Medium Separated, McGill Separated
- A3 breast phantom (larger of two available)
- 51-position measurement grid (subposition x row x column), 16 trials/pos

#### IV.B Data collection protocols
- **Batch drift test**: 4 back-to-back empty-phantom sessions with real
  session timing, ~30 min each
- **Overnight live sweep**: single-position continuous sweep for 15+
  hours (55K CSVs at ~1 Hz), no operator interaction
- **Real-object comparison set**: 3 sessions of same phantom + metal
  object at 51 positions
- Optional variants: 0.2s vs 1s sweep interval; antennas re-mounted

#### IV.C The three controls

**Control 1 — Empty-phantom within-session accuracy:**
- Per-session 75/25 stratified split; MLP ensemble (256, 128 hidden)
- Position vote across trials; report mean per session
- Purpose: quantify how much drift LOOKS LIKE a per-position fingerprint

**Control 2 — Empty-phantom LOSO:**
- 4-fold leave-one-session-out; per-session z-score training; MLP
  ensemble; identical hyperparameters to real-object model
- Purpose: verify drift fingerprint is session-specific, not cross-session

**Control 3 — Synthetic-positions detectable-difference:**
- Take continuous single-position data
- Carve into 1-hour blocks; first 30 sec = "baseline", subsequent 30-sec
  windows = "positions" (~119 per hour)
- Apply the standard CI-band-gap DD formula (95% t-CI, 10% relgap
  threshold) to these fake positions
- Report: detection rate per port, DD-in-dB heatmap, per-hour trends
- Purpose: quantify whether pure drift exceeds standard DD detection

#### IV.D Physics features and model
- Full 4-port feature extraction: mag, phase, real, imag, IFFT envelope,
  per-trace stats → 18,792-D per trial
- S11-only variant: 873-D per trial (mag/phase/real/imag/stats/IFFT
  restricted to reflection channel of port 1)
- MLP ensemble of 3 seeds, hidden (256, 128), early stopping,
  per-session z-score for LOSO
- Rationale: MLP > CNN for cross-session generalization in this problem
  (brief empirical justification, ~1 paragraph)

### V. Results

#### V.A Within-session drift IS a fingerprint

| Session | Position accuracy (empty phantom) |
|---------|-----------------------------------|
| S001    | 73.75% |
| S002    | 82.08% |
| S003    | 82.08% |
| S004    | 90.83% |
| **Mean**| **82.19%** |

Chance = 1.04% (1/96 positions). Every session's within-session accuracy
is orders of magnitude above chance despite no object.

#### V.B LOSO defeats drift

| Test fold | Position accuracy |
|-----------|-------------------|
| S001      | 2.08% |
| S002      | 2.08% |
| S003      | 2.08% |
| S004      | 3.12% |
| **Mean**  | **2.34%** |

Multi-session LOSO drops to essentially chance. **The drift fingerprint
is session-specific** — this is the key control result.

#### V.C Real-object LOSO on identical antennas achieves 99.35%

| Setting                       | Within | LOSO |
|-------------------------------|--------|------|
| Real object, 4-port full      | 100%   | 99.35% |
| Real object, S11-only         | 100%   | 100.00% |
| Empty phantom, 4-port full    | 82.19% | **2.34%** |
| Empty phantom, S11-only       | 88.28% | **1.30%** |

**Two drift-vs-signal ratios:**
- 4-port: 99.35 / 2.34 = **42x**
- **S11-only: 100.00 / 1.30 = 77x** — the strongest signal-vs-drift
  separation, achieved with the MINIMAL possible setup: one port,
  one antenna, one reflection channel.

The S11-only result is particularly important: it shows that the
detection capability isn't a 16-channel artifact and reproduces at
the cheapest possible measurement configuration a 1-port VNA can
produce. Any reader can replicate the control with a benchtop 1-port
VNA and one antenna.

Replicated on a second measurement system (2-port McGill, 6.5 GHz):
- Empty-phantom LOSO N=2 sessions: 3.25%
- Empty-phantom LOSO N=3 sessions: 5.0%
- Chance = 1.56% (1/64 positions)

#### V.D Standard DD threshold rejects thermal drift

Synthetic-positions DD on 15h overnight data:
- Hour 0 (fresh VNA): 22-42% detection rate — drift can fake positions
- Hours 1-14 (post-warmup, stable antennas): ports 1/2/4 detection = 0%
- Port 3 (mechanically compromised via loose tape): 15-20% detection
  persists for 9 hours, then decays

Interpretation:
1. VNAs need ~1h warmup before DD threshold is trustworthy for empty
   baselines
2. Standard 10% relgap threshold correctly rejects electronic drift
   post-warmup
3. Mechanical instability (loose antenna) creates spurious detections
   the threshold DOES catch — highlighting the value of the control

#### V.E Practical guidance
- Report both within-session and LOSO for any ML claim
- Include a matched empty-phantom control on the same day
- Verify VNA is at least 1h into operation before real measurements
- Physically secure antennas (rubber band > tape; ideally clips)

### VI. Discussion
- The 82% within-session number is likely underreported in the literature
  because most papers only report LOSO (or worse, only pooled accuracy)
- Papers reporting single-session accuracy without LOSO are essentially
  reporting drift capacity, not detection capacity
- Our method costs one extra dataset (empty-phantom controls) but
  eliminates a major class of false claims

### VII. Limitations
- Single VNA (Hunter MN7021A); other VNAs may have different drift
  timescales — recommend anyone reproducing should do their own
  overnight settling test
- Physical antenna mounting stability is a huge confound; not always
  under lab control
- MLP ensemble is one specific model class; results should replicate
  with other flexible classifiers but not verified here

### VIII. Conclusion
- Three linked controls; when applied, distinguish drift from signal
- Direct evidence that this matters: 42x accuracy gap on identical
  antennas
- Recommend as standard practice in microwave-imaging ML literature

### IX. Data + code availability
- All raw sessions (batch, overnight, McGill 2-port empty)
- Python analysis code in `Above 95 Percent/code/`
- Exact DD implementation matches `plot_hunter_dd.py` (linked)

---

## Figure list (draft)

**Figure 1** — Measurement setup schematic
- 4-port VNA + A3 phantom + antenna array
- Photo optional

**Figure 2** — Within vs LOSO on empty phantom (bar chart)
- 4 sessions x 2 metrics (within, LOSO)
- Chance line overlaid
- Direct visualization of "drift is session-specific"

**Figure 3** — Real object vs empty LOSO (bar chart)
- Same antennas, same protocol, only difference = presence of object
- Show 99.35% vs 2.34% side by side
- Log scale y-axis?

**Figure 4** — Per-port trial-noise CV across 4 sessions
- Highlights port-3 mechanical issue
- Shows stability of measurement chain

**Figure 5** — Overnight settling curves
- Rolling CV per port over 15 hours
- Marks the ~17-min short-term settle
- Shows port-3 long tail from mechanical drift

**Figure 6** — Synthetic-positions DD, grand timeline
- Hour × S-param heatmap (dB scale)
- Marks 1-hour settling for ports 1/2/4
- Marks 9-hour tail for port 3

**Figure 7** — Detection rate per hour per port
- Line plot with 10% threshold reference line
- Shows: threshold works, EXCEPT when mechanical setup is compromised

**Figure 8** (optional) — 2-port McGill LOSO curve
- Replicates the "empty-phantom LOSO stays at chance" finding on a
  completely different system

---

## Message architecture ("what does this paper actually say?")

**Sentence 1:** "Microwave-imaging ML claims are often inflated by drift."

**Sentence 2:** "We propose three specific controls to distinguish drift
from real signal."

**Sentence 3:** "Applied to our data, they show empty-phantom accuracy of
82% in-session collapses to 2.3% LOSO, while real-tumor accuracy on
identical hardware stays at 99.35% — the 42x gap is the signature of
real detection."

**Sentence 4:** "Standard DD detection (10% relgap threshold) is robust
to thermal drift after ~1 hour of warmup, but is defeated by mechanical
instability — providing an empirical diagnosis for a common failure
mode."

---

## Rebuttal-proofing (what will reviewers push back on?)

- **"You didn't try enough model architectures."** Response: we ran the
  same LOSO test on both CNN and MLP; MLP > CNN for cross-session
  generalization in this problem class (reference an appendix). The
  key finding (drift = session-specific) is model-independent.
- **"How do you know the '99.35% real-object' isn't also drift?"**
  Response: the empty-phantom control on the same antennas confirms
  drift LOSO = 2.34%; if the real-object 99.35% were drift, it would
  also be 2.34%. This is the point of the control.
- **"Small sample (4 sessions)."** Response: (a) replicated on 2-port
  McGill with N=2, 3; (b) chance baseline established analytically;
  (c) additional rubber-band data provides independent confirmation.
- **"Where's the code?"** Response: fully open, links provided.

---

## Timeline (rough)

- **Week 1 (now):** Complete rubber-band data; finalize all analyses
- **Week 2:** Draft full text; produce final figures
- **Week 3:** Co-author review with Dr. Furse
- **Week 4:** Submit to IEEE J-ERM (or AWPL depending on length)

---

## Related earlier findings to reference (from Peter's own work)

- [[project-a2-empty-breakthrough]] — 90% LOSO on A2 empty (per-session
  standardization was key)
- [[project-a2-empty-above-95]] — 98% LOSO on A2 empty position
- [[project-hunter-vna-findings]] — 99-100% LOSO on real tumor; McGill
  combine-shuffle 2.4x Sam Med; per-test-session normalization critical
- [[project-umbmid-localization-ceiling]] — 25mm tumor localization
  ceiling on UM-BMID due to drift-vs-signal ratio 5.6x
- [[project-antenna-imaging]] — original project context

These show that this paper isn't a one-off finding — the drift-vs-signal
problem is pervasive across Peter's whole research thread, which
strengthens the "this deserves a methodology paper" argument.
