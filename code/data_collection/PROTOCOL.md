# Hunter VNA Drift-Test Protocol

**Owner:** Peter Martin
**Instrument:** Hunter VNA (Keysight MN7021A, 4-port)
**Purpose:** Isolate drift from real per-position signal by running
fully-unattended empty-phantom sessions with the same timing as a real
data-collection session.

---

## 1. What this protocol actually tests

The phantom stays **empty for the entire test**. The batch program still
walks all 51 positions x N trials and produces a normal-looking session
folder, so the analysis pipeline can consume it unchanged. Everything the
CNN "sees" between trials/positions is therefore drift, cable flex,
thermal effects, and any interference in the room -- **not** any real
per-position signal.

Three back-to-back sessions let us:
- separate within-session drift (trials 1..N at the same position)
- from between-session drift (the same position two sessions later)
- and check that no single session is a wild outlier

---

## 2. One-time setup (do this once, then never again)

1. **Copy the folder to the Hunter VNA host** if the Hunter machine is not
   this Windows box:
   ```
   scp -r drift_test/ hunter@<host>:/home/hunter/HunterVNA/
   ```
2. **Compile:**
   ```bash
   cd /home/hunter/HunterVNA/drift_test
   chmod +x compile_drift.sh
   ./compile_drift.sh
   # -> produces ./batch_sweep_drift
   ```
3. **Sanity-check** by running one very short session (e.g., grid 2x2,
   1 trial, drift-mode delay 1s):
   ```bash
   ./batch_sweep_drift
   # answer prompts, pick model with a small grid, mode = D
   # confirm it runs the whole thing with no prompts and produces
   # ./Data/<name>_<timestamp>/ containing baseline_T01.csv and
   # R1C1P1_T01.csv ... R2C2P4_T01.csv
   ```

If the sanity-check works, you never need to touch the code again.

---

## 3. Test-day timing overview

Read this whole section BEFORE you start so you know what you're
committing to.

| Phase                         | Duration        | Notes                                    |
|-------------------------------|-----------------|------------------------------------------|
| Cold-VNA setup                | ~10 min         | Physical setup, empty-phantom check      |
| **COLD test (Session 1)**     | ~30-45 min      | VNA fully cold, structured drift mode    |
| VNA warmup wait               | 45 min          | Do nothing. Let VNA run baselines idle.  |
| **WARM test (Session 2)**     | ~30-45 min      | VNA now thermally stable                 |
| Inter-session settle          | 5-10 min        | Save data, log any notes                 |
| **WARM test (Session 3)**     | ~30-45 min      | Immediately after Session 2              |
| Data pull / verification      | ~10 min         | Confirm all 3 folders present            |
| **TOTAL time on the bench**   | **~3-4 hours**  | Mostly hands-off after Session 1 starts  |

**If you want to redo the whole thing on a different day** (e.g., because
Session 1 looked wild), you'll need the VNA to reach "cold" state again
- see Cooldown below.

---

## 4. Warmup and cooldown timing

Keysight does not publish an exact warmup curve for the MN7021A that I
can cite. The numbers below come from your own past drift work
(11-session LOSO convergence, drift-timescale analysis) and general
practice for X-band VNAs.

### Warmup (COLD -> stable WARM)

- **45 minutes minimum** of the VNA powered on and actively sweeping
- **90 minutes** if you want to be conservative
- **Verify empirically:** the drift-test itself will show whether the VNA
  reached equilibrium. If Session 1's magnitude traces drift a lot more
  than Session 2's, you know the 45-min warmup was needed.

**"Actively sweeping" matters**: leaving the VNA on but idle (no sweeps
running) warms it less than actively sweeping. The drift-test program
sweeps continuously, so once it starts, the VNA is warming up. This is
why we run Session 1 immediately after power-on: the drift-test itself
serves as a warmup log.

### Cooldown (WARM -> back to COLD)

The VNA does not need to be "at room temperature" to be COLD for this
protocol -- it needs to be **starting from off, without recent extended
use**. Practical thresholds:

- **2 hours powered off** = safely COLD for a follow-up test
- **1 hour powered off** = borderline, mark it in notes
- **< 30 min powered off** = still warm; do NOT run a "cold" session

**If you need a truly cold restart** (e.g., you want to redo Session 1
because it was corrupted): power off the VNA and physically leave it
alone for **at least 2 hours** before running the next "cold" session.

**Fastest way to test your own cooldown curve** (do this once, then
you'll know your VNA specifically):
1. Run a WARM session (45+ min continuous sweeping).
2. Power off the VNA. Note the time.
3. At t = 30 min: power on, immediately run a 5-min baseline-only session.
4. Repeat at t = 60 min, t = 120 min, t = 240 min.
5. Compare baseline magnitudes across the four points. Whichever
   power-off duration first shows baseline consistent with a
   "fully-cold" morning-after start is your cooldown time.

---

## 5. Session-by-session procedure

### Session 1 - COLD VNA (~30-45 min)

**Precondition:** VNA has been powered off for at least 2 hours (or
overnight -- easiest).

1. **Physical setup** (~5 min):
   - Confirm phantom is empty and lid is on if applicable
   - Confirm all 4 SMA cables are seated finger-tight, no cable bends
     have changed since your last real session
   - Confirm antennas are in place if you're testing "with antennas"
     (see Section 8 for the without-antennas variation)
2. **Power on VNA** and its host machine. Note the wall-clock time.
3. **Immediately** (within ~1 minute of power-on):
   ```bash
   cd /home/hunter/HunterVNA/drift_test
   ./batch_sweep_drift
   ```
4. Answer the initial prompts as you normally would:
   - Antenna name: whatever you're using (e.g., `Sam_Med` or `McGill`)
   - Model / grid: same as your real sessions
   - Object name: **`DriftCold_S1`** (distinguishes Session 1 as cold)
   - Operator: your name
   - Trial count: same as your real sessions (typically 5-16)
   - **Recording mode: `D`** (drift-test)
   - Trial delay: `0.2` (matches your real sessions)
   - Inter-position delay: `3.0` (matches typical operator-move time; see
     Section 7 to tune this)
   - Notes: `Session 1, VNA cold - powered on at HH:MM`
5. **Do not touch the bench.** The program will run all 51 positions x
   trials with zero further input. When it finishes, it writes:
   ```
   ./Data/<Model>_DriftCold_S1_YYYYMMDD_HHMM/
       baseline_T01.csv .. baseline_T{N}.csv
       R1C1P1_T01.csv .. R1C1P1_T{N}.csv
       R1C1P2_T01.csv ..
       ...
   ```

### Warmup wait (45 min)

**Session 1 already warmed the VNA** while it was sweeping. Total
warmup time so far = duration of Session 1 (~30-45 min). If Session 1
took at least 45 min, you can go directly to Session 2 with no wait.

If Session 1 was shorter than 45 min, add an idle-sweep wait to reach
45 min total on-time. Easiest: relaunch `./batch_sweep_drift` with the
same object name, let it start recording, then Ctrl+C after enough
minutes have elapsed and delete the partial folder. Or just wait.

### Session 2 - WARM VNA (~30-45 min)

**Precondition:** VNA has been powered on and sweeping for at least
45 minutes (Session 1 counts).

Repeat the "run batch_sweep_drift" flow from Session 1, but change:
- Object name: **`DriftWarm_S2`**
- Notes: `Session 2, VNA warm - session1 ended at HH:MM`

Everything else identical.

### Inter-session settle (5-10 min)

- Confirm Session 2's folder is complete (all CSVs present).
- Take any bench notes (temperature, cable movement, someone walked
  through the lab, etc.).
- **Do not power-cycle the VNA.**

### Session 3 - WARM VNA #2 (~30-45 min)

Same as Session 2, but:
- Object name: **`DriftWarm_S3`**
- Notes: `Session 3, VNA warm - session2 ended at HH:MM`

Session 3 exists so you have **two warm sessions** to compare against
each other (isolates "cold vs warm" from "session-to-session variation
regardless of temperature").

---

## 6. What to check before leaving the bench

After all 3 sessions complete, verify:

1. **Three data folders exist** under `./Data/`:
   - `<Model>_DriftCold_S1_<timestamp>/`
   - `<Model>_DriftWarm_S2_<timestamp>/`
   - `<Model>_DriftWarm_S3_<timestamp>/`
2. **Each folder has the expected CSV count**: `1 baseline + 51*4 positions = 205 position CSVs` (for a 51-position 4-subposition grid), times `N` trials each.
3. **Each folder has a `session_metadata.txt`** and a `README.md` with your notes.
4. **All 4 subpositions x all positions are present** (no gaps). Run:
   ```bash
   ls Data/*DriftCold_S1*/R*_T01.csv | wc -l   # should be 51*4 = 204
   ls Data/*DriftWarm_S2*/R*_T01.csv | wc -l   # same
   ls Data/*DriftWarm_S3*/R*_T01.csv | wc -l   # same
   ```

If any of the above fails, re-run just that session (start over from the
top of Section 5 - it's cheaper than trying to salvage a bad folder).

---

## 7. Tuning the inter-position delay (default 3.0s)

The 3.0-second default was chosen because in your normal
data-collection sessions with the 4-port config, an operator typically
takes 2-4 seconds to nudge the tumor phantom to the next
grid-and-subposition. If your operational cadence is different, adjust:

- **Faster real cadence** (2 sec/position): use `2.0`
- **Slower real cadence** (5 sec/position): use `5.0`
- **Match your own timing exactly:** the metadata files in your existing
  Sam_Med / McGill sessions have timestamps you can diff to compute the
  actual average. If you want a matching estimate, ask.

The **trial-to-trial delay stays at 0.2s** for all drift tests because
that's what all your existing sessions used.

---

## 8. Optional variations (run these LATER, not the first day)

Get the core 3-session cold+warm+warm working first. Then, on
subsequent days, run these to explore other drift dimensions:

| Variation                | Change                                      | Object-name convention |
|--------------------------|---------------------------------------------|------------------------|
| No antennas (cables only)| Physically remove all 4 antennas            | `DriftNoAnt_S1..3`     |
| Slow move                | Inter-position delay = 10s                  | `DriftSlow_S1..3`      |
| Fast move                | Inter-position delay = 1s                   | `DriftFast_S1..3`      |
| Warm-only 3-session      | Skip the cold session, all 3 while warm     | `DriftWarm_S1..3`      |
| Different antenna set    | Swap Sam_Med for McGill (or vice versa)     | `DriftMcG_S1..3` etc.  |

Each of these should reuse the exact same protocol (Sections 5-6) with
only the noted change. Consistent naming lets the analysis code
pattern-match on `DriftCold_*`, `DriftNoAnt_*`, etc.

---

## 9. Post-test analysis (what to do with the data)

Once you have the 3 folders, the standard LOSO / combine-shuffle
pipeline in `Above 95 Percent/code/` works unchanged. Suggested first
checks:

1. **Per-session baseline stability**: plot mean-baseline-magnitude
   over trial index for each session. Cold session should show a
   monotonic drift; warm sessions should be flat-ish.
2. **3-session combine-shuffle**: pool all 3 sessions position-by-position,
   shuffle, retrain. If accuracy is **still** ~chance (~2%), the sessions
   are pure drift. If accuracy is nonzero, there's a per-position
   fingerprint even from empty phantom.
3. **Cold-vs-warm baseline diff**: subtract Session 1 baseline from
   Session 2 baseline element-wise. The magnitude of that diff = your
   thermal drift signature.

Save any interesting findings to
`Above 95 Percent/results/drift_test_YYYYMMDD/` so they don't get lost.

---

## 10. Quick checklist (print this)

```
[ ] VNA off for 2+ hours before starting
[ ] Phantom empty, lid on
[ ] All 4 SMA cables finger-tight
[ ] Antennas in place (or removed for no-ant variation)
[ ] Power on VNA - note time: ______
[ ] Start Session 1 within 1 min of power-on
    Object name: DriftCold_S1
[ ] Session 1 complete - note end time: ______
[ ] Verify Session 1 folder has all CSVs
[ ] (If Session 1 was shorter than 45 min: wait)
[ ] Start Session 2
    Object name: DriftWarm_S2
[ ] Session 2 complete - note end time: ______
[ ] Verify Session 2 folder has all CSVs
[ ] Start Session 3 immediately
    Object name: DriftWarm_S3
[ ] Session 3 complete - note end time: ______
[ ] Verify all 3 folders complete
[ ] Copy folders to Windows box for analysis
```
