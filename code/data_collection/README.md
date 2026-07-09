# drift_test/

Fully-unattended drift-test rig for the Hunter VNA (4-port Keysight
MN7021A). Extends `batch_sweep.c` with a new **(D)rift-test** recording
mode that skips every operator prompt and inserts a configurable
inter-position delay to simulate normal session timing.

## Contents

| File                       | Purpose                                             |
|----------------------------|-----------------------------------------------------|
| `batch_sweep_drift.c`      | Modified `batch_sweep.c` with drift-test mode       |
| `compile_drift.sh`         | Builds `batch_sweep_drift` (Linux, Keysight libs)   |
| `run_drift_3sessions.sh`   | Chains 3 back-to-back sessions with a settle delay  |
| `PROTOCOL.md`              | **THE step-by-step drift-test procedure** (read this)|
| `README.md`                | You are here                                        |

## Quick start

On the Hunter VNA host:

```bash
cd /path/to/drift_test
chmod +x *.sh
./compile_drift.sh          # produces ./batch_sweep_drift
./batch_sweep_drift         # answer prompts; pick mode = D
```

For the full 3-session cold+warm+warm test with warmup/cooldown timing,
follow **`PROTOCOL.md`**. Don't skip it: cooldown especially is easy to
get wrong, which invalidates the "cold" session.

## What's different from `batch_sweep.c`

Six surgical edits, all guarded by `bs_driftMode`:

1. **Two new state variables** (`bs_driftMode`, `bs_driftMoveDelaySec`)
2. **New mode option** `(D)rift-test` in the recording-mode prompt
3. **"Press Enter to continue with VNA setup"** - auto-skipped in drift mode
4. **"Type '0' when ready for baseline"** - auto-skipped in drift mode
5. **Per-position prompts** (both the auto-skipped and place-object paths)
   - auto-skipped positions: silently skipped
   - other positions: `usleep(bs_driftMoveDelaySec)` then proceed to trials
6. **"Save skips to model?" prompt** - auto-skipped in drift mode

All other behavior is identical to `batch_sweep.c`. Interactive and
Automatic modes are unchanged.

## Data output

Same folder layout as normal sessions:

```
./Data/<model>_<object>_<YYYYMMDD>_<HHMM>/
    session_metadata.txt
    README.md
    baseline_T01.csv .. baseline_T{N}.csv
    R1C1P1_T01.csv .. R{rows}C{cols}P4_T{N}.csv
```

The existing loader in `data.py` (Hunter dispatcher) handles these
folders without any changes.

## Sanity-check the modified binary

Run once with a **small grid** (2x2, 1 trial, 1s inter-position delay)
to confirm the whole loop runs unattended in < 30 seconds:

```
Antenna:  test
Object:   sanity
Grid:     2x2, all cells
Trials:   1
Mode:     D
Trial delay: 0.2
Inter-position delay: 1.0
```

Should produce `./Data/test_sanity_*/` with 1 baseline + 16 position
CSVs and exit cleanly with no operator input beyond the initial config.
