# Data manifest

Raw measurement data used in this study, its structure, and how to obtain it.
Raw CSVs are ~5+ GB total; not stored in this repo. Aggregate results
(PNG plots, summary JSONs) are in `results/`.

## Datasets

All datasets were collected at the University of Utah RF group using a
Keysight MN7021A vector network analyzer (referred to internally as the
"Hunter VNA") on a 3D-printed acrylic A3 breast phantom shell.

### 4-port Sam Medium Separated antenna configuration

| Dataset name            | Sessions | Timing (UTC-6) | Mount | Antenna 3 solder | Total sweeps |
|-------------------------|----------|----------------|-------|------------------|--------------|
| `drift_test_tape`       | 4        | 2026-07-01 14:38 to 16:03 | Loose tape | Original | 4 x 1552 |
| `drift_test_tape_1sec`  | 2        | 2026-07-01 16:32 to 16:58 | Loose tape | Original | 2 x 1552, 1-sec inter-sweep |
| `drift_test_rubber` (pre-solder) | 8 | 2026-07-02 08:59 to 2026-07-03 09:36 | Rubber band | Original (bad joint) | 8 x 1552 |
| `drift_test_rubber` (post-solder) | 4 | 2026-07-03 10:49 to 13:04 | Rubber band | Re-soldered | 4 x 1552 |
| `empty_cables`          | 2        | 2026-07-02 12:23 and 13:11 | (no antennas) | – | 2 x 1552 |
| `overnight_live_tape`   | 1        | 2026-07-01 17:22 to 2026-07-02 08:32 (~15.2 h) | Loose tape | Original | ~55,298 (1 Hz) |
| `overnight_live_rubber` | 1        | 2026-07-02 14:24 to 2026-07-03 08:05 (~17.7 h) | Rubber band | Original (bad joint) | ~65,217 (1 Hz) |

### Real-target comparison (Sam Medium Separated, same antennas)

| Dataset name | Sessions | Timing | Object | Total sweeps |
|--------------|----------|--------|--------|--------------|
| `real_target_fishingweight` | 3 | 2026-06-18 15:30, 15:47, 16:11 | Lead fishing weight | 3 x (16 baseline + 51 pos x 16 trials) |

### Independent 2-port replication (McGill antennas, prior VNA)

Used only for the cross-system LOSO replication in Section IV-C.
Data source: prior study, University of Utah RF group, 2026-06-08.

## File-level layout inside each session folder

Standard batch-sweep output produced by `batch_sweep_drift.c`
(or `batch_sweep.c` for the sessions predating the drift-mode tool):

```
BreastPhantom_A3_Nothing_YYYYMMDD_HHMM/
├── session_metadata.txt          # antennas, phantom, grid, mode
├── README.md                     # session notes
├── baseline_T01.csv .. T{N}.csv  # N=16 empty-phantom sweeps at session start
└── R{r}C{c}P{p}_T{NN}.csv        # position r/c/subposition p, trial NN
```

Position labels: row 1..6, col 2..5, subposition 1..4 (top-left, top-right,
bottom-right, bottom-left within each 1.0-inch grid cell).

Overnight live-sweep folders have a different layout:

```
LiveData_<name>/LiveData/
└── SPARAM_ReArr_<M>-<D>-<Y>_<H>-<M>-<S>-<ms>_FreqSweep_MagPhs.csv
```

One CSV per sweep, ~1 sweep/second.

## CSV format (same for all files)

33 comma-separated columns per row (791 rows per file, one per freq point):

```
Frequency, S1-1, P1-1, S2-1, P2-1, S3-1, P3-1, S4-1, P4-1,
           S1-2, P1-2, ...
           ... (four blocks of 8 S/P pairs, receive port j fixed within block)
```

- Frequency: Hz, ~0.1-8 GHz
- `S<i>-<j>`: linear magnitude of S-parameter S_ij
- `P<i>-<j>`: phase of S_ij in degrees
- Trailing comma at end of each line

Loader implementation: `code/analysis/hunter_loader.py`

## Data availability

The raw CSV data is available on request from the corresponding
author (Peter Martin, University of Utah).

A public Zenodo DOI is planned; when available, the DOI will be
added here.

## Storage

Individual session folder: ~100 MB compressed, ~500 MB uncompressed.
Overnight live sweep: ~1.5 GB compressed, ~5 GB uncompressed.
Total dataset (all above): ~5-6 GB compressed.

## Ethical / consent notes

No human subjects, no biological samples. All measurements are of
3D-printed phantoms filled with canola oil and sugar water as
tissue-mimicking dielectrics, with a lead fishing weight as the
scattering-object surrogate.
