# Reproducing the results

Step-by-step guide from a fresh `git clone` to reproducing every
figure and table in the paper.

## 1. Set up Python

Any Python 3.10+ environment. Recommended: create a virtualenv or conda env.

```
pip install -r code/analysis/requirements.txt
```

Dependencies: numpy, scipy, matplotlib, scikit-learn, joblib.

## 2. Get the data

See [`DATA_MANIFEST.md`](DATA_MANIFEST.md). Raw CSVs are too large for
GitHub. Once downloaded, place them under a top-level `data/` folder
so that:

```
data/
├── 4port_sam_med/
│   ├── drift_test_tape/       # BreastPhantom_A3_Nothing_20260701_*
│   ├── drift_test_rubber/     # BreastPhantom_A3_Nothing_20260702-03_*
│   ├── overnight_live/        # LiveData_070126_Overnight/LiveData/
│   ├── overnight_rubber/      # LiveData_RubberBand/LiveData/
│   └── empty_cables/          # EmptyCables/BreastPhantom_A3_Nothing_20260702_*
└── real_target/               # BreastPhantom_A3_FishingWeight_20260618_*
```

## 3. Update paths in the analysis scripts

Every script has a `DATA_DIR` (or similar) constant near the top.
The current values point at the original absolute paths. Change
each to point at your `data/` folder.

A future release will move all paths into a single `config.py`.

## 4. Run the analyses

Each analysis writes its outputs to `results/drift_test_sam_med/`.
The `results/` folder already contains the outputs from our runs,
so you can also just inspect the JSONs and PNGs without re-running.

### Batch drift test (fastest, ~5 min per script)

```
python code/analysis/drift_analysis_by_port.py      # per-port trial-CV per session
python code/analysis/drift_session_to_session.py    # session-to-session drift
```

### Overnight settling analysis (~10 min including CSV parse)

```
python code/analysis/drift_settling_overnight.py         # tape overnight
python code/analysis/drift_settling_overnight_rubber.py  # rubber-band overnight
```

### Synthetic-positions detectable-difference (~15 min per overnight)

```
python code/analysis/drift_synthetic_positions_dd.py         # tape
python code/analysis/drift_synthetic_positions_dd_rubber.py  # rubber-band
```

### LOSO cross-validation tests (~30-90 min each due to MLP ensemble)

```
python code/analysis/drift_loso_test.py                # 4-session tape LOSO
python code/analysis/drift_loso_test_S11only.py        # S11-only tape LOSO
python code/analysis/drift_loso_test_rubber.py         # 6-session rubber LOSO
python code/analysis/drift_loso_test_rubber_all12.py   # 12-session pre/post-solder
```

### Mount comparisons (~15 min, mostly CSV loading)

```
python code/analysis/drift_rubber_band_compare.py      # tape vs rubber
python code/analysis/drift_cables_only_diagnostic.py   # antenna vs cable/VNA
python code/analysis/drift_three_mount_summary.py      # all three, both metrics
python code/analysis/drift_presolder_vs_postsolder.py  # antenna-3 fix effect
```

## 5. Regenerate the paper (optional)

The paper source is `paper/draft.docx`. It's regenerated from a
`build_drift_paper_v2.js` script (not committed here in this rev;
contact the authors if you want it).

Figures in the paper come directly from `results/`:

- Fig. 1 (per-port trial-CV, 4 tape sessions): `results/drift_test_sam_med/drift_over_time.png`
- Fig. 2 (overnight settling): `results/drift_test_sam_med/overnight/rolling_std.png`
- Fig. 3 (synthetic-positions DD grand timeline): `results/drift_test_sam_med/overnight/synthetic_positions_dd/grand_timeline_dB.png`
- Fig. 4 (detection rate per port over hours): `results/drift_test_sam_med/overnight/synthetic_positions_dd/per_port_dd_over_hours.png`
- Fig. 5 (three-mount two-metric comparison): `results/drift_test_sam_med/three_mount/three_mount_two_metric.png`
- Fig. 6 (reflection slow drift, three mounts): `results/drift_test_sam_med/three_mount/reflection_slow_drift_three_mounts.png`
- Fig. 7 (S44 antenna-vs-cable diagnostic): `results/drift_test_sam_med/cables_only/S44_diagnostic.png`

## 6. Collect your own drift data (optional)

If you have a Keysight MN7021A (Hunter VNA), you can collect your own
drift-test data using the modified `batch_sweep_drift.c` tool:

```
cd code/data_collection/
chmod +x compile_drift.sh
./compile_drift.sh
./batch_sweep_drift        # answer prompts, pick mode = D for drift-test
```

See `code/data_collection/PROTOCOL.md` for the full 3-session
warmup-and-drift protocol.
