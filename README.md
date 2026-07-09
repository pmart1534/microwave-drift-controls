# microwave-drift-controls

Three-experiment control framework for distinguishing measurement drift from
real signal in microwave-imaging machine learning.

Companion code, results, and paper draft for:

> P. Martin and C. M. Furse, "Distinguishing Drift from Signal in
> Microwave-Imaging Machine Learning: A Three-Experiment Control Framework"
> (in preparation).

## What's here

- `paper/` — draft manuscript (`draft.docx`) and full outline
- `code/analysis/` — Python analysis scripts (drift metrics, LOSO tests,
  detectable-difference on continuous data, mount comparisons)
- `code/data_collection/` — the modified `batch_sweep_drift.c` VNA
  acquisition tool with fully-unattended "drift-test" mode, plus its
  compile script and step-by-step data-collection protocol
- `results/` — analysis outputs (PNG plots, summary JSON) for every
  experiment reported in the paper. All small; safe to commit.
- `docs/` — reproduction guide, dataset manifest, install notes

## The three controls in one sentence each

1. **Empty-phantom within-session accuracy** — quantifies how strongly
   drift alone can be memorized as a per-position fingerprint (~82% on
   96 positions in our data).
2. **Empty-phantom multi-session LOSO** — verifies that the memorized
   fingerprint is session-specific rather than a stable hardware
   property (drops to 2.3%, chance = 1.04%).
3. **Synthetic-positions detectable-difference** — takes a continuous
   single-position sweep and tests whether pure drift exceeds the
   standard 10% relgap detection threshold at each hour of the run.

Applied to a 4-port VNA with a metal scattering object on identical
antennas, real-object LOSO reaches 99.35%, a 42x gap over the
drift-only case. Reducing the measurement to a single reflection
channel (S11) widens the gap to 77x.

## Data

Raw measurement data (~5+ GB of CSVs) is **not** in this repo.
See [`docs/DATA_MANIFEST.md`](docs/DATA_MANIFEST.md) for the
inventory of datasets used and how to obtain them.

## Getting started

1. Read [`docs/REPRODUCE.md`](docs/REPRODUCE.md) for a step-by-step
   guide from `git clone` through to reproducing every figure and
   table in the paper.
2. Read [`paper/OUTLINE.md`](paper/OUTLINE.md) for the paper's
   argument in outline form.
3. Read [`code/data_collection/PROTOCOL.md`](code/data_collection/PROTOCOL.md)
   for the drift-test data-collection procedure with warmup/cooldown
   timing.

## License

MIT.

## Citation

If you use this framework or code, please cite the paper (details
above; will be updated when published).

## Contact

Peter Martin, Department of Electrical and Computer Engineering,
University of Utah.
