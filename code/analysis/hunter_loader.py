"""Hunter VNA (4-port) data loader.

The original Above-95 pipeline was built around a 2-port VNA that produced
.mat / .npz files with shape (N, 8, 201) where the 8 rows are alternating
(mag_dB, phase_rad) for the four S-parameters S11, S12, S22, S21 at 201
frequency points (300 kHz - 6.5 GHz).

The Hunter MN7021A 4-port VNA produces a different layout:

  Folder per session:
    <SessionName>/
      session_metadata.txt          (plain-text plan)
      baseline_T01.csv .. T16.csv   (empty-phantom baselines)
      R<r>C<c>P<p>_T<NN>.csv        (row, column, sub-position 1-4, trial NN)

  Each CSV is:
    Frequency, S1-1, P1-1, S2-1, P2-1, S3-1, P3-1, S4-1, P4-1,
               S1-2, P1-2, S2-2, P2-2, S3-2, P3-2, S4-2, P4-2,
               S1-3, P1-3, S2-3, P2-3, S3-3, P3-3, S4-3, P4-3,
               S1-4, P1-4, S2-4, P2-4, S3-4, P3-4, S4-4, P4-4

  S<from>-<to> is LINEAR magnitude (not dB).
  P<from>-<to> is phase in DEGREES (not radians).
  Frequency is in Hz, typically ~691 points from 100 MHz to 7 GHz
  (NOT the 201 points from 300 kHz to 6.5 GHz that the old pipeline used).

This loader presents the new data in the SAME shape the rest of the
pipeline expects: (N, 8, 201) with rows = (mag_dB, phase_rad) of
S11, S12, S22, S21 at the original 201 freq grid.  Internally we resample
linearly from whatever native freq grid the file uses.

It also exposes a "full 4-port" mode that returns (N, 32, F) with rows =
(mag_dB, phase_rad) of all 16 S-parameters, for future models trained
on the full 4-port measurement.

Public API:
  parse_hunter_csv(path)            -> (freq_hz, S_complex)  S=(4,4,F)
  load_hunter_session(folder)       -> session dict (same keys as data.py)
  load_hunter_sessions(parent_dir)  -> {sid: session_dict, ...}
  is_hunter_session_folder(path)    -> bool
  is_hunter_parent_folder(path)     -> bool
"""
from __future__ import annotations
import glob, os, re
import numpy as np


# ----------------------------------------------------------------------------
#  constants matching the original 2-port pipeline
# ----------------------------------------------------------------------------
ORIG_FREQ_HZ = np.linspace(300e3, 6.5e9, 201)        # what models were trained on
ORIG_S_ORDER = ("S11", "S12", "S22", "S21")          # row order in legacy 8x201

# Hunter CSV column layout, mapping S-name -> (mag_col, phase_col).
# Header is "Frequency,S1-1,P1-1,S2-1,P2-1,S3-1,P3-1,S4-1,P4-1, S1-2,...".
# Column 0 is Frequency, columns are then S<i>-<j>, P<i>-<j> in blocks
# of (j fixed, i=1..4), in order j=1,2,3,4.
def _hunter_col_index(i: int, j: int) -> tuple[int, int]:
    """Return (mag_col, phase_col) for S<i>-<j> in the Hunter CSV.
    1-indexed ports; column 0 is Frequency."""
    block = j - 1                  # 0..3 across the 4 blocks of 8 cols
    within = i - 1                 # 0..3 within the block
    mag_col = 1 + block * 8 + within * 2
    ph_col  = mag_col + 1
    return mag_col, ph_col


# ----------------------------------------------------------------------------
#  CSV parser
# ----------------------------------------------------------------------------
def parse_hunter_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse one Hunter 4-port CSV file.

    Returns:
        freq_hz  (F,)        float64 frequency points (Hz)
        S        (4, 4, F)   complex64 S-parameter matrix.  S[i-1, j-1, f]
                             corresponds to S<i>-<j> at freq f.
                             Magnitude is LINEAR (as in the file), phase
                             is in RADIANS (converted from the file's degrees).
    """
    # Read raw numeric data.  Some lines may end in a stray trailing comma,
    # which np.loadtxt handles fine if we tell it the delimiter.
    raw = np.loadtxt(path, delimiter=",", skiprows=1, usecols=range(33))
    # raw shape: (F, 33).  Col 0 = freq.  Cols 1..32 = 16 (mag, phase deg).
    freq_hz = raw[:, 0].astype(np.float64)
    F = raw.shape[0]
    S = np.empty((4, 4, F), dtype=np.complex64)
    for i in range(1, 5):
        for j in range(1, 5):
            mc, pc = _hunter_col_index(i, j)
            mag = raw[:, mc].astype(np.float32)
            ph_rad = np.deg2rad(raw[:, pc].astype(np.float32))
            S[i - 1, j - 1, :] = (mag * np.exp(1j * ph_rad)).astype(np.complex64)
    return freq_hz, S


# ----------------------------------------------------------------------------
#  Re-sampling onto the 201-point pipeline grid
# ----------------------------------------------------------------------------
def _resample_complex(freq_native: np.ndarray, S_native: np.ndarray,
                      freq_target: np.ndarray) -> np.ndarray:
    """Resample S(*, *, f) onto a new freq grid using LINEAR interpolation
    of magnitude and unwrapped phase.  Out-of-range targets are clipped
    (held at edge value)."""
    # S_native may be (4,4,F) or (F,).  Work in flattened form.
    orig_shape = S_native.shape
    F_native = freq_native.shape[0]
    flat = S_native.reshape(-1, F_native)        # (K, F_native)
    K = flat.shape[0]
    F_t = freq_target.shape[0]
    out = np.empty((K, F_t), dtype=np.complex64)

    # clip targets to native range (no extrapolation in the wild)
    f_lo, f_hi = float(freq_native.min()), float(freq_native.max())
    ft = np.clip(freq_target, f_lo, f_hi)

    for k in range(K):
        mag = np.abs(flat[k])
        ph  = np.unwrap(np.angle(flat[k]))
        mag_t = np.interp(ft, freq_native, mag)
        ph_t  = np.interp(ft, freq_native, ph)
        out[k] = (mag_t * np.exp(1j * ph_t)).astype(np.complex64)
    return out.reshape(orig_shape[:-1] + (F_t,))


# ----------------------------------------------------------------------------
#  Mapping 4-port S -> legacy 2-port (S11, S12, S22, S21) at 201 points
# ----------------------------------------------------------------------------
def hunter_to_legacy_complex(freq_native: np.ndarray,
                             S_full: np.ndarray,
                             freq_target: np.ndarray = ORIG_FREQ_HZ
                             ) -> np.ndarray:
    """Take a Hunter full 4-port S-matrix (4,4,F_native) and return the
    same (4, F_target) complex array the legacy pipeline expects:
    rows = [S11, S12, S22, S21] at freq_target."""
    # Pull the 4 S-params used by the legacy 2-port pipeline.
    idx = {"S11": (1, 1), "S12": (1, 2), "S22": (2, 2), "S21": (2, 1)}
    F_t = freq_target.shape[0]
    out = np.empty((4, F_t), dtype=np.complex64)
    for r, name in enumerate(ORIG_S_ORDER):
        i, j = idx[name]
        out[r] = _resample_complex(freq_native, S_full[i - 1, j - 1, :], freq_target)
    return out


def hunter_to_full16_complex(freq_native: np.ndarray,
                             S_full: np.ndarray,
                             freq_target: np.ndarray = ORIG_FREQ_HZ
                             ) -> np.ndarray:
    """Resample the full 4-port matrix (4,4,F_native) onto freq_target.
    Returns complex (16, F_target) in row-order S11,S12,S13,S14,
    S21,S22,...,S44 (row-major over (from-port, to-port))."""
    F_t = freq_target.shape[0]
    out = np.empty((16, F_t), dtype=np.complex64)
    for i in range(1, 5):
        for j in range(1, 5):
            r = (i - 1) * 4 + (j - 1)
            out[r] = _resample_complex(freq_native, S_full[i - 1, j - 1, :], freq_target)
    return out


# ----------------------------------------------------------------------------
#  legacy (8, F) real-row format used by data.py
# ----------------------------------------------------------------------------
def _complex4_to_8rows(Sc: np.ndarray) -> np.ndarray:
    """(4, F) complex -> (8, F) real with rows alternating (mag_dB, phase_rad)
    in the SAME ORDER the legacy pipeline produced: S11, S12, S22, S21."""
    F = Sc.shape[1]
    out = np.empty((8, F), dtype=np.float32)
    for k in range(4):
        mag = np.abs(Sc[k])
        # guard against zero before log
        mag = np.where(mag <= 0, 1e-12, mag)
        out[2 * k, :]     = 20.0 * np.log10(mag)
        out[2 * k + 1, :] = np.angle(Sc[k])
    return out


# ----------------------------------------------------------------------------
#  Folder layout helpers
# ----------------------------------------------------------------------------
_RC_PAT = re.compile(r"^R(\d+)C(\d+)P(\d+)_T(\d+)\.csv$", re.IGNORECASE)
_BL_PAT = re.compile(r"^baseline_T(\d+)\.csv$", re.IGNORECASE)


def is_hunter_session_folder(path: str) -> bool:
    """A Hunter session folder has session_metadata.txt + at least one
    baseline_T*.csv + at least one RnCmPp_T*.csv."""
    if not os.path.isdir(path):
        return False
    has_meta = os.path.isfile(os.path.join(path, "session_metadata.txt"))
    files = os.listdir(path)
    has_bl = any(_BL_PAT.match(f) for f in files)
    has_pos = any(_RC_PAT.match(f) for f in files)
    return has_meta and has_bl and has_pos


def is_hunter_parent_folder(path: str) -> bool:
    """A parent folder that contains one or more Hunter session subfolders."""
    if not os.path.isdir(path):
        return False
    if is_hunter_session_folder(path):
        return False
    for name in os.listdir(path):
        sub = os.path.join(path, name)
        if is_hunter_session_folder(sub):
            return True
    return False


def parse_session_metadata(path: str) -> dict:
    """Parse a session_metadata.txt and return a dict of useful fields.
    Tolerates missing fields - returns sensible defaults."""
    meta = {
        "operator": "",
        "model": "",
        "antenna": "",
        "object": "",
        "total_rows": None, "total_cols": None,
        "measured_rows": [], "measured_cols": [],
        "cell_size_in": 1.0,
        "divider_in": 0.25,
        "n_positions": None, "n_trials": None,
        "mode": "",
    }
    if not os.path.isfile(path):
        return meta
    with open(path, "r") as f:
        text = f.read()

    def m(pat, default=None, cast=str):
        mm = re.search(pat, text, re.MULTILINE)
        if not mm: return default
        try:
            return cast(mm.group(1).strip())
        except Exception:
            return default

    meta["operator"] = m(r"^Operator:\s*(.+)$", "")
    meta["model"]    = m(r"^Model:\s*(.+)$", "")
    meta["antenna"]  = m(r"^Antenna:\s*(.+)$", "")
    meta["object"]   = m(r"^Object:\s*(.+)$", "")
    tg = re.search(r"Total grid:\s*(\d+)\s*rows\s*x\s*(\d+)\s*cols", text)
    if tg:
        meta["total_rows"] = int(tg.group(1)); meta["total_cols"] = int(tg.group(2))
    mr = re.search(r"Measured rows:\s*([\d\s]+)", text)
    if mr:
        meta["measured_rows"] = [int(x) for x in mr.group(1).split()]
    mc = re.search(r"Measured cols:\s*([\d\s]+)", text)
    if mc:
        meta["measured_cols"] = [int(x) for x in mc.group(1).split()]
    meta["cell_size_in"] = m(r"Cell size:\s*([\d.]+)\s*inches", 1.0, float)
    meta["divider_in"]   = m(r"Divider thick:\s*([\d.]+)\s*inches", 0.25, float)
    meta["n_positions"]  = m(r"Positions:\s*(\d+)", None, int)
    meta["n_trials"]     = m(r"Trials per position:\s*(\d+)", None, int)
    meta["mode"]         = m(r"Mode:\s*(\w+)", "")
    return meta


def _rcp_to_index(r: int, c: int, p: int, total_cols: int) -> int:
    """Encode (row, col, sub-position 1-4) into a single integer position id.
    Layout: pos = ((r-1) * total_cols + (c-1)) * 4 + (p-1).
    This is stable as long as you keep total_cols constant for the model."""
    return ((r - 1) * total_cols + (c - 1)) * 4 + (p - 1)


def _rcp_to_xy(r: int, c: int, p: int,
               cell_size: float, divider: float) -> tuple[float, float]:
    """Approximate (x, y) inches for a sub-position centre.
    Origin is top-left.  Sub-position layout (from BatchSweep_Guide):
      P1 = Top-Left   P2 = Top-Right
      P4 = Bot-Left   P3 = Bot-Right
    """
    pitch = cell_size + divider
    # cell centre
    cx = (c - 1) * pitch + cell_size / 2.0
    cy = (r - 1) * pitch + cell_size / 2.0
    # offset by sub-position
    dx = -cell_size / 4.0 if p in (1, 4) else +cell_size / 4.0
    dy = -cell_size / 4.0 if p in (1, 2) else +cell_size / 4.0
    return cx + dx, cy + dy


# ----------------------------------------------------------------------------
#  Whole-session loader (produces a data.py-compatible session dict)
# ----------------------------------------------------------------------------
def load_hunter_session(folder: str,
                        freq_target: np.ndarray = ORIG_FREQ_HZ,
                        mode: str = "legacy"):
    """Load one Hunter session folder.

    Args:
      folder       path to the session directory
      freq_target  freq grid to resample to (default 201-point legacy grid)
      mode         "legacy"  -> 4 S-params (S11,S12,S22,S21), rows=8
                   "full16"  -> 16 S-params, rows=32

    Returns a dict with the SAME KEYS as a single session in data.py's
    load_all_sessions output, so it can be dropped straight in:

      X_raw    (N, 8 or 32, F)   float32  mag_dB / phase_rad rows
      Xc       (N, 4 or 16, F)   complex  baseline-NOT-subtracted S-params
      base_c   (4 or 16, F)      complex  averaged-trial baseline
      y_pos    (N,)              int64    sparse position id (0..N_grid*4-1)
      y_cell   (N,)              int64    dense cell index 0..n_cells-1
      xy       (N, 2)            float32  inches
      meta     dict              metadata parsed from session_metadata.txt
      freq_hz  (F,)              float    the freq grid actually used
    """
    if not is_hunter_session_folder(folder):
        raise FileNotFoundError(f"{folder} is not a Hunter session folder")

    meta = parse_session_metadata(os.path.join(folder, "session_metadata.txt"))
    total_cols = meta["total_cols"] or 6

    # ---- baselines: average the complex baseline trials ----
    bl_paths = sorted([os.path.join(folder, f) for f in os.listdir(folder)
                       if _BL_PAT.match(f)])
    if not bl_paths:
        raise FileNotFoundError(f"No baseline_T*.csv in {folder}")

    freq_native, S_first = parse_hunter_csv(bl_paths[0])
    # accumulate full 4-port baseline at native freq, then resample once at the end
    base_accum = np.zeros((4, 4, freq_native.shape[0]), dtype=np.complex128)
    base_accum += S_first
    for p in bl_paths[1:]:
        _, S = parse_hunter_csv(p)
        if S.shape[-1] != freq_native.shape[0]:
            raise ValueError(f"Baseline freq-grid mismatch in {p}")
        base_accum += S
    base_native = (base_accum / len(bl_paths)).astype(np.complex64)

    if mode == "legacy":
        base_c = hunter_to_legacy_complex(freq_native, base_native, freq_target)
    elif mode == "full16":
        base_c = hunter_to_full16_complex(freq_native, base_native, freq_target)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    # ---- position CSVs ----
    pos_paths = sorted([f for f in os.listdir(folder) if _RC_PAT.match(f)])
    if not pos_paths:
        raise FileNotFoundError(f"No RnCmPp_T*.csv in {folder}")

    Xc_list, y_pos_list, xy_list, cell_list = [], [], [], []
    valid_cells = []
    cell_key_to_dense = {}

    for fname in pos_paths:
        mm = _RC_PAT.match(fname)
        r, c, p, _t = int(mm.group(1)), int(mm.group(2)), int(mm.group(3)), int(mm.group(4))
        path = os.path.join(folder, fname)
        try:
            f_native, S = parse_hunter_csv(path)
        except Exception as e:
            print(f"  [warn] could not parse {fname}: {e}")
            continue
        if f_native.shape[0] != freq_native.shape[0]:
            print(f"  [warn] freq-grid mismatch in {fname}, skipping")
            continue

        # if file is all-NaN (skipped position placeholder), skip
        if not np.isfinite(np.abs(S)).any():
            continue

        if mode == "legacy":
            Sc = hunter_to_legacy_complex(f_native, S, freq_target)
        else:
            Sc = hunter_to_full16_complex(f_native, S, freq_target)
        Xc_list.append(Sc)

        # sparse position id - one per sub-position in the entire grid
        pos_id = _rcp_to_index(r, c, p, total_cols)
        y_pos_list.append(pos_id)

        # dense cell index = unique (r,c) ordered by appearance
        ck = (r, c)
        if ck not in cell_key_to_dense:
            cell_key_to_dense[ck] = len(cell_key_to_dense)
            valid_cells.append(ck)
        cell_list.append(cell_key_to_dense[ck])

        x_in, y_in = _rcp_to_xy(r, c, p,
                                cell_size=meta["cell_size_in"],
                                divider=meta["divider_in"])
        xy_list.append((x_in, y_in))

    if not Xc_list:
        raise RuntimeError(f"No usable position CSVs loaded from {folder}")

    Xc = np.stack(Xc_list, axis=0)                        # (N, 4 or 16, F)
    # X_raw in (mag_dB, phase_rad) row form so the legacy code can keep
    # using it without conversion
    N = Xc.shape[0]; n_rows = 2 * Xc.shape[1]; F = Xc.shape[2]
    X_raw = np.empty((N, n_rows, F), dtype=np.float32)
    for i in range(N):
        X_raw[i] = _complex4_to_8rows(Xc[i]) if mode == "legacy" \
                   else _complex_n_to_2nrows(Xc[i])

    y_pos = np.array(y_pos_list, dtype=np.int64)
    y_cell = np.array(cell_list, dtype=np.int64)
    xy = np.array(xy_list, dtype=np.float32)

    return dict(
        Xc=Xc, base_c=base_c, y_pos=y_pos, y_cell=y_cell, xy=xy,
        X_raw=X_raw, meta=meta, freq_hz=np.asarray(freq_target, dtype=np.float64),
    )


def _complex_n_to_2nrows(Sc: np.ndarray) -> np.ndarray:
    """(K, F) complex -> (2K, F) real with rows alternating (mag_dB, phase_rad).
    Generalises _complex4_to_8rows to arbitrary K."""
    K, F = Sc.shape
    out = np.empty((2 * K, F), dtype=np.float32)
    for k in range(K):
        mag = np.abs(Sc[k])
        mag = np.where(mag <= 0, 1e-12, mag)
        out[2 * k]     = 20.0 * np.log10(mag)
        out[2 * k + 1] = np.angle(Sc[k])
    return out


# ----------------------------------------------------------------------------
#  Parent-folder loader (multiple sessions in subfolders)
# ----------------------------------------------------------------------------
def load_hunter_sessions(parent_dir: str,
                         freq_target: np.ndarray = ORIG_FREQ_HZ,
                         mode: str = "legacy"):
    """Find every Hunter session folder under parent_dir (or treat
    parent_dir itself as a single session if it is one).  Returns
    {session_id: session_dict, ...}.

    session_id is the leaf folder name (alphanumeric)."""
    if is_hunter_session_folder(parent_dir):
        sid = os.path.basename(os.path.normpath(parent_dir))
        return {sid: load_hunter_session(parent_dir, freq_target, mode)}
    if not is_hunter_parent_folder(parent_dir):
        raise FileNotFoundError(
            f"{parent_dir} is neither a Hunter session folder nor a parent of one")
    out = {}
    for name in sorted(os.listdir(parent_dir)):
        sub = os.path.join(parent_dir, name)
        if is_hunter_session_folder(sub):
            print(f"  loading session: {name}")
            out[name] = load_hunter_session(sub, freq_target, mode)
    if not out:
        raise FileNotFoundError(f"No Hunter sessions found in {parent_dir}")
    return out


# ----------------------------------------------------------------------------
#  CLI smoke test
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else \
        r"C:\Users\peter\Desktop\EM Imaging\HUNTER VNA\test_test_20260604_1235"
    print(f"loading {folder}")
    sess = load_hunter_session(folder)
    print("  meta:", sess["meta"])
    print(f"  Xc:      {sess['Xc'].shape}   dtype={sess['Xc'].dtype}")
    print(f"  base_c:  {sess['base_c'].shape}")
    print(f"  X_raw:   {sess['X_raw'].shape}")
    print(f"  y_pos:   {sess['y_pos'].shape}   {sess['y_pos'].min()}..{sess['y_pos'].max()}")
    print(f"  y_cell:  {sess['y_cell'].shape}   #cells={len(set(sess['y_cell'].tolist()))}")
    print(f"  xy:      {sess['xy'].shape}  range x[{sess['xy'][:,0].min():.2f}..{sess['xy'][:,0].max():.2f}]"
          f"  y[{sess['xy'][:,1].min():.2f}..{sess['xy'][:,1].max():.2f}]")
    print(f"  freq_hz: {sess['freq_hz'].shape}  "
          f"{sess['freq_hz'][0]/1e9:.3f}-{sess['freq_hz'][-1]/1e9:.3f} GHz")
