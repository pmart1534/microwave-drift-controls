"""Data loader for the A2_Empty cross-session task.

Reuses the cleaned per-session .npz files already produced by
`Above 80 Percent/src/export_sessions.py`. Each session npz contains:
  X        (N, 8, 201) float32  — 8 rows: alternating (mag_dB, phase_rad) for S11, S12, S22, S21
  y_pos    (N,)        int      — position index 0..63 (sparse; many never used)
  y_cell   (N,)        int      — 0..15 dense cell index
  baseline (8, 201)    float32  — empty-phantom baseline (saved per session)
  xy       (N, 2)      float32  — (x, y) inches

This module:
  1) reads all 7 sessions
  2) reconstructs complex S-parameters from (mag_dB, phase_rad) rows
  3) applies per-session calibration (Above 90 'sub_combo': subtract session
     baseline, then subtract the per-channel/per-freq mean of (X - baseline))
  4) splits Re/Im back into 8 real channels [8, 201]
  5) produces a stable dense label encoding for the *40-way position* task
     (24 of the 64 grid positions are skipped, so 64-way is impossible —
     the marble was never placed in those cells)

The output tensor layout matches PyTorch: (N, C=8, F=201).

NEW (June 2026): the same load_any() dispatcher now also reads Hunter MN7021A
4-port VNA data (folders containing session_metadata.txt + RnCmPp_T*.csv).
The Hunter loader resamples to the legacy 201-point grid and pulls just the
2-port subset (S11, S12, S22, S21) so the existing trained models keep
working.  See hunter_loader.py.
"""
from __future__ import annotations
import glob, os
import numpy as np

DATA_DIR_DEFAULT = r"C:\Users\peter\Desktop\EM Imaging\Above 80 Percent\data"


def _rows_to_complex(R: np.ndarray) -> np.ndarray:
    """(8, F) real mag_dB/phase_rad rows -> (4, F) complex S-params (S11, S12, S22, S21)."""
    F = R.shape[1]
    out = np.empty((4, F), dtype=np.complex64)
    for k in range(4):
        mag_db = R[2 * k, :]
        ph_rad = R[2 * k + 1, :]
        mag = 10.0 ** (mag_db / 20.0)
        out[k, :] = mag * np.exp(1j * ph_rad)
    return out


def _split_complex_to_8ch(Xc: np.ndarray) -> np.ndarray:
    """(N, 4, F) complex -> (N, 8, F) real with [Re, Im] interleaved per S-param."""
    N, _, F = Xc.shape
    out = np.empty((N, 8, F), dtype=np.float32)
    out[:, 0, :] = np.real(Xc[:, 0, :]); out[:, 1, :] = np.imag(Xc[:, 0, :])  # S11
    out[:, 2, :] = np.real(Xc[:, 1, :]); out[:, 3, :] = np.imag(Xc[:, 1, :])  # S12
    out[:, 4, :] = np.real(Xc[:, 2, :]); out[:, 5, :] = np.imag(Xc[:, 2, :])  # S22
    out[:, 6, :] = np.real(Xc[:, 3, :]); out[:, 7, :] = np.imag(Xc[:, 3, :])  # S21
    return out


def load_all_sessions(data_dir: str | None = None):
    """Load all sessions from `data_dir`. Returns a dict keyed by session id.

    The folder can be EITHER:
      - the legacy 2-port directory of session_*.npz files (original behavior), OR
      - a Hunter VNA 4-port session folder, OR
      - a parent folder containing Hunter VNA session subfolders.

    Resolution order for `data_dir`:
      1) explicit argument
      2) $EM_DATA_PATH environment variable
      3) if $EM_DATA_PICK is set, open a Tk folder picker
      4) DATA_DIR_DEFAULT (legacy A2_Empty .npz dir)
    """
    # ----- resolve the folder -----
    if data_dir is None:
        data_dir = os.environ.get("EM_DATA_PATH")
    if data_dir is None and os.environ.get("EM_DATA_PICK"):
        data_dir = pick_data_folder()
    if not data_dir:
        data_dir = DATA_DIR_DEFAULT

    # ----- dispatch on format -----
    fmt = detect_format(data_dir)
    if fmt in ("hunter_session", "hunter_parent"):
        from hunter_loader import load_hunter_sessions
        return load_hunter_sessions(data_dir)

    files = sorted(glob.glob(os.path.join(data_dir, "session_*.npz")))
    if not files:
        raise FileNotFoundError(
            f"No session_*.npz files in {data_dir}, and it is not a Hunter "
            "VNA session/parent folder either.")
    sessions = {}
    for f in files:
        sid = os.path.basename(f).split("_")[1].split(".")[0]  # '01'
        d = np.load(f)
        X_raw = d["X"].astype(np.float32)       # (N, 8, 201)
        baseline = d["baseline"].astype(np.float32)  # (8, 201)
        y_pos = d["y_pos"].astype(np.int64)
        y_cell = d["y_cell"].astype(np.int64)
        xy = d["xy"].astype(np.float32)

        # to complex
        N = X_raw.shape[0]
        Xc = np.empty((N, 4, X_raw.shape[2]), dtype=np.complex64)
        for i in range(N):
            Xc[i] = _rows_to_complex(X_raw[i])
        base_c = _rows_to_complex(baseline)     # (4, F)

        sessions[sid] = dict(
            Xc=Xc, base_c=base_c, y_pos=y_pos, y_cell=y_cell,
            xy=xy, X_raw=X_raw,
        )
    return sessions


def build_dataset(sessions, mode: str = "sub_combo"):
    """Apply per-session calibration, stack across sessions, build dense 40-way labels.

    mode:
      'none'      : raw real rows (no calibration) — bad baseline ablation
      'sub'       : Xc - baseline  (then split to 8-ch real)
      'sub_combo' : (Xc - baseline) - per-session mean of (Xc - baseline)

    Returns:
      X     (N, 8, 201) float32
      y     (N,) int64    — dense label 0..K-1 over the *valid* positions
      sess  (N,) int64    — dense session index 0..S-1
      pos   (N,) int64    — original sparse position index 0..63 (for grouping)
      xy    (N, 2) float32
      label_to_pos : list mapping dense label -> sparse position
      sids  : list of session id strings in dense-index order
    """
    sids = sorted(sessions.keys())
    # determine valid positions = positions that appear in EVERY session
    pos_sets = [set(int(p) for p in sessions[s]["y_pos"]) for s in sids]
    valid = sorted(pos_sets[0].intersection(*pos_sets[1:]))
    label_to_pos = valid
    pos_to_label = {p: i for i, p in enumerate(valid)}

    Xs, ys, sess_idx, pos_orig, xys = [], [], [], [], []
    for si, sid in enumerate(sids):
        S = sessions[sid]
        Xc = S["Xc"]; base = S["base_c"]
        if mode == "none":
            X = S["X_raw"].copy()
        elif mode == "sub":
            Y = Xc - base[None, :, :]
            X = _split_complex_to_8ch(Y)
        elif mode == "sub_combo":
            Y = Xc - base[None, :, :]
            mu = Y.mean(axis=0, keepdims=True)
            Y = Y - mu
            X = _split_complex_to_8ch(Y)
        else:
            raise ValueError(f"unknown mode {mode}")

        # keep only valid positions
        keep = np.array([(int(p) in pos_to_label) for p in S["y_pos"]], dtype=bool)
        X = X[keep]
        yp = np.array([pos_to_label[int(p)] for p in S["y_pos"][keep]], dtype=np.int64)
        Xs.append(X)
        ys.append(yp)
        sess_idx.append(np.full(yp.shape[0], si, dtype=np.int64))
        pos_orig.append(S["y_pos"][keep].astype(np.int64))
        xys.append(S["xy"][keep].astype(np.float32))
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    sess = np.concatenate(sess_idx, axis=0)
    pos = np.concatenate(pos_orig, axis=0)
    xy = np.concatenate(xys, axis=0)
    return dict(
        X=X, y=y, sess=sess, pos=pos, xy=xy,
        label_to_pos=label_to_pos, sids=sids,
        num_classes=len(label_to_pos), num_sessions=len(sids),
    )


def per_session_zscore(X: np.ndarray, sess: np.ndarray):
    """Per-session z-score (each session uses ONLY its own stats). The Above 80 'KEY' trick.
    Returns Xn (same shape) and a list of (mu, sd) per session for later reuse."""
    Xn = X.copy()
    stats = {}
    for s in np.unique(sess):
        m = sess == s
        flat = X[m].reshape(int(m.sum()), -1)
        mu = flat.mean(axis=0)
        sd = flat.std(axis=0) + 1e-8
        # reshape back
        shape = X.shape[1:]
        Xn[m] = ((flat - mu) / sd).reshape((-1,) + shape).astype(np.float32)
        stats[int(s)] = (mu.reshape(shape).astype(np.float32),
                         sd.reshape(shape).astype(np.float32))
    return Xn, stats


def global_zscore(X_train: np.ndarray, X_test: np.ndarray):
    """Global z-score using train stats only (after per-session)."""
    flat = X_train.reshape(X_train.shape[0], -1)
    mu = flat.mean(axis=0); sd = flat.std(axis=0) + 1e-8
    shape = X_train.shape[1:]
    mu_t = mu.reshape(shape); sd_t = sd.reshape(shape)
    return ((X_train - mu_t) / sd_t).astype(np.float32), \
           ((X_test  - mu_t) / sd_t).astype(np.float32)


# ----------------------------------------------------------------------------
#  Format dispatcher  (legacy 2-port .npz  OR  Hunter 4-port CSV folders)
# ----------------------------------------------------------------------------
def detect_format(path: str) -> str:
    """Inspect `path` and return one of:

      "legacy_npz"   : a directory containing one or more session_*.npz files
                       (or a single session_*.npz file)
      "hunter_session"
                     : a directory that IS a Hunter session
                       (has session_metadata.txt + RnCmPp_T*.csv + baseline)
      "hunter_parent"
                     : a directory containing one or more Hunter session
                       subfolders
      "unknown"      : nothing recognisable

    `path` may point to either a file or a folder.  This is what lets the
    user select "the overall folder that contains the data" without having
    to know which sub-layout it is.
    """
    if not os.path.exists(path):
        return "unknown"

    # late import so data.py still imports if hunter_loader has an issue
    try:
        from hunter_loader import (is_hunter_session_folder,
                                   is_hunter_parent_folder)
    except Exception:
        is_hunter_session_folder = lambda p: False     # noqa: E731
        is_hunter_parent_folder  = lambda p: False     # noqa: E731

    if os.path.isfile(path):
        if path.lower().endswith(".npz") and "session_" in os.path.basename(path).lower():
            return "legacy_npz"
        return "unknown"

    # path is a directory.  Test Hunter shapes first because they have
    # distinctive metadata files.
    if is_hunter_session_folder(path):
        return "hunter_session"
    if is_hunter_parent_folder(path):
        return "hunter_parent"
    # legacy: directory containing session_*.npz
    if glob.glob(os.path.join(path, "session_*.npz")):
        return "legacy_npz"
    return "unknown"


def load_any(path: str, mode: str = "legacy"):
    """Auto-detect and load whatever data format lives at `path`.

    Returns the SAME dict-of-sessions structure as load_all_sessions()
    so the rest of the pipeline (build_dataset, per_session_zscore,
    physics features, predict_live) doesn't need to care which VNA the
    measurements came from.

    Args:
      path  a file or folder.  Accepted layouts:
              - a directory with session_*.npz files (2-port legacy)
              - a single session_*.npz file
              - a Hunter session folder (4-port)
              - a parent folder containing Hunter session subfolders
      mode  for Hunter data, "legacy" (default) maps the 4-port matrix
            down to the 4 S-parameters the trained models expect
            (S11,S12,S22,S21 at 201 freqs).  "full16" returns the full
            4-port matrix for new models.
    """
    fmt = detect_format(path)
    if fmt == "legacy_npz":
        if os.path.isfile(path):
            # one .npz file -> wrap it as a single-session dict
            d = np.load(path)
            X_raw = d["X"].astype(np.float32)
            baseline = d["baseline"].astype(np.float32)
            y_pos = d["y_pos"].astype(np.int64)
            y_cell = d["y_cell"].astype(np.int64)
            xy = d["xy"].astype(np.float32)
            N = X_raw.shape[0]
            Xc = np.empty((N, 4, X_raw.shape[2]), dtype=np.complex64)
            for i in range(N):
                Xc[i] = _rows_to_complex(X_raw[i])
            base_c = _rows_to_complex(baseline)
            sid = os.path.basename(path).split("_")[1].split(".")[0]
            return {sid: dict(Xc=Xc, base_c=base_c, y_pos=y_pos, y_cell=y_cell,
                              xy=xy, X_raw=X_raw)}
        return load_all_sessions(path)

    if fmt in ("hunter_session", "hunter_parent"):
        from hunter_loader import load_hunter_sessions
        return load_hunter_sessions(path, mode=mode)

    raise FileNotFoundError(
        f"Could not detect a known data layout at {path!r}.\n"
        "Expected one of:\n"
        "  - a folder with session_*.npz files (legacy 2-port)\n"
        "  - a Hunter VNA session folder (with session_metadata.txt)\n"
        "  - a parent folder containing Hunter session subfolders.")


def pick_data_folder(initialdir: str | None = None) -> str:
    """Tkinter folder picker shared by the launcher scripts."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    if initialdir is None:
        initialdir = os.path.expanduser("~\\Desktop")
    path = filedialog.askdirectory(
        title="Select data folder (legacy .npz directory OR Hunter VNA folder)",
        initialdir=initialdir,
        mustexist=True,
    )
    root.destroy()
    return path


if __name__ == "__main__":
    print("Loading sessions...")
    sess = load_all_sessions()
    for sid, S in sess.items():
        print(f"  session {sid}: Xc{S['Xc'].shape}  positions={len(set(S['y_pos']))}")
    print("\nBuilding dataset (sub_combo)...")
    D = build_dataset(sess, mode="sub_combo")
    print(f"  X={D['X'].shape}  y={D['y'].shape}  classes={D['num_classes']}  sessions={D['num_sessions']}")
    print(f"  per-session counts: {np.bincount(D['sess'])}")
    print(f"  per-class counts: min={np.bincount(D['y']).min()} max={np.bincount(D['y']).max()}")
