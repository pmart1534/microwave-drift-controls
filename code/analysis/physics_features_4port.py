"""Physics-aware features for 4-port (16 S-parameter) Hunter VNA data.

Generalises `physics_features.physics_features_from_complex` from 4 channels
(S11, S12, S22, S21) to 16 channels (the full 4-port matrix).

Channel layout assumed: input is (N, 16, F) complex with the row order matching
hunter_loader's full16 output:
  rows = [S_{ij} for i in 1..4 for j in 1..4]
        = S11, S12, S13, S14, S21, S22, S23, S24, S31, ..., S44

Per-channel features (× 16 channels):
  raw mag / phase / Re / Im     (4 × F)
  per-trace stats (mean/std/med/max/min on |S|)   (5 scalars)
  IFFT envelope, first n_time_bins (default 64)   (64 floats)
  -> 4F + 5 + 64 features per channel = 877 at F = 201

Cross-channel features:
  6 reciprocity residuals  |S_ij - S_ji|  for the 6 (i<j) pairs   (× 2: mag & phase)
  6 log-ratios  log10(|S_ii|/|S_ij|)  per active reflection vs each transmission

Total feature count at F = 201, n_time_bins = 64:
  16 × 877 + 6 × 2 × F + 6 × F  =  14,032 + 3,618  =  ~17,650 features.

That's ~4x the original 4296-D feature vector, which is appropriate since we
have 4x as many S-parameters carrying information.
"""
from __future__ import annotations
import numpy as np


# 4-port: 6 unique transmission pairs (i < j), with their two row indices in
# the 16-row layout.  Row index = (i-1)*4 + (j-1).
_TRANSMIT_PAIRS = []
for i in range(4):
    for j in range(i + 1, 4):
        idx_ij = i * 4 + j   # row index of S_{i+1,j+1}
        idx_ji = j * 4 + i   # row index of S_{j+1,i+1}
        _TRANSMIT_PAIRS.append(((i + 1, j + 1), idx_ij, idx_ji))

_REFL_INDICES = [0, 5, 10, 15]   # diagonal: S11, S22, S33, S44


def _sparam_label(row_idx: int) -> str:
    i = row_idx // 4 + 1
    j = row_idx % 4 + 1
    return f"S{i}{j}"


def physics_features_4port(Xc_cal: np.ndarray, n_time_bins: int = 64):
    """Build 4-port physics features.

    Args:
      Xc_cal       (N, 16, F) complex   calibrated S-parameters
      n_time_bins  int                   IFFT envelope length per channel

    Returns:
      X            (N, D) float32        feature matrix
      names        list[str]             feature names (length D)
    """
    assert Xc_cal.ndim == 3 and Xc_cal.shape[1] == 16, \
        f"expected (N, 16, F) input, got {Xc_cal.shape}"
    N, _, F = Xc_cal.shape

    mag = np.abs(Xc_cal).astype(np.float32)      # (N, 16, F)
    ph  = np.angle(Xc_cal).astype(np.float32)
    re  = np.real(Xc_cal).astype(np.float32)
    im  = np.imag(Xc_cal).astype(np.float32)

    feats = []
    names = []

    # Per-channel: raw mag/phase/re/im + stats + IFFT envelope
    for k in range(16):
        sp = _sparam_label(k)
        for arr, suf in [(mag[:, k], "mag"), (ph[:, k], "ph"),
                          (re[:, k], "re"), (im[:, k], "im")]:
            feats.append(arr)
            names += [f"{sp}_{suf}_f{j}" for j in range(F)]
        for stat_fn, suf in [(np.mean, "mean"), (np.std, "std"),
                              (np.median, "med"), (np.max, "max"), (np.min, "min")]:
            v = stat_fn(mag[:, k], axis=1, keepdims=True)
            feats.append(v.astype(np.float32))
            names.append(f"{sp}_mag_{suf}")
        td = np.fft.ifft(Xc_cal[:, k, :], axis=-1).astype(np.complex64)
        env = np.abs(td[:, :n_time_bins]).astype(np.float32)
        feats.append(env)
        names += [f"{sp}_ifft{j}" for j in range(n_time_bins)]

    # Reciprocity residuals: |S_ij - S_ji|  for the 6 (i<j) pairs
    eps = 1e-9
    for (i, j), idx_ij, idx_ji in _TRANSMIT_PAIRS:
        rec_mag = np.abs(Xc_cal[:, idx_ij] - Xc_cal[:, idx_ji]).astype(np.float32)
        feats.append(rec_mag)
        names += [f"rec{i}{j}_mag_f{f}" for f in range(F)]
        rec_ph  = np.angle(Xc_cal[:, idx_ij] / (Xc_cal[:, idx_ji] + eps)).astype(np.float32)
        feats.append(rec_ph)
        names += [f"rec{i}{j}_ph_f{f}" for f in range(F)]

    # Log-ratios: at each port, |S_ii| / |S_ij| for each transmission off that port.
    # This captures how much energy stays vs leaves the antenna at each freq.
    for ii, refl_idx in enumerate(_REFL_INDICES):
        port = ii + 1
        for jj in range(4):
            if jj == ii: continue
            trans_idx = ii * 4 + jj    # S_{port, jj+1}
            ratio = (np.log10(mag[:, refl_idx] + eps) -
                     np.log10(mag[:, trans_idx] + eps)).astype(np.float32)
            feats.append(ratio)
            names += [f"r_S{port}{port}_S{port}{jj+1}_f{f}" for f in range(F)]

    flat = []
    for fa in feats:
        if fa.ndim == 1:
            flat.append(fa.reshape(-1, 1))
        else:
            flat.append(fa)
    X = np.concatenate(flat, axis=1).astype(np.float32)
    return X, names


if __name__ == "__main__":
    # Smoke test: random complex input
    N, F = 8, 201
    Xc = (np.random.randn(N, 16, F) + 1j * np.random.randn(N, 16, F)).astype(np.complex64)
    X, names = physics_features_4port(Xc)
    print(f"Input:  {Xc.shape}  ({Xc.dtype})")
    print(f"Output: {X.shape}   ({X.dtype})")
    print(f"Feature count breakdown:")
    print(f"  per-channel (16 × ({4*F} mag/ph/re/im + 5 stats + 64 IFFT)) = {16 * (4*F + 5 + 64)}")
    print(f"  reciprocity (6 pairs × 2 × {F})                              = {6 * 2 * F}")
    print(f"  log-ratios  (4 ports × 3 transmissions × {F})                = {4 * 3 * F}")
    print(f"  total expected = {16 * (4*F + 5 + 64) + 6 * 2 * F + 4 * 3 * F}")
