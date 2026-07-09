"""Minimal physics features for a SINGLE S-parameter channel.

For comparing what happens when you strip down from 4-port (16 channels,
18,792 features) or 2-port-equivalent (4 channels, 4,296 features) all the
way to a SINGLE channel of S-parameter data.

The interesting case: use only S11 (port 1's reflection coefficient) and
see how well a model can localize positions with that alone.

Input contract:
  Xc_cal  (N, 1, F) complex   one calibrated S-param channel, N trials, F freq points
  -- OR --
  Xc_cal  (N, F) complex      auto-promoted to (N, 1, F)

Per-channel features (× 1 channel):
  - magnitude (dB), phase (rad), real, imaginary   (4 × F)
  - per-trace stats on |S|: mean/std/median/max/min (5)
  - IFFT envelope, first n_time_bins (default 64)   (n_time_bins)

Total at F = 201, n_time_bins = 64:
  4 * 201 + 5 + 64 = 873 features
"""
from __future__ import annotations
import numpy as np


def physics_features_1channel(Xc_cal: np.ndarray, n_time_bins: int = 64,
                                channel_name: str = "S11"):
    """Build single-channel physics features.

    Args:
      Xc_cal       (N, 1, F) or (N, F) complex   calibrated S-parameter channel
      n_time_bins  int                           IFFT envelope length
      channel_name str                           label for feature names

    Returns:
      X     (N, D) float32
      names list[str]
    """
    if Xc_cal.ndim == 2:
        Xc_cal = Xc_cal[:, None, :]                   # promote to (N, 1, F)
    assert Xc_cal.ndim == 3 and Xc_cal.shape[1] == 1, \
        f"expected (N, 1, F) or (N, F) input, got {Xc_cal.shape}"
    N, _, F = Xc_cal.shape
    sp = channel_name

    mag = np.abs(Xc_cal).astype(np.float32)           # (N, 1, F)
    ph  = np.angle(Xc_cal).astype(np.float32)
    re  = np.real(Xc_cal).astype(np.float32)
    im  = np.imag(Xc_cal).astype(np.float32)

    feats = []
    names = []

    # Raw mag/ph/re/im
    for arr, suf in [(mag[:, 0], "mag"), (ph[:, 0], "ph"),
                      (re[:, 0], "re"),  (im[:, 0], "im")]:
        feats.append(arr)
        names += [f"{sp}_{suf}_f{j}" for j in range(F)]

    # Per-trace stats on magnitude
    for stat_fn, suf in [(np.mean, "mean"), (np.std, "std"),
                          (np.median, "med"), (np.max, "max"), (np.min, "min")]:
        v = stat_fn(mag[:, 0], axis=1, keepdims=True)
        feats.append(v.astype(np.float32))
        names.append(f"{sp}_mag_{suf}")

    # IFFT envelope (TDR proxy)
    td = np.fft.ifft(Xc_cal[:, 0, :], axis=-1).astype(np.complex64)
    env = np.abs(td[:, :n_time_bins]).astype(np.float32)
    feats.append(env)
    names += [f"{sp}_ifft{j}" for j in range(n_time_bins)]

    flat = []
    for fa in feats:
        if fa.ndim == 1:
            flat.append(fa.reshape(-1, 1))
        else:
            flat.append(fa)
    X = np.concatenate(flat, axis=1).astype(np.float32)
    return X, names


if __name__ == "__main__":
    # Smoke test
    N, F = 8, 201
    Xc = (np.random.randn(N, 1, F) + 1j * np.random.randn(N, 1, F)).astype(np.complex64)
    X, names = physics_features_1channel(Xc, channel_name="S11")
    print(f"Input:  {Xc.shape}")
    print(f"Output: {X.shape}  -- {len(names)} feature names")
    expected = 4 * F + 5 + 64
    print(f"Expected: {expected}")
    assert X.shape[1] == expected
    print("OK")
