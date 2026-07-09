"""Quantify drift timescales in the empty-phantom session.

Question: how often do we need to take a fresh baseline to cancel drift?
  - Compare DRIFT step between consecutive positions
  - to WITHIN-POSITION trial-to-trial noise (the floor)
  - to OBJECT effect (beet vs empty mean diff)

Reads:
  C:/.../Empty Phantom .../Imager_BreastPhantom_..._Empty_..._20260608_1047.mat
  C:/.../060426 Data/Beet 0.5cm/Updated_20260604_1251.mat

Writes:
  results/drift_timescale/per_channel_drift.png
  results/drift_timescale/summary.json
"""
from __future__ import annotations
import os, sys, json
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from data import _rows_to_complex

EMPTY = r"C:\Users\peter\Desktop\EM Imaging\BreastPhantom\Empty Phantom Test\Empty Phantom\Empty Phantom\Imager_BreastPhantom_A2_NoGland_ManetobaAntenna_Empty_64Pos_16Trials_20260608_1047.mat"
BEET  = r"C:\Users\peter\Desktop\EM Imaging\BreastPhantom\060426 Data\Beet 0.5cm\Updated_20260604_1251.mat"
OUT_DIR = os.path.join(ROOT, "results", "drift_timescale")
os.makedirs(OUT_DIR, exist_ok=True)


def load_in_acquisition_order(path):
    """Return (Xc, base_c, pos_acquisition_order).
       Xc.shape = (numPositions, 16, 4, 201) complex64 — kept grouped by position
       so we can compute within-pos and across-pos stats."""
    M = loadmat(path, squeeze_me=False)
    data = M["data"]
    base_c = _rows_to_complex(M["baseline"].astype(np.float32))
    Xc_pos = []        # list per position of (16, 4, 201)
    pos_idx = []
    for p in range(data.shape[0]):
        trials = []
        for t in range(data.shape[1]):
            cell = data[p, t]
            if cell is None: continue
            arr = np.asarray(cell)
            if arr.dtype.kind in ("U", "S", "O"): continue
            if arr.shape != (8, 201): continue
            arr = arr.astype(np.float32)
            if np.isnan(arr).any() or np.isinf(arr).any(): continue
            trials.append(_rows_to_complex(arr))
        if len(trials) == 16:
            Xc_pos.append(np.stack(trials, axis=0))
            pos_idx.append(p)
    Xc = np.stack(Xc_pos, axis=0)                    # (P, 16, 4, F)
    pos_idx = np.array(pos_idx, dtype=np.int64)
    return Xc, base_c, pos_idx


# ----------------------------------------------------------------------------
def analyze(label, Xc, base_c):
    """Return summary stats per S-param channel.

    For each S-param k:
      pos_mean[k, p, F]   = mean(magnitude over 16 trials)  at position p
      within_std[k, p, F] = std (over 16 trials)            at position p
      drift_step[k, p, F] = pos_mean[k, p] - pos_mean[k, p-1]

    We report scalars: median across positions and frequencies of
      (a) within-position trial std
      (b) inter-position drift step magnitude
    """
    P, T, K, F = Xc.shape
    mag = np.abs(Xc)                                  # (P, T, K, F)
    pos_mean = mag.mean(axis=1)                       # (P, K, F)
    within_std = mag.std(axis=1)                      # (P, K, F)
    drift_step = np.abs(np.diff(pos_mean, axis=0))    # (P-1, K, F)

    out = dict(
        per_channel=[],
        position_mean=pos_mean,
        within_std=within_std,
        drift_step=drift_step,
    )
    for k, sp in enumerate(["S11", "S12", "S22", "S21"]):
        within_med  = float(np.median(within_std[:, k, :]))
        drift_med   = float(np.median(drift_step[:, k, :]))
        signal_med  = float(np.median(pos_mean[:, k, :]))   # scale reference
        out["per_channel"].append(dict(
            sparam=sp,
            within_pos_trial_std_median=within_med,
            interpos_drift_step_median=drift_med,
            mean_magnitude=signal_med,
            drift_to_noise_ratio=drift_med / max(within_med, 1e-12),
        ))
        print(f"  {label:6s} {sp}: within-pos trial-std = {within_med:.4e}   "
              f"inter-pos drift = {drift_med:.4e}   "
              f"ratio = {drift_med / max(within_med,1e-12):6.2f}x   "
              f"(mean |S| = {signal_med:.3f})")
    return out


# ----------------------------------------------------------------------------
def compute_object_effect(empty_Xc, beet_Xc):
    """Object effect = |mean(beet, over trials and positions) - mean(empty,...)|
    per channel, per frequency.  This is the *typical* per-frequency
    perturbation the beet causes."""
    empty_mean = np.abs(empty_Xc).mean(axis=(0, 1))   # (K, F)
    beet_mean  = np.abs(beet_Xc ).mean(axis=(0, 1))
    obj = np.abs(beet_mean - empty_mean)              # (K, F)
    return obj


# ----------------------------------------------------------------------------
def plot_per_channel(empty_an, beet_an, obj_effect, out_path):
    """4-panel figure: for each S-param, plot
      - line: pos_mean magnitude vs position index (empty session)
      - bars at each position: within-pos trial std
    Lets the eye see whether drift is bigger than the trial-noise floor.
    """
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    P = empty_an["position_mean"].shape[0]
    pos_idx = np.arange(P)
    for k, sp in enumerate(["S11", "S12", "S22", "S21"]):
        ax = axes[k]
        # use median over freqs as a 1-number-per-position summary
        emp = np.median(empty_an["position_mean"][:, k, :], axis=1)
        std_emp = np.median(empty_an["within_std"][:, k, :], axis=1)
        bet = np.median(beet_an["position_mean"][:, k, :], axis=1)
        std_bet = np.median(beet_an["within_std"][:, k, :], axis=1)
        ax.plot(pos_idx, emp, color="#a14d3a", lw=1.5, label="EMPTY pos-mean |S|")
        ax.fill_between(pos_idx, emp - std_emp, emp + std_emp,
                         color="#a14d3a", alpha=0.20, label="EMPTY within-pos std")
        ax.plot(pos_idx, bet, color="#4a7c2a", lw=1.5, label="BEET 0.5 pos-mean |S|")
        ax.fill_between(pos_idx, bet - std_bet, bet + std_bet,
                         color="#4a7c2a", alpha=0.20, label="BEET within-pos std")
        ch_obj = float(np.median(obj_effect[k, :]))
        ax.set_title(f"{sp}  --  empty drift step (med) = "
                     f"{empty_an['per_channel'][k]['interpos_drift_step_median']:.2e}   "
                     f"|   beet object effect (med) = {ch_obj:.2e}",
                     fontsize=10)
        ax.set_ylabel("median |S| over freq")
        ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=2)
    axes[-1].set_xlabel("Position index (acquisition order)")
    fig.suptitle("Drift vs trial-noise vs object-effect, per S-parameter\n"
                 "Lines = position means.  Shaded = within-position trial std.  "
                 "Jagged line = inter-position drift step.", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_ratio_summary(empty_an, beet_an, obj_effect, out_path):
    """Bar chart: per channel, drift-step / trial-std and object-effect / trial-std.
    >> 1 means drift (or object) dominates noise; ~ 1 means they're comparable."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sparams = ["S11", "S12", "S22", "S21"]
    drift_ratio = [empty_an["per_channel"][k]["drift_to_noise_ratio"] for k in range(4)]
    obj_ratio = []
    for k in range(4):
        med_trial_std = empty_an["per_channel"][k]["within_pos_trial_std_median"]
        med_obj = float(np.median(obj_effect[k, :]))
        obj_ratio.append(med_obj / max(med_trial_std, 1e-12))
    x = np.arange(4); w = 0.38
    ax.bar(x - w/2, drift_ratio, w, color="#a14d3a", label="inter-pos drift / trial-std")
    ax.bar(x + w/2, obj_ratio,   w, color="#4a7c2a", label="object effect / trial-std")
    ax.axhline(1, color="black", ls="--", lw=0.8, alpha=0.6, label="ratio = 1 (drift/object = noise)")
    ax.set_xticks(x); ax.set_xticklabels(sparams, fontsize=11)
    ax.set_ylabel("magnitude / trial-to-trial noise floor")
    ax.set_title("How big is drift vs object effect, relative to trial noise?\n"
                 "Bars > 1 mean that signal stands out above the noise.",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    for i, (d, o) in enumerate(zip(drift_ratio, obj_ratio)):
        ax.text(x[i] - w/2, d + 0.05, f"{d:.1f}x", ha="center", fontsize=8)
        ax.text(x[i] + w/2, o + 0.05, f"{o:.1f}x", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ----------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("DRIFT TIMESCALE ANALYSIS")
    print("=" * 78)
    print(f"  EMPTY: {os.path.basename(EMPTY)}")
    print(f"  BEET : {os.path.basename(BEET)}")
    print()

    empty_Xc, empty_base_c, empty_pos = load_in_acquisition_order(EMPTY)
    beet_Xc,  beet_base_c,  beet_pos  = load_in_acquisition_order(BEET)
    print(f"  EMPTY shape: {empty_Xc.shape}  (positions, trials, sparams, freqs)")
    print(f"  BEET  shape: {beet_Xc.shape}")

    print("\nPer-channel statistics:")
    empty_an = analyze("EMPTY", empty_Xc, empty_base_c)
    print()
    beet_an  = analyze("BEET ", beet_Xc,  beet_base_c)
    print()

    obj_effect = compute_object_effect(empty_Xc, beet_Xc)
    print("Object effect (|<beet> - <empty>|) median per channel:")
    for k, sp in enumerate(["S11", "S12", "S22", "S21"]):
        print(f"  {sp}: {float(np.median(obj_effect[k, :])):.4e}")

    out_png = os.path.join(OUT_DIR, "per_channel_drift.png")
    plot_per_channel(empty_an, beet_an, obj_effect, out_png)

    out_ratio = os.path.join(OUT_DIR, "ratio_summary.png")
    plot_ratio_summary(empty_an, beet_an, obj_effect, out_ratio)

    # ---- printed recommendation ----
    print("\n" + "=" * 78)
    print("RECOMMENDATION")
    print("=" * 78)
    print(f"{'channel':<6s}  {'trial std':>10s}  {'drift step':>11s}  "
          f"{'object eff':>11s}  {'drift/obj':>9s}")
    print("-" * 78)
    avg_drift_over_obj = []
    for k, sp in enumerate(["S11", "S12", "S22", "S21"]):
        d = empty_an["per_channel"][k]["interpos_drift_step_median"]
        n = empty_an["per_channel"][k]["within_pos_trial_std_median"]
        o = float(np.median(obj_effect[k, :]))
        print(f"{sp:<6s}  {n:>10.3e}  {d:>10.3e}  {o:>10.3e}  {d / max(o, 1e-12):>7.2f}x")
        avg_drift_over_obj.append(d / max(o, 1e-12))
    avg = float(np.mean(avg_drift_over_obj))
    print("-" * 78)
    print(f"  Mean drift/object ratio across channels: {avg:.2f}x")
    if avg < 0.3:
        verdict = ("Drift is much SMALLER than the object effect.  Per-position\n"
                   "  baselines aren't critical; baseline every 5-10 positions OK.")
    elif avg < 1.0:
        verdict = ("Drift is COMPARABLE to but smaller than the object effect.\n"
                   "  Baseline every 2-3 positions, OR per-position if data is critical.")
    elif avg < 3.0:
        verdict = ("Drift is COMPARABLE to or LARGER than the object effect.\n"
                   "  Per-position baselines strongly recommended.")
    else:
        verdict = ("Drift DOMINATES the object effect.  Per-position baselines\n"
                   "  required OR switch to interleaved acquisition entirely.")
    print(f"  Verdict: {verdict}")

    out = dict(
        empty_per_channel=empty_an["per_channel"],
        beet_per_channel=beet_an["per_channel"],
        object_effect_per_channel_median={
            ["S11", "S12", "S22", "S21"][k]: float(np.median(obj_effect[k, :]))
            for k in range(4)
        },
        mean_drift_over_object_ratio=avg,
        verdict=verdict.replace("\n  ", " "),
    )
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  saved {os.path.join(OUT_DIR, 'summary.json')}")


if __name__ == "__main__":
    main()
