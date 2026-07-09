"""Honest drift-vs-object comparison.

Earlier version compared:
  empty session mean(|S|) vs beet session mean(|S|)
which conflates object effect with cross-day drift (4 days apart).

This version uses each session's OWN baseline:
  drift          = empty session: inter-position drift step (median)
  trial noise    = empty session: within-position 16-trial std (median)
  object effect  = beet  session: |mean(trial) - same-session baseline|
                                  (median across positions and freqs)
That isolates the within-session perturbation the beet causes from
cross-day drift contamination.

Three sizes of beet are evaluated so you can see if the object effect
scales with size as expected.

Outputs:
  results/drift_vs_object/comparison.png  -- log-scale per-channel chart
  results/drift_vs_object/summary.json
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
BEETS = {
    "Beet 1.0 cm":  r"C:\Users\peter\Desktop\EM Imaging\BreastPhantom\060426 Data\Beet 1cm\Imager_BreastPhantom_A2_NoGland_ManetobaAntenna_Beet(1cm)_64Pos_16Trials_20260604_1051.mat",
    "Beet 0.75 cm": r"C:\Users\peter\Desktop\EM Imaging\BreastPhantom\060426 Data\Beet 0.75cm\Imager_BreastPhantom_A2_NoGland_ManetobaAntenna_Beet(0.75cm)_64Pos_16Trials_20260604_1133.mat",
    "Beet 0.5 cm":  r"C:\Users\peter\Desktop\EM Imaging\BreastPhantom\060426 Data\Beet 0.5cm\Updated_20260604_1251.mat",
}
OUT_DIR = os.path.join(ROOT, "results", "drift_vs_object")
os.makedirs(OUT_DIR, exist_ok=True)
SPARAMS = ["S11", "S12", "S22", "S21"]


def load_per_pos(path):
    """Return Xc (P, 16, 4, F), baseline_c (4, F)."""
    M = loadmat(path, squeeze_me=False)
    data = M["data"]
    base_c = _rows_to_complex(M["baseline"].astype(np.float32))
    grouped = []
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
            grouped.append(np.stack(trials, axis=0))
    return np.stack(grouped, axis=0), base_c


# ----------------------------------------------------------------------------
def main():
    print("Loading EMPTY...")
    Xe, base_e = load_per_pos(EMPTY)
    mag_e = np.abs(Xe)                                # (P, 16, 4, F)

    # 1) trial-to-trial noise (within-position std), median across pos & freq
    trial_noise = np.median(mag_e.std(axis=1), axis=(0, 2))   # (4,)

    # 2) inter-position drift step, median across pos & freq
    pos_mean_e = mag_e.mean(axis=1)                          # (P, 4, F)
    drift_step = np.abs(np.diff(pos_mean_e, axis=0))         # (P-1, 4, F)
    drift_step = np.median(drift_step, axis=(0, 2))          # (4,)

    print(f"  trial noise (per S): {trial_noise}")
    print(f"  drift step (per S):  {drift_step}")

    # 3) object effect per beet, using each session's OWN baseline
    object_effects = {}
    for label, path in BEETS.items():
        print(f"Loading {label}...")
        Xb, base_b = load_per_pos(path)
        mag_b = np.abs(Xb)                                   # (P, 16, 4, F)
        base_mag = np.abs(base_b)                            # (4, F)
        per_pos_mean = mag_b.mean(axis=1)                    # (P, 4, F)
        obj_per_pos = np.abs(per_pos_mean - base_mag[None, :, :])  # (P, 4, F)
        obj_median = np.median(obj_per_pos, axis=(0, 2))     # (4,)
        object_effects[label] = obj_median
        print(f"  {label} object effect (per S): {obj_median}")

    # ---- print table ----
    print("\n" + "=" * 92)
    print(f"{'Channel':<8s}  {'Trial noise':>12s}  {'Drift step':>12s}  "
          f"{'Beet 1.0 obj':>14s}  {'Beet 0.75 obj':>14s}  {'Beet 0.5 obj':>14s}")
    print("-" * 92)
    for k, sp in enumerate(SPARAMS):
        row = f"{sp:<8s}  {trial_noise[k]:>11.3e}  {drift_step[k]:>11.3e}  "
        for label in BEETS:
            row += f"  {object_effects[label][k]:>12.3e}"
        print(row)
    print("=" * 92)

    # ---- ratios ----
    print("\nRATIOS (vs trial noise floor):")
    print(f"{'Channel':<8s}  {'drift/noise':>12s}  "
          f"{'B1.0/noise':>11s}  {'B0.75/noise':>11s}  {'B0.5/noise':>11s}")
    print("-" * 72)
    for k, sp in enumerate(SPARAMS):
        r_drift = drift_step[k] / trial_noise[k]
        r_objs = [object_effects[lab][k] / trial_noise[k] for lab in BEETS]
        print(f"{sp:<8s}  {r_drift:>11.2f}x  "
              + "  ".join(f"{r:>9.1f}x" for r in r_objs))
    print("-" * 72)

    # ---- per-S ratios object/drift ----
    print("\nObject-effect / drift-step (per channel):")
    print(f"{'Channel':<8s}  {'B1.0/drift':>12s}  {'B0.75/drift':>12s}  {'B0.5/drift':>12s}")
    print("-" * 56)
    for k, sp in enumerate(SPARAMS):
        ratios = [object_effects[lab][k] / drift_step[k] for lab in BEETS]
        print(f"{sp:<8s}  " + "  ".join(f"{r:>9.1f}x" for r in ratios))

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_groups = ["Trial-to-trial\nnoise (empty)",
                  "Inter-position\ndrift step (empty)",
                  "Object effect\nBeet 1.0 cm",
                  "Object effect\nBeet 0.75 cm",
                  "Object effect\nBeet 0.5 cm"]
    n_groups = len(bar_groups); n_channels = 4
    x = np.arange(n_groups); w = 0.18
    colors = ["#5b6c8a", "#a14d3a", "#4a7c2a", "#7c5b3a", "#3a5c7c"]

    for k, sp in enumerate(SPARAMS):
        vals = [trial_noise[k], drift_step[k]] + \
               [object_effects[lab][k] for lab in BEETS]
        offset = (k - 1.5) * w
        ax.bar(x + offset, vals, w, label=sp, color=colors[k % len(colors)])
        for i, v in enumerate(vals):
            ax.text(x[i] + offset, v * 1.15, f"{v:.1e}",
                    ha="center", fontsize=6.5, rotation=90, va="bottom")

    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(bar_groups, fontsize=9)
    ax.set_ylabel("|S| magnitude  (linear units, log-scale axis)", fontsize=11)
    ax.set_title("Drift vs object effect, per S-parameter\n"
                 "Object effect computed as |mean(trial) - same-session baseline|\n"
                 "(i.e. WITHIN-SESSION only -- no cross-day drift contamination)",
                 fontsize=11)
    ax.legend(title="S-parameter", loc="upper left", fontsize=9, ncol=1)
    ax.grid(alpha=0.3, axis="y", which="both")
    ax.set_ylim(1e-5, 1)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "comparison.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  saved {out}")

    # ---- also a "ratio" version that's easier to read ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # left: ratio vs trial noise (log scale)
    ax = axes[0]
    bar_groups2 = ["drift", "Beet 1.0 cm", "Beet 0.75 cm", "Beet 0.5 cm"]
    x2 = np.arange(len(bar_groups2)); w2 = 0.18
    for k, sp in enumerate(SPARAMS):
        vals = [drift_step[k] / trial_noise[k]] + \
               [object_effects[lab][k] / trial_noise[k] for lab in BEETS]
        offset = (k - 1.5) * w2
        ax.bar(x2 + offset, vals, w2, label=sp, color=colors[k % len(colors)])
        for i, v in enumerate(vals):
            ax.text(x2[i] + offset, v * 1.10, f"{v:.1f}x",
                    ha="center", fontsize=7, rotation=90, va="bottom")
    ax.axhline(1, color="black", ls="--", lw=0.8)
    ax.set_yscale("log")
    ax.set_xticks(x2); ax.set_xticklabels(bar_groups2, fontsize=9)
    ax.set_ylabel("ratio to trial-to-trial noise floor")
    ax.set_title("How big is each effect, in units of trial noise?")
    ax.legend(title="S-param", loc="upper left", fontsize=8)
    ax.grid(alpha=0.3, axis="y", which="both")
    ax.set_ylim(0.1, 1000)

    # right: object / drift ratio per beet
    ax = axes[1]
    x3 = np.arange(3); w3 = 0.18
    bar_groups3 = ["Beet 1.0 cm", "Beet 0.75 cm", "Beet 0.5 cm"]
    for k, sp in enumerate(SPARAMS):
        vals = [object_effects[lab][k] / drift_step[k] for lab in BEETS]
        offset = (k - 1.5) * w3
        ax.bar(x3 + offset, vals, w3, label=sp, color=colors[k % len(colors)])
        for i, v in enumerate(vals):
            ax.text(x3[i] + offset, v * 1.10, f"{v:.0f}x",
                    ha="center", fontsize=7, rotation=90, va="bottom")
    ax.axhline(1, color="black", ls="--", lw=0.8)
    ax.set_yscale("log")
    ax.set_xticks(x3); ax.set_xticklabels(bar_groups3, fontsize=9)
    ax.set_ylabel("ratio: object effect / drift step")
    ax.set_title("Object effect vs drift  (per channel)\n"
                 "Bar > 1 means object is bigger than per-position drift")
    ax.legend(title="S-param", loc="upper left", fontsize=8)
    ax.grid(alpha=0.3, axis="y", which="both")

    fig.tight_layout()
    out2 = os.path.join(OUT_DIR, "ratios.png")
    fig.savefig(out2, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out2}")

    # ---- save JSON ----
    out_data = dict(
        trial_noise={SPARAMS[k]: float(trial_noise[k]) for k in range(4)},
        drift_step ={SPARAMS[k]: float(drift_step [k]) for k in range(4)},
        object_effects={lab: {SPARAMS[k]: float(object_effects[lab][k]) for k in range(4)}
                        for lab in BEETS},
    )
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"  saved {os.path.join(OUT_DIR, 'summary.json')}")


if __name__ == "__main__":
    main()
