"""Session-to-session drift analysis - matches what detdifplot.py shows.

Trial-to-trial CV (see drift_analysis_by_port.py) measures noise WITHIN
a session at a fixed position. That's not what Peter saw in detdifplot -
he saw session MEANS shifting between sessions.

Metric here:
  For each position p, S-param k, freq f:
    mu_s = mean_trial(|S_k(p, f, trial)|)   in session s
  Session-to-session drift = mean over positions & freq of
    |mu_s - mu_{s-1}|    (magnitude of mean shift between consecutive sessions)

Then group by port involvement and compare port 3 vs 2 vs 4.
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hunter_loader import load_hunter_session

DATA_DIR = Path(r"C:\Users\peter\Desktop\EM Imaging\Detectable Difference\Data\DriftTest\Sam Med Antenna")
OUT_DIR  = Path(HERE).parent / "results" / "drift_test_sam_med"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SESSION_NAMES = [
    "BreastPhantom_A3_Nothing_20260701_1438_SamMedSep_001",
    "BreastPhantom_A3_Nothing_20260701_1509_SamMedSep_002",
    "BreastPhantom_A3_Nothing_20260701_1536_SamMedSep_003",
    "BreastPhantom_A3_Nothing_20260701_1603_SamMedSep_004",
]
S_NAMES = [f"S{i}{j}" for i in (1, 2, 3, 4) for j in (1, 2, 3, 4)]


def involves_port(sname, port):
    return str(port) in sname[1:]


def load_all():
    out = {}
    for name in SESSION_NAMES:
        folder = DATA_DIR / name
        if not folder.exists():
            print(f"[SKIP] {folder}"); continue
        print(f"[LOAD] {name.split('_')[-1]}")
        d = load_hunter_session(str(folder), mode="full16")
        out[name.split("_")[-1]] = d
    return out


def per_position_mean_mag(sess):
    """Return dict pos -> (16, F) array of |S| averaged over trials."""
    Xc = sess["Xc"]; ypos = sess["y_pos"]
    mag = np.abs(Xc)
    out = {}
    for p in np.unique(ypos):
        m = ypos == p
        out[int(p)] = mag[m].mean(0)   # (16, F)
    return out


def main():
    sessions = load_all()
    if len(sessions) < 2:
        print("Need >=2 sessions"); return
    sids = list(sessions.keys())

    # Compute per-session per-position mean |S|
    print("[COMPUTE] per-position mean |S| per session ...")
    per_sess_pos_mag = {sid: per_position_mean_mag(sess) for sid, sess in sessions.items()}

    # Positions common to all sessions
    common_pos = set.intersection(*[set(v.keys()) for v in per_sess_pos_mag.values()])
    common_pos = sorted(common_pos)
    print(f"[INFO] {len(common_pos)} positions common to all sessions")

    # For each consecutive session pair, compute per-position, per-S-param drift
    # drift[s_idx, k] = median over positions of mean-over-freq of |mu_s - mu_{s-1}|
    n_sess = len(sids)
    n_sp   = 16
    drift  = np.zeros((n_sess - 1, n_sp))
    rel_drift = np.zeros((n_sess - 1, n_sp))   # normalized by session-1 mean magnitude

    for si in range(1, n_sess):
        s_prev = sids[si - 1]; s_curr = sids[si]
        for k in range(n_sp):
            per_pos_abs = []
            per_pos_rel = []
            for p in common_pos:
                mu_prev = per_sess_pos_mag[s_prev][p][k]   # (F,)
                mu_curr = per_sess_pos_mag[s_curr][p][k]
                abs_diff = np.abs(mu_curr - mu_prev).mean()
                per_pos_abs.append(abs_diff)
                per_pos_rel.append(abs_diff / (mu_prev.mean() + 1e-15))
            drift[si - 1, k]     = np.median(per_pos_abs)
            rel_drift[si - 1, k] = np.median(per_pos_rel)

    # ----------------------------------------------------------------------
    # Aggregate per port
    # ----------------------------------------------------------------------
    per_port_rel = {p: np.zeros(n_sess - 1) for p in (1, 2, 3, 4)}
    per_port_abs = {p: np.zeros(n_sess - 1) for p in (1, 2, 3, 4)}
    for si in range(n_sess - 1):
        for p in (1, 2, 3, 4):
            mask = np.array([involves_port(sn, p) for sn in S_NAMES])
            per_port_rel[p][si] = rel_drift[si, mask].mean()
            per_port_abs[p][si] = drift[si, mask].mean()

    # ----------------------------------------------------------------------
    # PLOT 1: session-to-session drift per port
    # ----------------------------------------------------------------------
    pair_labels = [f"{sids[i]}->{sids[i+1]}" for i in range(n_sess - 1)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for p, color in zip((1, 2, 3, 4), ("steelblue", "olive", "firebrick", "seagreen")):
        lw = 2.5 if p == 3 else 1.5
        label = f"Port {p} (SUSPECT)" if p == 3 else f"Port {p}"
        axes[0].plot(pair_labels, per_port_rel[p], "o-", label=label, color=color, lw=lw)
    axes[0].set_ylabel("Relative session-to-session drift\n(mean |ΔμS| / μS)")
    axes[0].set_xlabel("Session pair")
    axes[0].set_title("Session-to-session drift per port\n(higher = larger mean shift between sessions)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Ratio port3 vs mean(1,2,4)
    baseline = np.mean([per_port_rel[1], per_port_rel[2], per_port_rel[4]], axis=0)
    ratio = per_port_rel[3] / baseline
    axes[1].bar(pair_labels, ratio, color="firebrick")
    axes[1].axhline(1.0, color="k", ls="--", lw=1, label="Equal to other ports")
    axes[1].set_ylabel("Port-3 drift / mean(port 1,2,4)")
    axes[1].set_title("Antenna-3 excess drift ratio\n(session-to-session)")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "session_to_session_drift.png", dpi=140)
    plt.close()
    print(f"[SAVE] {OUT_DIR/'session_to_session_drift.png'}")

    # ----------------------------------------------------------------------
    # PLOT 2: heatmap 16 S-params x session pairs
    # ----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 8))
    im = ax.imshow(rel_drift.T, aspect="auto", cmap="magma")
    ax.set_yticks(range(16)); ax.set_yticklabels(S_NAMES, fontsize=9)
    ax.set_xticks(range(n_sess - 1)); ax.set_xticklabels(pair_labels, rotation=30, ha="right")
    ax.set_xlabel("Session pair"); ax.set_title("Session-to-session relative drift\n(per S-parameter)")
    for k, sn in enumerate(S_NAMES):
        if involves_port(sn, 3):
            ax.get_yticklabels()[k].set_color("red")
            ax.get_yticklabels()[k].set_fontweight("bold")
    plt.colorbar(im, ax=ax, label="Relative drift")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "session_to_session_heatmap.png", dpi=140)
    plt.close()
    print(f"[SAVE] {OUT_DIR/'session_to_session_heatmap.png'}")

    # ----------------------------------------------------------------------
    # PRINT SUMMARY
    # ----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SESSION-TO-SESSION DRIFT (matches what detdifplot.py shows)")
    print("=" * 70)
    print(f"{'Pair':<12} {'Port1':<10} {'Port2':<10} {'Port3':<10} {'Port4':<10} {'P3/mean(P1,2,4)':<15}")
    for si in range(n_sess - 1):
        p3 = per_port_rel[3][si]
        base = np.mean([per_port_rel[1][si], per_port_rel[2][si], per_port_rel[4][si]])
        print(f"{pair_labels[si]:<12} "
              f"{per_port_rel[1][si]:<10.4f} "
              f"{per_port_rel[2][si]:<10.4f} "
              f"{per_port_rel[3][si]:<10.4f} "
              f"{per_port_rel[4][si]:<10.4f} "
              f"{p3/base:<15.2f}")

    # Save JSON
    result = {
        "pair_labels": pair_labels,
        "per_port_relative_drift": {p: per_port_rel[p].tolist() for p in (1, 2, 3, 4)},
        "port3_excess_ratio": (per_port_rel[3] / baseline).tolist(),
    }
    (OUT_DIR / "session_to_session_summary.json").write_text(json.dumps(result, indent=2))
    print(f"\n[SAVE] {OUT_DIR/'session_to_session_summary.json'}")


if __name__ == "__main__":
    main()
