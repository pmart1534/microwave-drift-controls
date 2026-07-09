"""Per-port drift analysis on the 4-session Sam Med drift test.

Loads the 4 back-to-back empty-phantom drift sessions and asks:
  - How much does each S-parameter drift trial-to-trial and session-to-session?
  - When you group by port involvement, does port 3 stand out?
  - Does drift decrease from session 1 -> 2 -> 3 -> 4 as Peter observed?

The hypothesis: antenna 3 tape was loose, so any S-param INVOLVING port 3
(S13, S23, S33, S31, S32, S34, S43) should have much higher trial-to-trial
variance than S-params that don't touch port 3 (S22, S24, S44, S42, etc.).

Two drift metrics per (session, position, S-param):
  1. Trial-to-trial std of |S| over the 16 trials
  2. Baseline drift = mean |S| over first-half trials minus second-half

We aggregate across positions (median) and freq (mean over full band) so
each (session, S-param) collapses to one number.

Output: results/drift_test_sam_med/{summary.csv, per_port_bar.png,
    per_session_heatmap.png, drift_over_time.png}
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

# ----------------------------------------------------------------------------
DATA_DIR = Path(r"C:\Users\peter\Desktop\EM Imaging\Detectable Difference\Data\DriftTest\Sam Med Antenna")
OUT_DIR  = Path(HERE).parent / "results" / "drift_test_sam_med"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Only the 4 main drift sessions (skip the 1-second-sweep variants for now)
SESSION_NAMES = [
    "BreastPhantom_A3_Nothing_20260701_1438_SamMedSep_001",
    "BreastPhantom_A3_Nothing_20260701_1509_SamMedSep_002",
    "BreastPhantom_A3_Nothing_20260701_1536_SamMedSep_003",
    "BreastPhantom_A3_Nothing_20260701_1603_SamMedSep_004",
]

# S-parameter label order matching hunter_loader's full16 mode
# S_names[k] is the k-th S-param channel in Xc[:, k, :]
S_NAMES = [f"S{i}{j}" for i in (1, 2, 3, 4) for j in (1, 2, 3, 4)]  # S11..S44


def involves_port(sname: str, port: int) -> bool:
    return str(port) in sname[1:]  # skip the leading 'S'


# ----------------------------------------------------------------------------
def load_all_sessions():
    sessions = {}
    for name in SESSION_NAMES:
        folder = DATA_DIR / name
        if not folder.exists():
            print(f"[SKIP] {folder} missing"); continue
        print(f"[LOAD] {name} ...")
        out = load_hunter_session(str(folder), mode="full16")
        Xc   = out["Xc"]           # (N, 16, F) complex
        ypos = out["y_pos"]        # (N,)
        base_c = out["base_c"]     # (16, F) complex baseline
        sessions[name.split("_")[-1]] = {
            "Xc": Xc, "ypos": ypos, "base_c": base_c,
            "session_label": name.split("_")[-1],
        }
    return sessions


def per_session_drift_metrics(sess):
    """Return dict of per-S-param drift metrics for one session.
       trial_std   : median-over-positions of std(|S|) over the 16 trials, mean over freq
       trial_cv    : same but coefficient-of-variation (std/mean)
       within_pos_range : median (peak-to-peak of trial |S|) / mean |S|, per freq averaged
    """
    Xc = sess["Xc"]                    # (N, 16, F)
    ypos = sess["ypos"]
    unique_pos = np.unique(ypos)
    mag = np.abs(Xc)                   # (N, 16, F)
    n_sp = mag.shape[1]

    trial_std = np.zeros(n_sp)
    trial_cv  = np.zeros(n_sp)
    ptp_norm  = np.zeros(n_sp)

    for k in range(n_sp):
        # For each position, compute trial-axis metrics, then median across positions
        per_pos_std = []
        per_pos_cv  = []
        per_pos_ptp = []
        for p in unique_pos:
            m = ypos == p
            # mag[m, k, :] shape = (T, F)
            trial_traces = mag[m, k, :]
            trial_mean = trial_traces.mean(0) + 1e-15
            s = trial_traces.std(0)                 # (F,)
            per_pos_std.append(s.mean())
            per_pos_cv.append((s / trial_mean).mean())
            ptp = trial_traces.max(0) - trial_traces.min(0)
            per_pos_ptp.append((ptp / trial_mean).mean())
        trial_std[k] = np.median(per_pos_std)
        trial_cv[k]  = np.median(per_pos_cv)
        ptp_norm[k]  = np.median(per_pos_ptp)

    return {
        "trial_std":  trial_std,
        "trial_cv":   trial_cv,
        "ptp_norm":   ptp_norm,
    }


def main():
    sessions = load_all_sessions()
    if not sessions:
        print("No sessions loaded, aborting"); return

    # ------------------------------------------------------------------------
    # per-session drift metrics
    # ------------------------------------------------------------------------
    metrics = {}
    for sid, sess in sessions.items():
        print(f"[METRIC] {sid} ...")
        metrics[sid] = per_session_drift_metrics(sess)

    # ------------------------------------------------------------------------
    # Save summary CSV
    # ------------------------------------------------------------------------
    csv_lines = ["session,sparam,involves_port1,involves_port2,involves_port3,involves_port4,trial_std,trial_cv,ptp_norm"]
    for sid, md in metrics.items():
        for k, sn in enumerate(S_NAMES):
            row = [
                sid, sn,
                int(involves_port(sn, 1)),
                int(involves_port(sn, 2)),
                int(involves_port(sn, 3)),
                int(involves_port(sn, 4)),
                f"{md['trial_std'][k]:.6e}",
                f"{md['trial_cv'][k]:.6f}",
                f"{md['ptp_norm'][k]:.6f}",
            ]
            csv_lines.append(",".join(str(x) for x in row))
    (OUT_DIR / "summary.csv").write_text("\n".join(csv_lines))
    print(f"[SAVE] {OUT_DIR/'summary.csv'}")

    # ------------------------------------------------------------------------
    # Aggregate: per-session, per-port-involvement mean trial_cv
    # ------------------------------------------------------------------------
    sids = list(metrics.keys())
    n_sess = len(sids)
    per_port_cv = {p: np.zeros(n_sess) for p in (1, 2, 3, 4)}
    per_port_std_lin = {p: np.zeros(n_sess) for p in (1, 2, 3, 4)}
    for si, sid in enumerate(sids):
        md = metrics[sid]
        for p in (1, 2, 3, 4):
            mask = np.array([involves_port(sn, p) for sn in S_NAMES])
            per_port_cv[p][si]     = md["trial_cv"][mask].mean()
            per_port_std_lin[p][si] = md["trial_std"][mask].mean()

    # ------------------------------------------------------------------------
    # BAR CHART: mean CV per port across sessions
    # ------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(n_sess)
    width = 0.2
    for i, p in enumerate((1, 2, 3, 4)):
        axes[0].bar(x + (i - 1.5) * width, per_port_cv[p], width, label=f"Port {p}")
    axes[0].set_xticks(x); axes[0].set_xticklabels(sids)
    axes[0].set_ylabel("Mean trial-to-trial CV of |S|")
    axes[0].set_xlabel("Session")
    axes[0].set_title("Drift by port involvement, across sessions")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Ratio: port 3 vs mean of ports 1,2,4
    ratio = per_port_cv[3] / np.mean([per_port_cv[1], per_port_cv[2], per_port_cv[4]], axis=0)
    axes[1].bar(sids, ratio, color="firebrick")
    axes[1].axhline(1.0, color="k", ls="--", lw=1)
    axes[1].set_ylabel("Port-3 CV / mean(port 1,2,4 CV)")
    axes[1].set_xlabel("Session")
    axes[1].set_title("Antenna-3 excess drift ratio\n(1.0 = same as other antennas)")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "per_port_bar.png", dpi=140)
    plt.close()
    print(f"[SAVE] {OUT_DIR/'per_port_bar.png'}")

    # ------------------------------------------------------------------------
    # HEATMAP: 16 S-params (rows) x sessions (cols), CV metric
    # ------------------------------------------------------------------------
    heat = np.zeros((16, n_sess))
    for si, sid in enumerate(sids):
        heat[:, si] = metrics[sid]["trial_cv"]

    fig, ax = plt.subplots(figsize=(6, 8))
    im = ax.imshow(heat, aspect="auto", cmap="magma")
    ax.set_yticks(range(16)); ax.set_yticklabels(S_NAMES, fontsize=9)
    ax.set_xticks(range(n_sess)); ax.set_xticklabels(sids)
    ax.set_xlabel("Session"); ax.set_title("Trial-to-trial CV of |S|\nper S-parameter x session")
    # Highlight rows containing port 3
    for k, sn in enumerate(S_NAMES):
        if involves_port(sn, 3):
            ax.get_yticklabels()[k].set_color("red")
            ax.get_yticklabels()[k].set_fontweight("bold")
    plt.colorbar(im, ax=ax, label="Trial-to-trial CV")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "per_session_heatmap.png", dpi=140)
    plt.close()
    print(f"[SAVE] {OUT_DIR/'per_session_heatmap.png'}")

    # ------------------------------------------------------------------------
    # DRIFT-OVER-TIME plot: mean CV of port 3 vs port 2 vs port 4 across sessions
    # ------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sids, per_port_cv[1], "o-", label="Port 1", color="steelblue")
    ax.plot(sids, per_port_cv[2], "o-", label="Port 2", color="olive")
    ax.plot(sids, per_port_cv[3], "o-", label="Port 3 (suspect)", color="firebrick", lw=2.5)
    ax.plot(sids, per_port_cv[4], "o-", label="Port 4", color="seagreen")
    ax.set_xlabel("Session (chronological)")
    ax.set_ylabel("Mean trial-to-trial CV of |S|")
    ax.set_title("Drift over session sequence\n(higher = worse mechanical/electronic stability)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "drift_over_time.png", dpi=140)
    plt.close()
    print(f"[SAVE] {OUT_DIR/'drift_over_time.png'}")

    # ------------------------------------------------------------------------
    # Print top-line results
    # ------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PER-PORT DRIFT SUMMARY (mean CV over all S-params involving that port)")
    print("=" * 60)
    print(f"{'Session':<8} {'Port1':<10} {'Port2':<10} {'Port3':<10} {'Port4':<10} {'P3/mean(P1,2,4)':<15}")
    for si, sid in enumerate(sids):
        r = per_port_cv[3][si] / np.mean([per_port_cv[1][si], per_port_cv[2][si], per_port_cv[4][si]])
        print(f"{sid:<8} {per_port_cv[1][si]:<10.4f} {per_port_cv[2][si]:<10.4f} "
              f"{per_port_cv[3][si]:<10.4f} {per_port_cv[4][si]:<10.4f} {r:<15.2f}")

    # Save top-line summary as JSON
    result = {
        "sessions": sids,
        "per_port_cv": {p: per_port_cv[p].tolist() for p in (1, 2, 3, 4)},
        "port3_excess_ratio": ratio.tolist(),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(result, indent=2))
    print(f"\n[SAVE] {OUT_DIR/'summary.json'}")
    print(f"\nAll outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
