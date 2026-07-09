"""Rubber-band vs tape drift analysis + side-by-side comparison.

Loads:
  - 4 TAPE sessions   (Sam Med Antenna/ BreastPhantom_A3_Nothing_20260701_1438..1603_SamMedSep_001..004)
  - 6 RUBBER sessions (Sam Med Antenna/Rubber_Band/ BreastPhantom_A3_Nothing_20260702_*)

Computes per-port trial-to-trial CV (same metric as drift_analysis_by_port.py):
  For each session, S-param:
    per position: std over 16 trials / mean over trials, at each freq -> mean over freq
    median across positions
  Then group S-params by port involvement, mean.

Then plots side-by-side comparison so we can answer:
  1. Did rubber band fix port 3?
  2. Is port 4 misbehaving in rubber-band data?
  3. What's the overall drift level compared to tape?
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

DATA_ROOT   = Path(r"C:\Users\peter\Desktop\EM Imaging\Detectable Difference\Data\DriftTest\Sam Med Antenna")
TAPE_DIR    = DATA_ROOT
RUBBER_DIR  = DATA_ROOT / "Rubber_Band"
OUT_DIR     = Path(HERE).parent / "results" / "drift_test_sam_med" / "rubber_vs_tape"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TAPE_SESSIONS = [
    "BreastPhantom_A3_Nothing_20260701_1438_SamMedSep_001",
    "BreastPhantom_A3_Nothing_20260701_1509_SamMedSep_002",
    "BreastPhantom_A3_Nothing_20260701_1536_SamMedSep_003",
    "BreastPhantom_A3_Nothing_20260701_1603_SamMedSep_004",
]

RUBBER_SESSIONS = [
    "BreastPhantom_A3_Nothing_20260702_0859",
    "BreastPhantom_A3_Nothing_20260702_0928",
    "BreastPhantom_A3_Nothing_20260702_0958",
    "BreastPhantom_A3_Nothing_20260702_1029",
    "BreastPhantom_A3_Nothing_20260702_1058",
    "BreastPhantom_A3_Nothing_20260702_1139",
]

S_NAMES = [f"S{i}{j}" for i in (1, 2, 3, 4) for j in (1, 2, 3, 4)]


def involves_port(sname, port):
    return str(port) in sname[1:]


def per_session_metrics(sess):
    Xc = sess["Xc"]
    ypos = sess["y_pos"]
    unique_pos = np.unique(ypos)
    mag = np.abs(Xc)
    n_sp = mag.shape[1]

    trial_cv = np.zeros(n_sp)
    for k in range(n_sp):
        per_pos_cv = []
        for p in unique_pos:
            m = ypos == p
            traces = mag[m, k, :]     # (T, F)
            mn = traces.mean(0) + 1e-15
            s = traces.std(0)
            per_pos_cv.append((s / mn).mean())
        trial_cv[k] = np.median(per_pos_cv)
    return trial_cv


def load_all(dirbase, names, tag):
    metrics = {}
    for name in names:
        folder = dirbase / name
        if not folder.exists():
            print(f"[SKIP {tag}] {folder} missing"); continue
        print(f"[LOAD {tag}] {name.split('_')[-1] if 'Sam' in name else name.split('_')[-2]+'_'+name.split('_')[-1]} ...", flush=True)
        d = load_hunter_session(str(folder), mode="full16")
        sid = name.split("_")[-1] if "SamMedSep" in name else name.split("_")[-1]  # timestamp e.g. "0859"
        metrics[sid] = per_session_metrics(d)
    return metrics


def per_port_from_metrics(metrics_dict):
    """Return {port: array over sessions of mean-CV over S-params involving that port}."""
    sids = list(metrics_dict.keys())
    out = {p: np.zeros(len(sids)) for p in (1, 2, 3, 4)}
    for si, sid in enumerate(sids):
        cv = metrics_dict[sid]
        for p in (1, 2, 3, 4):
            mask = np.array([involves_port(sn, p) for sn in S_NAMES])
            out[p][si] = cv[mask].mean()
    return sids, out


def main():
    print("=" * 70)
    print("Loading TAPE data (4 sessions) ...")
    tape_metrics = load_all(TAPE_DIR, TAPE_SESSIONS, "TAPE")

    print("=" * 70)
    print("Loading RUBBER data (6 sessions) ...")
    rubber_metrics = load_all(RUBBER_DIR, RUBBER_SESSIONS, "RUBBER")

    if not tape_metrics or not rubber_metrics:
        print("Missing data"); return

    tape_sids, tape_pp = per_port_from_metrics(tape_metrics)
    rubber_sids, rubber_pp = per_port_from_metrics(rubber_metrics)

    # ------------------------------------------------------------------------
    # Print numeric comparison
    # ------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PER-PORT MEAN TRIAL-CV  (each session)")
    print("=" * 70)
    print("\n>>> TAPE (older/loose antenna 3) <<<")
    print(f"{'Session':<10}{'P1':>10}{'P2':>10}{'P3':>10}{'P4':>10}   P3/mean(P1,P2,P4)")
    for i, sid in enumerate(tape_sids):
        v = [tape_pp[p][i] for p in (1, 2, 3, 4)]
        ratio = v[2] / np.mean([v[0], v[1], v[3]])
        print(f"{sid:<10}{v[0]:>10.5f}{v[1]:>10.5f}{v[2]:>10.5f}{v[3]:>10.5f}   {ratio:.2f}")

    print("\n>>> RUBBER BAND <<<")
    print(f"{'Session':<10}{'P1':>10}{'P2':>10}{'P3':>10}{'P4':>10}   P3/mean(P1,P2,P4)  P4/mean(P1,P2,P3)")
    for i, sid in enumerate(rubber_sids):
        v = [rubber_pp[p][i] for p in (1, 2, 3, 4)]
        ratio3 = v[2] / np.mean([v[0], v[1], v[3]])
        ratio4 = v[3] / np.mean([v[0], v[1], v[2]])
        print(f"{sid:<10}{v[0]:>10.5f}{v[1]:>10.5f}{v[2]:>10.5f}{v[3]:>10.5f}   {ratio3:.2f}              {ratio4:.2f}")

    # ------------------------------------------------------------------------
    # PLOT 1: per-port CV over session index, side by side
    # ------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    colors = {1: "steelblue", 2: "olive", 3: "firebrick", 4: "seagreen"}
    for ax, (title, sids, pp) in zip(
        axes, [("TAPE (4 sessions, loose antenna 3)", tape_sids, tape_pp),
               ("RUBBER BAND (6 sessions)", rubber_sids, rubber_pp)]):
        x = list(range(len(sids)))
        for port in (1, 2, 3, 4):
            lw = 2.5 if port == 3 else 1.5
            ax.plot(x, pp[port], "o-", label=f"Port {port}", color=colors[port], lw=lw)
        ax.set_xticks(x); ax.set_xticklabels(sids, rotation=30, ha="right", fontsize=9)
        ax.set_xlabel("Session")
        ax.set_ylabel("Mean trial-to-trial CV of |S|  (lower = better)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "per_port_cv_side_by_side.png", dpi=140)
    plt.close()
    print(f"\n[SAVE] {OUT_DIR/'per_port_cv_side_by_side.png'}")

    # ------------------------------------------------------------------------
    # PLOT 2: bar chart of MEAN per-port CV, tape vs rubber
    # ------------------------------------------------------------------------
    tape_mean = {p: tape_pp[p].mean() for p in (1, 2, 3, 4)}
    rubber_mean = {p: rubber_pp[p].mean() for p in (1, 2, 3, 4)}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(4); w = 0.35
    ax.bar(x - w/2, [tape_mean[p] for p in (1, 2, 3, 4)], width=w,
           label=f"TAPE (n={len(tape_sids)})", color="tab:blue", alpha=0.85)
    ax.bar(x + w/2, [rubber_mean[p] for p in (1, 2, 3, 4)], width=w,
           label=f"RUBBER (n={len(rubber_sids)})", color="tab:green", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([f"Port {p}" for p in (1, 2, 3, 4)])
    ax.set_ylabel("Mean trial-CV averaged over sessions")
    ax.set_title("Per-port drift: tape vs rubber-band mounting")
    ax.legend()
    ax.grid(True, alpha=0.3)
    for i, p in enumerate((1, 2, 3, 4)):
        change_pct = 100 * (rubber_mean[p] - tape_mean[p]) / tape_mean[p]
        color = "green" if change_pct < -5 else ("red" if change_pct > 5 else "gray")
        ax.annotate(f"{change_pct:+.0f}%", xy=(i, max(tape_mean[p], rubber_mean[p]) * 1.02),
                    ha="center", fontsize=10, color=color, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "mean_cv_bar_tape_vs_rubber.png", dpi=140)
    plt.close()
    print(f"[SAVE] {OUT_DIR/'mean_cv_bar_tape_vs_rubber.png'}")

    # ------------------------------------------------------------------------
    # PLOT 3: per-S-param heatmap for rubber-band (to spot outlier S-params)
    # ------------------------------------------------------------------------
    heat = np.zeros((16, len(rubber_sids)))
    for si, sid in enumerate(rubber_sids):
        heat[:, si] = rubber_metrics[sid]
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(heat, aspect="auto", cmap="magma")
    ax.set_yticks(range(16)); ax.set_yticklabels(S_NAMES, fontsize=9)
    ax.set_xticks(range(len(rubber_sids))); ax.set_xticklabels(rubber_sids, rotation=30, ha="right")
    ax.set_xlabel("Rubber-band session")
    ax.set_title("Rubber-band per-S-param trial-CV heatmap\n(bold red = involves port 3, blue = involves port 4)")
    for k, sn in enumerate(S_NAMES):
        if involves_port(sn, 3):
            ax.get_yticklabels()[k].set_color("red")
            ax.get_yticklabels()[k].set_fontweight("bold")
        elif involves_port(sn, 4):
            ax.get_yticklabels()[k].set_color("blue")
    plt.colorbar(im, ax=ax, label="Trial-CV of |S|")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "rubber_per_sparam_heatmap.png", dpi=140)
    plt.close()
    print(f"[SAVE] {OUT_DIR/'rubber_per_sparam_heatmap.png'}")

    # ------------------------------------------------------------------------
    # SUMMARY JSON
    # ------------------------------------------------------------------------
    summary = {
        "tape_sessions": tape_sids,
        "tape_per_port_mean": {str(p): float(tape_mean[p]) for p in (1, 2, 3, 4)},
        "tape_per_port_per_session": {str(p): tape_pp[p].tolist() for p in (1, 2, 3, 4)},
        "rubber_sessions": rubber_sids,
        "rubber_per_port_mean": {str(p): float(rubber_mean[p]) for p in (1, 2, 3, 4)},
        "rubber_per_port_per_session": {str(p): rubber_pp[p].tolist() for p in (1, 2, 3, 4)},
        "port_change_pct": {str(p): float(100 * (rubber_mean[p] - tape_mean[p]) / tape_mean[p]) for p in (1, 2, 3, 4)},
    }
    (OUT_DIR / "rubber_vs_tape_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[SAVE] {OUT_DIR/'rubber_vs_tape_summary.json'}")

    # Top-line print
    print("\n" + "=" * 70)
    print("HEADLINE: mean per-port trial-CV")
    print("=" * 70)
    print(f"{'Port':<6}{'TAPE':>12}{'RUBBER':>12}{'CHANGE':>12}")
    for p in (1, 2, 3, 4):
        change = 100 * (rubber_mean[p] - tape_mean[p]) / tape_mean[p]
        print(f"Port{p:<3}{tape_mean[p]:>12.5f}{rubber_mean[p]:>12.5f}{change:>11.1f}%")


if __name__ == "__main__":
    main()
