"""Cables-only drift diagnostic.

Loads 2 EmptyCables sessions (no antennas, no phantom, just SMA cables
sitting there) and asks:

  Q1: Is S44 drift specific to port 4 (cable/VNA) or does it disappear
      when antennas are removed?
  Q2: Do all reflection channels (S11/S22/S33/S44) drift the same, or
      does one stand out?
  Q3: How does the drift behavior of transmission channels (S12, S13,
      S14, S23, S24, S34) compare to reflection channels?

For direct comparability, we ALSO reload the rubber-band data (6 sessions,
antennas + rubber band mount) and compute the same metrics on it. If
S44 drifts in rubber but is quiet in cables-only, S44 drift is caused
by the antenna. If S44 drifts in BOTH, the source is cable/VNA/connector.

Metrics:
  1. Trial-to-trial CV (noise floor) — same as drift_analysis_by_port.py
  2. Within-session slow drift = |mean(last-10%-of-trials) - mean(first-10%)|,
     normalized by overall mean. Captures the DD-style drift Peter is
     seeing in detdifplot.

Output: results/drift_test_sam_med/cables_only/
  {per_sparam_cv.png, per_sparam_slow_drift.png, summary.json}
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
CABLE_DIR   = DATA_ROOT / "EmptyCables"
RUBBER_DIR  = DATA_ROOT / "Rubber_Band"
OUT_DIR     = Path(HERE).parent / "results" / "drift_test_sam_med" / "cables_only"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CABLE_SESSIONS = [
    "BreastPhantom_A3_Nothing_20260702_1223",
    "BreastPhantom_A3_Nothing_20260702_1311",
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


def is_reflection(sname):
    return sname[1] == sname[2]      # S11, S22, S33, S44


def per_session_metrics(sess):
    """Return (trial_cv_per_sp, slow_drift_per_sp).
       trial_cv:   median across positions of trial-CV per freq, mean over freq.
       slow_drift: |mean(last-10% trials) - mean(first-10% trials)| /
                   overall_mean, per S-param.
    """
    Xc = sess["Xc"]         # (N, 16, F)
    ypos = sess["y_pos"]
    mag = np.abs(Xc)        # (N, 16, F)
    n = mag.shape[0]; n_sp = mag.shape[1]

    # ---- Trial-CV (same as before) ----
    trial_cv = np.zeros(n_sp)
    unique_pos = np.unique(ypos)
    for k in range(n_sp):
        per_pos_cv = []
        for p in unique_pos:
            m = ypos == p
            traces = mag[m, k, :]
            mn = traces.mean(0) + 1e-15
            s = traces.std(0)
            per_pos_cv.append((s / mn).mean())
        trial_cv[k] = np.median(per_pos_cv)

    # ---- Slow drift: first 10% of trials vs last 10% ----
    n_edge = max(1, n // 10)
    early = mag[:n_edge].mean(0)         # (16, F)
    late  = mag[-n_edge:].mean(0)        # (16, F)
    overall_mean = mag.mean(0) + 1e-15
    slow_drift_pf = np.abs(late - early) / overall_mean   # (16, F)
    slow_drift = slow_drift_pf.mean(axis=-1)              # (16,)
    return trial_cv, slow_drift


def load_dataset(dirbase, names, tag):
    metrics_cv = {}
    metrics_slow = {}
    for name in names:
        folder = dirbase / name
        if not folder.exists():
            print(f"[SKIP {tag}] {folder} missing"); continue
        sid = name.split("_")[-1]
        print(f"[LOAD {tag}] {sid} ...", flush=True)
        d = load_hunter_session(str(folder), mode="full16")
        cv, sd = per_session_metrics(d)
        metrics_cv[sid] = cv
        metrics_slow[sid] = sd
    return metrics_cv, metrics_slow


def stack_over_sessions(m_dict):
    return np.stack([m_dict[k] for k in m_dict.keys()], axis=0)  # (n_sess, 16)


def main():
    print("Loading CABLES-ONLY sessions ...")
    cable_cv, cable_slow = load_dataset(CABLE_DIR, CABLE_SESSIONS, "CABLE")
    print("Loading RUBBER-BAND (antennas) sessions ...")
    rubber_cv, rubber_slow = load_dataset(RUBBER_DIR, RUBBER_SESSIONS, "RUBBER")

    cable_cv_arr   = stack_over_sessions(cable_cv)
    cable_slow_arr = stack_over_sessions(cable_slow)
    rubber_cv_arr   = stack_over_sessions(rubber_cv)
    rubber_slow_arr = stack_over_sessions(rubber_slow)

    # ------------------------------------------------------------------------
    # Print per-S-param comparison
    # ------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("MEAN TRIAL-CV per S-parameter  (noise floor within session)")
    print("=" * 78)
    print(f"{'S-param':<8}{'CABLE mean':>14}{'RUBBER mean':>14}   {'ratio C/R':>12}   flags")
    for k, sn in enumerate(S_NAMES):
        c = cable_cv_arr[:, k].mean()
        r = rubber_cv_arr[:, k].mean()
        ratio = c / r
        flags = []
        if is_reflection(sn):
            flags.append("refl")
        if involves_port(sn, 3):
            flags.append("P3")
        if involves_port(sn, 4):
            flags.append("P4")
        print(f"{sn:<8}{c:>14.5f}{r:>14.5f}   {ratio:>12.2f}   {' '.join(flags)}")

    print("\n" + "=" * 78)
    print("SLOW DRIFT per S-parameter  (|late mean - early mean| / overall mean)")
    print("This is the metric that best matches what Peter sees in detdifplot")
    print("=" * 78)
    print(f"{'S-param':<8}{'CABLE mean':>14}{'RUBBER mean':>14}   {'ratio C/R':>12}   flags")
    for k, sn in enumerate(S_NAMES):
        c = cable_slow_arr[:, k].mean()
        r = rubber_slow_arr[:, k].mean()
        ratio = c / r
        flags = []
        if is_reflection(sn):
            flags.append("refl")
        if involves_port(sn, 3):
            flags.append("P3")
        if involves_port(sn, 4):
            flags.append("P4")
        print(f"{sn:<8}{c:>14.5f}{r:>14.5f}   {ratio:>12.2f}   {' '.join(flags)}")

    # ------------------------------------------------------------------------
    # PLOT 1: Trial-CV per S-param, bar comparison
    # ------------------------------------------------------------------------
    x = np.arange(16); w = 0.4
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - w/2, cable_cv_arr.mean(0), width=w, label="CABLES-ONLY (no antennas)",
           color="tab:orange", alpha=0.85)
    ax.bar(x + w/2, rubber_cv_arr.mean(0), width=w, label="RUBBER (antennas)",
           color="tab:green", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(S_NAMES, rotation=45, ha="right")
    ax.set_ylabel("Mean trial-CV of |S|  (noise floor)")
    ax.set_title("Per-S-param trial noise: cables-only vs antennas + rubber-band")
    ax.legend()
    ax.grid(True, alpha=0.3)
    for k, sn in enumerate(S_NAMES):
        if is_reflection(sn):
            ax.get_xticklabels()[k].set_fontweight("bold")
        if involves_port(sn, 4):
            ax.get_xticklabels()[k].set_color("blue")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "per_sparam_cv_cables_vs_rubber.png", dpi=140)
    plt.close()
    print(f"\n[SAVE] {OUT_DIR/'per_sparam_cv_cables_vs_rubber.png'}")

    # ------------------------------------------------------------------------
    # PLOT 2: Slow drift per S-param, bar comparison — THIS matches DD story
    # ------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - w/2, cable_slow_arr.mean(0), width=w, label="CABLES-ONLY (no antennas)",
           color="tab:orange", alpha=0.85)
    ax.bar(x + w/2, rubber_slow_arr.mean(0), width=w, label="RUBBER (antennas)",
           color="tab:green", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(S_NAMES, rotation=45, ha="right")
    ax.set_ylabel("|late-mean - early-mean| / overall-mean")
    ax.set_title("Per-S-param SLOW drift (first vs last 10% of session)\n"
                 "matches session-to-session drift Peter sees in detdifplot")
    ax.legend()
    ax.grid(True, alpha=0.3)
    for k, sn in enumerate(S_NAMES):
        if is_reflection(sn):
            ax.get_xticklabels()[k].set_fontweight("bold")
        if involves_port(sn, 4):
            ax.get_xticklabels()[k].set_color("blue")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "per_sparam_slow_drift_cables_vs_rubber.png", dpi=140)
    plt.close()
    print(f"[SAVE] {OUT_DIR/'per_sparam_slow_drift_cables_vs_rubber.png'}")

    # ------------------------------------------------------------------------
    # PLOT 3: Focus on S44 — does it drift with just cables?
    # ------------------------------------------------------------------------
    k44 = S_NAMES.index("S44")
    k11 = S_NAMES.index("S11")
    k22 = S_NAMES.index("S22")
    k33 = S_NAMES.index("S33")
    keys = [k11, k22, k33, k44]
    names = ["S11", "S22", "S33", "S44"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    xr = np.arange(4)
    ax.bar(xr - w/2, [cable_slow_arr[:, k].mean() for k in keys], width=w,
           label="CABLES-ONLY", color="tab:orange", alpha=0.85)
    ax.bar(xr + w/2, [rubber_slow_arr[:, k].mean() for k in keys], width=w,
           label="RUBBER (antennas)", color="tab:green", alpha=0.85)
    ax.set_xticks(xr); ax.set_xticklabels(names)
    ax.set_ylabel("Slow drift |late-early|/overall")
    ax.set_title("Reflection channels: S44 diagnostic\n"
                 "(if S44 drift is high in cables-only -> cable/VNA; if not -> antenna)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "S44_diagnostic.png", dpi=140)
    plt.close()
    print(f"[SAVE] {OUT_DIR/'S44_diagnostic.png'}")

    # ------------------------------------------------------------------------
    # Numerical verdict
    # ------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("S44 DIAGNOSTIC VERDICT")
    print("=" * 78)
    s44_cable  = cable_slow_arr[:, k44].mean()
    s44_rubber = rubber_slow_arr[:, k44].mean()
    other_refl_cable  = np.mean([cable_slow_arr[:, k].mean()  for k in [k11, k22, k33]])
    other_refl_rubber = np.mean([rubber_slow_arr[:, k].mean() for k in [k11, k22, k33]])
    print(f"S44 slow drift:                CABLES {s44_cable:.5f}   RUBBER {s44_rubber:.5f}")
    print(f"Other reflections (S11/22/33): CABLES {other_refl_cable:.5f}   RUBBER {other_refl_rubber:.5f}")
    print(f"S44 excess over other refls:   CABLES {s44_cable/other_refl_cable:.2f}x   "
          f"RUBBER {s44_rubber/other_refl_rubber:.2f}x")
    if s44_cable > 1.5 * other_refl_cable:
        print("\n=> S44 IS elevated in the cables-only case.")
        print("   The excess drift is coming from the CABLE, CONNECTOR, or VNA port-4 channel.")
        print("   Swapping the port-4 antenna will NOT fix it.")
    else:
        print("\n=> S44 is NOT elevated in the cables-only case.")
        print("   The excess drift in the antennas case is coming from the ANTENNA itself")
        print("   (mechanical, impedance mismatch that drifts, or antenna-to-cable connector).")

    (OUT_DIR / "summary.json").write_text(json.dumps({
        "cable_sessions":  list(cable_cv.keys()),
        "rubber_sessions": list(rubber_cv.keys()),
        "trial_cv":  {"cable_mean_per_sp":  {sn: float(cable_cv_arr[:, k].mean())  for k, sn in enumerate(S_NAMES)},
                      "rubber_mean_per_sp": {sn: float(rubber_cv_arr[:, k].mean()) for k, sn in enumerate(S_NAMES)}},
        "slow_drift": {"cable_mean_per_sp":  {sn: float(cable_slow_arr[:, k].mean())  for k, sn in enumerate(S_NAMES)},
                       "rubber_mean_per_sp": {sn: float(rubber_slow_arr[:, k].mean()) for k, sn in enumerate(S_NAMES)}},
        "s44_verdict": {
            "s44_cable_slow":            float(s44_cable),
            "s44_rubber_slow":           float(s44_rubber),
            "other_refls_cable_slow":    float(other_refl_cable),
            "other_refls_rubber_slow":   float(other_refl_rubber),
            "s44_excess_cable":          float(s44_cable / other_refl_cable),
            "s44_excess_rubber":         float(s44_rubber / other_refl_rubber),
        },
    }, indent=2))
    print(f"\n[SAVE] {OUT_DIR/'summary.json'}")


if __name__ == "__main__":
    main()
