"""Three-mount summary: tape / rubber / cables, both metrics.

Loads tape data and computes slow-drift metric (rubber + cables already
computed by drift_cables_only_diagnostic.py, reads those JSONs).

Produces the paper's central three-mount x two-metric comparison figure.
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
CABLES_DIR  = DATA_ROOT / "EmptyCables"
CABLES_JSON = Path(HERE).parent / "results" / "drift_test_sam_med" / "cables_only" / "summary.json"
OUT_DIR     = Path(HERE).parent / "results" / "drift_test_sam_med" / "three_mount"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TAPE_SESSIONS = [
    "BreastPhantom_A3_Nothing_20260701_1438_SamMedSep_001",
    "BreastPhantom_A3_Nothing_20260701_1509_SamMedSep_002",
    "BreastPhantom_A3_Nothing_20260701_1536_SamMedSep_003",
    "BreastPhantom_A3_Nothing_20260701_1603_SamMedSep_004",
]

S_NAMES = [f"S{i}{j}" for i in (1, 2, 3, 4) for j in (1, 2, 3, 4)]


def involves_port(sname, port):
    return str(port) in sname[1:]

def is_reflection(sname):
    return sname[1] == sname[2]


def per_session_slow_drift(sess):
    Xc = sess["Xc"]; mag = np.abs(Xc); n = mag.shape[0]
    n_edge = max(1, n // 10)
    early = mag[:n_edge].mean(0)
    late  = mag[-n_edge:].mean(0)
    overall = mag.mean(0) + 1e-15
    return (np.abs(late - early) / overall).mean(axis=-1)  # (16,)


def per_session_trial_cv(sess):
    Xc = sess["Xc"]; mag = np.abs(Xc); ypos = sess["y_pos"]
    n_sp = mag.shape[1]
    cv = np.zeros(n_sp)
    for k in range(n_sp):
        per_pos = []
        for p in np.unique(ypos):
            m = ypos == p
            tr = mag[m, k, :]
            per_pos.append((tr.std(0) / (tr.mean(0) + 1e-15)).mean())
        cv[k] = np.median(per_pos)
    return cv


def main():
    print("Loading TAPE data (4 sessions) ...")
    tape_cv = []
    tape_slow = []
    for name in TAPE_SESSIONS:
        folder = TAPE_DIR / name
        if not folder.exists():
            print(f"[SKIP] {folder}"); continue
        print(f"[LOAD] {name.split('_')[-1]} ...", flush=True)
        d = load_hunter_session(str(folder), mode="full16")
        tape_cv.append(per_session_trial_cv(d))
        tape_slow.append(per_session_slow_drift(d))
    tape_cv_mean   = np.mean(tape_cv, axis=0)   # (16,)
    tape_slow_mean = np.mean(tape_slow, axis=0)

    print("\nLoading cables/rubber summary from existing JSON ...")
    with open(CABLES_JSON) as f:
        cables_summary = json.load(f)
    cable_cv_mean   = np.array([cables_summary["trial_cv"]["cable_mean_per_sp"][sn]   for sn in S_NAMES])
    rubber_cv_mean  = np.array([cables_summary["trial_cv"]["rubber_mean_per_sp"][sn]  for sn in S_NAMES])
    cable_slow_mean  = np.array([cables_summary["slow_drift"]["cable_mean_per_sp"][sn]  for sn in S_NAMES])
    rubber_slow_mean = np.array([cables_summary["slow_drift"]["rubber_mean_per_sp"][sn] for sn in S_NAMES])

    # -------------------------------------------------------------------
    # Per-port aggregation
    # -------------------------------------------------------------------
    def per_port(arr):
        return {p: arr[[k for k, sn in enumerate(S_NAMES) if involves_port(sn, p)]].mean() for p in (1, 2, 3, 4)}

    # Reflection-only aggregation (S11/22/33/44)
    def refl_only(arr):
        return {p: arr[[k for k, sn in enumerate(S_NAMES) if is_reflection(sn) and involves_port(sn, p)]].mean()
                for p in (1, 2, 3, 4)}

    tape_cv_pp   = per_port(tape_cv_mean)
    rubber_cv_pp = per_port(rubber_cv_mean)
    cable_cv_pp  = per_port(cable_cv_mean)
    tape_slow_pp   = per_port(tape_slow_mean)
    rubber_slow_pp = per_port(rubber_slow_mean)
    cable_slow_pp  = per_port(cable_slow_mean)

    tape_slow_refl   = refl_only(tape_slow_mean)
    rubber_slow_refl = refl_only(rubber_slow_mean)
    cable_slow_refl  = refl_only(cable_slow_mean)

    # -------------------------------------------------------------------
    # Print summary
    # -------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PER-PORT TRIAL-CV  (noise floor within session, mean across S-params w/ port)")
    print("=" * 80)
    print(f"{'Port':<6}{'TAPE':>14}{'RUBBER':>14}{'CABLES':>14}")
    for p in (1, 2, 3, 4):
        print(f"P{p:<5}{tape_cv_pp[p]:>14.5f}{rubber_cv_pp[p]:>14.5f}{cable_cv_pp[p]:>14.5f}")

    print("\n" + "=" * 80)
    print("PER-PORT SLOW-DRIFT  (mean shift over session, all S-params w/ port)")
    print("=" * 80)
    print(f"{'Port':<6}{'TAPE':>14}{'RUBBER':>14}{'CABLES':>14}")
    for p in (1, 2, 3, 4):
        print(f"P{p:<5}{tape_slow_pp[p]:>14.5f}{rubber_slow_pp[p]:>14.5f}{cable_slow_pp[p]:>14.5f}")

    print("\n" + "=" * 80)
    print("PER-PORT SLOW-DRIFT  (REFLECTION channel only: S11/22/33/44)")
    print("=" * 80)
    print(f"{'Port':<6}{'TAPE':>14}{'RUBBER':>14}{'CABLES':>14}")
    for p in (1, 2, 3, 4):
        print(f"S{p}{p:<4}{tape_slow_refl[p]:>14.5f}{rubber_slow_refl[p]:>14.5f}{cable_slow_refl[p]:>14.5f}")

    # -------------------------------------------------------------------
    # Combined comparison figure: 2 subplots, one per metric
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    x = np.arange(4); w = 0.27

    ax = axes[0]
    ax.bar(x - w, [tape_cv_pp[p]  for p in (1, 2, 3, 4)], width=w, label="TAPE (n=4)",   color="tab:red", alpha=0.85)
    ax.bar(x,     [rubber_cv_pp[p] for p in (1, 2, 3, 4)], width=w, label="RUBBER (n=6)", color="tab:green", alpha=0.85)
    ax.bar(x + w, [cable_cv_pp[p]  for p in (1, 2, 3, 4)], width=w, label="CABLES (n=2)",  color="tab:orange", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([f"Port {p}" for p in (1, 2, 3, 4)])
    ax.set_ylabel("Mean trial-CV of |S|")
    ax.set_title("TRIAL NOISE per port\n(limits within-session accuracy)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.bar(x - w, [tape_slow_pp[p]  for p in (1, 2, 3, 4)], width=w, label="TAPE (n=4)",   color="tab:red", alpha=0.85)
    ax.bar(x,     [rubber_slow_pp[p] for p in (1, 2, 3, 4)], width=w, label="RUBBER (n=6)", color="tab:green", alpha=0.85)
    ax.bar(x + w, [cable_slow_pp[p]  for p in (1, 2, 3, 4)], width=w, label="CABLES (n=2)",  color="tab:orange", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([f"Port {p}" for p in (1, 2, 3, 4)])
    ax.set_ylabel("|late-early|/overall (slow drift)")
    ax.set_title("SLOW DRIFT per port\n(limits LOSO accuracy)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "three_mount_two_metric.png", dpi=140)
    plt.close()
    print(f"\n[SAVE] {OUT_DIR/'three_mount_two_metric.png'}")

    # -------------------------------------------------------------------
    # Reflection-only slow drift figure (SNN diagnostic)
    # -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - w, [tape_slow_refl[p]  for p in (1, 2, 3, 4)], width=w, label="TAPE",   color="tab:red", alpha=0.85)
    ax.bar(x,     [rubber_slow_refl[p] for p in (1, 2, 3, 4)], width=w, label="RUBBER", color="tab:green", alpha=0.85)
    ax.bar(x + w, [cable_slow_refl[p]  for p in (1, 2, 3, 4)], width=w, label="CABLES", color="tab:orange", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([f"S{p}{p}" for p in (1, 2, 3, 4)])
    ax.set_ylabel("Slow drift")
    ax.set_title("Reflection-channel slow drift, three mount conditions\n"
                 "S44 elevated even in CABLES-only → port-4 cable/VNA contribution")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "reflection_slow_drift_three_mounts.png", dpi=140)
    plt.close()
    print(f"[SAVE] {OUT_DIR/'reflection_slow_drift_three_mounts.png'}")

    # -------------------------------------------------------------------
    # Save JSON
    # -------------------------------------------------------------------
    (OUT_DIR / "three_mount_summary.json").write_text(json.dumps({
        "per_port_trial_cv": {
            "tape":   {str(p): float(tape_cv_pp[p])   for p in (1, 2, 3, 4)},
            "rubber": {str(p): float(rubber_cv_pp[p]) for p in (1, 2, 3, 4)},
            "cables": {str(p): float(cable_cv_pp[p])  for p in (1, 2, 3, 4)},
        },
        "per_port_slow_drift": {
            "tape":   {str(p): float(tape_slow_pp[p])   for p in (1, 2, 3, 4)},
            "rubber": {str(p): float(rubber_slow_pp[p]) for p in (1, 2, 3, 4)},
            "cables": {str(p): float(cable_slow_pp[p])  for p in (1, 2, 3, 4)},
        },
        "reflection_slow_drift": {
            "tape":   {f"S{p}{p}": float(tape_slow_refl[p])   for p in (1, 2, 3, 4)},
            "rubber": {f"S{p}{p}": float(rubber_slow_refl[p]) for p in (1, 2, 3, 4)},
            "cables": {f"S{p}{p}": float(cable_slow_refl[p])  for p in (1, 2, 3, 4)},
        },
        "tape_slow_drift_per_sparam":   {sn: float(tape_slow_mean[k])   for k, sn in enumerate(S_NAMES)},
        "rubber_slow_drift_per_sparam": {sn: float(rubber_slow_mean[k]) for k, sn in enumerate(S_NAMES)},
        "cable_slow_drift_per_sparam":  {sn: float(cable_slow_mean[k])  for k, sn in enumerate(S_NAMES)},
    }, indent=2))
    print(f"[SAVE] {OUT_DIR/'three_mount_summary.json'}")


if __name__ == "__main__":
    main()
