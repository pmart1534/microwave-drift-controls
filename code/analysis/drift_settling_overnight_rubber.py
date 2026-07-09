"""Overnight settling analysis on the RUBBER-BAND live sweep.

Same as drift_settling_overnight.py but points at LiveData_RubberBand
(65K sweeps, ~17.7h, July 2 14:24 -> July 3 08:05).

Output:  results/drift_test_sam_med/overnight_rubber/
"""
from __future__ import annotations
import os, sys, re, json
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hunter_loader import parse_hunter_csv

DATA_DIR = Path(r"C:\Users\peter\Desktop\EM Imaging\Detectable Difference\Data\DriftTest\Sam Med Antenna\LiveData_RubberBand\LiveData")
OUT_DIR  = Path(HERE).parent / "results" / "drift_test_sam_med" / "overnight_rubber"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOWNSAMPLE_SEC = 30
BAND_LO_HZ = 2e9
BAND_HI_HZ = 8e9

S_NAMES = [f"S{i}{j}" for i in (1, 2, 3, 4) for j in (1, 2, 3, 4)]


def involves_port(sname, port):
    return str(port) in sname[1:]


TS_RE = re.compile(r"SPARAM_ReArr_(\d+)-(\d+)-(\d+)_(\d+)-(\d+)-(\d+)-(\d+)_")


def parse_ts(name):
    m = TS_RE.search(name)
    if not m: return None
    mo, dy, yr, hh, mm, ss, ms = (int(x) for x in m.groups())
    try:
        return datetime(yr, mo, dy, hh, mm, ss, ms * 1000)
    except ValueError:
        return None


def main():
    print(f"[SCAN] {DATA_DIR}")
    files = list(DATA_DIR.glob("SPARAM_ReArr_*.csv"))
    print(f"       {len(files)} files")
    ts_files = [(parse_ts(p.name), p) for p in files]
    ts_files = [(t, p) for t, p in ts_files if t is not None]
    ts_files.sort(key=lambda tp: tp[0])
    t0 = ts_files[0][0]
    duration_h = (ts_files[-1][0] - t0).total_seconds() / 3600.0
    print(f"       start = {t0}, duration = {duration_h:.2f} h")

    kept = []
    last = None
    for t, p in ts_files:
        if last is None or (t - last).total_seconds() >= DOWNSAMPLE_SEC:
            kept.append((t, p)); last = t
    print(f"[DOWNSAMPLE] {len(kept)} sweeps kept")

    n = len(kept)
    times_sec = np.zeros(n)
    mean_mag = np.zeros((n, 16))

    for i, (t, p) in enumerate(kept):
        if i % 100 == 0:
            print(f"    [{i}/{n}]  t={(t - t0).total_seconds()/3600:.2f}h", flush=True)
        freq, S = parse_hunter_csv(str(p))
        band_mask = (freq >= BAND_LO_HZ) & (freq <= BAND_HI_HZ)
        mag = np.abs(S)
        for k, sn in enumerate(S_NAMES):
            i_port = int(sn[1]); j_port = int(sn[2])
            trace = mag[i_port - 1, j_port - 1, :]
            mean_mag[i, k] = trace[band_mask].mean() if band_mask.any() else trace.mean()
        times_sec[i] = (t - t0).total_seconds()
    hours = times_sec / 3600.0

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    port_colors = {1: "steelblue", 2: "olive", 3: "firebrick", 4: "seagreen"}
    for pi, port in enumerate((1, 2, 3, 4)):
        ax = axes[pi // 2, pi % 2]
        for k, sn in enumerate(S_NAMES):
            if involves_port(sn, port):
                ax.plot(hours, mean_mag[:, k], lw=0.8, label=sn, alpha=0.8)
        ax.set_title(f"Port {port}  (2-8 GHz band-mean |S|, RUBBER BAND)")
        ax.set_xlabel("Hours from start"); ax.set_ylabel("|S|")
        ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "mean_mag_over_time.png", dpi=140)
    plt.close()
    print(f"[SAVE] {OUT_DIR/'mean_mag_over_time.png'}")

    # Rolling CV
    window_samples = max(3, int(300 / DOWNSAMPLE_SEC))
    def rolling_cv(x):
        out = np.full_like(x, np.nan)
        for i in range(len(x)):
            lo = max(0, i - window_samples + 1)
            w = x[lo:i+1]
            m = w.mean()
            out[i] = w.std() / (abs(m) + 1e-15)
        return out

    fig, ax = plt.subplots(figsize=(11, 5.5))
    per_port_cv = {}
    for port in (1, 2, 3, 4):
        idx = [k for k, sn in enumerate(S_NAMES) if involves_port(sn, port)]
        cv_stack = np.stack([rolling_cv(mean_mag[:, k]) for k in idx], axis=0)
        per_port_cv[port] = cv_stack.mean(0)
        lw = 2.5 if port == 3 else 1.5
        label = f"Port {port}"
        if port == 3: label += " (loose ant. tape)"
        ax.plot(hours, per_port_cv[port], label=label, color=port_colors[port], lw=lw)
    ax.set_xlabel("Hours from start"); ax.set_ylabel("Rolling CV of |S| (~5 min)")
    ax.set_title("Overnight drift settling per port -- RUBBER BAND (pre-solder)\n"
                 "Compare to first overnight (loose tape) for effect of mount improvement")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "rolling_std.png", dpi=140)
    plt.close()
    print(f"[SAVE] {OUT_DIR/'rolling_std.png'}")

    # First 2h zoom
    zoom = hours <= 2.0
    if zoom.any():
        fig, ax = plt.subplots(figsize=(11, 5.5))
        for port in (1, 2, 3, 4):
            lw = 2.5 if port == 3 else 1.5
            label = f"Port {port}"
            if port == 3: label += " (loose ant. tape)"
            ax.plot(hours[zoom], per_port_cv[port][zoom], label=label,
                    color=port_colors[port], lw=lw)
        ax.set_xlabel("Hours from start"); ax.set_ylabel("Rolling CV of |S|")
        ax.set_title("First 2 hours -- RUBBER BAND")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "per_port_zoom.png", dpi=140)
        plt.close()
        print(f"[SAVE] {OUT_DIR/'per_port_zoom.png'}")

    # Settling report
    late_mask = hours >= 10.0
    settle_report = {}
    for port in (1, 2, 3, 4):
        cv = per_port_cv[port]
        baseline = np.nanmean(cv[late_mask]) if late_mask.any() else np.nanmean(cv[-max(3, len(cv)//8):])
        threshold = 2.0 * baseline
        need = max(3, int(15 * 60 / DOWNSAMPLE_SEC))
        settle_at = None
        for i in range(len(cv) - need):
            if np.all(cv[i:i+need] < threshold):
                settle_at = hours[i]; break
        settle_report[port] = dict(
            asymptote_cv=float(baseline),
            settle_time_hr=None if settle_at is None else float(settle_at),
        )

    print("\n" + "=" * 60)
    print("SETTLING REPORT (rubber-band overnight)")
    print("=" * 60)
    for port, r in settle_report.items():
        s = f"{r['settle_time_hr']:.2f}" if r['settle_time_hr'] is not None else "never"
        print(f"Port {port}:  asymptote CV {r['asymptote_cv']:.5f}   settled at {s} hr")

    (OUT_DIR / "settling_summary.json").write_text(json.dumps({
        "n_files": len(files), "n_kept": n, "duration_hours": float(hours[-1]),
        "band_hz": [BAND_LO_HZ, BAND_HI_HZ],
        "per_port": {str(p): r for p, r in settle_report.items()},
    }, indent=2))
    print(f"\n[SAVE] {OUT_DIR/'settling_summary.json'}")


if __name__ == "__main__":
    main()
