"""Overnight drift-settling analysis.

Reads the ~55K continuous single-position sweeps from
LiveData_070126_Overnight/, downsamples to one sweep every N seconds
(default 30), and plots |S| over time for every S-parameter so we can see
when (or whether) drift settles.

Also computes a rolling std over a moving window and reports the time at
which the rolling std drops below various thresholds -- an empirical
"settling time" per port.

Output: results/drift_test_sam_med/overnight/
  {mean_mag_over_time.png, rolling_std.png, per_port_zoom.png,
   settling_summary.json}
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

DATA_DIR = Path(r"C:\Users\peter\Desktop\EM Imaging\Detectable Difference\Data\DriftTest\Sam Med Antenna\LiveData_070126_Overnight\LiveData")
OUT_DIR  = Path(HERE).parent / "results" / "drift_test_sam_med" / "overnight"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOWNSAMPLE_SEC = 30      # sample one sweep per this many seconds
BAND_LO_HZ = 2e9         # 2-8 GHz -- Peter's usable band for Sam antennas
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
    # -------- Enumerate + timestamp + downsample --------
    print(f"[SCAN] {DATA_DIR}")
    files = list(DATA_DIR.glob("SPARAM_ReArr_*.csv"))
    print(f"       {len(files)} files")
    ts_files = [(parse_ts(p.name), p) for p in files]
    ts_files = [(t, p) for t, p in ts_files if t is not None]
    ts_files.sort(key=lambda tp: tp[0])
    t0 = ts_files[0][0]
    print(f"       start = {t0}, end = {ts_files[-1][0]}, "
          f"duration = {(ts_files[-1][0]-t0).total_seconds()/3600:.2f} h")

    # Downsample: keep first sweep, then next when at least DOWNSAMPLE_SEC has passed
    kept = []
    last_kept_time = None
    for t, p in ts_files:
        if last_kept_time is None or (t - last_kept_time).total_seconds() >= DOWNSAMPLE_SEC:
            kept.append((t, p))
            last_kept_time = t
    print(f"[DOWNSAMPLE] {len(kept)} sweeps kept (one every ~{DOWNSAMPLE_SEC}s)")

    # -------- Parse each kept sweep --------
    n = len(kept)
    times_sec = np.zeros(n)                    # seconds from start
    mean_mag = np.zeros((n, 16))               # mean |S| over band per S-param
    mean_mag_full = np.zeros((n, 16))          # mean |S| over full sweep

    for i, (t, p) in enumerate(kept):
        if i % 100 == 0:
            print(f"    [{i}/{n}]  t={((t - t0).total_seconds()/3600):.2f}h ...", flush=True)
        freq, S = parse_hunter_csv(str(p))     # freq (F,), S (4,4,F) complex
        # linearize S into 16-channel order matching S_NAMES
        # S_NAMES[k] = S{i}{j} for i in 1..4, j in 1..4 -> S[i-1, j-1, :]
        mag = np.abs(S)                        # (4, 4, F)
        band_mask = (freq >= BAND_LO_HZ) & (freq <= BAND_HI_HZ)
        for k, sn in enumerate(S_NAMES):
            i_port = int(sn[1]); j_port = int(sn[2])
            trace = mag[i_port - 1, j_port - 1, :]
            mean_mag_full[i, k] = trace.mean()
            mean_mag[i, k]      = trace[band_mask].mean() if band_mask.any() else trace.mean()
        times_sec[i] = (t - t0).total_seconds()

    hours = times_sec / 3600.0

    # -------- Plot: |S| vs time, one line per S-param --------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    port_colors = {1: "steelblue", 2: "olive", 3: "firebrick", 4: "seagreen"}
    for pi, port in enumerate((1, 2, 3, 4)):
        ax = axes[pi // 2, pi % 2]
        for k, sn in enumerate(S_NAMES):
            if involves_port(sn, port):
                ax.plot(hours, mean_mag[:, k], lw=0.8, label=sn, alpha=0.8)
        ax.set_title(f"Port {port} S-params  (band-mean |S|, {BAND_LO_HZ/1e9:.0f}-{BAND_HI_HZ/1e9:.0f} GHz)")
        ax.set_xlabel("Hours from start")
        ax.set_ylabel("|S|")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "mean_mag_over_time.png", dpi=140)
    plt.close()
    print(f"[SAVE] {OUT_DIR/'mean_mag_over_time.png'}")

    # -------- Plot: rolling std per port (settling curve) --------
    # For a rolling window, compute std of |S| (relative to its mean) within the window
    window_samples = max(3, int(300 / DOWNSAMPLE_SEC))    # ~5-min window
    def rolling_cv(x):
        out = np.full_like(x, np.nan)
        for i in range(len(x)):
            lo = max(0, i - window_samples + 1)
            w = x[lo:i+1]
            m = w.mean()
            out[i] = w.std() / (abs(m) + 1e-15)
        return out

    # Per-port aggregated rolling CV = mean rolling CV over S-params involving that port
    fig, ax = plt.subplots(figsize=(11, 5.5))
    per_port_cv = {}
    for port in (1, 2, 3, 4):
        idx = [k for k, sn in enumerate(S_NAMES) if involves_port(sn, port)]
        cv_stack = np.stack([rolling_cv(mean_mag[:, k]) for k in idx], axis=0)
        per_port_cv[port] = cv_stack.mean(0)
        lw = 2.5 if port == 3 else 1.5
        label = f"Port {port} (SUSPECT)" if port == 3 else f"Port {port}"
        ax.plot(hours, per_port_cv[port], label=label, color=port_colors[port], lw=lw)
    ax.set_xlabel("Hours from start")
    ax.set_ylabel(f"Rolling CV of |S| (window ~{window_samples * DOWNSAMPLE_SEC / 60:.1f} min)")
    ax.set_title("Overnight drift settling per port\n(lower = more stable)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "rolling_std.png", dpi=140)
    plt.close()
    print(f"[SAVE] {OUT_DIR/'rolling_std.png'}")

    # -------- Zoom on the first 2 hours --------
    zoom_mask = hours <= 2.0
    if zoom_mask.any():
        fig, ax = plt.subplots(figsize=(11, 5.5))
        for port in (1, 2, 3, 4):
            lw = 2.5 if port == 3 else 1.5
            label = f"Port {port} (SUSPECT)" if port == 3 else f"Port {port}"
            ax.plot(hours[zoom_mask], per_port_cv[port][zoom_mask], label=label,
                    color=port_colors[port], lw=lw)
        ax.set_xlabel("Hours from start"); ax.set_ylabel("Rolling CV of |S|")
        ax.set_title("First 2 hours (warmup zoom)")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "per_port_zoom.png", dpi=140)
        plt.close()
        print(f"[SAVE] {OUT_DIR/'per_port_zoom.png'}")

    # -------- Settling-time report --------
    # For each port: baseline CV = mean rolling CV over hours 10-14 (long-term asymptote)
    # Settling time = first time the rolling CV stays within 2x baseline for >= 15 min
    late_mask = hours >= 10.0
    settle_report = {}
    for port in (1, 2, 3, 4):
        cv = per_port_cv[port]
        if late_mask.any():
            baseline = np.nanmean(cv[late_mask])
        else:
            baseline = np.nanmean(cv[-max(3, len(cv)//8):])
        threshold = 2.0 * baseline
        # First time cv stays under threshold for a sustained window
        need_samples = max(3, int(15 * 60 / DOWNSAMPLE_SEC))   # 15-min sustained
        settled_at = None
        for i in range(len(cv) - need_samples):
            if np.all(cv[i:i+need_samples] < threshold):
                settled_at = hours[i]
                break
        settle_report[port] = dict(
            asymptote_cv=float(baseline),
            settle_threshold=float(threshold),
            settle_time_hr=None if settled_at is None else float(settled_at),
        )

    print("\n" + "=" * 60)
    print("SETTLING REPORT (rolling CV settles within 2x long-term baseline)")
    print("=" * 60)
    print(f"{'Port':<6} {'Asymptote CV':<15} {'Settle time (hr)':<18}")
    for port, r in settle_report.items():
        settle_str = f"{r['settle_time_hr']:.2f}" if r['settle_time_hr'] is not None else "never"
        print(f"{port:<6} {r['asymptote_cv']:<15.5f} {settle_str:<18}")

    (OUT_DIR / "settling_summary.json").write_text(json.dumps({
        "n_files": len(files),
        "n_kept": n,
        "duration_hours": float(hours[-1]),
        "band_hz": [BAND_LO_HZ, BAND_HI_HZ],
        "window_sec": window_samples * DOWNSAMPLE_SEC,
        "per_port": {str(p): r for p, r in settle_report.items()},
    }, indent=2))
    print(f"\n[SAVE] {OUT_DIR/'settling_summary.json'}")


if __name__ == "__main__":
    main()
