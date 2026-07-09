"""Synthetic-positions DD analysis on the RUBBER-BAND overnight (65K sweeps).

Same CI-band-gap + 10% relgap detection formula as
drift_synthetic_positions_dd.py, but points at LiveData_RubberBand.

The comparison of interest: how does the detection rate curve differ
from the first overnight (tape)?

  First overnight:  P3 42% in hour 0, dropping to ~15% for 9 hours
  This overnight:   Should show P3 elevated but not as bad (mount was
                    rubber-band, only solder joint remained bad).
"""
from __future__ import annotations
import os, sys, re, json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hunter_loader import parse_hunter_csv

DATA_DIR = Path(r"C:\Users\peter\Desktop\EM Imaging\Detectable Difference\Data\DriftTest\Sam Med Antenna\LiveData_RubberBand\LiveData")
OUT_DIR  = Path(HERE).parent / "results" / "drift_test_sam_med" / "overnight_rubber" / "synthetic_positions_dd"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HOUR_SEC = 3600
BASELINE_SEC = 30
POSITION_SEC = 30
BAND_LO_HZ = 2e9
BAND_HI_HZ = 8e9
MIN_RELATIVE_GAP = 0.10

S_ALL = [(i, j) for i in (1, 2, 3, 4) for j in (1, 2, 3, 4)]
UNIQ = [(i, j) for i in (1, 2, 3, 4) for j in (1, 2, 3, 4) if i >= j]
UNIQ_NAMES = [f"S{i}{j}" for i, j in UNIQ]


def involves_port(i, j, port):
    return i == port or j == port


TS_RE = re.compile(r"SPARAM_ReArr_(\d+)-(\d+)-(\d+)_(\d+)-(\d+)-(\d+)-(\d+)_")


def parse_ts(name):
    m = TS_RE.search(name)
    if not m: return None
    mo, dy, yr, hh, mm, ss, ms = (int(x) for x in m.groups())
    try:
        return datetime(yr, mo, dy, hh, mm, ss, ms * 1000)
    except ValueError:
        return None


def ci95(W):
    n = W.shape[-1]
    mu  = W.mean(axis=-1)
    if n < 2:
        return np.abs(mu), np.abs(mu)
    sig = W.std(axis=-1, ddof=1)
    tc  = stats.t.ppf(0.975, n - 1)
    half = np.abs(tc * sig / np.sqrt(n))
    return np.abs(mu) - half, np.abs(mu) + half


def compute_delta(base, obj):
    b_lo, b_hi = ci95(base); o_lo, o_hi = ci95(obj)
    ss = (b_hi - b_lo) + (o_hi - o_lo)
    diff1 = o_lo - b_hi; diff2 = b_lo - o_hi
    out = np.zeros_like(diff1)
    above = diff1 >= 0
    out[above] = diff1[above]
    below = (~above) & (np.abs(diff1) >= ss)
    out[below] = diff2[below]
    return out


def load_all_metadata():
    files = list(DATA_DIR.glob("SPARAM_ReArr_*.csv"))
    ts_files = [(parse_ts(p.name), p) for p in files]
    ts_files = [(t, p) for t, p in ts_files if t is not None]
    ts_files.sort(key=lambda tp: tp[0])
    return ts_files[0][0], ts_files


def compute_hour_dd(hour_files, band_mask):
    if not hour_files: return None
    hour_start = hour_files[0][0]

    windows = defaultdict(list)
    for t, p in hour_files:
        dt = (t - hour_start).total_seconds()
        wi = int(dt // POSITION_SEC)
        windows[wi].append(p)
    if 0 not in windows or len(windows) < 3:
        return None

    baseline_paths = windows[0]
    position_indices = sorted(k for k in windows if k > 0)
    if len(baseline_paths) < 3 or len(position_indices) < 5:
        return None

    band_freqs = np.where(band_mask)[0]
    F = len(band_freqs); n_uniq = len(UNIQ)

    nTb = len(baseline_paths)
    base_trials = np.zeros((n_uniq, F, nTb), dtype=np.float32)
    for si, path in enumerate(baseline_paths):
        _, S = parse_hunter_csv(str(path))
        mag = np.abs(S)
        for k, (i, j) in enumerate(UNIQ):
            base_trials[k, :, si] = mag[i - 1, j - 1, band_freqs]

    nP = len(position_indices)
    delta = np.zeros((nP, n_uniq, F), dtype=np.float32)
    for pi, wi in enumerate(position_indices):
        paths = windows[wi]
        nT = len(paths)
        obj_trials = np.zeros((n_uniq, F, nT), dtype=np.float32)
        for si, path in enumerate(paths):
            _, S = parse_hunter_csv(str(path))
            mag = np.abs(S)
            for k, (i, j) in enumerate(UNIQ):
                obj_trials[k, :, si] = mag[i - 1, j - 1, band_freqs]
        delta[pi] = compute_delta(base_trials, obj_trials)

    mu_b = base_trials.mean(axis=-1)
    relgap = delta / np.maximum(mu_b[None], 1e-12)

    det_per_pos_sp = (relgap >= MIN_RELATIVE_GAP).any(axis=-1)
    detection_rate_sp = det_per_pos_sp.mean(axis=0) * 100.0

    band_avg_delta = delta.mean(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        db_per_pos_sp = np.where(band_avg_delta > 0,
                                 20 * np.log10(band_avg_delta), np.nan)
    dd_dB_per_sp = np.nanmean(db_per_pos_sp, axis=0)
    relgap_per_sp = relgap.mean(axis=(0, 2))

    per_port_det = {}; per_port_relgap = {}
    for port in (1, 2, 3, 4):
        idx = [k for k, (i, j) in enumerate(UNIQ) if involves_port(i, j, port)]
        per_port_det[port] = float(np.mean(detection_rate_sp[idx]))
        per_port_relgap[port] = float(np.mean(relgap_per_sp[idx]))

    return dict(
        n_baseline=nTb, n_positions=nP,
        detection_rate_sp=detection_rate_sp.tolist(),
        dd_dB_per_sp=dd_dB_per_sp.tolist(),
        relgap_per_sp=relgap_per_sp.tolist(),
        per_port_detection=per_port_det,
        per_port_relgap=per_port_relgap,
    )


def main():
    print(f"[SCAN] {DATA_DIR}")
    t0, ts_files = load_all_metadata()
    n_files = len(ts_files)
    duration_h = (ts_files[-1][0] - t0).total_seconds() / 3600.0
    print(f"       {n_files} files, {duration_h:.2f} h")

    freq0, _ = parse_hunter_csv(str(ts_files[0][1]))
    band_mask = (freq0 >= BAND_LO_HZ) & (freq0 <= BAND_HI_HZ)

    n_hours = int(np.ceil(duration_h))
    hour_files = [[] for _ in range(n_hours)]
    for t, p in ts_files:
        h = int((t - t0).total_seconds() // HOUR_SEC)
        if 0 <= h < n_hours:
            hour_files[h].append((t, p))

    hour_results = []
    for h in range(n_hours):
        if len(hour_files[h]) < 60:
            print(f"[SKIP] hour {h}"); continue
        print(f"[HOUR {h}] {len(hour_files[h])} sweeps ...", flush=True)
        res = compute_hour_dd(hour_files[h], band_mask)
        if res is None:
            print("        skipped"); continue
        p_det = res["per_port_detection"]
        print(f"    positions={res['n_positions']}  det per port: "
              f"P1={p_det[1]:5.1f}%  P2={p_det[2]:5.1f}%  "
              f"P3={p_det[3]:5.1f}%  P4={p_det[4]:5.1f}%")
        hour_results.append((h, res))

    if not hour_results:
        print("No usable hours"); return

    hours = [h for h, _ in hour_results]
    n_h = len(hour_results)

    # Detection rate heatmap
    det_grid = np.zeros((len(UNIQ), n_h))
    for hi, (_, r) in enumerate(hour_results):
        det_grid[:, hi] = r["detection_rate_sp"]

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(det_grid, aspect="auto", cmap="magma", vmin=0, vmax=100)
    ax.set_yticks(range(len(UNIQ))); ax.set_yticklabels(UNIQ_NAMES, fontsize=9)
    ax.set_xticks(range(n_h)); ax.set_xticklabels([str(h) for h in hours])
    ax.set_xlabel("Hour of overnight run")
    ax.set_title(f"Detection rate per S-param per hour (RUBBER BAND, pre-solder)\n"
                 f"% of fake positions with relgap >= {MIN_RELATIVE_GAP:.0%}")
    for k, (i, j) in enumerate(UNIQ):
        if involves_port(i, j, 3):
            ax.get_yticklabels()[k].set_color("red")
            ax.get_yticklabels()[k].set_fontweight("bold")
    plt.colorbar(im, ax=ax, label="Detection rate (%)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "detection_rate_per_hour.png", dpi=140)
    plt.close()
    print(f"[SAVE] {OUT_DIR/'detection_rate_per_hour.png'}")

    # Per-port trends
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {1: "steelblue", 2: "olive", 3: "firebrick", 4: "seagreen"}
    for port in (1, 2, 3, 4):
        y_det = [r["per_port_detection"][port] for _, r in hour_results]
        y_rel = [r["per_port_relgap"][port] for _, r in hour_results]
        lw = 2.5 if port == 3 else 1.5
        label = f"Port {port}"
        if port == 3: label += " (bad solder joint)"
        axes[0].plot(hours, y_det, "o-", label=label, color=colors[port], lw=lw)
        axes[1].plot(hours, y_rel, "o-", label=label, color=colors[port], lw=lw)
    axes[0].set_ylabel("Detection rate (%)"); axes[0].set_xlabel("Hour")
    axes[0].set_title("Detection rate per port over hours (rubber band)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].axhline(MIN_RELATIVE_GAP, color="k", ls="--", lw=1, alpha=0.5, label="10% threshold")
    axes[1].set_ylabel("Mean relgap"); axes[1].set_xlabel("Hour")
    axes[1].set_title("Mean relgap per port over hours (rubber band)")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "per_port_dd_over_hours.png", dpi=140)
    plt.close()
    print(f"[SAVE] {OUT_DIR/'per_port_dd_over_hours.png'}")

    # Summary
    overall_det = float(np.mean(det_grid))
    early_det = float(np.mean(det_grid[:, :2])) if n_h >= 2 else float(np.mean(det_grid[:, :1]))
    late_det  = float(np.mean(det_grid[:, -2:])) if n_h >= 2 else float(np.mean(det_grid[:, -1:]))
    print("\n" + "=" * 70)
    print("SUMMARY (RUBBER BAND OVERNIGHT)")
    print("=" * 70)
    print(f"Overall mean detection rate:        {overall_det:5.1f}%")
    print(f"First 2 hours mean:                 {early_det:5.1f}%")
    print(f"Last 2 hours mean:                  {late_det:5.1f}%")

    (OUT_DIR / "summary.json").write_text(json.dumps({
        "mount": "rubber_band_pre_solder",
        "band_hz": [BAND_LO_HZ, BAND_HI_HZ],
        "detection_threshold": MIN_RELATIVE_GAP,
        "n_hours": n_h,
        "hours": hours,
        "uniq_sparams": UNIQ_NAMES,
        "per_hour": [{"hour": h, **{k: v for k, v in r.items() if not isinstance(v, np.ndarray)}}
                     for h, r in hour_results],
        "overall_mean_detection_rate": overall_det,
        "early_mean_detection_rate": early_det,
        "late_mean_detection_rate": late_det,
    }, indent=2))
    print(f"\n[SAVE] {OUT_DIR/'summary.json'}")


if __name__ == "__main__":
    main()
