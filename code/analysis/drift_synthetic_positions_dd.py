"""Synthetic-positions DD on the overnight live data (uses SAME formula as
plot_hunter_dd.py phantom plots).

Carves the 15h continuous overnight sweep into "sessions" of 1 hour each:

  For each 1-hour block:
    - first 30 sec = "baseline"          (~30 sweeps @ 1 Hz)
    - rest of hour = "positions"         (~119 positions x 30 sec x ~30 sweeps)

DD per S-param per freq at each fake position, using the EXACT ci95 +
compute_delta formula from Detectable Difference/plot_hunter_dd.py:

    baseline: (nU, nF, nTb) linear magnitude trials
    obj:      (nU, nF, nT)   linear magnitude trials at this position

    ci95(W): mu = mean; half = t_crit * std/sqrt(n); returns (|mu|-half, |mu|+half)
    delta:   o_lo - b_hi   (obj above baseline)
             or b_lo - o_hi (obj below, only if gap >= sum of widths)
             or 0          (overlap)
    relgap = delta / mean_baseline
    Detected = (relgap >= 0.10) somewhere in the freq band (Peter's 10% threshold)

Reciprocal S-params dropped (keep i >= j) -> 10 uniques: S11, S21, S22,
S31, S32, S33, S41, S42, S43, S44.

Output: results/drift_test_sam_med/overnight/synthetic_positions_dd/
    detection_rate_per_hour.png  - per-S-param detection rate per hour
    best_case_dd_dB.png          - per-hour best-case DD in dB (20log10)
    per_port_dd_over_hours.png   - relgap trends per port
    summary.json                 - numeric summary
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

DATA_DIR = Path(r"C:\Users\peter\Desktop\EM Imaging\Detectable Difference\Data\DriftTest\Sam Med Antenna\LiveData_070126_Overnight\LiveData")
OUT_DIR  = Path(HERE).parent / "results" / "drift_test_sam_med" / "overnight" / "synthetic_positions_dd"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HOUR_SEC = 3600
BASELINE_SEC = 30
POSITION_SEC = 30
BAND_LO_HZ = 2e9
BAND_HI_HZ = 8e9
MIN_RELATIVE_GAP = 0.10       # Peter's phantom threshold

# All 16 S-params in order S11..S44 (matches 4x4 matrix flattened i*4+j)
S_ALL = [(i, j) for i in (1, 2, 3, 4) for j in (1, 2, 3, 4)]
# Unique lower-triangle S-params (i >= j): S11, S21, S22, S31, S32, S33, S41, S42, S43, S44
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


# ---- ci95 and compute_delta -- EXACT copy of plot_hunter_dd.py ----
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
    b_lo, b_hi = ci95(base)
    o_lo, o_hi = ci95(obj)
    ss = (b_hi - b_lo) + (o_hi - o_lo)
    diff1 = o_lo - b_hi
    diff2 = b_lo - o_hi
    out = np.zeros_like(diff1)
    above = diff1 >= 0
    out[above] = diff1[above]
    below = (~above) & (np.abs(diff1) >= ss)
    out[below] = diff2[below]
    return out


# ---- load metadata ----
def load_all_metadata():
    files = list(DATA_DIR.glob("SPARAM_ReArr_*.csv"))
    ts_files = [(parse_ts(p.name), p) for p in files]
    ts_files = [(t, p) for t, p in ts_files if t is not None]
    ts_files.sort(key=lambda tp: tp[0])
    return ts_files[0][0], ts_files


# ---- per-hour DD ----
def compute_hour_dd(hour_files, band_mask):
    """Return dict with delta (nU, nF), relgap (nU, nF, nP), etc.
       delta shape here: (nP, nU, nF).
    """
    if not hour_files:
        return None
    hour_start = hour_files[0][0]

    # split by 30-sec windows
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
    F = len(band_freqs)
    n_uniq = len(UNIQ)

    # ---- Baseline trials: (nU, F, nTb) ----
    nTb = len(baseline_paths)
    base_trials = np.zeros((n_uniq, F, nTb), dtype=np.float32)
    for si, path in enumerate(baseline_paths):
        _, S = parse_hunter_csv(str(path))
        mag = np.abs(S)      # (4, 4, F_full)
        for k, (i, j) in enumerate(UNIQ):
            base_trials[k, :, si] = mag[i - 1, j - 1, band_freqs]

    # ---- Position trials + delta ----
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

    # ---- relgap ----
    mu_b = base_trials.mean(axis=-1)   # (nU, F)
    relgap = delta / np.maximum(mu_b[None], 1e-12)

    # ---- Detection rate per S-param (frac of positions where any freq in band > 10%) ----
    # This is per-position detection: is this pseudo-position detectably different from baseline?
    det_per_pos_sp = (relgap >= MIN_RELATIVE_GAP).any(axis=-1)   # (nP, nU) bool
    detection_rate_sp = det_per_pos_sp.mean(axis=0) * 100.0      # (nU,)  percent

    # ---- Best-case DD in dB (band-avg delta, then 20 log10, then aggregate) ----
    band_avg_delta = delta.mean(axis=-1)                          # (nP, nU)
    # per S-param: mean over positions of dB
    with np.errstate(divide="ignore", invalid="ignore"):
        db_per_pos_sp = np.where(band_avg_delta > 0,
                                 20 * np.log10(band_avg_delta), np.nan)
    dd_dB_per_sp = np.nanmean(db_per_pos_sp, axis=0)               # (nU,)

    # ---- relgap avg per S-param (band + position aggregate) ----
    relgap_per_sp = relgap.mean(axis=(0, 2))                       # (nU,)

    # ---- per-port aggregation ----
    per_port_det = {}
    per_port_relgap = {}
    for port in (1, 2, 3, 4):
        idx = [k for k, (i, j) in enumerate(UNIQ) if involves_port(i, j, port)]
        per_port_det[port]    = float(np.mean(detection_rate_sp[idx]))
        per_port_relgap[port] = float(np.mean(relgap_per_sp[idx]))

    # ---- per-position, per-S-param DD in dB (kept for per-hour position-timeline plots)
    with np.errstate(divide="ignore", invalid="ignore"):
        db_pos_sp = np.where(band_avg_delta > 0,
                             20 * np.log10(band_avg_delta), np.nan)   # (nP, nU)
    # ---- per-position, per-S-param relgap (band-mean)
    relgap_pos_sp = relgap.mean(axis=-1)                              # (nP, nU)

    return dict(
        n_baseline=nTb,
        n_positions=nP,
        detection_rate_sp=detection_rate_sp.tolist(),   # (nU,)
        dd_dB_per_sp=dd_dB_per_sp.tolist(),
        relgap_per_sp=relgap_per_sp.tolist(),
        per_port_detection=per_port_det,
        per_port_relgap=per_port_relgap,
        db_pos_sp=db_pos_sp,                            # (nP, nU) DD dB per fake position x S-param
        relgap_pos_sp=relgap_pos_sp,                    # (nP, nU) relgap per fake position x S-param
    )


# ---- main ----
def main():
    print(f"[SCAN] {DATA_DIR}")
    t0, ts_files = load_all_metadata()
    n_files = len(ts_files)
    duration_h = (ts_files[-1][0] - t0).total_seconds() / 3600.0
    print(f"       {n_files} files, {duration_h:.2f} h total")

    # Determine band from first file
    freq0, _ = parse_hunter_csv(str(ts_files[0][1]))
    band_mask = (freq0 >= BAND_LO_HZ) & (freq0 <= BAND_HI_HZ)
    print(f"[BAND] {BAND_LO_HZ/1e9:.1f}-{BAND_HI_HZ/1e9:.1f} GHz -> {int(band_mask.sum())} freq points")

    # Group by hour
    n_hours = int(np.ceil(duration_h))
    hour_files = [[] for _ in range(n_hours)]
    for t, p in ts_files:
        h = int((t - t0).total_seconds() // HOUR_SEC)
        if 0 <= h < n_hours:
            hour_files[h].append((t, p))

    # Compute per hour
    hour_results = []
    for h in range(n_hours):
        if len(hour_files[h]) < 60:
            print(f"[SKIP] hour {h} too small ({len(hour_files[h])} sweeps)")
            continue
        print(f"[HOUR {h}] {len(hour_files[h])} sweeps ...", flush=True)
        res = compute_hour_dd(hour_files[h], band_mask)
        if res is None:
            print(f"        skipped"); continue
        p_det = res["per_port_detection"]
        p_rel = res["per_port_relgap"]
        print(f"    positions={res['n_positions']}  det-rate per port: "
              f"P1={p_det[1]:5.1f}%  P2={p_det[2]:5.1f}%  "
              f"P3={p_det[3]:5.1f}%  P4={p_det[4]:5.1f}%   "
              f"relgap-mean: P1={p_rel[1]:.4f} P2={p_rel[2]:.4f} "
              f"P3={p_rel[3]:.4f} P4={p_rel[4]:.4f}")
        hour_results.append((h, res))

    if not hour_results:
        print("No usable hours"); return

    hours    = [h for h, _ in hour_results]
    n_h = len(hour_results)

    # ---- PLOT 1: detection-rate heatmap, hours x uniq S-params ----
    det_grid = np.zeros((len(UNIQ), n_h))
    for hi, (_, r) in enumerate(hour_results):
        det_grid[:, hi] = r["detection_rate_sp"]

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(det_grid, aspect="auto", cmap="magma", vmin=0, vmax=100)
    ax.set_yticks(range(len(UNIQ))); ax.set_yticklabels(UNIQ_NAMES, fontsize=9)
    ax.set_xticks(range(n_h)); ax.set_xticklabels([str(h) for h in hours])
    ax.set_xlabel("Hour of overnight run")
    ax.set_title(f"Detection rate per S-param per hour\n"
                 f"(% of fake positions with relgap >= {MIN_RELATIVE_GAP:.0%})")
    for k, (i, j) in enumerate(UNIQ):
        if involves_port(i, j, 3):
            ax.get_yticklabels()[k].set_color("red")
            ax.get_yticklabels()[k].set_fontweight("bold")
    plt.colorbar(im, ax=ax, label="Detection rate (%)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "detection_rate_per_hour.png", dpi=140)
    plt.close()
    print(f"[SAVE] {OUT_DIR/'detection_rate_per_hour.png'}")

    # ---- PLOT 2: best-case DD in dB, hours x uniq ----
    db_grid = np.full((len(UNIQ), n_h), np.nan)
    for hi, (_, r) in enumerate(hour_results):
        db_grid[:, hi] = r["dd_dB_per_sp"]

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(db_grid, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(UNIQ))); ax.set_yticklabels(UNIQ_NAMES, fontsize=9)
    ax.set_xticks(range(n_h)); ax.set_xticklabels([str(h) for h in hours])
    ax.set_xlabel("Hour of overnight run")
    ax.set_title("Best-case detectable difference per S-param per hour\n"
                 "20 log10(band-avg delta), NaN where no gap")
    for k, (i, j) in enumerate(UNIQ):
        if involves_port(i, j, 3):
            ax.get_yticklabels()[k].set_color("red")
            ax.get_yticklabels()[k].set_fontweight("bold")
    plt.colorbar(im, ax=ax, label="DD (dB)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "best_case_dd_dB.png", dpi=140)
    plt.close()
    print(f"[SAVE] {OUT_DIR/'best_case_dd_dB.png'}")

    # ---- PLOT 3: per-port trends ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {1: "steelblue", 2: "olive", 3: "firebrick", 4: "seagreen"}
    for port in (1, 2, 3, 4):
        y_det = [r["per_port_detection"][port] for _, r in hour_results]
        y_rel = [r["per_port_relgap"][port] for _, r in hour_results]
        lw = 2.5 if port == 3 else 1.5
        label = f"Port {port} (suspect)" if port == 3 else f"Port {port}"
        axes[0].plot(hours, y_det, "o-", label=label, color=colors[port], lw=lw)
        axes[1].plot(hours, y_rel, "o-", label=label, color=colors[port], lw=lw)
    axes[0].axhline(0, color="k", ls="--", lw=1, alpha=0.5)
    axes[0].set_ylabel(f"Detection rate (%)")
    axes[0].set_xlabel("Hour"); axes[0].legend()
    axes[0].set_title(f"Detection rate per port over hours\n(relgap >= {MIN_RELATIVE_GAP:.0%})")
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(MIN_RELATIVE_GAP, color="k", ls="--", lw=1, alpha=0.5,
                    label=f"{MIN_RELATIVE_GAP:.0%} threshold")
    axes[1].set_ylabel("Mean relgap over S-params in band")
    axes[1].set_xlabel("Hour"); axes[1].legend()
    axes[1].set_title("Mean relgap per port over hours")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "per_port_dd_over_hours.png", dpi=140)
    plt.close()
    print(f"[SAVE] {OUT_DIR/'per_port_dd_over_hours.png'}")

    # ----------------------------------------------------------------------
    # PLOT 4: Per-hour position-timeline heatmaps (dB) - similar layout to
    #         the phantom DD plots, but with fake positions on the x-axis
    #         ordered by TIME instead of grid location.
    # ----------------------------------------------------------------------
    # global vmin/vmax across all hours for consistent colorbar
    all_db = np.concatenate([r["db_pos_sp"].ravel() for _, r in hour_results])
    vmin = float(np.nanpercentile(all_db, 5))
    vmax = float(np.nanpercentile(all_db, 99))

    for h, r in hour_results:
        db = r["db_pos_sp"]                        # (nP, nU)
        nP = db.shape[0]
        fig, ax = plt.subplots(figsize=(10, 5.5))
        im = ax.imshow(db.T, aspect="auto", cmap="viridis",
                       vmin=vmin, vmax=vmax, origin="lower",
                       extent=[0, nP * POSITION_SEC / 60.0, -0.5, len(UNIQ) - 0.5])
        ax.set_yticks(range(len(UNIQ))); ax.set_yticklabels(UNIQ_NAMES, fontsize=9)
        for k, (i, j) in enumerate(UNIQ):
            if involves_port(i, j, 3):
                ax.get_yticklabels()[k].set_color("red")
                ax.get_yticklabels()[k].set_fontweight("bold")
        ax.set_xlabel(f"Fake-position time within hour (minutes; 1 position = {POSITION_SEC}s)")
        ax.set_title(f"HOUR {h}  -  Detectable difference per fake position (dB)\n"
                     f"20 log10 of band-avg delta; 2-8 GHz; empty phantom (drift only)")
        plt.colorbar(im, ax=ax, label="DD (dB)")
        plt.tight_layout()
        out = OUT_DIR / f"hour_{h:02d}_positions_dB.png"
        plt.savefig(out, dpi=140)
        plt.close()
        if h in (0, 1, 4, 8, 14):
            print(f"[SAVE] {out}")

    # ----------------------------------------------------------------------
    # PLOT 5: Grand timeline - all hours concatenated, DD in dB
    # ----------------------------------------------------------------------
    concat_db = np.concatenate([r["db_pos_sp"] for _, r in hour_results], axis=0)  # (nP_total, nU)
    total_positions = concat_db.shape[0]
    total_minutes = total_positions * POSITION_SEC / 60.0

    fig, ax = plt.subplots(figsize=(15, 5))
    im = ax.imshow(concat_db.T, aspect="auto", cmap="viridis",
                   vmin=vmin, vmax=vmax, origin="lower",
                   extent=[0, total_minutes / 60.0, -0.5, len(UNIQ) - 0.5])
    ax.set_yticks(range(len(UNIQ))); ax.set_yticklabels(UNIQ_NAMES, fontsize=9)
    for k, (i, j) in enumerate(UNIQ):
        if involves_port(i, j, 3):
            ax.get_yticklabels()[k].set_color("red")
            ax.get_yticklabels()[k].set_fontweight("bold")
    # vertical divider between hours
    for hi in range(1, len(hour_results)):
        ax.axvline(hi * (119 * POSITION_SEC / 3600.0), color="white",
                   lw=0.5, alpha=0.4)
    ax.set_xlabel(f"Time from start (hours; each column = one {POSITION_SEC}-sec fake position)")
    ax.set_title("Grand timeline - DD (dB) per fake position, all hours concatenated\n"
                 "White divider = new hour (baseline resets)")
    plt.colorbar(im, ax=ax, label="DD (dB)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "grand_timeline_dB.png", dpi=140)
    plt.close()
    print(f"[SAVE] {OUT_DIR/'grand_timeline_dB.png'}")

    # ----------------------------------------------------------------------
    # PLOT 6: Grand timeline - relgap (10% threshold overlay)
    # ----------------------------------------------------------------------
    concat_rel = np.concatenate([r["relgap_pos_sp"] for _, r in hour_results], axis=0)
    fig, ax = plt.subplots(figsize=(15, 5))
    # Cap colorscale at 3x threshold for visibility of moderate signals
    vcap = MIN_RELATIVE_GAP * 3
    im = ax.imshow(concat_rel.T, aspect="auto", cmap="magma",
                   vmin=0, vmax=vcap, origin="lower",
                   extent=[0, total_minutes / 60.0, -0.5, len(UNIQ) - 0.5])
    ax.set_yticks(range(len(UNIQ))); ax.set_yticklabels(UNIQ_NAMES, fontsize=9)
    for k, (i, j) in enumerate(UNIQ):
        if involves_port(i, j, 3):
            ax.get_yticklabels()[k].set_color("red")
            ax.get_yticklabels()[k].set_fontweight("bold")
    for hi in range(1, len(hour_results)):
        ax.axvline(hi * (119 * POSITION_SEC / 3600.0), color="white",
                   lw=0.5, alpha=0.4)
    ax.set_xlabel("Time from start (hours)")
    ax.set_title(f"Grand timeline - relgap per fake position\n"
                 f"Colorbar capped at {vcap:.0%}; 10% = detection threshold")
    cb = plt.colorbar(im, ax=ax, label="relgap")
    cb.ax.axhline(MIN_RELATIVE_GAP / vcap, color="red", lw=1.5)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "grand_timeline_relgap.png", dpi=140)
    plt.close()
    print(f"[SAVE] {OUT_DIR/'grand_timeline_relgap.png'}")

    # ---- Summary ----
    overall_det = float(np.mean(det_grid))
    early_det = float(np.mean(det_grid[:, :2])) if n_h >= 2 else float(np.mean(det_grid[:, :1]))
    late_det  = float(np.mean(det_grid[:, -2:])) if n_h >= 2 else float(np.mean(det_grid[:, -1:]))
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Overall mean detection rate:        {overall_det:5.1f}%")
    print(f"First 2 hours mean detection rate:  {early_det:5.1f}%")
    print(f"Last 2 hours mean detection rate:   {late_det:5.1f}%")
    if overall_det > 50:
        print("\n=> Pure drift within an hour is DETECTABLE at >10% relgap on many/most")
        print("   fake positions. A CNN training on batch sessions would see this as")
        print("   real per-position signal and learn to overfit it.")
    elif overall_det > 10:
        print("\n=> Pure drift produces SOME detectable fake-position signal.")
    else:
        print("\n=> Pure drift within an hour is BELOW the 10%-relgap threshold.")
        print("   Real batch-session differences must come from other sources.")

    # Strip numpy arrays from per-hour dicts before JSON-serializing
    per_hour_json = []
    for h, r in hour_results:
        clean = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
        per_hour_json.append({"hour": h, **clean})

    (OUT_DIR / "summary.json").write_text(json.dumps({
        "band_hz": [BAND_LO_HZ, BAND_HI_HZ],
        "detection_threshold": MIN_RELATIVE_GAP,
        "n_hours": n_h,
        "hours": hours,
        "uniq_sparams": UNIQ_NAMES,
        "per_hour": per_hour_json,
        "overall_mean_detection_rate": overall_det,
        "early_mean_detection_rate": early_det,
        "late_mean_detection_rate": late_det,
    }, indent=2))
    print(f"\n[SAVE] {OUT_DIR/'summary.json'}")


if __name__ == "__main__":
    main()
