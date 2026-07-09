"""Pre-solder vs post-solder antenna 3 fix analysis.

Peter re-soldered antenna 3 between sessions 0936 and 1049 on July 3.
This script splits the 12 rubber-band sessions into pre-solder (8) and
post-solder (4), computes per-port trial-CV and slow-drift for each
group, and reports:

  1. Did antenna 3 fix reduce port-3 drift further beyond just rubber
     band? (Compare pre/post S33 numbers)
  2. Did anything change on the OTHER ports? (Peter noticed port 1
     variability afterward)
  3. Overall drift-improvement summary.

Also produces a session-by-session timeline plot so we can see any
sudden shifts at the re-solder timestamp.
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

DATA_DIR = Path(r"C:\Users\peter\Desktop\EM Imaging\Detectable Difference\Data\DriftTest\Sam Med Antenna\Rubber_Band")
OUT_DIR  = Path(HERE).parent / "results" / "drift_test_sam_med" / "solder_fix"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Ordered by time. Peter re-soldered antenna 3 AFTER session 0936 on July 3.
PRE_SOLDER = [
    ("20260702_0859", "BreastPhantom_A3_Nothing_20260702_0859"),
    ("20260702_0928", "BreastPhantom_A3_Nothing_20260702_0928"),
    ("20260702_0958", "BreastPhantom_A3_Nothing_20260702_0958"),
    ("20260702_1029", "BreastPhantom_A3_Nothing_20260702_1029"),
    ("20260702_1058", "BreastPhantom_A3_Nothing_20260702_1058"),
    ("20260702_1139", "BreastPhantom_A3_Nothing_20260702_1139"),
    ("20260703_0808", "BreastPhantom_A3_Nothing_20260703_0808"),
    ("20260703_0936", "BreastPhantom_A3_Nothing_20260703_0936"),
]
POST_SOLDER = [
    ("20260703_1049", "BreastPhantom_A3_Nothing_20260703_1049"),
    ("20260703_1124", "BreastPhantom_A3_Nothing_20260703_1124"),
    ("20260703_1154", "BreastPhantom_A3_Nothing_20260703_1154"),
    ("20260703_1304", "BreastPhantom_A3_Nothing_20260703_1304"),
]

S_NAMES = [f"S{i}{j}" for i in (1, 2, 3, 4) for j in (1, 2, 3, 4)]


def involves_port(sname, port):
    return str(port) in sname[1:]


def is_reflection(sname):
    return sname[1] == sname[2]


def per_session_metrics(sess):
    Xc = sess["Xc"]; mag = np.abs(Xc); ypos = sess["y_pos"]
    n_sp = mag.shape[1]; n = mag.shape[0]

    # Trial-CV
    trial_cv = np.zeros(n_sp)
    for k in range(n_sp):
        per_pos = []
        for p in np.unique(ypos):
            m = ypos == p
            tr = mag[m, k, :]
            per_pos.append((tr.std(0) / (tr.mean(0) + 1e-15)).mean())
        trial_cv[k] = np.median(per_pos)

    # Slow drift (first 10% vs last 10%)
    n_edge = max(1, n // 10)
    early = mag[:n_edge].mean(0)
    late  = mag[-n_edge:].mean(0)
    overall = mag.mean(0) + 1e-15
    slow_drift = (np.abs(late - early) / overall).mean(axis=-1)

    return trial_cv, slow_drift


def load_group(sessions, label):
    print(f"\nLoading {label} ({len(sessions)} sessions)...")
    cvs, slows = [], []
    for sid, name in sessions:
        folder = DATA_DIR / name
        if not folder.exists():
            print(f"[SKIP] {folder}"); continue
        print(f"  [LOAD] {sid} ...", flush=True)
        d = load_hunter_session(str(folder), mode="full16")
        cv, sd = per_session_metrics(d)
        cvs.append(cv); slows.append(sd)
    return np.array(cvs), np.array(slows)


def per_port_from_arr(arr):
    """arr: (n_sess, 16) -> {port: (n_sess,) mean over S-params involving port}"""
    out = {}
    for p in (1, 2, 3, 4):
        idx = [k for k, sn in enumerate(S_NAMES) if involves_port(sn, p)]
        out[p] = arr[:, idx].mean(axis=1)
    return out


def refl_from_arr(arr):
    """arr: (n_sess, 16) -> {port: (n_sess,) values of Sii for that port}"""
    out = {}
    for p in (1, 2, 3, 4):
        k = S_NAMES.index(f"S{p}{p}")
        out[p] = arr[:, k]
    return out


def main():
    pre_cv, pre_slow = load_group(PRE_SOLDER, "PRE-SOLDER")
    post_cv, post_slow = load_group(POST_SOLDER, "POST-SOLDER")

    pre_cv_pp   = per_port_from_arr(pre_cv)
    post_cv_pp  = per_port_from_arr(post_cv)
    pre_slow_pp  = per_port_from_arr(pre_slow)
    post_slow_pp = per_port_from_arr(post_slow)

    pre_slow_refl  = refl_from_arr(pre_slow)
    post_slow_refl = refl_from_arr(post_slow)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PER-PORT TRIAL-CV (mean across sessions +/- std)")
    print("=" * 80)
    print(f"{'Port':<6}{'PRE (n=8)':>22}{'POST (n=4)':>22}{'CHANGE':>12}")
    for p in (1, 2, 3, 4):
        m_pre  = pre_cv_pp[p].mean();  s_pre  = pre_cv_pp[p].std()
        m_post = post_cv_pp[p].mean(); s_post = post_cv_pp[p].std()
        change = 100 * (m_post - m_pre) / m_pre
        marker = "  <=" if abs(change) > 15 else ""
        print(f"P{p:<5}{m_pre:>12.5f} +/- {s_pre:.5f}  {m_post:>12.5f} +/- {s_post:.5f}  {change:>10.1f}%{marker}")

    print("\n" + "=" * 80)
    print("PER-PORT SLOW-DRIFT (mean over all Sxx involving port)")
    print("=" * 80)
    print(f"{'Port':<6}{'PRE (n=8)':>22}{'POST (n=4)':>22}{'CHANGE':>12}")
    for p in (1, 2, 3, 4):
        m_pre  = pre_slow_pp[p].mean();  s_pre  = pre_slow_pp[p].std()
        m_post = post_slow_pp[p].mean(); s_post = post_slow_pp[p].std()
        change = 100 * (m_post - m_pre) / m_pre
        marker = "  <=" if abs(change) > 20 else ""
        print(f"P{p:<5}{m_pre:>12.5f} +/- {s_pre:.5f}  {m_post:>12.5f} +/- {s_post:.5f}  {change:>10.1f}%{marker}")

    print("\n" + "=" * 80)
    print("REFLECTION-ONLY SLOW-DRIFT (Sii for each port)")
    print("=" * 80)
    print(f"{'Sxx':<6}{'PRE (n=8)':>22}{'POST (n=4)':>22}{'CHANGE':>12}")
    for p in (1, 2, 3, 4):
        m_pre  = pre_slow_refl[p].mean();  s_pre  = pre_slow_refl[p].std()
        m_post = post_slow_refl[p].mean(); s_post = post_slow_refl[p].std()
        change = 100 * (m_post - m_pre) / m_pre
        marker = "  <=" if abs(change) > 20 else ""
        print(f"S{p}{p}{'':<3}{m_pre:>12.5f} +/- {s_pre:.5f}  {m_post:>12.5f} +/- {s_post:.5f}  {change:>10.1f}%{marker}")

    # ------------------------------------------------------------------
    # PLOT 1: session-by-session timeline
    # ------------------------------------------------------------------
    all_sids  = [sid for sid, _ in PRE_SOLDER] + [sid for sid, _ in POST_SOLDER]
    all_cv    = np.vstack([pre_cv,   post_cv])
    all_slow  = np.vstack([pre_slow, post_slow])
    all_cv_pp   = per_port_from_arr(all_cv)
    all_slow_pp = per_port_from_arr(all_slow)
    solder_idx = len(PRE_SOLDER) - 0.5

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = {1: "steelblue", 2: "olive", 3: "firebrick", 4: "seagreen"}
    x = np.arange(len(all_sids))
    for port in (1, 2, 3, 4):
        lw = 2.5 if port in (1, 3) else 1.5
        label = f"Port {port}"
        if port == 1: label += " (new issue?)"
        if port == 3: label += " (soldered)"
        axes[0].plot(x, all_cv_pp[port], "o-", label=label, color=colors[port], lw=lw)
        axes[1].plot(x, all_slow_pp[port], "o-", label=label, color=colors[port], lw=lw)
    for ax in axes:
        ax.axvline(solder_idx, color="k", ls="--", lw=1.5, alpha=0.6, label="ANTENNA 3 RE-SOLDER")
        ax.set_xticks(x); ax.set_xticklabels(all_sids, rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Trial-CV of |S|"); axes[0].set_title("Trial noise per port across sessions")
    axes[1].set_ylabel("Slow drift |late-early|/overall"); axes[1].set_title("Slow drift per port across sessions")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "session_timeline.png", dpi=140)
    plt.close()
    print(f"\n[SAVE] {OUT_DIR/'session_timeline.png'}")

    # ------------------------------------------------------------------
    # PLOT 2: bar chart pre vs post
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    xr = np.arange(4); w = 0.35
    axes[0].bar(xr - w/2, [pre_cv_pp[p].mean() for p in (1, 2, 3, 4)],  yerr=[pre_cv_pp[p].std() for p in (1, 2, 3, 4)],
                width=w, label="PRE-SOLDER (n=8)",  color="tab:red",   alpha=0.85, capsize=4)
    axes[0].bar(xr + w/2, [post_cv_pp[p].mean() for p in (1, 2, 3, 4)], yerr=[post_cv_pp[p].std() for p in (1, 2, 3, 4)],
                width=w, label="POST-SOLDER (n=4)", color="tab:green", alpha=0.85, capsize=4)
    axes[0].set_xticks(xr); axes[0].set_xticklabels([f"Port {p}" for p in (1, 2, 3, 4)])
    axes[0].set_ylabel("Mean trial-CV of |S|"); axes[0].set_title("Trial noise: pre vs post antenna-3 re-solder")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].bar(xr - w/2, [pre_slow_pp[p].mean() for p in (1, 2, 3, 4)],  yerr=[pre_slow_pp[p].std() for p in (1, 2, 3, 4)],
                width=w, label="PRE-SOLDER (n=8)",  color="tab:red",   alpha=0.85, capsize=4)
    axes[1].bar(xr + w/2, [post_slow_pp[p].mean() for p in (1, 2, 3, 4)], yerr=[post_slow_pp[p].std() for p in (1, 2, 3, 4)],
                width=w, label="POST-SOLDER (n=4)", color="tab:green", alpha=0.85, capsize=4)
    axes[1].set_xticks(xr); axes[1].set_xticklabels([f"Port {p}" for p in (1, 2, 3, 4)])
    axes[1].set_ylabel("Slow drift"); axes[1].set_title("Slow drift: pre vs post antenna-3 re-solder")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "pre_vs_post_bar.png", dpi=140)
    plt.close()
    print(f"[SAVE] {OUT_DIR/'pre_vs_post_bar.png'}")

    # Save JSON
    (OUT_DIR / "summary.json").write_text(json.dumps({
        "pre_solder_sessions":  [sid for sid, _ in PRE_SOLDER],
        "post_solder_sessions": [sid for sid, _ in POST_SOLDER],
        "trial_cv_per_port": {
            "pre":  {str(p): {"mean": float(pre_cv_pp[p].mean()),  "std": float(pre_cv_pp[p].std())}  for p in (1, 2, 3, 4)},
            "post": {str(p): {"mean": float(post_cv_pp[p].mean()), "std": float(post_cv_pp[p].std())} for p in (1, 2, 3, 4)},
        },
        "slow_drift_per_port": {
            "pre":  {str(p): {"mean": float(pre_slow_pp[p].mean()),  "std": float(pre_slow_pp[p].std())}  for p in (1, 2, 3, 4)},
            "post": {str(p): {"mean": float(post_slow_pp[p].mean()), "std": float(post_slow_pp[p].std())} for p in (1, 2, 3, 4)},
        },
        "reflection_slow_drift": {
            "pre":  {f"S{p}{p}": float(pre_slow_refl[p].mean())  for p in (1, 2, 3, 4)},
            "post": {f"S{p}{p}": float(post_slow_refl[p].mean()) for p in (1, 2, 3, 4)},
        },
    }, indent=2))
    print(f"[SAVE] {OUT_DIR/'summary.json'}")


if __name__ == "__main__":
    main()
