"""LOSO on all 12 rubber-band sessions (8 pre-solder + 4 post-solder).

Bonus: also runs LOSO WITHIN each subgroup to see if pre-solder and
post-solder data are internally consistent.

The interesting question: does mixing pre- and post-solder sessions
in one LOSO pool hurt accuracy (because the drift regime changed)?
"""
from __future__ import annotations
import os, sys, json
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hunter_loader import load_hunter_session
from physics_features_4port import physics_features_4port
from data import per_session_zscore
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from joblib import Parallel, delayed


DATA_DIR = Path(r"C:\Users\peter\Desktop\EM Imaging\Detectable Difference\Data\DriftTest\Sam Med Antenna\Rubber_Band")
OUT_DIR  = Path(HERE).parent / "results" / "drift_test_sam_med" / "rubber_all12_loso"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRE_SOLDER = [
    "BreastPhantom_A3_Nothing_20260702_0859",
    "BreastPhantom_A3_Nothing_20260702_0928",
    "BreastPhantom_A3_Nothing_20260702_0958",
    "BreastPhantom_A3_Nothing_20260702_1029",
    "BreastPhantom_A3_Nothing_20260702_1058",
    "BreastPhantom_A3_Nothing_20260702_1139",
    "BreastPhantom_A3_Nothing_20260703_0808",
    "BreastPhantom_A3_Nothing_20260703_0936",
]
POST_SOLDER = [
    "BreastPhantom_A3_Nothing_20260703_1049",
    "BreastPhantom_A3_Nothing_20260703_1124",
    "BreastPhantom_A3_Nothing_20260703_1154",
    "BreastPhantom_A3_Nothing_20260703_1304",
]
ALL_SESSIONS = PRE_SOLDER + POST_SOLDER

SPLIT_SEEDS    = (42, 7, 13, 99, 2025)
ENSEMBLE_SEEDS = (42, 7, 13)
TEST_PER_POS_WITHIN = 4


def load_session(folder):
    out = load_hunter_session(str(folder), mode="full16")
    Xc = out["Xc"]; base_c = out["base_c"]; ypos = out["y_pos"]
    per_pos = {}
    for p in np.unique(ypos):
        per_pos[int(p)] = Xc[ypos == p]
    return per_pos, base_c


def calibrate(Xc, base_c):
    Y = Xc - base_c[None, :, :]
    return Y - Y.mean(axis=0, keepdims=True)

def physics(Y):
    X, _ = physics_features_4port(Y)
    return X

def _fit_predict(seed, X_tr, y_tr, X_te):
    clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=200,
                        early_stopping=True, validation_fraction=0.1,
                        n_iter_no_change=15, random_state=seed, verbose=False)
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)

def train_predict(X_tr, y_tr, X_te):
    mu = X_tr.mean(0); sd = X_tr.std(0) + 1e-8
    X_tr_z = (X_tr - mu) / sd; X_te_z = (X_te - mu) / sd
    sc = StandardScaler().fit(X_tr_z)
    X_tr_s = sc.transform(X_tr_z); X_te_s = sc.transform(X_te_z)
    probs_list = Parallel(n_jobs=len(ENSEMBLE_SEEDS), backend="threading")(
        delayed(_fit_predict)(es, X_tr_s, y_tr, X_te_s) for es in ENSEMBLE_SEEDS)
    return np.mean(probs_list, axis=0).argmax(1)


def stratified_split(y_sparse, seed, n_test):
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for p in np.unique(y_sparse):
        idxs = np.where(y_sparse == p)[0]; rng.shuffle(idxs)
        if len(idxs) <= n_test: tr.extend(idxs.tolist()); continue
        te.extend(idxs[:n_test].tolist())
        tr.extend(idxs[n_test:].tolist())
    return np.array(sorted(tr)), np.array(sorted(te))


def per_pos_vote(ypos_te, pred, label_to_pos):
    by_pos = defaultdict(list)
    for tp, pp in zip(ypos_te, pred): by_pos[int(tp)].append(int(pp))
    correct = 0
    for tp, preds in by_pos.items():
        u, c = np.unique(preds, return_counts=True)
        winner = label_to_pos[int(u[c.argmax()])]
        if winner == int(tp): correct += 1
    return correct / max(1, len(by_pos))


def eval_within(per_pos, base_c):
    sorted_pos = sorted(per_pos.keys())
    Xc = np.concatenate([per_pos[p] for p in sorted_pos], axis=0)
    ypos = np.concatenate([np.full(per_pos[p].shape[0], p) for p in sorted_pos]).astype(np.int64)
    Y = calibrate(Xc, base_c); X = physics(Y)
    pos_to_label = {p: i for i, p in enumerate(sorted_pos)}
    label_to_pos = sorted_pos
    y_dense = np.array([pos_to_label[int(p)] for p in ypos], dtype=np.int64)
    accs = []
    for s in SPLIT_SEEDS:
        tr, te = stratified_split(ypos, seed=s, n_test=TEST_PER_POS_WITHIN)
        pred = train_predict(X[tr], y_dense[tr], X[te])
        accs.append(per_pos_vote(ypos[te], pred, label_to_pos))
    return float(np.mean(accs)), float(np.std(accs)), len(sorted_pos)


def eval_loso(sessions, label_tag):
    sids = list(sessions.keys())
    pos_sets = [set(per_pos.keys()) for per_pos, _ in sessions.values()]
    valid = sorted(set.intersection(*pos_sets))
    pos_to_label = {p: i for i, p in enumerate(valid)}
    label_to_pos = valid

    feats, labels, yposes = {}, {}, {}
    for sid, (per_pos, base_c) in sessions.items():
        Xc = np.concatenate([per_pos[p] for p in valid], axis=0)
        ypos = np.concatenate([np.full(per_pos[p].shape[0], p) for p in valid]).astype(np.int64)
        Y = calibrate(Xc, base_c); X = physics(Y)
        feats[sid] = X
        labels[sid] = np.array([pos_to_label[int(p)] for p in ypos], dtype=np.int64)
        yposes[sid] = ypos

    fold_results = {}
    for test_sid in sids:
        train_sids = [s for s in sids if s != test_sid]
        X_tr = np.concatenate([feats[s] for s in train_sids])
        y_tr = np.concatenate([labels[s] for s in train_sids])
        sess_tr = np.concatenate([np.full(feats[s].shape[0], j, dtype=np.int64)
                                   for j, s in enumerate(train_sids)])
        X_te = feats[test_sid]; y_te = labels[test_sid]; ypos_te = yposes[test_sid]

        X_tr_z, _ = per_session_zscore(X_tr, sess_tr)
        mu = X_te.mean(0); sd = X_te.std(0) + 1e-8
        X_te_z = (X_te - mu) / sd
        pred = train_predict(X_tr_z, y_tr, X_te_z)
        trial = float((pred == y_te).mean())
        pos = per_pos_vote(ypos_te, pred, label_to_pos)
        fold_results[test_sid] = dict(trial=trial, pos=pos)
        print(f"    [{label_tag}] fold test={test_sid}:  trial={trial*100:5.2f}%   pos={pos*100:5.2f}%", flush=True)
    return fold_results, len(valid)


def load_group(names, tag):
    print(f"\nLoading {tag} ({len(names)} sessions)...")
    sess = {}
    for n in names:
        folder = DATA_DIR / n
        if not folder.exists():
            print(f"[SKIP] {folder}"); continue
        sid = n.split("_")[-1] if "SamMedSep" not in n else n.split("_")[-2] + "_" + n.split("_")[-1]
        sid = n.split("_")[-1]
        print(f"  [LOAD] {sid} ...", flush=True)
        sess[sid] = load_session(folder)
    return sess


def group_summary(sessions, label_tag):
    within_results = {}
    n_pos_all = None
    for sid, (per_pos, base_c) in sessions.items():
        mean, std, n_pos = eval_within(per_pos, base_c)
        within_results[sid] = dict(mean=mean, std=std, n_positions=n_pos)
        n_pos_all = n_pos
    within_mean = np.mean([r["mean"] for r in within_results.values()])
    print(f"    [{label_tag}]  WITHIN mean: {within_mean*100:.2f}% (n={len(sessions)} sessions)")

    loso_folds, n_pos_loso = eval_loso(sessions, label_tag)
    loso_mean_pos = np.mean([r["pos"] for r in loso_folds.values()])
    print(f"    [{label_tag}]  LOSO   mean: {loso_mean_pos*100:.2f}%")

    return dict(within=within_results, within_mean=within_mean,
                loso_folds=loso_folds, loso_mean=loso_mean_pos,
                n_positions=n_pos_all)


def main():
    print("=" * 70)
    print("DRIFT TEST: pre-solder / post-solder / all-12 LOSO on rubber-band")
    print("=" * 70)

    all_sessions = load_group(ALL_SESSIONS, "all12")
    pre_sess  = {sid: v for sid, v in all_sessions.items() if any(sid in p for p in PRE_SOLDER)}
    post_sess = {sid: v for sid, v in all_sessions.items() if any(sid in p for p in POST_SOLDER)}

    print(f"\nPRE-SOLDER: {list(pre_sess.keys())}")
    print(f"POST-SOLDER: {list(post_sess.keys())}")

    print("\n" + "=" * 70)
    print("GROUP A: PRE-SOLDER (n=8)")
    print("=" * 70)
    pre_summary = group_summary(pre_sess, "PRE")

    print("\n" + "=" * 70)
    print("GROUP B: POST-SOLDER (n=4)")
    print("=" * 70)
    post_summary = group_summary(post_sess, "POST")

    print("\n" + "=" * 70)
    print("GROUP C: ALL 12 SESSIONS COMBINED")
    print("=" * 70)
    all_summary = group_summary(all_sessions, "ALL12")

    chance = 1.0 / all_summary["n_positions"]
    print(f"\nCHANCE = 1/{all_summary['n_positions']} = {chance*100:.2f}%")

    print("\n" + "=" * 70)
    print("HEADLINE COMPARISON")
    print("=" * 70)
    print(f"{'Group':<20}{'N':>4}{'WITHIN':>12}{'LOSO':>12}")
    print(f"{'Tape (prior)':<20}{4:>4}{82.19:>11.2f}%{2.34:>11.2f}%")
    print(f"{'Rubber pre-solder':<20}{len(pre_sess):>4}{pre_summary['within_mean']*100:>11.2f}%{pre_summary['loso_mean']*100:>11.2f}%")
    print(f"{'Rubber post-solder':<20}{len(post_sess):>4}{post_summary['within_mean']*100:>11.2f}%{post_summary['loso_mean']*100:>11.2f}%")
    print(f"{'Rubber all 12':<20}{len(all_sessions):>4}{all_summary['within_mean']*100:>11.2f}%{all_summary['loso_mean']*100:>11.2f}%")

    result = {
        "chance_pct": chance * 100,
        "n_positions": all_summary["n_positions"],
        "pre_solder":   {k: v for k, v in pre_summary.items() if k != "loso_folds"},
        "post_solder":  {k: v for k, v in post_summary.items() if k != "loso_folds"},
        "all12":        {k: v for k, v in all_summary.items() if k != "loso_folds"},
    }
    (OUT_DIR / "loso_all12_summary.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"\n[SAVE] {OUT_DIR/'loso_all12_summary.json'}")


if __name__ == "__main__":
    main()
