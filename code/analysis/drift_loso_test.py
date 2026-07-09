"""Within-session 75/25 vs LOSO on the 4-session drift test.

The empty phantom has NO real per-position signal. Anything the model
learns is drift. This script asks:

  (A) WITHIN  For each session alone, split 75/25 stratified by position.
              How well can the model classify position from the empty-phantom
              data? Non-chance = there IS a per-position drift fingerprint
              within a session.

  (B) LOSO    4-fold leave-one-session-out (train on 3 sessions, test on 4th).
              If chance -> the fingerprint from (A) doesn't survive across
              sessions -> multi-session training WOULD suppress drift.
              If nonzero -> some drift is session-consistent -> adds bias.

Reports per-fold and mean accuracies. Chance = 1 / (# positions).
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


# ----------------------------------------------------------------------------
DATA_DIR = Path(r"C:\Users\peter\Desktop\EM Imaging\Detectable Difference\Data\DriftTest\Sam Med Antenna")
OUT_DIR  = Path(HERE).parent / "results" / "drift_test_sam_med"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SESSION_NAMES = [
    "BreastPhantom_A3_Nothing_20260701_1438_SamMedSep_001",
    "BreastPhantom_A3_Nothing_20260701_1509_SamMedSep_002",
    "BreastPhantom_A3_Nothing_20260701_1536_SamMedSep_003",
    "BreastPhantom_A3_Nothing_20260701_1603_SamMedSep_004",
]

SPLIT_SEEDS    = (42, 7, 13, 99, 2025)
ENSEMBLE_SEEDS = (42, 7, 13)
TEST_PER_POS_WITHIN = 4     # ~25% if 16 trials per position


# ----------------------------------------------------------------------------
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
    probs = np.mean(probs_list, axis=0)
    return probs.argmax(1)


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


# ----------------------------------------------------------------------------
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


def eval_loso(sessions):
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
        print(f"    fold test={test_sid}:  trial={trial*100:5.2f}%   pos-vote={pos*100:5.2f}%", flush=True)
    return fold_results, len(valid)


# ----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("DRIFT-TEST WITHIN vs LOSO   (empty phantom, 4 sessions)")
    print("=" * 70)

    sessions = {}
    for name in SESSION_NAMES:
        folder = DATA_DIR / name
        if not folder.exists():
            print(f"[SKIP] {folder} missing"); continue
        sid = name.split("_")[-1]
        print(f"[LOAD] {sid} ...")
        sessions[sid] = load_session(folder)

    if len(sessions) < 2:
        print("Not enough sessions to LOSO"); return

    # ------------------- WITHIN --------------------
    print("\n--- WITHIN (per-session 75/25) ---")
    within_results = {}
    n_positions_all = None
    for sid, (per_pos, base_c) in sessions.items():
        mean, std, n_pos = eval_within(per_pos, base_c)
        within_results[sid] = dict(mean=mean, std=std, n_positions=n_pos)
        n_positions_all = n_pos
        print(f"    {sid}:  {mean*100:5.2f}% +/- {std*100:.2f}%  (n_pos={n_pos})")
    within_mean = np.mean([r["mean"] for r in within_results.values()])
    print(f"    ---------  MEAN: {within_mean*100:.2f}%")

    # ------------------- LOSO ----------------------
    print("\n--- LOSO (train 3 sessions, test 1) ---")
    loso_folds, n_pos_loso = eval_loso(sessions)
    loso_mean_trial = np.mean([r["trial"] for r in loso_folds.values()])
    loso_mean_pos   = np.mean([r["pos"]   for r in loso_folds.values()])
    print(f"    ---------  MEAN trial: {loso_mean_trial*100:.2f}%   pos-vote: {loso_mean_pos*100:.2f}%")

    chance = 1.0 / n_positions_all
    print(f"\nCHANCE = 1/{n_positions_all} = {chance*100:.2f}%")

    # ------------------- INTERPRETATION ------------
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print(f"Within-session (drift-only) accuracy:  {within_mean*100:.2f}%")
    print(f"LOSO (across-session) accuracy:        {loso_mean_pos*100:.2f}%")
    print(f"Chance baseline:                        {chance*100:.2f}%")
    if within_mean > 5 * chance and loso_mean_pos < 2 * chance:
        print("\n=> Within-session drift IS a per-position fingerprint,")
        print("   but LOSO collapses to near-chance.")
        print("   => The drift fingerprint is SESSION-SPECIFIC.")
        print("   => Multi-session training will SUPPRESS this leakage.")
    elif within_mean > 5 * chance and loso_mean_pos > 5 * chance:
        print("\n=> BOTH within and LOSO show fingerprint.")
        print("   => The drift fingerprint is SESSION-CONSISTENT.")
        print("   => Multi-session training will NOT easily suppress it.")
        print("   => Real accuracy on real data may include drift signal.")
    else:
        print("\n=> Both metrics near chance -- no per-position leakage from drift.")

    # ------------------- SAVE ----------------------
    result = {
        "within":  {sid: {**r, "mean_pct": r["mean"] * 100, "std_pct": r["std"] * 100}
                    for sid, r in within_results.items()},
        "within_mean_pct": within_mean * 100,
        "loso_folds": {sid: {"trial_pct": r["trial"] * 100, "pos_pct": r["pos"] * 100}
                        for sid, r in loso_folds.items()},
        "loso_mean_trial_pct": loso_mean_trial * 100,
        "loso_mean_pos_pct":   loso_mean_pos * 100,
        "chance_pct":          chance * 100,
        "n_positions":         n_positions_all,
    }
    (OUT_DIR / "loso_summary.json").write_text(json.dumps(result, indent=2))
    print(f"\n[SAVE] {OUT_DIR/'loso_summary.json'}")


if __name__ == "__main__":
    main()
