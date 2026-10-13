"""
03_loro_cv.py
=============

Step 3 of the Figure 4 pipeline: leave-one-repetition-out (LORO)
cross-validation on the step-02 RMS features.

RMS features z-scored with the training data, CatBoost with
``_emg.CATBOOST_PARAMS`` (CPU, fixed seed, so every fit is bit-for-bit
reproducible). Train/test split:

    unit   one detected gesture attempt (one region of the feature file). The
           rest window is split into n_rest equal parts, n_rest = median number
           of repetitions of the session's gesture classes (at least 2), each
           treated as one repetition.
    fold   every unit is held out once: train on all other frames, predict the
           unit's frames. For a rest unit, rest frames within BUFFER_S of it are
           also left out of training (adjacent rest frames are near-duplicates).

Out-of-fold predictions are pooled per session. Metrics: accuracy, balanced
accuracy, macro F1 over the classes present, confusion matrix, one-vs-rest
AUROC per class, chance levels (most frequent class; 1/K), and 95 % intervals
for accuracy and macro F1 from a class-stratified bootstrap over repetitions
(N_BOOT resamples, seed 0 per session).

CatBoost becomes unstable after many fits in one process on some machines, so
the folds run in worker processes of at most MAX_FITS fits each; progress is
saved after every fit and this script restarts workers until every session is
done (the result does not depend on where a worker stops).

Input / Output
--------------
    in : data/<S>_rms_features.npz (step 02)
    out: data/<S>_loro.npz            (true class, prediction, class probabilities
                                       and repetition of every frame)
         results/loro_results.json    (metrics per session)

Dependencies
------------
    numpy, scikit-learn, catboost

Usage
-----
    uv run python figure-four/03_loro_cv.py

Author:  Pauline Wittermann (pauline.wittermann@fau.de) and Dominik I. Braun (dome.braun@fau.de)
"""

import json
import os
import subprocess
import sys
import time

import numpy as np

import _common as C
import _emg as E

BUFFER_S = 1.0
MAX_FITS = 6            # CatBoost fits per worker process
N_BOOT = 2000
THREADS = 8             # CatBoost thread_count of the published run
MAX_WORKER_RUNS = 500
RESULTS_PATH = os.path.join(C.RESULTS_DIR, "loro_results.json")


def loro_path(session: str) -> str:
    return os.path.join(C.DATA_DIR, f"{session}_loro.npz")


def load_results() -> dict:
    if not os.path.isfile(RESULTS_PATH):
        return {}
    with open(RESULTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def units_for(d):
    """Return (unit id per frame, X, y, n_rest) for one session."""
    bad = E.keep_mask_from_bad_spec(d["kept_channels_1idx"], d["bad_channels"])
    fm = d["region_ids_per_frame"] >= 0
    X = d["features"][bad][:, fm].T.astype(np.float32)
    y = d["labels"][fm].astype(np.int64)
    reg = d["region_ids_per_frame"][fm]
    gest_reps = [len(np.unique(reg[y == k])) for k in range(1, 4) if (y == k).any()]
    n_rest = max(2, int(round(np.median(gest_reps))))
    unit = np.full(len(y), -1)
    u = 0
    for r in np.unique(reg):
        idx = np.where(reg == r)[0]
        if y[idx[0]] == 0:                                    # rest: split into pseudo-repetitions
            for part in np.array_split(idx, n_rest):
                unit[part] = u
                u += 1
        else:
            unit[idx] = u
            u += 1
    return unit, X, y, n_rest


def metrics(y, p, proba, unit, rng):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
    G = C.GESTURES
    present = sorted(set(y.tolist()))
    out = dict(accuracy=float(accuracy_score(y, p)), balanced_accuracy=float(balanced_accuracy_score(y, p)),
               f1_macro=float(f1_score(y, p, labels=present, average="macro", zero_division=0)),
               f1_per_class={G[k]: float(v) for k, v in
                             zip(present, f1_score(y, p, labels=present, average=None, zero_division=0))},
               confusion_matrix=confusion_matrix(y, p, labels=range(4)).tolist(),
               auroc={G[k]: float(roc_auc_score((y == k).astype(int), proba[:, k])) for k in present},
               chance_accuracy=float(max(np.bincount(y)) / len(y)), chance_f1=1 / len(present), K=len(present))
    # bootstrap over units (repetitions), stratified by class so every class stays represented
    ucls = {u: int(y[unit == u][0]) for u in np.unique(unit)}
    by_cls = {k: [u for u, c in ucls.items() if c == k] for k in present}
    frames = {u: np.where(unit == u)[0] for u in ucls}
    acc_b, f1_b = [], []
    for _ in range(N_BOOT):
        idx = np.concatenate([frames[u] for k in present for u in rng.choice(by_cls[k], len(by_cls[k]))])
        acc_b.append(accuracy_score(y[idx], p[idx]))
        f1_b.append(f1_score(y[idx], p[idx], labels=present, average="macro", zero_division=0))
    out["accuracy_ci95"] = [float(np.percentile(acc_b, 2.5)), float(np.percentile(acc_b, 97.5))]
    out["f1_macro_ci95"] = [float(np.percentile(f1_b, 2.5)), float(np.percentile(f1_b, 97.5))]
    out["unit_accuracy"] = {int(u): float((p[frames[u]] == y[frames[u]]).mean()) for u in ucls}
    out["unit_class"] = {int(u): c for u, c in ucls.items()}
    out["n_units_per_class"] = {G[k]: len(by_cls[k]) for k in present}
    return out


def worker():
    """Run up to MAX_FITS folds of the first unfinished session, then exit."""
    from catboost import CatBoostClassifier
    params = {**E.CATBOOST_PARAMS, "thread_count": THREADS}
    results = load_results()
    todo = [s for s in C.SESSIONS if s not in results]
    if not todo:
        return
    sess = todo[0]
    d = E.load_rms_dataset(os.path.join(C.DATA_DIR, f"{sess}_rms_features.npz"))
    unit, X, y, n_rest = units_for(d)
    rate = d["feature_rate_hz"]
    buf = int(round(BUFFER_S * rate))
    part = os.path.join(C.DATA_DIR, f"{sess}_loro_partial.npz")
    if os.path.isfile(part):
        with np.load(part) as z:                    # closed again, so the file can be removed at the end
            oof, proba, done = z["oof"], z["proba"], set(z["done"].tolist())
    else:
        oof, proba, done = np.full(len(y), -1), np.full((len(y), 4), np.nan), set()
    units = [u for u in np.unique(unit) if u not in done]
    print(f"{sess}: {len(np.unique(unit))} units (rest split into {n_rest}), {len(units)} to go", flush=True)
    t0 = time.time()
    for n_fit, u in enumerate(units):
        if n_fit >= MAX_FITS:
            print(f"{sess}: pausing after {MAX_FITS} fits", flush=True)
            return
        te = np.where(unit == u)[0]
        train = unit != u
        if y[te[0]] == 0:                                          # rest unit: buffer against neighbouring rest
            lo, hi = te[0] - buf, te[-1] + buf
            near = (np.arange(len(y)) >= lo) & (np.arange(len(y)) <= hi) & (y == 0)
            train &= ~near
        tr = np.where(train)[0]
        mu = X[tr].mean(0, keepdims=True)
        sd = X[tr].std(0, keepdims=True)
        sd[sd == 0] = 1.0
        m = CatBoostClassifier(**params)
        m.fit((X[tr] - mu) / sd, y[tr])
        Xt = (X[te] - mu) / sd
        oof[te] = m.predict(Xt).reshape(-1).astype(int)
        proba[np.ix_(te, np.asarray(m.classes_, int))] = m.predict_proba(Xt)
        done.add(int(u))
        np.savez_compressed(part, oof=oof, proba=proba, done=np.array(sorted(done)))
        print(f"{sess} unit {u} ({C.GESTURES[y[te[0]]]}, {len(te)} frames): acc {np.mean(oof[te] == y[te]):.2f} "
              f"[{len(done)}/{len(np.unique(unit))}, {time.time() - t0:.0f}s]", flush=True)

    proba = np.nan_to_num(proba, nan=0.0)
    r = metrics(y, oof, proba, unit, np.random.default_rng(0))
    r.update(session=sess, n_rest_units=n_rest, buffer_s=BUFFER_S, n_frames=int(len(y)),
             n_channels=int(X.shape[1]))
    np.savez_compressed(loro_path(sess), y=y, oof=oof, proba=proba, unit=unit, feature_rate=rate)
    results = load_results()
    results[sess] = r
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    os.remove(part)
    print(f"SESSION DONE {sess}: acc {r['accuracy']:.3f} [{r['accuracy_ci95'][0]:.2f}-{r['accuracy_ci95'][1]:.2f}]  "
          f"F1 {r['f1_macro']:.3f} [{r['f1_macro_ci95'][0]:.2f}-{r['f1_macro_ci95'][1]:.2f}]  "
          f"bAcc {r['balanced_accuracy']:.3f}", flush=True)


def main():
    if "--worker" in sys.argv:
        worker()
        return
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    for _ in range(MAX_WORKER_RUNS):
        if all(s in load_results() for s in C.SESSIONS):
            print("ALL DONE ->", RESULTS_PATH)
            return
        r = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "--worker"])
        if r.returncode != 0:
            print(f"worker exited with code {r.returncode}; restarting (progress is kept)", flush=True)
    raise RuntimeError(f"LORO not finished after {MAX_WORKER_RUNS} worker runs")


if __name__ == "__main__":
    main()
