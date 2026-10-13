"""
04_statistics.py
================

Step 4 of the Figure 4 pipeline: statistics on the saved leave-one-repetition-out
predictions (no retraining).

Per session, from data/<S>_loro.npz (true class, prediction, class
probabilities and repetition of every frame):

    permutation test   the true classes are permuted between held-out
                       repetitions (gesture attempts and rest parts); the
                       predictions stay fixed. N_PERM permutations. Statistics:
                       accuracy, macro F1 (over the classes present) and
                       one-vs-rest AUROC per class.
                       p = (1 + #{null >= observed}) / (1 + N_PERM), one-sided.
    bootstrap CI       one-vs-rest AUROC per class, 95 % percentile interval,
                       class-stratified bootstrap over repetitions (N_BOOT
                       resamples), same scheme as the accuracy / F1 intervals of
                       step 03.

Holm correction across the AUROC tests of all sessions and classes. One random
stream (seed SEED) runs through the sessions in the order of ``SESSIONS``.

Input / Output
--------------
    in : data/<S>_loro.npz (step 03)
    out: results/loro_stats.json, results/loro_stats.csv

Dependencies
------------
    numpy, scipy, scikit-learn

Usage
-----
    uv run python figure-four/04_statistics.py

Author:  Pauline Wittermann (pauline.wittermann@fau.de) and Dominik I. Braun (dome.braun@fau.de)
"""

import csv
import json
import os
from math import factorial

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

import _common as C

N_PERM, N_BOOT = 10000, 2000
SEED = 20261008
STATS_JSON = os.path.join(C.RESULTS_DIR, "loro_stats.json")
STATS_CSV = os.path.join(C.RESULTS_DIR, "loro_stats.csv")


def acc_f1(y, p, present):
    acc = np.mean(y == p)
    f1 = []
    for k in present:
        tp = np.sum((y == k) & (p == k))
        fp = np.sum((y != k) & (p == k))
        fn = np.sum((y == k) & (p != k))
        f1.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return acc, float(np.mean(f1))


def auc_from_ranks(ranks, pos):
    n1 = pos.sum()
    n0 = len(pos) - n1
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (len(ps) - i) * ps[j])
        adj[j] = min(1.0, run)
    return adj


def session_stats(sess, rng):
    z = np.load(os.path.join(C.DATA_DIR, f"{sess}_loro.npz"))
    y, p, proba, unit = z["y"], z["oof"], z["proba"], z["unit"]
    units = np.unique(unit)
    assert np.array_equal(units, np.arange(len(units))), "units are not 0..U-1"
    ucls = np.array([y[unit == u][0] for u in units])
    present = sorted(set(y.tolist()))
    acc, f1 = acc_f1(y, p, present)
    ranks = {k: rankdata(proba[:, k]) for k in present}
    auc = {k: auc_from_ranks(ranks[k], y == k) for k in present}
    for k in present:                                     # rank shortcut == sklearn
        assert abs(auc[k] - roc_auc_score(y == k, proba[:, k])) < 1e-9
    n_ge = {"accuracy": 0, "f1_macro": 0, **{k: 0 for k in present}}
    for _ in range(N_PERM):
        yp = rng.permutation(ucls)[unit]
        a, f = acc_f1(yp, p, present)
        n_ge["accuracy"] += a >= acc - 1e-12
        n_ge["f1_macro"] += f >= f1 - 1e-12
        for k in present:
            n_ge[k] += auc_from_ranks(ranks[k], yp == k) >= auc[k] - 1e-12
    pval = {key: (1 + n) / (1 + N_PERM) for key, n in n_ge.items()}
    # bootstrap CI for AUROC, stratified by class over repetitions
    by_cls = {k: units[ucls == k] for k in present}
    frames = {u: np.where(unit == u)[0] for u in units}
    boot = {k: [] for k in present}
    for _ in range(N_BOOT):
        idx = np.concatenate([frames[u] for k in present for u in rng.choice(by_cls[k], len(by_cls[k]))])
        for k in present:
            boot[k].append(roc_auc_score(y[idx] == k, proba[idx, k]))
    # number of distinct label arrangements (limits the smallest attainable p)
    n_arr = factorial(len(units))
    for k in present:
        n_arr //= factorial(int(np.sum(ucls == k)))
    G = C.GESTURES
    return dict(session=sess, n_units=int(len(units)), units_per_class={G[k]: int(np.sum(ucls == k)) for k in present},
                n_distinct_permutations=int(n_arr), accuracy=float(acc), p_accuracy=pval["accuracy"],
                f1_macro=float(f1), p_f1_macro=pval["f1_macro"],
                auroc={G[k]: dict(value=float(auc[k]), ci95=[float(np.percentile(boot[k], 2.5)),
                                                             float(np.percentile(boot[k], 97.5))],
                                  p=pval[k]) for k in present})


def main():
    rng = np.random.default_rng(SEED)
    res = {s: session_stats(s, rng) for s in C.SESSIONS}
    tests = [(s, g) for s in C.SESSIONS for g in res[s]["auroc"]]
    for (s, g), q in zip(tests, holm(np.array([res[s]["auroc"][g]["p"] for s, g in tests]))):
        res[s]["auroc"][g]["p_holm"] = float(q)
    meta = dict(n_perm=N_PERM, n_boot=N_BOOT, seed=SEED,
                test="one-sided permutation of class labels between held-out repetitions, predictions fixed; "
                     "p = (1 + #null >= observed) / (1 + n_perm)",
                ci="95% percentile, class-stratified bootstrap over repetitions", holm="across all AUROC tests")
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    with open(STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(dict(meta=meta, sessions=res), f, indent=2)
    with open(STATS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["session", "metric", "value", "ci_low", "ci_high", "p_perm", "p_holm", "distinct_perms"])
        for s in C.SESSIONS:
            r = res[s]
            w.writerow([s, "accuracy", r["accuracy"], "", "", r["p_accuracy"], "", r["n_distinct_permutations"]])
            w.writerow([s, "f1_macro", r["f1_macro"], "", "", r["p_f1_macro"], "", r["n_distinct_permutations"]])
            for g, a in r["auroc"].items():
                w.writerow([s, f"auroc_{g}", a["value"], a["ci95"][0], a["ci95"][1], a["p"], a["p_holm"],
                            r["n_distinct_permutations"]])
    for s in C.SESSIONS:
        r = res[s]
        print(f"{s:7s} units {r['n_units']:2d} (distinct perms {r['n_distinct_permutations']:>9d})  "
              f"acc {r['accuracy']:.2f} p={r['p_accuracy']:.4f}  F1 {r['f1_macro']:.2f} p={r['p_f1_macro']:.4f}")
        for g, a in r["auroc"].items():
            print(f"        AUROC {g:13s} {a['value']:.2f} [{a['ci95'][0]:.2f}, {a['ci95'][1]:.2f}]  "
                  f"p={a['p']:.4f}  p_holm={a['p_holm']:.4f}")
    print("saved", STATS_JSON)


if __name__ == "__main__":
    main()
