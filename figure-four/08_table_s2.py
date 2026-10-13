"""
08_table_s2.py
==============

Step 8 of the Figure 4 pipeline: Supplementary Table S2, offline gesture
classification performance per session and per class (leave-one-repetition-out).
Reads only the step-03 and step-04 results; no raw data needed.

Per session
    K                       number of classes
    majority baseline       accuracy of always predicting the most frequent class (%)
    Rep.                    held-out repetitions (detected attempts + rest parts)
    accuracy, macro F1      with the 95 % bootstrap intervals of step 03
    p                       permutation test of the accuracy (step 04, uncorrected)
Per class
    Rep., recall            recall = diagonal of the confusion matrix (% of true class)
    F1                      per-class F1
    AUROC                   one-vs-rest, with the 95 % bootstrap interval of step 04
    p                       permutation test of the AUROC, Holm-adjusted across all
                            class-wise tests (step 04)

Input / Output
--------------
    in : results/loro_results.json (step 03), results/loro_stats.json (step 04)
    out: results/table_s2_sessions.csv, results/table_s2_classes.csv   (full precision)
         results/table_s2.md                                           (as printed)

Dependencies
------------
    numpy

Usage
-----
    uv run python figure-four/08_table_s2.py

Author:  Pauline Wittermann (pauline.wittermann@fau.de) and Dominik I. Braun (dome.braun@fau.de)
"""

import csv
import json
import os

import numpy as np

import _common as C

NAME = {"P01-S1": "P01 Session 1", "P01-S2": "P01 Session 2"}
CLASS = {"Rest": "Rest", "Power Grasp": "Power grasp", "Pinch": "Pinch", "Tripod Pinch": "Tripod pinch"}
OUT_SESSIONS = os.path.join(C.RESULTS_DIR, "table_s2_sessions.csv")
OUT_CLASSES = os.path.join(C.RESULTS_DIR, "table_s2_classes.csv")
OUT_MD = os.path.join(C.RESULTS_DIR, "table_s2.md")


def p_txt(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def load(name):
    with open(os.path.join(C.RESULTS_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def main():
    res, stats = load("loro_results.json"), load("loro_stats.json")["sessions"]
    sessions, classes = [], []
    for s in C.SESSIONS:
        r, st = res[s], stats[s]
        assert abs(r["accuracy"] - st["accuracy"]) < 1e-12, f"{s}: steps 03 and 04 disagree"
        sessions.append(dict(session=s, K=r["K"], majority_baseline=r["chance_accuracy"],
                             repetitions=sum(r["n_units_per_class"].values()), accuracy=r["accuracy"],
                             accuracy_ci_low=r["accuracy_ci95"][0], accuracy_ci_high=r["accuracy_ci95"][1],
                             f1_macro=r["f1_macro"], f1_macro_ci_low=r["f1_macro_ci95"][0],
                             f1_macro_ci_high=r["f1_macro_ci95"][1], p_accuracy=st["p_accuracy"]))
        cm = np.array(r["confusion_matrix"], float)
        for k, g in enumerate(C.GESTURES):
            if g not in r["f1_per_class"]:
                continue
            a = st["auroc"][g]
            assert abs(a["value"] - r["auroc"][g]) < 1e-9, f"{s} {g}: steps 03 and 04 disagree"
            classes.append(dict(session=s, cls=g, repetitions=r["n_units_per_class"][g], recall=cm[k, k] / cm[k].sum(),
                                f1=r["f1_per_class"][g], auroc=a["value"], auroc_ci_low=a["ci95"][0],
                                auroc_ci_high=a["ci95"][1], p_auroc=a["p"], p_auroc_holm=a["p_holm"]))

    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    for path, rows in ((OUT_SESSIONS, sessions), (OUT_CLASSES, classes)):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    md = ["**Performance per session**", "",
          "| Participant / session | K | Majority-class baseline (%) | Rep. | Accuracy (%) [95% CI] "
          "| Macro-F1 [95% CI] | p (uncorrected) |", "|---|---|---|---|---|---|---|"]
    for r in sessions:
        md.append(f"| {NAME.get(r['session'], r['session'])} | {r['K']} | {100 * r['majority_baseline']:.1f} "
                  f"| {r['repetitions']} | {100 * r['accuracy']:.2f} [{100 * r['accuracy_ci_low']:.2f}–"
                  f"{100 * r['accuracy_ci_high']:.2f}] | {r['f1_macro']:.3f} [{r['f1_macro_ci_low']:.3f}–"
                  f"{r['f1_macro_ci_high']:.3f}] | {p_txt(r['p_accuracy'])} |")
    md += ["", "**Performance per class**", "",
           "| Participant / session | Class | Rep. | Recall (%) | F1 | AUROC [95% CI] | p (Holm-adjusted) |",
           "|---|---|---|---|---|---|---|"]
    prev = None
    for r in classes:
        name = NAME.get(r["session"], r["session"]) if r["session"] != prev else ""
        prev = r["session"]
        md.append(f"| {name} | {CLASS[r['cls']]} | {r['repetitions']} | {100 * r['recall']:.1f} | {r['f1']:.3f} "
                  f"| {r['auroc']:.2f} [{r['auroc_ci_low']:.2f}–{r['auroc_ci_high']:.2f}] "
                  f"| {p_txt(r['p_auroc_holm'])} |")
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print("saved", OUT_SESSIONS, OUT_CLASSES, OUT_MD)


if __name__ == "__main__":
    main()
