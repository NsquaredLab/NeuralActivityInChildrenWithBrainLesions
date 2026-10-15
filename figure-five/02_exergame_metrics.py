"""
02_exergame_metrics.py
======================

Step 2 of the Figure 5 pipeline: the numbers of one exergame run -- accuracy,
reaction latency and the event-based detection
(``_metrics.compute_recording_metrics``).

    accuracy (%)         share of control frames in which the binary prediction
                         equals the cue (rest frames count as correct, so it is
                         high by construction)
    sample-wise          precision / recall / F1 over the control frames
    event-wise           TP / FN / FP, precision / recall / F1 -- a cue is hit if
                         a prediction burst of >= MIN_BURST_SECONDS overlaps it
    reaction latency (s) per cue: cue onset to the first rising edge of the
                         prediction after it; a cue without an edge within
                         MAX_LATENCY s counts as missed and gets no latency.
                         Summary over the hit cues: n, mean, SD, median, min, max.

SD: ``latency_sd_s`` is the population SD (ddof = 0); ``latency_sd_ddof1_s`` is
the sample SD (ddof = 1, empty for n < 2) -- the one to report.

Input / Output
--------------
    in : the exergame recording of _common.EXERGAME_SESSION (path in sessions.local.json)
    out: results/Fig5_exergame_metrics_values.csv   (one row: every number above)
         results/Fig5_exergame_per_cue_values.csv   (one row per cue: window, hit, latency)

The recording is not distributed with this repository (identifiable human EMG
of a vulnerable participant group); it is available from the authors on
reasonable request.

Dependencies
------------
    numpy

Usage
-----
    uv run python figure-five/02_exergame_metrics.py

Author:  Pauline Wittermann (pauline.wittermann@fau.de) and Dominik I. Braun (dome.braun@fau.de)
"""

import os

import numpy as np

import _common as C
from _metrics import (compute_confusion_matrix_event, compute_latency,
                      compute_recording_metrics)

C.set_level("INFO")

# Activations shorter than this are treated as noise by the event-based counts.
MIN_BURST_SECONDS = 0.5
# Cues whose first reaction comes later than this count as missed, not as slow.
MAX_LATENCY = 2.0
GERMAN_EXCEL = True

COLUMNS = [
    "participant", "visit", "session",
    "n_cues", "n_frames", "duration_s",
    "active_fraction_gt", "active_fraction_pred",
    "accuracy",
    "sample_precision", "sample_recall", "sample_f1",
    "sample_tp", "sample_fp", "sample_fn", "sample_tn",
    "n_cues_detected", "event_tp", "event_fp", "event_fn",
    "event_precision", "event_recall", "event_f1",
    "n_latencies", "latency_mean_s", "latency_sd_s", "latency_sd_ddof1_s",
    "latency_median_s", "latency_min_s", "latency_max_s",
    "threshold", "min_burst_s", "max_latency_s", "fs", "frame",
]
SUMMARY_COLUMNS = [
    "session", "n_cues", "accuracy", "sample_f1", "n_cues_detected",
    "event_f1", "n_latencies", "latency_mean_s", "latency_sd_ddof1_s",
]
COLUMNS_CUE = ["cue", "start_s", "end_s", "detected", "latency_s"]


def per_cue(path, ground_truth):
    """One row per cue: hit (event-based) and its latency (empty if none)."""
    rec = C.load_exergame_recording(path)
    _, _, detected = compute_confusion_matrix_event(
        ground_truth, rec.predictions, C.FS, C.FRAME,
        min_burst_seconds=MIN_BURST_SECONDS)
    rows = []
    for k, (window, hit) in enumerate(zip(ground_truth, detected), start=1):
        # The latency function on this one window: [] if missed.
        lat = compute_latency([window], rec.predictions, C.FS, C.FRAME,
                              max_latency=MAX_LATENCY)
        rows.append({"cue": k, "start_s": window[0], "end_s": window[1],
                     "detected": bool(hit),
                     "latency_s": lat[0] if lat else float("nan")})
    return rows


def main():
    cfg = C.load_config()
    path = C.exergame_file(cfg)
    out = C.out_dir()
    session = C.EXERGAME_SESSION
    ground_truth = C.EXERGAME_GROUND_TRUTH

    row = compute_recording_metrics(
        path, ground_truth, name=session,
        min_burst_seconds=MIN_BURST_SECONDS, max_latency=MAX_LATENCY)
    cues = per_cue(path, ground_truth)

    lat = np.array([c["latency_s"] for c in cues], dtype=float)
    lat = lat[np.isfinite(lat)]
    if lat.size != row["n_latencies"]:
        raise SystemExit("Per-cue latencies do not match the summary -- check.")
    row["latency_sd_ddof1_s"] = (float(np.std(lat, ddof=1)) if lat.size > 1
                                 else float("nan"))
    row["participant"], row["visit"] = session[:3], session[4:]

    print(f"\n=== Exergame run {session} (min_burst = {MIN_BURST_SECONDS} s, "
          f"max_latency = {MAX_LATENCY} s) ===")
    print(C.format_table([row], SUMMARY_COLUMNS))
    print()
    print(C.format_table(cues, COLUMNS_CUE))

    C.write_csv([row], os.path.join(out, "Fig5_exergame_metrics_values.csv"),
                columns=COLUMNS, german_excel=GERMAN_EXCEL)
    C.write_csv(cues, os.path.join(out, "Fig5_exergame_per_cue_values.csv"),
                columns=COLUMNS_CUE, german_excel=GERMAN_EXCEL)
    print(f"\nWrote Fig5_exergame_metrics_values.csv and "
          f"Fig5_exergame_per_cue_values.csv to {out}")


if __name__ == "__main__":
    main()
