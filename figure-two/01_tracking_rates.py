"""
01_tracking_rates.py
====================

Step 1 of the Figure 2 pipeline: motor-unit tracking panel per recording day,
with the hold labels as the gray cue.

One column per gesture: RMS envelope (mean EMG behind it) on top, below it per
tracked motor unit its discharge rate (1.5 s Hann window, pps) with its pulses
underneath (``_tracking.make_tracking_rates_figure``). The gray shading is a
0/1 mask built from the ``*.labels_auto_v2.json`` segments (``quality ==
"good"`` and ``gesture`` matching the task).

Settings
--------
    RATE_WINDOW        1.5 s   Hann window of the discharge rate
    RATES_RMS_WINDOW   1.0 s   RMS window
    RATES_RMS_SMOOTH   0.5 s   Hann smoothing of the RMS envelope
    DROP_DISCARDED     True    channels flagged in discardChannelsVec left out

Input / Output
--------------
    in : per recording day the recording-day pickle, the tracking .mat and the
         hold labels (paths in sessions.local.json)
    out: results/tracking_rates/<S>_labels_tracking_rates.svg   (one per recording day)
         results/tracking_rates/tracking_rates_labels_values.csv
                                                     (per label segment, used or not)

The recordings are not distributed with this repository (identifiable human
EMG of a vulnerable participant group); they are available from the authors on
reasonable request.

Dependencies
------------
    numpy, scipy, matplotlib

Usage
-----
    uv run python figure-two/01_tracking_rates.py

Author:  Pauline Wittermann (pauline.wittermann@fau.de) and Dominik I. Braun (dome.braun@fau.de)
"""

import json
import os
import warnings

import matplotlib as mpl
mpl.use("Agg")           # save only, no window (the plot function calls plt.show())
# plt.show() inside the plot functions does nothing under Agg -- no warning.
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
import matplotlib.pyplot as plt   # noqa: E402

import _common as C               # noqa: E402
from _tracking import make_tracking_rates_figure   # noqa: E402

C.set_level("WARNING")

DROP_DISCARDED = True
SHOW_EMG = True
RATE_WINDOW = 1.5        # Hann window of the discharge rate (s)
RATES_RMS_WINDOW = 1.0   # RMS window (s)
RATES_RMS_SMOOTH = 0.5   # Hann smoothing of the RMS envelope (s)
BARE = False             # no copy without text
QUALITY = "good"         # only these label segments enter the gray mask

# Text stays text in the SVG (not paths). Set before every figure: the
# plotting function restores matplotlib's defaults after drawing.
FONT_RC = {"svg.fonttype": "none", "pdf.fonttype": 42}


def run_session(session, pkl, mat, labels_json, gestures, out):
    print(f"\n--- {session} ---")
    tracking = C.load_tracking(mat, drop_discarded=DROP_DISCARDED,
                               gestures=gestures)
    blocks = C.recording_day_blocks(pkl)
    with open(labels_json, encoding="utf-8") as f:
        labels = json.load(f)["tasks"]

    # Copies of the pkl blocks, only "labels" swapped for the label mask.
    cue_blocks, seg_rows = [], []
    for block in blocks:
        key = block["task"]
        if key not in labels:
            raise ValueError(f"{session}: task {key} not in {os.path.basename(labels_json)}")
        mask, rows = C.label_mask(labels[key], key,
                                  block["stop"] - block["start"], session,
                                  quality=QUALITY)
        cue_blocks.append({**block, "labels": mask})
        seg_rows += rows
        print(f"  {key}: {sum(r['used'] for r in rows)}/{len(rows)} "
              f"segments in the mask")
    if len(cue_blocks) != len(tracking["blocks"]):
        # make_tracking_rates_figure would silently fall back to the trigger.
        raise ValueError(f"{session}: {len(cue_blocks)} pkl task(s) for "
                         f"{len(tracking['blocks'])} tracking block(s)")

    mpl.rcParams.update(FONT_RC)
    # The function appends "_tracking_rates.svg".
    make_tracking_rates_figure(
        tracking, save_prefix=os.path.join(out, f"{session}_labels"),
        show_emg=SHOW_EMG, rate_window=RATE_WINDOW,
        rms_window=RATES_RMS_WINDOW, rms_smooth=RATES_RMS_SMOOTH,
        cue_blocks=cue_blocks, bare_copy=BARE)
    plt.close("all")
    return seg_rows


def main():
    cfg = C.load_config()
    out = C.out_dir("tracking_rates")
    seg_rows = []
    for session, gestures in C.SESSIONS.items():
        pkl, mat, labels_json = C.session_files(cfg, session)
        seg_rows += run_session(session, pkl, mat, labels_json, gestures, out)
    out_values = os.path.join(out, "tracking_rates_labels_values.csv")
    C.write_csv(seg_rows, out_values)
    print(f"\nWrote the tracking rate panels and {os.path.basename(out_values)} to {out}")


if __name__ == "__main__":
    main()
