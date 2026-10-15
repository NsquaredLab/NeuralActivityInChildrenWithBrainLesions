"""
01_exergame_run.py
==================

Step 1 of the Figure 5 pipeline: one run of the bracelet exergame -- the
control signal against the cues, and its event-based confusion matrix.

    run plot           RMS envelope of the bracelet EMG (left axis, µV), cue
                       windows shaded, binary prediction of the myocontrol
                       (right axis) and the activation threshold
                       ``rms_reference * prediction_threshold`` (dashed)
    confusion matrix   event-based: a cue is hit (TP) if a prediction burst of
                       >= MIN_BURST_SECONDS overlaps it, else missed (FN); a
                       burst that overlaps no cue is a false alarm (FP); TN is
                       undefined for events

The cue windows of this run are transcribed from the protocol
(``_common.EXERGAME_GROUND_TRUTH``); the older myocontrol software stored no
cue track. The numbers (accuracy, latency, F1) come from step 02.

Input / Output
--------------
    in : the exergame recording of _common.EXERGAME_SESSION (path in sessions.local.json)
    out: results/Fig5_exergame_run.svg
         results/Fig5_exergame_confusion.svg

The recording is not distributed with this repository (identifiable human EMG
of a vulnerable participant group); it is available from the authors on
reasonable request.

Dependencies
------------
    numpy, matplotlib

Usage
-----
    uv run python figure-five/01_exergame_run.py

Author:  Pauline Wittermann (pauline.wittermann@fau.de) and Dominik I. Braun (dome.braun@fau.de)
"""

import os
import warnings

import matplotlib as mpl
mpl.use("Agg")           # save only, no window (the plot functions call plt.show())
# plt.show() inside the plot functions does nothing under Agg -- no warning.
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
import matplotlib.pyplot as plt   # noqa: E402

import _common as C               # noqa: E402
from _plot import make_confusion_figure, make_exergame_figure   # noqa: E402

C.set_level("INFO")

CUT_X = None              # crop the run plot to [0, CUT_X] s; None = whole run
MIN_BURST_SECONDS = 0.5   # shorter prediction bursts are noise (confusion)

# Text stays text in the SVG (not paths).
FONT_RC = {"svg.fonttype": "none", "pdf.fonttype": 42}


def main():
    cfg = C.load_config()
    path = C.exergame_file(cfg)
    out = C.out_dir()
    ground_truth = C.EXERGAME_GROUND_TRUTH

    mpl.rcParams.update(FONT_RC)
    make_exergame_figure(path, ground_truth, cutX=CUT_X,
                         savepath=os.path.join(out, "Fig5_exergame_run.svg"))
    plt.close("all")

    mpl.rcParams.update(FONT_RC)
    make_confusion_figure(path, ground_truth, title=None,
                          savepath=os.path.join(out, "Fig5_exergame_confusion.svg"),
                          min_burst_seconds=MIN_BURST_SECONDS)
    plt.close("all")
    print(f"Wrote Fig5_exergame_run.svg and Fig5_exergame_confusion.svg "
          f"({C.EXERGAME_SESSION}) to {out}")


if __name__ == "__main__":
    main()
