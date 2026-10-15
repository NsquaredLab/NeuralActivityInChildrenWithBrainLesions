"""
02_heatmaps_muap.py
===================

Step 2 of the Figure 2 pipeline: P2P heatmaps and mean MUAP shapes of every
tracked motor unit, per recording day.

    heatmap     one SVG per unit: every discharge in a gesture is cut out of
                the filtered grid (50 Hz notch + 20-500 Hz band-pass,
                +-SHIMMER_WIN_MS) and averaged; the peak-to-peak amplitude of
                that mean MUAP on every electrode is the map. One colour scale
                per recording day (0 to the largest P2P of the file); title:
                spatial correlation of every pair of gestures.
    mean MUAP   one SVG per day: per unit the mean MUAP on its strongest
                electrode over all discharges, below it one row per gesture on
                the same electrode; one shared y-axis.

A gesture with fewer than SHIMMER_MIN_SPIKES discharges gets no map / an empty
panel. Electrodes listed in ``_common.EXTRA_DISCARDED_CHANNELS`` are discarded
on top of the file's own ``discardChannelsVec``.

Input / Output
--------------
    in : per recording day the recording-day pickle (gesture segments) and the
         tracking .mat (paths in sessions.local.json)
    out: results/heatmaps/<S>_MU_<id>_heatmap.svg   (one per unit)
         results/heatmaps/<S>_mean_muap.svg         (one per recording day)

The recordings are not distributed with this repository (identifiable human
EMG of a vulnerable participant group); they are available from the authors on
reasonable request.

Dependencies
------------
    numpy, scipy, matplotlib

Usage
-----
    uv run python figure-two/02_heatmaps_muap.py

Author:  Pauline Wittermann (pauline.wittermann@fau.de) and Dominik I. Braun (dome.braun@fau.de)
"""

import matplotlib as mpl
mpl.use("Agg")           # save only, no window
import matplotlib.pyplot as plt   # noqa: E402

import _common as C               # noqa: E402
from _tracking import (           # noqa: E402
    make_heatmap_figure, make_mean_muap_figure, prepare_tracking_grid)

C.set_level("WARNING")

SHIMMER_WIN_MS = 25      # half window around a discharge (ms)
SHIMMER_MIN_SPIKES = 10  # fewest discharges a gesture needs for a MUAP

# Text stays text in the SVG (not paths).
FONT_RC = {"svg.fonttype": "none", "pdf.fonttype": 42}


def run_session(session, pkl, mat, out):
    print(f"--- {session} ---")
    # The gesture segments come from the pkl: the tracking file holds its
    # tasks one after the other, in pkl order.
    blocks = C.recording_day_blocks(pkl)
    grid = prepare_tracking_grid(mat, blocks, attribute=C.attribute(session))
    mpl.rcParams.update(FONT_RC)
    heat = make_heatmap_figure(mat, blocks, out_dir=out,
                               save_prefix=session, win_ms=SHIMMER_WIN_MS,
                               min_spikes=SHIMMER_MIN_SPIKES, grid=grid)
    plt.close("all")
    mpl.rcParams.update(FONT_RC)
    muap = make_mean_muap_figure(mat, blocks, out_dir=out,
                                 save_prefix=session, win_ms=SHIMMER_WIN_MS,
                                 min_spikes=SHIMMER_MIN_SPIKES, grid=grid)
    plt.close("all")
    print(f"  {len(heat)} heatmap(s), {len(muap)} unit(s) in the mean MUAP "
          f"figure")


def main():
    cfg = C.load_config()
    out = C.out_dir("heatmaps")
    for session in C.SESSIONS:
        pkl, mat, _ = C.session_files(cfg, session)
        run_session(session, pkl, mat, out)
    print(f"Wrote the heatmaps and mean MUAPs to {out}")


if __name__ == "__main__":
    main()
