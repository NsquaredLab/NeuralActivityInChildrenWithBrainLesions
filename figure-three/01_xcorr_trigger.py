"""
01_xcorr_trigger.py
===================

Step 1 of the Figure 3 pipeline: cross-correlation of every tracked motor
unit's discharge rate with the recorded trigger -- per hold and over the whole
gesture.

Per recording day, task and tracked unit (``_common.tracked_tasks``): the
discharge rate (Hann window of XCORR_RATE_WINDOW s) and the trigger are cut to
each hold (from the task's ``labels``), mean-subtracted and cross-correlated as
MATLAB's ``xcorr(rate, trigger, maxlag, 'normalized')``, maxlag =
XCORR_MAX_LAG s (``_xcorr.compute_task_hold_xcorr``). A hold counts as active
for a unit with >= MIN_DISCHARGES_PER_RAMP discharges inside it. Plus the same
xcorr over the whole gesture, rest included (``xcorr_whole_*``).

Overview table (input of step 03): per unit and task the mean and SD (ddof = 1,
empty for a single hold) of the lag-0 xcorr over the unit's valid holds
(active, finite xcorr) and the lag-0 xcorr over the whole gesture.

Input / Output
--------------
    in : per recording day the recording-day pickle and the tracking .mat
         (paths in sessions.local.json)
    out: data/<S>_xcorr_per_hold_tracked.csv   (one row per unit and hold)
         data/<S>_xcorr_per_mu_tracked.csv     (one row per unit and task)
         data/<S>_overview.csv                 (the overview table)

The recordings are not distributed with this repository (identifiable human
EMG of a vulnerable participant group); they are available from the authors on
reasonable request.

Dependencies
------------
    numpy, scipy

Usage
-----
    uv run python figure-three/01_xcorr_trigger.py

Author:  Pauline Wittermann (pauline.wittermann@fau.de) and Dominik I. Braun (dome.braun@fau.de)
"""

import os

import _common as C
import _xcorr as X

C.set_level("WARNING")

# Largest shift of the rate against the trigger, in seconds, both directions
# (MATLAB's maxlag). Positive lag = the rate trails the trigger.
XCORR_MAX_LAG = 1.5
# Fewest discharges of a unit inside a hold for the hold to count as active.
MIN_DISCHARGES_PER_RAMP = 5
# Hann window of the discharge rate, in seconds -- the RATE_WINDOW of the
# tracking rate panel (Figure 2), so the rate correlated here is the one drawn
# there.
XCORR_RATE_WINDOW = 1.5
GERMAN_EXCEL = True
SOURCE = "tracked"


def run_session(session, pkl, mat, out):
    recording = C.load_recording_day(pkl)
    # blocks_mat=None: the task boundaries come from the pkl.
    tracking = C.load_tracking(mat, blocks_mat=None)

    hold_all, mu_all = [], []
    for task_key, task in C.tracked_tasks(recording, tracking):
        hold_rows, mu_rows = X.compute_task_hold_xcorr(
            task, task_name=task_key, rate_window=XCORR_RATE_WINDOW,
            max_lag=XCORR_MAX_LAG,
            min_discharges_per_hold=MIN_DISCHARGES_PER_RAMP)
        # Pseudonym and visit from the session key, never the recording's
        # own date.
        for row in hold_rows + mu_rows:
            row.update({"patient_id": session[:3],
                        "visit": session[4:],
                        "max_lag_s": XCORR_MAX_LAG,
                        "min_discharges_per_ramp": MIN_DISCHARGES_PER_RAMP,
                        "rate_window_s": XCORR_RATE_WINDOW,
                        "source": SOURCE})
        hold_all += hold_rows
        mu_all += mu_rows

    C.write_csv(hold_all, os.path.join(out, f"{session}_xcorr_per_hold_tracked.csv"),
                columns=X.COLUMNS_XCORR_HOLD, german_excel=GERMAN_EXCEL)
    C.write_csv(mu_all, os.path.join(out, f"{session}_xcorr_per_mu_tracked.csv"),
                columns=X.COLUMNS_XCORR_MU, german_excel=GERMAN_EXCEL)
    overview = X.build_overview(hold_all, mu_all)
    C.write_csv(overview, os.path.join(out, f"{session}_overview.csv"),
                columns=X.COLUMNS_OVERVIEW, german_excel=GERMAN_EXCEL)
    print(f"{session}: {len(mu_all)} unit x task rows, "
          f"{len(hold_all)} hold rows")


def main():
    cfg = C.load_config()
    out = C.data_dir()
    for session in C.SESSIONS:
        pkl, mat, _ = C.session_files(cfg, session)
        run_session(session, pkl, mat, out)
    print(f"Wrote the trigger xcorr tables to {out}")


if __name__ == "__main__":
    main()
