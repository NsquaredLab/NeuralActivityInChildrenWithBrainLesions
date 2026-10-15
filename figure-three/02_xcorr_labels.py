"""
02_xcorr_labels.py
==================

Step 2 of the Figure 3 pipeline: cross-correlation of the tracked motor units
with the hold labels instead of the trigger.

Per recording day, task and tracked motor unit: Pearson r at lag 0
(``xcorr_coeff``, value at lag 0) of the 1.5 s Hann discharge rate with a
**box cue built from the labels** (1 inside a segment, 0 outside), over the
**whole task** (rest included) -- the same computation as ``xcorr_whole_lag0``
in step 01, only the cue differs.

Label segments used: ``quality == "good"`` and ``gesture`` matching the task
(``Power Grasp`` -> ``power_grasp`` ...). A task without such a segment gets
no label-r (left empty, ``n_segments_used = 0``).

The two r's are kept strictly apart:

    LABELS_r_whole_lag0    rate vs. label box cue
    TRIGGER_r_whole_lag0   rate vs. recorded trigger (cross-check against
                           <S>_xcorr_per_mu_tracked.csv of step 01)

Per label segment (hold-wise): the xcorr runs over the segment's cycle window
(middle of the pause before to middle of the pause after, see
``cycle_windows``); ``active`` counts the discharges inside the segment.

Input / Output
--------------
    in : per recording day the recording-day pickle, the tracking .mat and the
         hold labels (paths in sessions.local.json)
    out: data/mu_corr_labels_values.csv            (per day, task and MU)
         data/mu_corr_labels_segments_values.csv   (per label segment, used or not)
         data/labels_xcorr_per_mu.csv, data/labels_xcorr_per_hold.csv,
         data/labels_overview.csv                  (the tables of step 01, cue =
                                                    labels, all days in one file;
                                                    input of step 04)

The recordings are not distributed with this repository (identifiable human
EMG of a vulnerable participant group); they are available from the authors on
reasonable request.

Dependencies
------------
    numpy, scipy, pandas

Usage
-----
    uv run python figure-three/02_xcorr_labels.py

Author:  Pauline Wittermann (pauline.wittermann@fau.de) and Dominik I. Braun (dome.braun@fau.de)
"""

import os

import numpy as np
import pandas as pd

import _common as C
import _xcorr as X

C.set_level("WARNING")

RATE_WINDOW = 1.5        # Hann window of the discharge rate (s)
MAX_LAG = 1.5            # xcorr maxlag (s); lag 0 is read from it
MIN_DISCHARGES_PER_RAMP = 5   # discharges in a segment for it to be active
QUALITY = "good"         # only these label segments enter the cue
SOURCE = "tracked"
CUE = "labels_auto_v2"   # extra column of the overview-format tables


# ---------------------------------------------------------------- labels -> box cue
def label_cue(label_task, task_key, n_samples, session):
    """Box cue (0/1) from the good, gesture-matching segments + segment rows."""
    if int(label_task["n_samples"]) != n_samples:
        raise ValueError(f"{session} {task_key}: labels have "
                         f"{label_task['n_samples']} samples, pkl {n_samples}")
    g = C.task_gesture(task_key)
    cue = np.zeros(n_samples)
    rows = []
    for k, s in enumerate(label_task["segments"], start=1):
        used = s["quality"] == QUALITY and s["gesture"] == g
        a, b = int(s["start_sample"]), int(s["end_sample"])
        if not 0 <= a < b <= n_samples:
            raise ValueError(f"{session} {task_key} segment {k}: "
                             f"{a}-{b} outside 0-{n_samples}")
        if used:
            cue[a:b] = 1.0
        rows.append(dict(session=session, task=task_key, segment=k,
                         start_s=s["start_s"], end_s=s["end_s"],
                         start_sample=a, end_sample=b, gesture=s["gesture"],
                         quality=s["quality"], source=s.get("source"),
                         used=used))
    return cue, rows


def cycle_windows(segments, n_samples):
    """
    One window per used segment: from the middle of the pause before it to
    the middle of the pause after it (task start / end for the first / last
    segment). Inside a segment the box cue is constant, so r is only defined
    over a window that contains its edges.
    """
    segments = sorted(segments)
    out = []
    for k, (a, b) in enumerate(segments):
        start = 0 if k == 0 else (segments[k - 1][1] + a) // 2
        end = n_samples if k == len(segments) - 1 else (b + segments[k + 1][0]) // 2
        out.append((start, end))
    return out


def hold_xcorr_cycles(task, cue, segments, task_key):
    """
    Per MU and segment, as ``compute_task_hold_xcorr`` -- but the
    correlation runs over the cycle window, and ``active`` / ``n_discharges``
    count the discharges inside the **segment** itself (>= MIN_DISCHARGES).
    Returns hold rows and, per MU, the means over the active holds.
    """
    fsamp = float(task["fsamp"])
    n = cue.size
    segments = sorted(segments)
    windows = cycle_windows(segments, n)
    pnr = task.get("PNR")
    mu_pulses = list(task["MUPulses"])
    pnr = (np.full(len(mu_pulses), np.nan) if pnr is None
           else np.atleast_1d(np.asarray(pnr, dtype=float)).ravel())
    hold_rows, means = [], {}
    for i, pulses in enumerate(mu_pulses):
        pulses = np.asarray(pulses, dtype=int).ravel()
        rate = X.hann_discharge_rate(pulses, n, fsamp, window=RATE_WINDOW,
                                     mode="same")
        unit_pnr = float(pnr[0] if pnr.size == 1 else pnr[i])
        active = []
        for h, ((a, b), (ws, we)) in enumerate(zip(segments, windows)):
            n_dis = int(np.count_nonzero((pulses >= a) & (pulses < b)))
            is_active = n_dis >= MIN_DISCHARGES_PER_RAMP
            peak = lag_s = lag0 = float("nan")
            if is_active:
                lags_s, c = X.xcorr_coeff(rate[ws:we], cue[ws:we], fsamp,
                                          max_lag=MAX_LAG)
                if np.any(np.isfinite(c)):
                    best = int(np.nanargmax(c))
                    peak, lag_s = float(c[best]), float(lags_s[best])
                    lag0 = float(c[lags_s.size // 2])
                    active.append((peak, lag_s, lag0))
            hold_rows.append(dict(
                task=task_key, mu=i + 1, pnr=unit_pnr, hold=h + 1,
                n_discharges=n_dis, active=is_active, xcorr_peak=peak,
                xcorr_lag_s=lag_s, xcorr_lag0=lag0,
                segment_start_s=a / fsamp, segment_end_s=b / fsamp,
                window_start_s=ws / fsamp, window_end_s=we / fsamp))
        m = np.mean(active, axis=0) if active else [float("nan")] * 3
        means[i + 1] = dict(n_holds=len(segments), n_holds_active=len(active),
                            xcorr_peak_mean=float(m[0]),
                            xcorr_lag_mean_s=float(m[1]),
                            xcorr_lag0_mean=float(m[2]))
    return hold_rows, means


def r_lag0(rate, cue, fsamp):
    """Pearson r at lag 0 (xcorr_coeff, centre value); nan if one is flat."""
    lags, c = X.xcorr_coeff(rate, cue, fsamp, max_lag=MAX_LAG)
    return float(c[lags.size // 2])


# ---------------------------------------------------------------- per recording day
def run_session(session, pkl, mat, labels_json):
    print(f"\n--- {session} ---")
    recording = C.load_recording_day(pkl)
    tracking = C.load_tracking(mat, blocks_mat=None)
    labels = C.read_labels(labels_json)

    mu_rows, seg_rows, x_hold, x_mu = [], [], [], []
    for task_key, task in C.tracked_tasks(recording, tracking):
        fsamp = float(task["fsamp"])
        trigger = np.asarray(task["trigger"], dtype=float).ravel()
        n = trigger.size
        if task_key not in labels:
            print(f"  {task_key}: no labels -> skipped")
            continue
        cue, rows = label_cue(labels[task_key], task_key, n, session)
        seg_rows.extend(rows)
        n_used = sum(r["used"] for r in rows)
        print(f"  {task_key}: {n_used}/{len(rows)} segments used, "
              f"cue {cue.sum() / fsamp:.1f} s of {n / fsamp:.1f} s")

        # Whole task: the trigger pipeline itself, label box as trigger.
        _, m_rows = X.compute_task_hold_xcorr(
            {**task, "labels": cue, "trigger": cue}, task_name=task_key,
            rate_window=RATE_WINDOW, max_lag=MAX_LAG,
            min_discharges_per_hold=MIN_DISCHARGES_PER_RAMP)
        # Per segment: over its cycle window (see cycle_windows).
        segments = [(r["start_sample"], r["end_sample"]) for r in rows
                    if r["used"]]
        h_rows, means = hold_xcorr_cycles(task, cue, segments, task_key)
        for r in m_rows:
            r.update(means[r["mu"]])
        for r in h_rows + m_rows:
            r.update(patient_id=session[:3], visit=session[4:],
                     max_lag_s=MAX_LAG,
                     min_discharges_per_ramp=MIN_DISCHARGES_PER_RAMP,
                     rate_window_s=RATE_WINDOW, source=SOURCE, cue=CUE)
        x_hold += h_rows
        x_mu += m_rows

        for i, pulses in enumerate(task["MUPulses"]):
            pulses = np.asarray(pulses, dtype=int).ravel()
            rate = X.hann_discharge_rate(pulses, n, fsamp, window=RATE_WINDOW,
                                         mode="same")
            mu_rows.append(dict(
                session=session, patient_id=session[:3], visit=session[4:],
                task=task_key, gesture=task_key.split("_")[0], mu=i + 1,
                n_discharges=int(pulses.size),
                n_segments_used=n_used, cue_on_s=cue.sum() / fsamp,
                task_s=n / fsamp,
                LABELS_r_whole_lag0=(r_lag0(rate, cue, fsamp) if n_used
                                     else np.nan),
                TRIGGER_r_whole_lag0=r_lag0(rate, trigger, fsamp),
                rate_window_s=RATE_WINDOW))
    return mu_rows, seg_rows, x_hold, x_mu


def main():
    cfg = C.load_config()
    out = C.data_dir()
    out_stem = os.path.join(out, "mu_corr_labels")
    out_xcorr_mu = os.path.join(out, "labels_xcorr_per_mu.csv")
    out_xcorr_hold = os.path.join(out, "labels_xcorr_per_hold.csv")
    out_overview = os.path.join(out, "labels_overview.csv")

    mu_rows, seg_rows, x_hold, x_mu, overview = [], [], [], [], []
    for session in C.SESSIONS:
        paths = C.session_files(cfg, session)
        m, s, h, x = run_session(session, *paths)
        mu_rows += m
        seg_rows += s
        x_hold += h
        x_mu += x
        # per day: build_overview keys on (task, mu) only, and task names
        # repeat across days
        overview += X.build_overview(h, x)
    for r in overview:
        r["cue"] = CUE
    C.write_csv(x_mu, out_xcorr_mu, columns=X.COLUMNS_XCORR_MU + ["cue"])
    C.write_csv(x_hold, out_xcorr_hold, columns=X.COLUMNS_XCORR_HOLD + [
        "cue", "segment_start_s", "segment_end_s", "window_start_s",
        "window_end_s"])
    C.write_csv(overview, out_overview, columns=X.COLUMNS_OVERVIEW + ["cue"])

    mu = pd.DataFrame(mu_rows)
    seg = pd.DataFrame(seg_rows)
    kw = dict(sep=";", decimal=",", index=False, encoding="utf-8-sig")
    mu.to_csv(f"{out_stem}_values.csv", float_format="%.5f", **kw)
    seg.to_csv(f"{out_stem}_segments_values.csv", float_format="%.4f", **kw)

    pd.set_option("display.width", 200)
    print("\n=== LABELS: r(lag 0), rate vs. label box cue, whole task ===")
    print(mu.pivot_table(index=["session", "mu"], columns="gesture",
                         values="LABELS_r_whole_lag0").round(2))
    print(f"\nWrote mu_corr_labels_values.csv, mu_corr_labels_segments_values.csv, "
          f"{os.path.basename(out_xcorr_mu)}, {os.path.basename(out_xcorr_hold)}, "
          f"{os.path.basename(out_overview)} to {out}")


if __name__ == "__main__":
    main()
