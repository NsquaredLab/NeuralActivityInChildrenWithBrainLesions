"""
06_attempt_summary.py
=====================

Step 6 of the Figure 4 pipeline: how often the children produced a detectable
attempt per cue, and how the activity was coupled to the cue (numbers for the
Results text). Reads only the step-01 outputs; no raw data needed.

A cue is counted when at least half of its closing + hold phase lies within the
recording, measured against the median duration of that phase over the
completely recorded cues of the same recording (this keeps a cue that had
started shortly before the recording began and drops one cut off by the end of
the recording). Per cue:
    detected        at least one attempt segment assigned to the cue
    latency_s       onset of its first segment minus the closing-cue onset
                    (negative = the attempt started before the cue)
    hold_s          duration of the cue's closing + hold phase (closing onset to
                    opening-ramp start)
    coverage        fraction of that phase covered by the cue's segments
    active_s        summed duration of the cue's segments
    n_segments      > 1 means the contraction was interrupted (separate bursts)
    peak_rel_rest   peak of the smoothed envelope within the segments / rest level
Envelope around the cue (smoothed envelope of step 01, R = rest level; K_ON,
K_OFF, HOLD and QUIET_GAP are the rule parameters of step 01):
    closing_peak_rel_rest / closing_peak_time_s
                    peak / R in the closing + hold phase, and its time after the
                    closing-cue onset
    released        the envelope is below K_OFF x R throughout the last RELEASE_S
                    of the hold, i.e. the contraction ended before the opening cue
    opening_peak_rel_rest / opening_peak_time_s
                    peak / R from the opening-ramp start until QUIET_GAP after its
                    end (at the latest the next cue), and its time after the
                    opening-ramp start
    opening_burst   a separate burst in that window: the envelope is below
                    K_ON x R at the opening-ramp start and then rises above it for
                    >= HOLD

Cue timing (the presented close-hold-open cycle), from the cue trace of every
completely recorded cue; the P02 pinch recording is left out because its
timeline has export gaps. Phases as the rule sees them (5 % / 95 % levels):
    closing_s       closing onset (cue > 5 %) to the first sample at
                    >= PLATEAU_LEVEL of the cue maximum
    hold_s          from there to the opening-ramp start
    opening_s       opening-ramp start to cue < 5 %
    cycle_s         closing + hold + opening
    period_s        closing onset to the next closing onset (cue repetition period)
Full phases of the presented cue (amplitude A above the inter-cue baseline):
    rise_s          cue leaves the baseline to the first sample at its maximum
    plateau_s       time at the maximum
    fall_s          last sample at the maximum to the cue back at <= END_LEVEL x A
                    (some cues keep a residue of up to ~1 % of A for up to ~1 s)
    rise_sine_dev / fall_sine_dev
                    largest deviation (fraction of A) of the normalised ramps from
                    a quarter sine wave, sin(pi/2 x) for the rise and cos(pi/2 x)
                    for the fall (x = normalised ramp time)

Input / Output
--------------
    in : data/labels/<S>.labels_auto_v2.json, data/labels/<S>_envelopes.npz (step 01)
    out: results/attempt_summary_cues.csv   (one row per cue)
         results/attempt_summary.csv        (one row per session and task)
         results/attempt_summary_by_task.csv (pooled over sessions)
         results/cue_timing.csv             (phase durations of every complete cue)

Dependencies
------------
    numpy

Usage
-----
    uv run python figure-four/06_attempt_summary.py

Author:  Pauline Wittermann (pauline.wittermann@fau.de) and Dominik I. Braun (dome.braun@fau.de)
"""

import csv
import os

import numpy as np

import _common as C

TASK_ORDER = ["Power Grasp", "Pinch", "Tripod Pinch"]
MIN_RECORDED = 0.5      # a cue counts when at least this fraction of its closing + hold phase was recorded
RELEASE_S = 0.5         # s at the end of the hold checked for a released contraction
PLATEAU_LEVEL = 0.95    # the cue plateau starts where the cue first reaches this fraction of its maximum
END_LEVEL = 0.02        # the full opening ramp ends where the cue is back at this fraction of its amplitude


def runs(mask):
    d = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def envelope_measures(te, sm, R, prm, c, op, end, nxt) -> dict:
    """Closing- and opening-phase measures of one cue (NaN where a window is empty)."""
    dt = float(np.median(np.diff(te)))
    mc = (te >= c) & (te < op)
    mo = (te >= op) & (te < min(end + prm["QUIET_GAP"], nxt))
    out = dict(closing_peak_rel_rest=np.nan, closing_peak_time_s=np.nan, released=False,
               opening_peak_rel_rest=np.nan, opening_peak_time_s=np.nan, opening_burst=False)
    if mc.any():
        out["closing_peak_rel_rest"] = float(sm[mc].max() / R)
        out["closing_peak_time_s"] = float(te[mc][np.argmax(sm[mc])] - c)
        last = np.where(mc)[0][-max(1, int(round(RELEASE_S / dt))):]
        out["released"] = bool(sm[last].max() < prm["K_OFF"] * R)
    if mo.any():
        idx = np.where(mo)[0]
        out["opening_peak_rel_rest"] = float(sm[mo].max() / R)
        out["opening_peak_time_s"] = float(te[mo][np.argmax(sm[mo])] - op)
        above = sm > prm["K_ON"] * R
        nh = max(1, int(round(prm["HOLD"] / dt)))
        out["opening_burst"] = bool(not above[idx[0]] and
                                    any(idx[0] <= a <= idx[-1] and b - a >= nh for a, b in runs(above)))
    return out


def cue_rows(session: str) -> list[dict]:
    lab = C.load_labels(session)
    prm = lab["rule_parameters"]
    env = np.load(C.envelope_path(session))
    rows = []
    for i_task, t in enumerate(env["tasks"]):
        td = lab["tasks"][t]
        te, sm, R = env[f"te_{i_task}"], env[f"sm_{i_task}"], td["rest_level"]
        gesture = t.split("_")[0]
        cues = td["cues"]
        T = td["duration_s"]
        # nominal closing + hold duration: median over the cues recorded completely (start and opening ramp inside)
        nominal = float(np.median([q["opening_start_s"] - q["onset_s"] for q in cues
                                   if q["onset_s"] >= 0 and q["opening_end_s"] < T - 1e-3]))
        for k, cue in enumerate(cues):
            segs = [s for s in td["segments"] if s["cue"] == cue["cue"]]
            c, op, end = cue["onset_s"], cue["opening_start_s"], cue["opening_end_s"]
            nxt = cues[k + 1]["onset_s"] if k + 1 < len(cues) else T
            recorded = (min(c + nominal, T) - max(c, 0.0)) / nominal
            covered = sum(max(0.0, min(s["end_s"], op) - max(s["start_s"], c)) for s in segs)
            rows.append(dict(
                session=session, task=gesture, cue=cue["cue"], counted=recorded >= MIN_RECORDED,
                recorded_fraction=recorded,
                analysed=all(s["quality"] == "good" for s in segs) and t not in C.EXCLUDED_TASKS.get(session, {}),
                detected=bool(segs), n_segments=len(segs),
                latency_s=min(s["start_s"] for s in segs) - c if segs else np.nan,
                hold_s=op - c, coverage=covered / (op - c) if segs else 0.0,
                active_s=sum(s["end_s"] - s["start_s"] for s in segs),
                peak_rel_rest=max(s["peak_rel_rest"] for s in segs) if segs else np.nan,
                **envelope_measures(te, sm, R, prm, c, op, end, nxt)))
    return rows


def summarize(rows: list[dict]) -> dict:
    rows = [r for r in rows if r["counted"]]
    det = [r for r in rows if r["detected"]]
    lat = np.array([r["latency_s"] for r in det])
    return dict(n_cues=len(rows), n_detected=len(det), detection_rate=len(det) / len(rows) if rows else np.nan,
                n_interrupted=sum(r["n_segments"] > 1 for r in det),
                n_anticipatory=int(np.sum(lat < 0)),
                latency_median_s=float(np.median(lat)) if len(det) else np.nan,
                latency_min_s=float(lat.min()) if len(det) else np.nan,
                latency_max_s=float(lat.max()) if len(det) else np.nan,
                coverage_median=float(np.median([r["coverage"] for r in det])) if det else np.nan,
                active_median_s=float(np.median([r["active_s"] for r in det])) if det else np.nan,
                hold_median_s=float(np.median([r["hold_s"] for r in rows])),
                peak_rel_rest_median=float(np.median([r["peak_rel_rest"] for r in det])) if det else np.nan,
                closing_peak_time_median_s=float(np.nanmedian([r["closing_peak_time_s"] for r in rows])),
                closing_peak_rel_rest_median=float(np.nanmedian([r["closing_peak_rel_rest"] for r in rows])),
                n_released=sum(r["released"] for r in rows),
                n_opening_burst=sum(r["opening_burst"] for r in rows),
                n_opening_above_closing=sum(r["opening_peak_rel_rest"] > r["closing_peak_rel_rest"] for r in rows),
                opening_peak_rel_rest_median=float(np.nanmedian([r["opening_peak_rel_rest"] for r in rows])),
                opening_peak_time_median_s=float(np.nanmedian([r["opening_peak_time_s"] for r in rows])))


def cue_timing_rows(session: str) -> list[dict]:
    """Phase durations of every completely recorded cue (recordings without export gaps)."""
    lab = C.load_labels(session)
    env = np.load(C.envelope_path(session))
    rows = []
    for i_task, t in enumerate(env["tasks"]):
        td = lab["tasks"][t]
        if td["cue_corrected"]:
            continue
        tt, cue, T = env[f"tt_{i_task}"], env[f"cue_{i_task}"], td["duration_s"]
        cues = td["cues"]
        for k, q in enumerate(cues):
            c, op, end = q["onset_s"], q["opening_start_s"], q["opening_end_s"]
            if c < 0 or end >= T - 1e-3:
                continue
            m = (tt >= c) & (tt <= end)
            plateau = float(tt[m][np.argmax(cue[m] >= PLATEAU_LEVEL * cue[m].max())])
            nxt = cues[k + 1]["onset_s"] if k + 1 < len(cues) else np.nan
            row = dict(session=session, task=t.split("_")[0], cue=q["cue"], closing_s=plateau - c,
                       hold_s=op - plateau, opening_s=end - op, cycle_s=end - c, period_s=nxt - c)
            # full phases: 1 s around the cue, baseline = cue level in the 0.5 s before it starts moving
            if c < 1.0 or end > T - 1.0:
                row.update(rise_s=np.nan, plateau_s=np.nan, fall_s=np.nan, rise_sine_dev=np.nan, fall_sine_dev=np.nan)
                rows.append(row)
                continue
            w = (tt >= c - 1.0) & (tt <= end + 1.0)
            seg, ts = cue[w], tt[w]
            lo = float(np.median(cue[(tt >= c - 1.0) & (tt < c - 0.5)]))
            amp = seg.max() - lo
            top = np.where(seg >= seg.max() - 1e-4 * amp)[0]
            t0 = ts[np.where(seg[:top[0]] <= lo + 1e-4 * amp)[0][-1]]
            t1, t2 = ts[top[0]], ts[top[-1]]
            t3 = ts[top[-1] + np.argmax(seg[top[-1]:] <= lo + END_LEVEL * amp)]
            r, f = (ts >= t0) & (ts <= t1), (ts >= t2) & (ts <= t3)
            row.update(rise_s=t1 - t0, plateau_s=t2 - t1, fall_s=t3 - t2,
                       rise_sine_dev=float(np.abs((seg[r] - lo) / amp - np.sin(np.pi / 2 * (ts[r] - t0) / (t1 - t0))).max()),
                       fall_sine_dev=float(np.abs((seg[f] - lo) / amp - np.cos(np.pi / 2 * (ts[f] - t2) / (t3 - t2))).max()))
            rows.append(row)
    return rows


def write_csv(path: str, rows: list[dict]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})


def main():
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    cues = [r for s in C.SESSIONS for r in cue_rows(s)]
    per_task = [dict(session=s, task=g, **summarize([r for r in cues if r["session"] == s and r["task"] == g]))
                for s in C.SESSIONS for g in TASK_ORDER if any(r["session"] == s and r["task"] == g for r in cues)]
    by_task = [dict(task=g, n_sessions=len({r["session"] for r in cues if r["task"] == g}),
                    **summarize([r for r in cues if r["task"] == g])) for g in TASK_ORDER]
    write_csv(os.path.join(C.RESULTS_DIR, "attempt_summary_cues.csv"), cues)
    write_csv(os.path.join(C.RESULTS_DIR, "attempt_summary.csv"), per_task)
    write_csv(os.path.join(C.RESULTS_DIR, "attempt_summary_by_task.csv"), by_task)

    def line(r):
        return (f"{r['n_detected']}/{r['n_cues']} cues ({r['detection_rate']:.0%}), interrupted {r['n_interrupted']}, "
                f"before cue {r['n_anticipatory']}, latency median {r['latency_median_s']:+.2f} s "
                f"[{r['latency_min_s']:+.2f}, {r['latency_max_s']:+.2f}], hold covered {r['coverage_median']:.0%}, "
                f"active {r['active_median_s']:.1f} s of {r['hold_median_s']:.1f} s hold, peak {r['peak_rel_rest_median']:.1f} x rest\n"
                f"{'':22s}closing peak {r['closing_peak_rel_rest_median']:.1f} x rest at {r['closing_peak_time_median_s']:.1f} s, "
                f"released before opening {r['n_released']}/{r['n_cues']}, opening burst {r['n_opening_burst']}/{r['n_cues']}, "
                f"opening > closing peak {r['n_opening_above_closing']}/{r['n_cues']}, opening peak "
                f"{r['opening_peak_rel_rest_median']:.1f} x rest at +{r['opening_peak_time_median_s']:.1f} s")
    for r in per_task:
        print(f"{r['session']:7s} {r['task']:13s} " + line(r))
    print()
    for r in by_task:
        print(f"{r['task']:13s} ({r['n_sessions']} sessions) " + line(r))
    for r in cues:
        if r["recorded_fraction"] < 1:
            print(f"{'counted' if r['counted'] else 'not counted'}: {r['session']} {r['task']} cue {r['cue']} "
                  f"({r['recorded_fraction']:.0%} of its closing + hold phase recorded)")

    timing = [r for s in C.SESSIONS for r in cue_timing_rows(s)]
    write_csv(os.path.join(C.RESULTS_DIR, "cue_timing.csv"), timing)
    print(f"\ncue timing ({len(timing)} complete cues): median [min, max]")
    for key in ("closing_s", "hold_s", "opening_s", "cycle_s", "period_s",
                "rise_s", "plateau_s", "fall_s", "rise_sine_dev", "fall_sine_dev"):
        v = np.array([r[key] for r in timing], dtype=float)
        v = v[~np.isnan(v)]
        print(f"  {key:10s} {np.median(v):6.2f} s  [{v.min():.2f}, {v.max():.2f}]  (n = {len(v)})")
    for s in C.SESSIONS:
        v = {k: np.nanmedian([r[k] for r in timing if r["session"] == s]) for k in ("closing_s", "hold_s", "opening_s", "period_s")}
        print(f"  {s:7s} closing {v['closing_s']:.2f}  hold {v['hold_s']:.2f}  opening {v['opening_s']:.2f}  period {v['period_s']:.2f} s")


if __name__ == "__main__":
    main()
