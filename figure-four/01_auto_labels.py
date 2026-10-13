"""
01_auto_labels.py
=================

Step 1 of the Figure 4 pipeline: rule-based labels of every attempted gesture
("automatic labels v2"). No segment is drawn, moved or edited by hand; the only
per-session decisions are the ones listed in ``_common.py`` (analysed task
recordings, excluded tasks, the P02 cue correction).

Signal
------
Per task recording, the channel-averaged RMS envelope of the session's analysed
channels (``_common.session_channels``), computed with the filters and RMS
window of ``_emg`` (2nd-order Butterworth band-pass 20-500 Hz and band-stop
45-55 Hz, 360-sample window, 64-sample stride), smoothed with a SMOOTH s moving
average. The cue is the stored kinematic target trace of the recording.

Rule (per task recording: one instructed gesture, five cues)
------------------------------------------------------------
    cues         closing onset c (cue rises above 5 %), opening-ramp start op
                 (last sample at >= 95 % of the cue maximum), opening-ramp end
                 (cue back below 5 %). A cue already running at t = 0 (recording
                 started late) is kept as cue 1; its onset is reconstructed from
                 its opening-ramp start and the median closing + plateau duration
                 of the other cues.
    rest level   R = median of the envelope in the quiet phases: cue < 5 %, not
                 within QUIET_GAP s after an opening ramp.
    candidates   per cue, the search starts at max(c - ANT, end of the previous
                 opening ramp, end of the previous segment). Activity still
                 running there (above K_ON x R) is carry-over and skipped. A rise
                 needs >= HOLD s above K_ON x R; its onset is moved back by at most
                 BACK s while above K_OFF x R. The segment runs while above
                 K_OFF x R (dips shorter than GAP s bridged), at the latest until
                 the next closing cue. Onsets after op do not start a segment.
    trimming     inside the cue [c, op] everything above threshold stays. Before c
                 and after op the segment keeps only the part above
                 max(K_OFF x R, R + FRAC x (peak - R)), peak = maximum before op.
    selection    a segment must overlap [c, op] and last >= MIN_DUR s. The one
                 with the largest area above R is the attempt; further ones are
                 kept with >= MIN_SECOND of its area.
    label        the instructed gesture of the recording (quality "good");
                 segments of an excluded task get quality "reject".

Input / Output
--------------
    in : recording-day pickles, rest recordings and the raw P02 pinch recording
         (paths in sessions.local.json)
    out: data/labels/<S>.labels_auto_v2.json   (one per session, emg_labeler/1 layout; per task
                                                also the cue phases and the rest level, per
                                                segment its cue and peak envelope / rest level)
         data/labels/<S>_envelopes.npz         (the smoothed envelope of every task recording,
                                                frame times te_<i> and values sm_<i> for
                                                tasks[i], and the cue trace the rule used,
                                                times tt_<i> and values cue_<i>; used by step 06)

The recordings are not distributed with this repository (identifiable human
EMG of a vulnerable participant group); they are available from the authors on
reasonable request.

Dependencies
------------
    numpy, scipy

Usage
-----
    uv run python figure-four/01_auto_labels.py

Author:  Pauline Wittermann (pauline.wittermann@fau.de) and Dominik I. Braun (dome.braun@fau.de)
"""

import json
import os

import numpy as np

import _common as C
import _emg as E

SMOOTH = 0.25       # s, moving average of the envelope
RP = 50             # percentile of the quiet phases used as rest level
K_ON, K_OFF = 1.5, 1.4
HOLD = 0.25         # s above K_ON to start a segment
GAP = 0.75          # s, dips shorter than this are bridged
MIN_DUR = 0.5       # s
ANT = 5.0           # s, onsets up to this long before the closing cue count (early attempts)
BACK = 1.0          # s, the onset may be moved back at most this far to the off-threshold crossing
FRAC = 0.3          # trim level outside the cue: R + FRAC x (peak - R)
MIN_SECOND = 0.25   # further segments of a cue need this fraction of the main segment's area
QUIET_GAP = 1.5     # s after an opening ramp before the quiet phase starts (opening bursts)
CUE_THR = 0.05      # cue level that marks a cue as on
CUE_RATE_HZ = 200   # the cue phases are read from the cue decimated to about this rate

# P02 cue correction: block length and tolerance of the sample matching, and the
# sampling rate of the kinematic target trace in the raw recording
MATCH_W = 256
MATCH_TOL = 1e-2
GT_RATE_HZ = 60.0

PARAMS = dict(SMOOTH=SMOOTH, RP=RP, K_ON=K_ON, K_OFF=K_OFF, HOLD=HOLD, GAP=GAP, MIN_DUR=MIN_DUR, ANT=ANT,
              BACK=BACK, FRAC=FRAC, MIN_SECOND=MIN_SECOND, QUIET_GAP=QUIET_GAP, CUE_THR=CUE_THR)


# ---------------------------------------------------------------- cue correction (P02 pinch)
def corrected_cue(td: dict, raw: dict, fs: int, raw_channel: int) -> np.ndarray:
    """Cue of a task recording, rebuilt from its raw recording.

    The recording-day file holds the task as concatenated pieces (1 s of the raw
    recording missing at each joint), while its stored cue is the complete raw
    cue resampled onto the shorter recording (up to ~1.2 s off). Every block of
    MATCH_W samples of the first EMG channel is located in the raw recording
    (``raw_channel``: the same electrode there) by an exact, mean-free match,
    which maps each sample to its raw sample; the raw kinematic target trace
    (mean over its degrees of freedom, min-max normalised, GT_RATE_HZ) is then
    read at those raw times."""
    u = td["emg"][0].astype(np.float64)
    rr = E.flatten_biosignal(raw["biosignal"])[raw_channel].astype(np.float64)

    def match(i, guess):
        a = u[i:i + MATCH_W] - u[i:i + MATCH_W].mean()
        for d in range(0, 3 * fs):
            for k in (guess + d, guess - d):
                if 0 <= k <= len(rr) - MATCH_W:
                    b = rr[k:k + MATCH_W] - rr[k:k + MATCH_W].mean()
                    if np.abs(a - b).max() < MATCH_TOL:
                        return k
        return None

    mapping, shift, cuts = [], 0, []
    for i in range(0, len(u) - MATCH_W, MATCH_W):
        k = match(i, i + shift)
        if k is None:
            continue
        if k - i != shift:
            cuts.append((i, k - i - shift))
            shift = k - i
        mapping.append((i, k))
    print("  cue correction: joints at " + ", ".join(f"{i / fs:.2f} s ({n / fs:.3f} s missing)" for i, n in cuts))
    mi, mk = np.array(mapping).T
    raw_idx = np.interp(np.arange(len(u)), mi, mk)          # piecewise exact, linear only inside blocks
    gt = np.asarray(raw["ground_truth"]).mean(0)
    g = (gt - gt.min()) / (gt.max() - gt.min())
    return np.interp(raw_idx / fs, np.arange(len(g)) / GT_RATE_HZ, g)


# ---------------------------------------------------------------- rule
def task_signals(td: dict, kept: np.ndarray, trig: np.ndarray) -> dict:
    """Envelope and cue of one task recording, as used by the rule."""
    fs = int(td["fsamp"])
    te, env = C.mean_rms_envelope(td["emg"], kept, fs)
    b = (trig > CUE_THR).astype(int)
    ds = max(1, fs // CUE_RATE_HZ)
    return dict(te=te, env=env, tt=np.arange(0, len(trig), ds) / fs, trig=trig[::ds],
                up=np.where(np.diff(b) == 1)[0] / fs, down=np.where(np.diff(b) == -1)[0] / fs, fs=fs)


def cue_phases(d):
    """[(c, op, end, partial)] per cue: closing onset, opening-ramp start, opening-ramp end"""
    tt, trig = d["tt"], d["trig"]

    def phase(c, end):
        m = (tt >= max(c, 0)) & (tt <= end)
        seg = trig[m]
        hi = np.where(seg >= 0.95 * seg.max())[0] if seg.size else []
        return tt[m][hi[-1]] if len(hi) else end

    out = [(c, phase(c, e), e, False) for c in d["up"]
           for e in [min([x for x in d["down"] if x > c], default=tt[-1])]]
    if trig[0] > CUE_THR:      # recording started inside a cue
        e0 = min([x for x in d["down"]], default=tt[-1])
        op0 = phase(0.0, e0)
        rise = float(np.median([op - c for c, op, _, _ in out])) if out else 5.0
        out.insert(0, (op0 - rise, op0, e0, True))
    return out


def rest_level(d, sm, phases):
    te = d["te"]
    trig = np.interp(te, d["tt"], d["trig"])
    quiet = trig < CUE_THR
    for _, _, end, _ in phases:
        quiet &= ~((te >= end) & (te < end + QUIET_GAP))
    return float(np.percentile(sm[quiet], RP)), quiet


def bridged_run(mask, k, lo, hi, ng):
    """[a, b] around index k of the True run in mask[lo:hi], bridging False gaps shorter than ng"""
    a = b = k
    while a > lo:
        j = a - 1
        while j >= lo and not mask[j]:
            j -= 1
        if j >= lo and (a - 1 - j) < ng:
            a = j
        else:
            break
    while b < hi - 1:
        j = b + 1
        while j < hi and not mask[j]:
            j += 1
        if j < hi and (j - b - 1) < ng:
            b = j
        else:
            break
    return a, b


def label_task(d):
    """Attempt segments [(start_s, end_s, cue index)] of one task recording."""
    te, env = d["te"], d["env"]
    dt = float(np.median(np.diff(te)))
    k = max(1, int(round(SMOOTH / dt)))
    sm = np.convolve(env, np.ones(k) / k, mode="same")
    phases = cue_phases(d)
    R, _ = rest_level(d, sm, phases)
    on_thr, off_thr = K_ON * R, K_OFF * R
    nh, ng = max(1, int(round(HOLD / dt))), max(1, int(round(GAP / dt)))
    hi, lo_ = sm > on_thr, sm > off_thr
    idx = lambda t: int(np.clip(np.searchsorted(te, t), 0, len(te) - 1))  # noqa: E731
    out = []
    busy_until = 0
    for ci, (c, op, end, _) in enumerate(phases):
        prev_end = phases[ci - 1][2] if ci > 0 else 0.0
        nxt = phases[ci + 1][0] if ci + 1 < len(phases) else te[-1]
        i0 = i = max(idx(max(c - ANT, prev_end)), busy_until)
        while i < len(te) and hi[i]:         # carry-over of the previous attempt
            i += 1
        i_op, i_cap, i_c = idx(op), idx(nxt), idx(c)
        cands = []
        while i <= i_op:
            if not (hi[i] and hi[i:i + nh].all()):
                i += 1
                continue
            a = i
            while a > i0 and lo_[a - 1] and te[i] - te[a - 1] <= BACK:
                a -= 1
            _, b = bridged_run(lo_, i, a, i_cap + 1, ng)
            # trimming outside the cue, relative to the attempt's peak (before the opening ramp)
            pk = a + int(np.argmax(sm[a:max(a + 1, min(b + 1, i_op + 1))]))
            lvl = max(off_thr, R + FRAC * (sm[pk] - R))
            ta, tb = bridged_run(sm >= lvl, pk, a, b + 1, ng)
            a2 = a if a >= i_c else min(max(ta, a), i_c)       # before the cue: trimmed
            b2 = b if b <= i_op else max(min(tb, b), i_op)      # after the opening-ramp start: trimmed
            a2, b2 = max(a2, a), min(b2, b)
            if te[b2] > c and te[a2] < op and te[b2] - te[a2] >= MIN_DUR:   # must overlap the cue itself
                cands.append((float(np.sum(sm[a2:b2 + 1] - R) * dt), a2, b2))
            busy_until = max(busy_until, b)
            i = b + 1
        if cands:
            main = max(x[0] for x in cands)
            for area, a2, b2 in cands:
                if area >= MIN_SECOND * main:
                    out.append((float(te[a2]), float(te[b2]), ci))
    return out, dict(R=R, on=on_thr, off=off_thr, sm=sm, phases=phases)


# ---------------------------------------------------------------- per session
def label_session(cfg: dict, session: str) -> tuple[dict, dict]:
    """Labels of one session, and its smoothed envelopes (for step 06)."""
    day = C.load_pickle(C.session_path(cfg, session, "recording"))
    rest_raw = C.load_pickle(C.session_path(cfg, session, "rest_recording"))
    kept, bad = C.session_channels(day, rest_raw, session)
    del rest_raw
    excluded = C.EXCLUDED_TASKS.get(session, {})
    out = dict(schema="emg_labeler/1", session=session, labeler="auto_rule_v2", rule_parameters=PARAMS,
               channels_kept_1idx=[int(c) + 1 for c in kept], tasks={})
    envelopes = {"tasks": np.array(C.analysed_tasks(day, session))}
    print(f"\n=== {session}: {len(kept)} channels analysed ({len(bad)} bad) ===", flush=True)
    for i_task, t in enumerate(C.analysed_tasks(day, session)):
        td = day["tasks"][t]
        fs = int(td["fsamp"])
        if C.CUE_CORRECTION.get(session) == t:
            raw = C.load_pickle(C.session_path(cfg, session, "raw_task_recordings", t))
            trig = corrected_cue(td, raw, fs, raw_channel=C.BLOCKS[C.REST_BLOCK[session]].start)
            del raw
        else:
            trig = np.asarray(td["trigger"], dtype=float)
        d = task_signals(td, kept, trig)
        segs, info = label_task(d)
        gesture = t.split("_")[0]
        quality, why = ("reject", excluded[t]) if t in excluded else ("good", "")
        sm, te = info["sm"], d["te"]
        envelopes[f"te_{i_task}"], envelopes[f"sm_{i_task}"] = te, sm
        envelopes[f"tt_{i_task}"], envelopes[f"cue_{i_task}"] = d["tt"], d["trig"]
        out["tasks"][t] = dict(
            fs=float(fs), n_samples=int(td["emg"].shape[1]), n_channels=int(td["emg"].shape[0]),
            duration_s=td["emg"].shape[1] / fs, gesture_hint=C.GESTURE_TO_JSON[gesture],
            rest_level=info["R"], cue_corrected=C.CUE_CORRECTION.get(session) == t,
            cues=[dict(cue=ci + 1, onset_s=float(c), opening_start_s=float(op), opening_end_s=float(e),
                       started_before_recording=bool(p)) for ci, (c, op, e, p) in enumerate(info["phases"])],
            segments=[dict(start_s=s, end_s=e, start_sample=int(round(s * fs)), end_sample=int(round(e * fs)),
                           gesture=C.GESTURE_TO_JSON[gesture], quality=quality, source="auto_rule_v2",
                           note=f"cue {ci + 1}" + (f"; {why}" if why else ""), cue=ci + 1,
                           peak_rel_rest=float(sm[(te >= s) & (te <= e)].max() / info["R"]))
                      for s, e, ci in segs])
        part = " (cue 1 started before the recording)" if info["phases"][0][3] else ""
        print(f"{t:28s} R={info['R']:5.1f} on={info['on']:5.1f} off={info['off']:5.1f}{part}  "
              + " ".join(f"c{ci + 1}:{s:.1f}-{e:.1f}" for s, e, ci in segs)
              + (f"  [{quality}: {why}]" if why else ""), flush=True)
    return out, envelopes


def main():
    cfg = C.load_config()
    os.makedirs(C.LABEL_DIR, exist_ok=True)
    for session in C.SESSIONS:
        labels, envelopes = label_session(cfg, session)
        with open(C.label_path(session), "w", encoding="utf-8") as f:
            json.dump(labels, f, indent=2)
        np.savez_compressed(C.envelope_path(session), **envelopes)
        n = sum(len(t["segments"]) for t in labels["tasks"].values())
        print(f"  {n} segments -> {C.label_path(session)}", flush=True)


if __name__ == "__main__":
    main()
