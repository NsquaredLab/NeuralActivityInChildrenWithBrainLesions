"""
02_dataset.py
=============

Step 2 of the Figure 4 pipeline: build one labelled dataset per session and
compute its RMS features (``_emg.write_rms_features``).

Dataset per session
-------------------
    gestures   every segment of quality "good" from the step-01 labels, cut out
               of the session's task recordings (pickle order, then cue order)
    rest       the separate rest recording of the same session (matching
               forearm block), channel means removed like the gesture
               recordings, REST_TRIM_S dropped at both ends, then a centred
               window as long as the median duration of the session's gesture
               classes (so rest does not dominate the class balance)
    channels   all channels except those discarded in the motor-unit
               decomposition or saturated in any analysed task recording, or
               saturated in the rest recording (``_common.session_channels``)
    features   2nd-order Butterworth band-pass 20-500 Hz + band-stop 45-55 Hz,
               RMS over 360 samples with a 64-sample stride, computed within
               each segment (no window crosses a segment boundary)

Input / Output
--------------
    in : data/labels/<S>.labels_auto_v2.json (step 01), recording-day pickles and
         rest recordings (paths in sessions.local.json)
    out: data/<S>_rms_features.npz   (RMS features, consumed by steps 03, 05 and 07)
         data/<S>_dataset.json       (channels, class durations, rest window)

The labelled raw signal is kept in memory only. None of these files are
distributed with this repository (identifiable human EMG of a vulnerable
participant group); the data are available from the authors on reasonable
request.

Dependencies
------------
    numpy, scipy

Usage
-----
    uv run python figure-four/02_dataset.py

Author:  Pauline Wittermann (pauline.wittermann@fau.de) and Dominik I. Braun (dome.braun@fau.de)
"""

import json
import os

import numpy as np

import _common as C
import _emg as E

REST_TRIM_S = 1.0       # s dropped at both ends of the rest recording
REST_LIMIT = "median"   # rest window = median duration of the gesture classes ("median" or "mean")


def dataset_path(session: str) -> str:
    return os.path.join(C.DATA_DIR, f"{session}_dataset.json")


def build_session(cfg: dict, session: str) -> dict:
    labels = C.load_labels(session)
    day = C.load_pickle(C.session_path(cfg, session, "recording"))
    rest_raw = C.load_pickle(C.session_path(cfg, session, "rest_recording"))
    kept, bad = C.session_channels(day, rest_raw, session)

    segments, seg_labels, seg_sources, fs, n_ch = [], [], [], None, None
    for task in C.analysed_tasks(day, session):
        td = day["tasks"][task]
        fs = fs or int(td["fsamp"])
        assert fs == int(td["fsamp"]), f"{session}: sampling rates differ"
        emg = td["emg"].astype(np.float32)
        n_ch = n_ch or emg.shape[0]
        for i, sg in enumerate(labels["tasks"][task]["segments"]):
            if sg["quality"] != "good":
                continue
            a, b = max(0, int(round(sg["start_s"] * fs))), min(emg.shape[1], int(round(sg["end_s"] * fs)))
            if b > a:
                segments.append(emg[:, a:b].copy())
                seg_labels.append(C.JSON_TO_GESTURE[sg["gesture"]])
                seg_sources.append((task, i))
    del day

    assert int(rest_raw["device_information"]["sampling_frequency"]) == fs, f"{session}: rest sampling rate"
    rest = C.rest_signal(rest_raw, session)
    del rest_raw
    assert rest.shape[0] == n_ch, f"{session}: rest block has {rest.shape[0]} channels, gestures {n_ch}"
    trim = int(REST_TRIM_S * fs)
    rest = rest[:, trim:rest.shape[1] - trim]
    durs = {}
    for seg, g in zip(segments, seg_labels):
        durs[g] = durs.get(g, 0) + seg.shape[1]
    target = int(round((np.median if REST_LIMIT == "median" else np.mean)(list(durs.values()))))
    target = min(target, rest.shape[1])
    a = (rest.shape[1] - target) // 2                        # centred window, fixed rule
    rest = rest[:, a:a + target]
    rest_window_s = [(a + trim) / fs, (a + trim + target) / fs]
    print(f"[{session}] rest limited to the {REST_LIMIT} gesture-class duration: {target / fs:.1f} s, "
          f"window {rest_window_s[0]:.1f}-{rest_window_s[1]:.1f} s of the rest recording", flush=True)
    segments.append(rest)
    seg_labels.append("Rest")
    seg_sources.append(("rest_recording", 0))

    signal = np.concatenate(segments, axis=1)
    ends = np.cumsum([s.shape[1] for s in segments]).astype(np.int64)
    starts = np.concatenate(([0], ends[:-1]))
    del segments, rest
    bad_spec = E.channels_to_spec(bad, E.N_BIOSIGNAL_CHANNELS)
    secs = {g: round(float(((ends - starts)[np.array(seg_labels) == g]).sum() / fs), 1) for g in C.GESTURES}
    nseg = {g: seg_labels.count(g) for g in C.GESTURES}
    print(f"[{session}] {len(seg_labels)} segments, seconds/class {secs}, bad ({len(bad)}): {bad_spec}", flush=True)
    E.write_rms_features(os.path.join(C.DATA_DIR, f"{session}_rms_features.npz"), signal, fs, starts, ends,
                         seg_labels, [s[0] for s in seg_sources], [s[1] for s in seg_sources], bad_spec)
    del signal
    return dict(session=session, n_channels=int(n_ch), n_bad=len(bad), bad_spec=bad_spec,
                kept_channels_1idx=[int(c) + 1 for c in kept], segments_per_class=nseg, seconds_per_class=secs,
                rest_trim_s=REST_TRIM_S, rest_limit=REST_LIMIT, rest_window_s=rest_window_s)


def main():
    cfg = C.load_config()
    os.makedirs(C.DATA_DIR, exist_ok=True)
    for session in C.SESSIONS:
        print(f"\n################ {session} ################", flush=True)
        info = build_session(cfg, session)
        with open(dataset_path(session), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()
