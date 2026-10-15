"""
_metrics.py
===========

Performance metrics of the exergame myocontrol predictions. All functions work
on the (frame-downsampled) binary prediction stream; cue windows
(``ground_truth``) are ``(start, end)`` pairs in **seconds**.

    - ``compute_latency``                 reaction latency per cue
    - ``compute_accuracy``                sample-wise agreement with the cue (%)
    - ``compute_f1_sample``               sample-based precision / recall / F1
    - ``compute_confusion_matrix_event``  event-based (burst) confusion counts
    - ``compute_recording_metrics``       all of the above for one recording,
                                          as one flat row

Author:  Pauline Wittermann (pauline.wittermann@fau.de) and Dominik I. Braun (dome.braun@fau.de)
"""

import os

import numpy as np

import _common as C

log = C.get_logger(__name__)

# A prediction onset later than this after the cue is not a reaction any more,
# but the next (spontaneous or mistimed) activation -- such cues are discarded.
MAX_LATENCY_SECONDS = 2.0

# Activations shorter than this are noise to the event-based counts.
DEFAULT_MIN_BURST_SECONDS = 0.5


# ---------------------------------------------------------------- latency
def find_onsets(prediction):
    """Frame indices where the binary ``prediction`` rises from 0 to 1."""
    prediction = np.asarray(prediction)
    # prepend=0 makes a leading 1 count as a 0->1 transition.
    diff = np.diff(prediction, prepend=0)
    return np.where(diff == 1)[0]


def compute_latency(ground_truth, prediction, fs, frame,
                    max_latency=MAX_LATENCY_SECONDS):
    """
    Reaction latency between every cue onset and the first prediction onset that
    follows it.

    Parameters
    ----------
    ground_truth : list of (float, float)
        Cue windows ``(start, end)`` in seconds.
    prediction : array_like
        Binary prediction signal (0/1) at control cadence.
    fs : float
        EMG sampling frequency in Hz.
    frame : int
        EMG samples per control frame.
    max_latency : float, optional
        Cut-off in seconds (default :data:`MAX_LATENCY_SECONDS`, 2 s).

    Returns
    -------
    list of float
        One latency (in seconds) per correctly performed cue; shorter than
        ``ground_truth`` when cues were missed. A latency is only kept if it is
        <= ``max_latency``.
    """
    latencies = []
    onset_pred = find_onsets(prediction)
    f = frame / fs  # seconds per control frame

    n_too_late = 0      # reacted, but outside the plausible window
    n_no_onset = 0      # no activation at all after this cue

    for (start, end) in ground_truth:
        gt_start = int(start / f)  # cue onset in frame indices
        following = onset_pred[onset_pred >= gt_start]
        if not len(following):
            n_no_onset += 1
            log.debug("Cue at %.2f s: no prediction onset follows it -- skipped.",
                      start)
            continue

        latency = (following[0] - gt_start) * f              # seconds
        if latency <= max_latency:
            latencies.append(float(latency))
        else:
            n_too_late += 1
            log.debug("Cue at %.2f s: first onset %.2f s later (> %.1f s) -- "
                      "treated as a missed cue.", start, latency, max_latency)

    log.info("Latency: %d/%d cues used (%d too late, %d without any onset)",
             len(latencies), len(ground_truth), n_too_late, n_no_onset)
    if latencies:
        log.debug("Latencies (s): %s", [f"{v:.3f}" for v in latencies])
    else:
        log.warning("No usable reaction latency in this recording -- check that "
                    "the ground-truth cue windows match the recording.")

    return latencies


# ---------------------------------------------------------------- sample-based
def _ground_truth_vector(ground_truth, n_frames, fs, frame):
    """Boolean cue vector at control cadence from ``(start, end)`` windows in seconds."""
    gt_vector = np.zeros(n_frames, dtype=bool)
    for (start, end) in ground_truth:
        s = int(start * fs / frame)
        e = int(end * fs / frame)
        if s < n_frames:
            gt_vector[s:min(e, n_frames)] = True
    return gt_vector


def compute_accuracy(ground_truth, prediction, fs, frame):
    """
    Sample-wise accuracy (%) between the binary prediction and the binarised
    cue: the percentage of frames at which prediction and cue agree.
    """
    prediction = np.asarray(prediction).astype(bool)
    gt_vector = _ground_truth_vector(ground_truth, len(prediction), fs, frame)
    correct = np.sum(gt_vector == prediction)
    return (correct / len(prediction)) * 100.0


def compute_f1_sample(ground_truth, prediction, fs, frame):
    """
    Sample-based precision, recall and F1: every control frame is scored
    against the binarised cue.

    Returns
    -------
    metrics : dict
        ``{"precision", "recall", "f1"}``.
    counts : dict
        ``{"tp", "fp", "fn", "tn"}`` (sample-wise).
    """
    prediction = np.asarray(prediction).astype(bool)
    gt_vector = _ground_truth_vector(ground_truth, len(prediction), fs, frame)

    tp = int(np.sum(prediction & gt_vector))
    fp = int(np.sum(prediction & ~gt_vector))
    fn = int(np.sum(~prediction & gt_vector))
    tn = int(np.sum(~prediction & ~gt_vector))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    metrics = {"precision": precision, "recall": recall, "f1": f1}
    counts = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return metrics, counts


# ---------------------------------------------------------------- event-based
def compute_confusion_matrix_event(ground_truth, prediction, fs, frame,
                                   min_burst_seconds=0.5):
    """
    Event-based confusion-matrix counts and metrics.

    A *burst* is a maximal contiguous run of active frames in the prediction
    stream. Bursts shorter than ``min_burst_seconds`` are treated as noise and
    ignored. Then:

    * **TP** = number of cue windows overlapped by at least one valid burst;
    * **FN** = number of cue windows with no overlapping valid burst;
    * **FP** = number of valid bursts that overlap no cue window;
    * **TN** is undefined for event-based metrics.

    Returns
    -------
    counts : dict
        ``{"tp", "fn", "fp"}``.
    metrics : dict
        ``{"precision", "recall", "f1"}``.
    detected : list of bool
        Per-cue detection flag, one entry per ground-truth window.
    """
    prediction = np.asarray(prediction).astype(bool)
    max_frame = len(prediction)

    gt_frames = [(int(start * fs / frame), int(end * fs / frame))
                 for (start, end) in ground_truth]
    min_burst_samples = round(min_burst_seconds * fs / frame)

    # Ground-truth activation vector (for the false-alarm / overlap check).
    gt_vector = np.zeros(max_frame, dtype=bool)
    for (start, end) in gt_frames:
        if start < max_frame:
            gt_vector[start:min(end, max_frame)] = True

    # Find all bursts (maximal contiguous runs of active frames).
    bursts = []
    i = 0
    while i < max_frame:
        if prediction[i]:
            j = i
            while j < max_frame and prediction[j]:
                j += 1
            bursts.append((i, j))
            i = j
        else:
            i += 1

    # Keep only bursts long enough to count as a genuine activation.
    valid_bursts = [(s, e) for (s, e) in bursts if (e - s) >= min_burst_samples]
    n_short = len(bursts) - len(valid_bursts)

    # TP / FN: a cue window is detected if any valid burst overlaps it.
    detected = []
    for (start, end) in gt_frames:
        end_clipped = min(end, max_frame)
        if start >= max_frame:
            detected.append(False)
            continue
        is_detected = any(bs < end_clipped and be > start
                          for (bs, be) in valid_bursts)
        detected.append(is_detected)
    tp = int(sum(detected))
    fn = int(len(detected) - tp)

    # FP: valid bursts that don't overlap any cue window (false alarms).
    fp = int(sum(1 for (bs, be) in valid_bursts if not np.any(gt_vector[bs:be])))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    log.info("Event-based: TP=%d/%d detected, FN=%d, FP=%d "
             "(%d bursts shorter than %.2f s ignored)",
             tp, len(gt_frames), fn, fp, n_short, min_burst_seconds)

    counts = {"tp": tp, "fn": fn, "fp": fp}
    metrics = {"precision": precision, "recall": recall, "f1": f1}
    return counts, metrics, detected


# ---------------------------------------------------------------- one recording
def compute_recording_metrics(pkl_path, ground_truth=None, name=None,
                              min_burst_seconds=DEFAULT_MIN_BURST_SECONDS,
                              max_latency=MAX_LATENCY_SECONDS):
    """
    Score one exergame recording -- sample-wise, event-wise and by latency.

    ``accuracy`` counts rest frames as correct too, and a recording is mostly
    rest -- so it is high by construction. Read ``event_f1`` for "did the
    control work", ``accuracy`` for "how much of the time was the control in
    the right state".

    Parameters
    ----------
    pkl_path : str
        Path to the ``myocontrol_data*.pkl`` recording.
    ground_truth : list of (float, float) or None, optional
        Cue windows ``(start, end)`` in seconds; ``None`` reads them from the
        recording's ``metronome`` cue track.
    name : str or None, optional
        Label for this recording, copied into the row as ``session``.
    min_burst_seconds : float, optional
        Minimum duration of a valid activation for the event-based counts.
    max_latency : float, optional
        Cues whose first reaction comes later than this are counted as missed.

    Returns
    -------
    dict
        One flat row: identification, ``accuracy``, the sample-based and
        event-based metrics with their counts, the latency summary (SD with
        ddof = 0), the recording's calibration and the settings used.
    """
    if name is None:
        name = os.path.basename(os.path.dirname(pkl_path)) or \
            os.path.splitext(os.path.basename(pkl_path))[0]

    with C.step(log, f"metrics for {name}"):
        rec = C.load_exergame_recording(pkl_path)
        windows = C.resolve_ground_truth(rec, ground_truth,
                                         name=f"ground_truth for {name}")

        prediction = np.asarray(rec.predictions).astype(bool)
        n_frames = int(prediction.size)

        accuracy = compute_accuracy(windows, prediction, C.FS, C.FRAME)
        sample_metrics, sample_counts = compute_f1_sample(
            windows, prediction, C.FS, C.FRAME)
        event_counts, event_metrics, detected = compute_confusion_matrix_event(
            windows, prediction, C.FS, C.FRAME, min_burst_seconds=min_burst_seconds)
        latencies = compute_latency(windows, prediction, C.FS, C.FRAME,
                                    max_latency=max_latency)

    # A recording in which nothing was ever detected has no latency; report that
    # as "no value" rather than as a zero.
    latencies = np.asarray(latencies, dtype=float)
    has_latency = latencies.size > 0

    # One control frame in seconds.
    frame_seconds = C.FRAME / C.FS

    row = {
        "session": name,
        "file": os.path.basename(pkl_path),
        "path": pkl_path,

        # ---- what the recording contains -----------------------------------
        "n_cues": len(windows),
        "n_frames": n_frames,
        "duration_s": n_frames * frame_seconds,
        "active_fraction_gt": float(
            (sample_counts["tp"] + sample_counts["fn"]) / n_frames)
        if n_frames else float("nan"),
        "active_fraction_pred": float(prediction.mean()) if n_frames
        else float("nan"),

        # ---- sample-wise ----------------------------------------------------
        "accuracy": accuracy,
        "sample_precision": sample_metrics["precision"],
        "sample_recall": sample_metrics["recall"],
        "sample_f1": sample_metrics["f1"],
        "sample_tp": sample_counts["tp"],
        "sample_fp": sample_counts["fp"],
        "sample_fn": sample_counts["fn"],
        "sample_tn": sample_counts["tn"],

        # ---- event-wise -----------------------------------------------------
        "n_cues_detected": int(sum(detected)),
        "event_tp": event_counts["tp"],
        "event_fp": event_counts["fp"],
        "event_fn": event_counts["fn"],
        "event_precision": event_metrics["precision"],
        "event_recall": event_metrics["recall"],
        "event_f1": event_metrics["f1"],

        # ---- latency --------------------------------------------------------
        "n_latencies": int(latencies.size),
        "latency_mean_s": float(np.mean(latencies)) if has_latency
        else float("nan"),
        "latency_sd_s": float(np.std(latencies)) if has_latency
        else float("nan"),
        "latency_median_s": float(np.median(latencies)) if has_latency
        else float("nan"),
        "latency_min_s": float(np.min(latencies)) if has_latency
        else float("nan"),
        "latency_max_s": float(np.max(latencies)) if has_latency
        else float("nan"),

        # ---- calibration (nan for recordings that store none) ----------------
        "threshold": rec.threshold_value,

        # ---- settings the numbers were produced with -------------------------
        "min_burst_s": float(min_burst_seconds),
        "max_latency_s": float(max_latency),
        "fs": float(C.FS),
        "frame": int(C.FRAME),
    }

    log.info("%s: accuracy %.1f%%, sample F1 %.3f, event F1 %.3f "
             "(%d/%d cues), latency %s",
             name, row["accuracy"], row["sample_f1"], row["event_f1"],
             row["n_cues_detected"], row["n_cues"],
             f"{row['latency_mean_s']:.3f} s +/- {row['latency_sd_s']:.3f} s"
             if has_latency else "--")
    return row
