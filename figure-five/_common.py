"""
_common.py
==========

Shared definitions for the Figure 5 pipeline (one run of the bracelet
exergame).

    - ``FS`` / ``FRAME``         EMG sampling rate and the frame size of the
                                 real-time control loop
    - ``EXERGAME_SESSION`` / ``EXERGAME_GROUND_TRUTH``
                                 the analysed run and its cue windows
    - ``load_config`` / ``exergame_file``
                                 the private mapping from the session name to
                                 the data file (``sessions.local.json``)
    - ``ExergameRecording`` / ``load_exergame_recording``
                                 reader of the exergame ``myocontrol_data*.pkl``
    - ``resolve_ground_truth``   cue windows, checked against the recording
    - ``write_csv`` / ``format_table``
                                 result tables (German Excel dialect) and their
                                 console view
    - ``get_logger`` / ``set_level`` / ``step`` / ``log_array``
                                 console logging
    - ``RESULTS_DIR``            output folder (not distributed, see below)

The steps are numbered ``01`` and ``02``. The metrics are in ``_metrics.py``,
the figures in ``_plot.py``.

Data
----
No participant data are distributed with this repository (see the top-level
README). ``sessions.local.json`` (git-ignored) maps the pseudonymised session
name used here to the exergame recording; its layout is shown in
``sessions.example.json``. Everything a step writes goes to ``results/`` next
to the scripts (git-ignored).

Author:  Pauline Wittermann (pauline.wittermann@fau.de) and Dominik I. Braun (dome.braun@fau.de)
"""

import csv
import json
import logging
import math
import os
import pickle
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np

FIG5_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(FIG5_DIR, "results")
CONFIG_PATH = os.path.join(FIG5_DIR, "sessions.local.json")

# EMG sampling frequency of the bracelet.
FS = 2048          # [Hz]  EMG samples per second

# Decimation factor of the real-time myocontrol loop: the control pipeline emits
# exactly one prediction / RMS value per FRAME EMG samples, so the prediction
# stream runs at FS / FRAME Hz (~114 Hz).
FRAME = 18         # [-]   EMG samples per control-loop frame

# One run of the bracelet exergame: P01, power grasp, 2nd run. Older myocontrol
# software: the pkl stores rms / predictions / calibration but no cue track, so
# the cue windows (s) are transcribed from the protocol (2 s activation every
# 7 s, six times, first cue at 0.1 s).
EXERGAME_SESSION = "P01-S1"
EXERGAME_GROUND_TRUTH = [(0.1, 2.1), (7.1, 9.1), (14.1, 16.1),
                         (21.1, 23.1), (28.1, 30.1), (35.1, 37.1)]


# ---------------------------------------------------------------- configuration
def load_config() -> dict:
    """Read ``sessions.local.json`` (data root + exergame recording)."""
    if not os.path.isfile(CONFIG_PATH):
        raise FileNotFoundError(
            f"Missing {CONFIG_PATH}\n"
            "The Figure 5 steps read the participant recording, which is not distributed with this "
            "repository (identifiable EMG of a vulnerable participant group; available from the authors on "
            "reasonable request). Copy sessions.example.json to sessions.local.json and fill in the paths.")
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def exergame_file(cfg: dict, session: str = EXERGAME_SESSION) -> str:
    """The exergame ``myocontrol_data*.pkl`` of ``session``."""
    path = os.path.join(cfg["data_root"], cfg["sessions"][session]["exergame"])
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing exergame recording: {path}")
    return path


def out_dir():
    """RESULTS_DIR, created if missing."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


# ---------------------------------------------------------------- logging
# Root logger of the pipeline; every module logger is a child of it, so one
# set_level() call covers all of them.
ROOT_LOGGER_NAME = "figure5"
_FORMAT = "%(levelname).1s %(short_name)s | %(message)s"


class _ShortNameFilter(logging.Filter):
    """Add ``short_name``: the logger name without the leading root name."""

    def filter(self, record):
        prefix = ROOT_LOGGER_NAME + "."
        record.short_name = (record.name[len(prefix):]
                             if record.name.startswith(prefix) else record.name)
        return True


def _configure():
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        handler.addFilter(_ShortNameFilter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def set_level(level):
    """Set the log level of the whole pipeline (``"DEBUG"`` ... ``"ERROR"``)."""
    _configure().setLevel(logging.getLevelName(level.upper())
                          if isinstance(level, str) else int(level))


def get_logger(name):
    """Logger of one module, ``get_logger(__name__)``."""
    _configure()
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name.lstrip('_')}")


def log_array(log, name, array, level=logging.DEBUG):
    """One-line summary of an array: shape, dtype, value range and NaN count."""
    if not log.isEnabledFor(level):
        return
    arr = np.asarray(array)
    if arr.dtype == object:
        log.log(level, "%s: object array, shape %s, %d entries",
                name, arr.shape, arr.size)
        return
    if arr.size == 0:
        log.log(level, "%s: EMPTY, shape %s, dtype %s", name, arr.shape, arr.dtype)
        return
    n_nan = int(np.count_nonzero(np.isnan(arr))) if arr.dtype.kind == "f" else 0
    finite = arr[np.isfinite(arr)] if arr.dtype.kind == "f" else arr
    span = (f"[{np.min(finite):.4g}, {np.max(finite):.4g}]" if finite.size
            else "[all non-finite]")
    nan_note = f", {n_nan} NaN" if n_nan else ""
    log.log(level, "%s: shape %s, dtype %s, range %s%s",
            name, arr.shape, arr.dtype, span, nan_note)


@contextmanager
def step(log, description, level=logging.DEBUG):
    """Bracket one pipeline step with start / end messages and its duration."""
    log.log(level, "-> %s", description)
    started = time.perf_counter()
    try:
        yield
    except Exception:
        log.error("FAILED after %.2f s: %s", time.perf_counter() - started,
                  description)
        raise
    log.log(level, "   done: %s (%.2f s)", description,
            time.perf_counter() - started)


log = get_logger(__name__)


# ---------------------------------------------------------------- exergame recording
@dataclass
class ExergameRecording:
    """
    Exergame (myocontrol) recording.

    Attributes
    ----------
    emg : ndarray
        Raw EMG stored by the exergame software (channels x samples).
    rms : ndarray, shape (n_frames,)
        RMS envelope of the EMG, one value per control-loop frame.
    predictions : ndarray, shape (n_frames,)
        Binary control signal (0 = rest, 1 = active), one value per frame;
        reconstructed from ``myocontrol`` for recordings that do not store it.
    prediction_threshold : float or None
        Dimensionless scaling factor applied to ``rms_reference`` to obtain the
        absolute activation threshold.
    rms_reference : float or None
        Baseline RMS level measured during calibration.
    prediction_buffer_length : int
        Number of predictions considered for the exergame control signal.
    myocontrol : ndarray, shape (n_frames, n_dof)
        Control signal actually sent to the exergame, one column per degree of
        freedom.
    timings : ndarray
        Per-frame time vector.
    metronome : ndarray or None
        Boolean cue track of the newer exergame recordings (``None`` for the
        older ones, whose cue windows come from the protocol).
    """

    emg: Any
    rms: np.ndarray
    predictions: np.ndarray
    prediction_threshold: float | None = None
    rms_reference: float | None = None
    prediction_buffer_length: Any = None
    myocontrol: Any = None
    timings: Any = None
    metronome: Any = None

    @property
    def threshold(self):
        """Absolute activation threshold ``rms_reference * prediction_threshold``, or ``None``."""
        if self.rms_reference is None or self.prediction_threshold is None:
            return None
        return self.rms_reference * self.prediction_threshold

    @property
    def threshold_value(self) -> float:
        """The activation threshold as a single ``float``, ``nan`` if unknown."""
        threshold = self.threshold
        if threshold is None:
            return float("nan")
        return float(np.ravel(np.asarray(threshold, dtype=float))[0])


def _read_pickle(pkl_path):
    """Unpickle ``pkl_path`` and log what came out."""
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"Recording not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        recording = pickle.load(f)

    log.debug("Read %s (%.1f MB), keys: %s", os.path.basename(pkl_path),
              os.path.getsize(pkl_path) / 1e6,
              sorted(recording) if isinstance(recording, dict) else type(recording))
    log.info("Model threshold: %s", recording.get("prediction_threshold"))
    return recording


def _require(recording, key, pkl_path):
    """Fetch ``key`` from a recording dict, or explain what is actually in the file."""
    if key not in recording:
        raise KeyError(
            f"{os.path.basename(pkl_path)} has no {key!r} field -- is this the "
            f"right kind of recording? Available keys: {sorted(recording)}")
    return recording[key]


def _predictions_from_myocontrol(myocontrol):
    """
    Rebuild the binary prediction stream from the myocontrol command: a frame
    is *active* when any degree of freedom is driven.
    """
    myocontrol = np.atleast_2d(np.asarray(myocontrol, dtype=float))
    return np.any(myocontrol != 0.0, axis=1)


def load_exergame_recording(pkl_path):
    """
    Load an exergame ``myocontrol_data*.pkl`` recording into an
    :class:`ExergameRecording`.

    Older files store ``predictions`` plus the calibration
    (``prediction_threshold`` / ``rms_reference``) and carry no cue track;
    newer files store neither, but do store ``metronome``. The predictions are
    then rebuilt from ``myocontrol`` and the threshold stays ``None``.
    """
    recording = _read_pickle(pkl_path)

    # Predictions: stored directly by the older software, reconstructed from the
    # myocontrol command by the newer one.
    if "predictions" in recording:
        predictions = np.asarray(recording["predictions"])
    else:
        myocontrol = _require(recording, "myocontrol", pkl_path)
        predictions = _predictions_from_myocontrol(myocontrol)
        log.info("%s stores no 'predictions' -- reconstructed %d frames from "
                 "'myocontrol'.", os.path.basename(pkl_path), len(predictions))

    metronome = recording.get("metronome")
    rec = ExergameRecording(
        emg=recording.get("emg"),
        rms=np.asarray(_require(recording, "rms", pkl_path)),
        predictions=predictions,
        prediction_threshold=recording.get("prediction_threshold"),
        rms_reference=recording.get("rms_reference"),
        prediction_buffer_length=recording.get("prediction_buffer_length"),
        myocontrol=recording.get("myocontrol"),
        timings=recording.get("timings"),
        metronome=None if metronome is None else np.asarray(metronome),
    )

    n_active = int(np.count_nonzero(rec.predictions))
    threshold = ("threshold %.4g mV" % rec.threshold_value
                 if rec.threshold is not None else "no threshold recorded")
    log.info("Exergame recording: %d frames, %s, %d active frames (%.1f %%)%s",
             len(rec.rms), threshold, n_active,
             100.0 * n_active / len(rec.predictions) if len(rec.predictions) else 0.0,
             "" if rec.metronome is None else ", cue track included")
    log_array(log, "rms", rec.rms)
    log_array(log, "predictions", rec.predictions)
    return rec


# ---------------------------------------------------------------- cue windows
def check_ground_truth(ground_truth, name="ground_truth"):
    """
    Validate hand-typed cue windows ``[(start, end), ...]`` in seconds and
    raise a message that names the culprit (empty list, not a pair, a window
    ending before it starts, a negative start). Overlaps are only warned about.
    """
    if not ground_truth:
        raise ValueError(f"{name} is empty: pass the cue windows "
                         f"[(start, end), ...] in seconds.")

    for i, window in enumerate(ground_truth):
        if len(window) != 2:
            raise ValueError(f"{name}[{i}] = {window!r} is not a (start, end) pair.")
        start, end = window
        if end <= start:
            raise ValueError(f"{name}[{i}] = ({start}, {end}): the cue ends "
                             f"before it starts.")
        if start < 0:
            raise ValueError(f"{name}[{i}] = ({start}, {end}): negative start time.")

    for i in range(1, len(ground_truth)):
        if ground_truth[i][0] < ground_truth[i - 1][1]:
            log.warning("%s[%d] starts at %.2f s, before %s[%d] ends at %.2f s "
                        "-- overlapping cue windows.", name, i,
                        ground_truth[i][0], name, i - 1, ground_truth[i - 1][1])

    log.debug("%s: %d cue windows spanning %.1f-%.1f s", name, len(ground_truth),
              ground_truth[0][0], ground_truth[-1][1])


def check_covers_recording(ground_truth, n_frames, name="ground_truth"):
    """Warn when the last cue ends after the recording stops."""
    recording_seconds = n_frames * FRAME / FS
    last_cue = ground_truth[-1][1]
    if last_cue > recording_seconds:
        log.warning("%s: the last cue ends at %.1f s but the recording is only "
                    "%.1f s long -- cue windows and recording probably do not "
                    "belong together.", name, last_cue, recording_seconds)


def cue_windows_from_metronome(metronome, fs=FS, name="metronome"):
    """
    Cue windows ``[(start, end), ...]`` in seconds from the boolean
    ``metronome`` track of the newer exergame recordings: every rising edge
    opens a cue, the next falling edge closes it.
    """
    track = np.asarray(metronome).ravel().astype(bool)
    if not track.any():
        raise ValueError(
            f"{name} never goes high: this recording's cue track is empty, so "
            f"the cue windows have to be passed in from the protocol.")

    edges = np.diff(track.astype(np.int8))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1) + 1
    if track[0]:                       # cue already running when recording began
        starts = np.concatenate(([0], starts))
    if track[-1]:                      # cue still running when recording stopped
        ends = np.concatenate((ends, [len(track)]))

    windows = [(float(start / fs), float(end / fs))
               for start, end in zip(starts, ends)]

    lengths = [end - start for start, end in windows]
    log.info("%s: %d cues read from the recording, first at %.2f s, "
             "%.2f s long on average.", name, len(windows), windows[0][0],
             float(np.mean(lengths)))
    return windows


def resolve_ground_truth(recording, ground_truth=None, name="ground_truth"):
    """
    The validated cue windows for one recording: ``ground_truth`` if given,
    otherwise read from the recording's ``metronome`` track. Either way the
    windows go through :func:`check_ground_truth` and
    :func:`check_covers_recording`.
    """
    if ground_truth is None:
        if recording.metronome is None:
            raise ValueError(
                f"{name} is None, but this recording stores no 'metronome' cue "
                f"track -- it was made with the older myocontrol software. Pass "
                f"its cue windows [(start, end), ...] from the protocol.")
        ground_truth = cue_windows_from_metronome(recording.metronome, name=name)

    check_ground_truth(ground_truth, name=name)
    check_covers_recording(ground_truth, len(recording.predictions), name=name)
    return ground_truth


# ---------------------------------------------------------------- result tables
def _format_value(value, precision=3):
    """Render one cell for the text table: short, aligned, never in exponent form."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if value == int(value) and abs(value) < 1e6:
            return str(int(value))
        if abs(value) >= 1000:
            return f"{value:.0f}"
        return f"{value:.{precision}g}"
    return str(value)


def format_table(rows, columns=None, precision=3, max_width=36):
    """Column-aligned plain-text table of result rows (list of dict)."""
    if not rows:
        return "(no rows)"

    columns = list(columns) if columns is not None else list(rows[0])

    def cell(value):
        text = _format_value(value, precision)
        return text if len(text) <= max_width else text[:max_width - 3] + "..."

    table = [[cell(row.get(col)) for col in columns] for row in rows]
    widths = [max(len(col), *(len(line[i]) for line in table))
              for i, col in enumerate(columns)]

    def render(cells):
        return "  ".join(text.rjust(widths[i]) for i, text in enumerate(cells))

    lines = [render(columns), "  ".join("-" * w for w in widths)]
    lines.extend(render(line) for line in table)
    return "\n".join(lines)


def write_csv(rows, path, columns=None, german_excel=True, precision=6):
    """
    Write result rows (list of dict) to a CSV file.

    ``german_excel=True`` (default) writes semicolon-separated values with a
    decimal comma; floats are rounded to ``precision`` significant digits and
    NaN becomes an empty cell. Returns the number of rows written.
    """
    if not rows:
        raise ValueError("No rows to write.")

    columns = list(columns) if columns is not None else list(rows[0])
    delimiter = ";" if german_excel else ","

    def cell(value):
        if isinstance(value, float):
            if math.isnan(value):
                return ""            # an empty cell, not the string "nan"
            text = f"{value:.{precision}g}"
            return text.replace(".", ",") if german_excel else text
        return "" if value is None else value

    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        # utf-8-sig: without the BOM, Excel reads the file as ANSI.
        writer = csv.writer(handle, delimiter=delimiter)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([cell(row.get(col)) for col in columns])

    log.debug("Wrote %d row(s) x %d column(s) to %s", len(rows), len(columns), path)
    return len(rows)
