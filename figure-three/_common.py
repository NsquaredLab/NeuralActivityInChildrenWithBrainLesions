"""
_common.py
==========

Shared definitions for the Figure 3 pipeline (cross-correlation of the tracked
motor units' discharge rates with the cue).

    - ``SESSIONS`` / ``BY_EYE`` / ``BY_EYE_MAYBE``
                                 the analysed recording days and the by-eye
                                 verdict on cue modulation
    - ``load_config`` / ``session_files``
                                 the private mapping from session names to the
                                 data files (``sessions.local.json``)
    - ``load_recording_day`` / ``load_tracking`` / ``tracked_tasks``
                                 readers of the recording-day pickle and of the
                                 tracking decomposition
    - ``read_labels`` / ``task_gesture``
                                 the hold labels
    - ``write_csv``              result tables (German Excel dialect)
    - ``get_logger`` / ``set_level`` / ``step`` / ``log_array``
                                 console logging
    - ``DATA_DIR`` / ``RESULTS_DIR``
                                 output folders (not distributed, see below)

The steps are numbered ``01`` ... ``04``; their run order is the numeric order.
The discharge rate and the cross-correlation are in ``_xcorr.py``.

Data
----
No participant data are distributed with this repository (see the top-level
README). ``sessions.local.json`` (git-ignored) maps the pseudonymised session
names used here to the files; its layout is shown in ``sessions.example.json``.
Per recording day three files are read:

    recording   the unified ``*_raw_and_decomp.pkl`` (per task: EMG, trigger,
                labels, decomposition)
    tracking    the tracking ``.mat``: the concatenated DEMUSE decomposition of
                all gestures of the day (``SIG`` / ``MUPulses`` / ``PNR`` /
                ``fsamp`` / ``discardChannelsVec``)
    labels      the ``*.labels_auto_v2.json``: per task the hold segments
                (``start_sample`` / ``end_sample`` / ``gesture`` / ``quality``)

Steps 01 and 02 write their tables to ``data/``, steps 03 and 04 the figures
and value tables to ``results/``, both next to the scripts and git-ignored.

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

import numpy as np
import scipy.io

FIG3_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(FIG3_DIR, "data")
RESULTS_DIR = os.path.join(FIG3_DIR, "results")
CONFIG_PATH = os.path.join(FIG3_DIR, "sessions.local.json")

# Recording days in the paper: session -> gestures in recording order.
# P01 was recorded twice (S1, S2).
SESSIONS = {
    "P01-S1": ["Power Grasp", "Pinch", "Tripod Pinch"],
    "P01-S2": ["Power Grasp", "Pinch", "Tripod Pinch"],
    "P02-S1": ["Power Grasp", "Pinch"],
    "P03-S1": ["Power Grasp", "Pinch", "Tripod Pinch"],
    "P04-S1": ["Power Grasp", "Pinch", "Tripod Pinch"],
    "P05-S1": ["Power Grasp", "Pinch", "Tripod Pinch"],
}

# By-eye verdict: cue-modulated MUs per recording day, 1-based numbering of
# the tracking file. 'maybe' is counted as not cue-modulated.
BY_EYE = {
    "P01-S1": [1, 3, 4, 5, 6, 9, 11, 12],
    "P01-S2": [1, 2, 3],
    "P02-S1": [1, 2, 3, 5, 6],
    "P03-S1": [1, 2, 3, 5, 6, 7, 8],
    "P04-S1": [1, 2, 4, 5, 6, 7],
    "P05-S1": [],
}
BY_EYE_MAYBE = {"P04-S1": [3]}


# ---------------------------------------------------------------- configuration
def load_config() -> dict:
    """Read ``sessions.local.json`` (data root + files of every session)."""
    if not os.path.isfile(CONFIG_PATH):
        raise FileNotFoundError(
            f"Missing {CONFIG_PATH}\n"
            "The Figure 3 steps 01 and 02 read the participant recordings, which are not distributed with "
            "this repository (identifiable EMG of a vulnerable participant group; available from the authors "
            "on reasonable request). Copy sessions.example.json to sessions.local.json and fill in the paths.")
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def session_files(cfg: dict, session: str) -> tuple[str, str, str]:
    """(recording pkl, tracking mat, labels json) of one recording day."""
    entry = cfg["sessions"][session]
    paths = tuple(os.path.join(cfg["data_root"], entry[key])
                  for key in ("recording", "tracking", "labels"))
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"{session}: missing {missing}")
    return paths


def data_dir():
    """DATA_DIR, created if missing (tables of steps 01 and 02)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    return DATA_DIR


def results_dir():
    """RESULTS_DIR, created if missing (figures of steps 03 and 04)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


# ---------------------------------------------------------------- logging
# Root logger of the pipeline; every module logger is a child of it, so one
# set_level() call covers all of them.
ROOT_LOGGER_NAME = "figure3"
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


# ---------------------------------------------------------------- recording day
def load_recording_day(path):
    """
    Load a unified ``*_raw_and_decomp.pkl`` file.

    Returns
    -------
    dict
        The recording-day dict (``patient_id`` / ``date`` / ``tasks``); per task
        ``emg`` (channels, samples), ``emg_grid``, ``fsamp``, ``trigger``
        (resampled + normalised to 0..1), ``MUPulses``, ``discardChannelsVec``,
        ``labels`` (0 = rest, 1 = activation) and ``PNR``.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Recording day not found: {path}")

    with open(path, "rb") as f:
        recording = pickle.load(f)

    tasks = recording.get("tasks", {})
    log.info("Loaded recording day %s (%s): %d tasks",
             recording.get("patient_id"), recording.get("date"), len(tasks))
    log.debug("Tasks: %s", list(tasks))
    return recording


def iter_tasks(recording):
    """Yield ``(task_key, task_dict)`` pairs of a recording day, in insertion order."""
    for task_key, task in recording.get("tasks", {}).items():
        yield task_key, task


# ---------------------------------------------------------------- tracking file
def blocks_path_for(tracking_mat):
    """
    The companion ``*_blocks.mat`` of a concatenated decomposition.

    ``..._concat_decomp.mat`` -> ``..._concat_decomp_blocks.mat``, next to it.
    """
    stem, _ = os.path.splitext(tracking_mat)
    return stem + "_blocks.mat"


def grid_to_channels(sig, discard=None, drop_discarded=True):
    """
    Flatten a DEMUSE ``SIG`` grid into a plain ``(channels, samples)`` array.

    The grid is an object array with one cell per electrode position; the gap
    columns between the physical grids are empty and are skipped. Channels are
    returned in column-major order of the grid. Channels flagged in
    ``discard`` are left out when ``drop_discarded`` is True (default).
    Empty ``(0, 0)`` if the grid holds no usable channel.
    """
    sig = np.asarray(sig, dtype=object)
    discard = None if discard is None else np.asarray(discard)

    channels = []
    dropped = 0
    short = 0
    lengths = [np.asarray(cell).size for cell in sig.ravel()]
    n_samples = max(lengths) if lengths else 0

    for col in range(sig.shape[1]):
        for row in range(sig.shape[0]):
            channel = np.asarray(sig[row, col], dtype=float).ravel()
            if channel.size == 0:
                continue                      # gap column between the grids
            if (drop_discarded and discard is not None
                    and row < discard.shape[0] and col < discard.shape[1]
                    and discard[row, col]):
                dropped += 1
                continue
            if channel.size != n_samples:
                # Should not happen -- the concatenation zero-pads every channel
                # to the full length.
                short += 1
                channel = np.pad(channel, (0, n_samples - channel.size))
            channels.append(channel)

    if dropped:
        log.info("SIG: %d discarded channel(s) left out, %d kept.",
                 dropped, len(channels))
    if short:
        log.warning("SIG: %d channel(s) were shorter than %d samples and were "
                    "zero-padded -- check the concatenation.", short, n_samples)
    if not channels:
        log.warning("SIG holds no usable channel.")
        return np.zeros((0, 0))
    return np.vstack(channels)


def _strings(value):
    """A MATLAB cell of char (however scipy squeezed it) as a list of str."""
    if value is None:
        return []
    items = np.atleast_1d(np.asarray(value, dtype=object)).ravel()
    return [str(np.asarray(item).ravel()[0]) if np.asarray(item).size else ""
            for item in items]


def _int_row(value):
    """A MATLAB 1 x n double row as a list of int."""
    if value is None:
        return []
    return [int(round(float(x)))
            for x in np.atleast_1d(np.asarray(value, dtype=float)).ravel()]


def load_tracking(tracking_mat, blocks_mat="auto", drop_discarded=True):
    """
    Load a concatenated decomposition (and, if present, its block boundaries).

    Parameters
    ----------
    tracking_mat : str
        Path to the concatenated decomposition ``.mat``.
    blocks_mat : str or None, optional
        Path to the companion ``*_blocks.mat``. ``"auto"`` (default) looks for
        it next to ``tracking_mat``; ``None`` skips it (the task boundaries
        then come from the recording-day pickle, see :func:`tracked_tasks`).
    drop_discarded : bool, optional
        Passed to :func:`grid_to_channels` (default True).

    Returns
    -------
    dict
        ``path``, ``name``, ``fsamp``, ``emg`` (the flattened ``SIG``),
        ``emg_grid``, ``discardChannelsVec``, ``MUPulses`` (object array, the
        discharge samples of every unit in the file's own 1-based time base),
        ``PNR``, ``ref_signal``, ``trigger`` and ``blocks``.
    """
    if not os.path.isfile(tracking_mat):
        raise FileNotFoundError(f"Tracking MAT not found: {tracking_mat}")

    mat = scipy.io.loadmat(tracking_mat, squeeze_me=True, struct_as_record=False)
    missing = [key for key in ("SIG", "MUPulses", "fsamp") if key not in mat]
    if missing:
        available = sorted(key for key in mat if not key.startswith("__"))
        raise KeyError(
            f"{os.path.basename(tracking_mat)} is missing {missing} -- is this "
            f"a concatenated DEMUSE decomposition? It holds: {available}")

    fsamp = float(np.asarray(mat["fsamp"]).ravel()[0])
    sig = np.asarray(mat["SIG"], dtype=object)
    discard = mat.get("discardChannelsVec")
    emg = grid_to_channels(sig, discard, drop_discarded=drop_discarded)

    mu_pulses = mat["MUPulses"]
    if not (isinstance(mu_pulses, np.ndarray) and mu_pulses.dtype == object):
        # squeeze_me unwrapped a 1x1 cell (a single MU) to its contents --
        # wrap it again, or ravel() would turn every discharge into a unit.
        single = np.empty(1, dtype=object)
        single[0] = np.atleast_1d(mu_pulses)
        mu_pulses = single
    mu_pulses = mu_pulses.ravel()

    pnr = mat.get("PNR")
    if pnr is not None:
        pnr = np.atleast_1d(np.asarray(pnr, dtype=float))

    ref_signal = mat.get("ref_signal")
    if ref_signal is not None:
        ref_signal = np.asarray(ref_signal, dtype=float).ravel()

    n_samples = emg.shape[1] if emg.size else 0
    trigger = None
    blocks = []

    if blocks_mat == "auto":
        blocks_mat = blocks_path_for(tracking_mat)
    if blocks_mat is not None and not os.path.isfile(blocks_mat):
        log.warning("No block file next to the tracking MAT (%s).",
                    os.path.basename(blocks_mat))
        blocks_mat = None

    if blocks_mat is not None:
        companion = scipy.io.loadmat(blocks_mat, squeeze_me=True,
                                     struct_as_record=False)
        if "trigger" in companion:
            trigger = np.asarray(companion["trigger"], dtype=float).ravel()

        names = _strings(companion.get("blockNames"))
        block_gestures = _strings(companion.get("blockGestures"))
        starts = _int_row(companion.get("blockStartSamples"))
        stops = _int_row(companion.get("blockStopSamples"))
        counts = _int_row(companion.get("blockMUCounts"))
        offsets = _int_row(companion.get("blockPulseOffsets"))

        for i, (start, stop) in enumerate(zip(starts, stops)):
            gesture = block_gestures[i] if i < len(block_gestures) else ""
            name = names[i] if i < len(names) else ""
            blocks.append({
                # MATLAB's sample numbers are 1-based and inclusive, python
                # slices are 0-based and half-open.
                "start": start - 1,
                "stop": stop,
                "gesture": gesture,
                "name": name,
                "label": gesture.title() if gesture else (name or f"Block {i + 1}"),
                "n_mu": counts[i] if i < len(counts) else 0,
                "mu_offset": offsets[i] if i < len(offsets) else 0,
            })

    if not blocks:
        blocks = [{"start": 0, "stop": n_samples, "gesture": "", "name": "",
                   "label": os.path.basename(tracking_mat),
                   "n_mu": int(mu_pulses.size), "mu_offset": 0}]

    log.info("Tracking: %s -- %d channel(s), %d MU(s), %d block(s) (%s), "
             "fsamp %g Hz", os.path.basename(tracking_mat), emg.shape[0],
             mu_pulses.size, len(blocks),
             ", ".join(block["label"] for block in blocks), fsamp)
    log_array(log, "tracking.emg", emg)

    return {
        "path": tracking_mat,
        "name": os.path.basename(tracking_mat),
        "fsamp": fsamp,
        "emg": emg,
        "emg_grid": sig,
        "discardChannelsVec": discard,
        "MUPulses": mu_pulses,
        "PNR": pnr,
        "ref_signal": ref_signal,
        "trigger": trigger,
        "blocks": blocks,
    }


def tracked_tasks(recording, tracking):
    """
    The tasks of a recording day with their motor units swapped for the
    tracked ones.

    The tracking file holds the tasks of the recording day one after the
    other, in the order the pkl stores them, so every task's length says where
    it sits in the tracking file. Each task is returned as a shallow copy in
    which ``MUPulses`` and ``PNR`` come from the tracking file, cut to the
    task's own stretch -- everything else (``emg``, ``labels``, ``trigger``)
    stays the task's own. Unit ``k`` is therefore the same unit in every task.

    Returns
    -------
    list of (str, dict)
        ``(task_key, task)`` in pkl order.

    Raises
    ------
    ValueError
        If the tasks do not fit the tracking file (sampling rate or total
        length) -- then the pkl is not the one the file was built from.
    """
    fsamp = float(tracking["fsamp"])
    n_samples = np.asarray(tracking["emg"]).shape[1]
    mu_pulses = [np.asarray(pulses, dtype=int).ravel()
                 for pulses in tracking["MUPulses"]]

    tasks = list(iter_tasks(recording))
    rates = {float(task["fsamp"]) for _, task in tasks}
    if rates != {fsamp}:
        raise ValueError(f"The tasks were recorded at {sorted(rates)} Hz, "
                         f"the tracking file at {fsamp:g} Hz.")
    total = sum(int(np.asarray(task["emg"]).shape[1]) for _, task in tasks)
    if total != n_samples:
        raise ValueError(f"The tasks cover {total} samples, the tracking file "
                         f"holds {n_samples} -- is the pkl the one this "
                         f"tracking file was built from?")

    result = []
    start = 0
    for task_key, task in tasks:
        stop = start + int(np.asarray(task["emg"]).shape[1])
        # MUPulses stay 1-based, as in the per-task decompositions the pkl
        # stores: the task's samples 1..length are start+1..stop of the
        # tracking file.
        pulses = [p[(p > start) & (p <= stop)] - start for p in mu_pulses]
        # Filled one by one: np.array(pulses, dtype=object) would stack units
        # of equal length into a 2-D array instead.
        mu_array = np.empty(len(pulses), dtype=object)
        for k, p in enumerate(pulses):
            mu_array[k] = p
        swapped = dict(task)
        swapped["MUPulses"] = mu_array
        swapped["PNR"] = tracking.get("PNR")
        log.debug("%s: samples %d-%d, discharges per MU %s", task_key, start,
                  stop, [int(p.size) for p in pulses])
        result.append((task_key, swapped))
        start = stop
    return result


# ---------------------------------------------------------------- hold labels
def task_gesture(task_key):
    """'Tripod Pinch_trial1_left' -> 'tripod_pinch' (the label spelling)."""
    return task_key.split("_")[0].strip().lower().replace(" ", "_")


def read_labels(path):
    """The ``tasks`` of a ``*.labels_auto_v2.json``."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d["tasks"]


# ---------------------------------------------------------------- result tables
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
