"""
_common.py
==========

Shared definitions for the Figure 2 pipeline (motor-unit tracking per
recording day).

    - ``SESSIONS`` / ``ATTRIBUTE`` / ``EXTRA_DISCARDED_CHANNELS`` / ``GRIDS``
                                 the analysed recording days and the analysis
                                 decisions made per recording day
    - ``load_config`` / ``session_files``
                                 the private mapping from session names to the
                                 data files (``sessions.local.json``)
    - ``load_recording_day`` / ``recording_day_blocks`` / ``load_tracking`` /
      ``load_tracking_grid``     readers of the recording-day pickle and of the
                                 tracking decomposition
    - ``label_mask``             hold labels -> 0/1 cue mask
    - ``write_csv``              result tables (German Excel dialect)
    - ``get_logger`` / ``set_level`` / ``step`` / ``log_array``
                                 console logging
    - ``RESULTS_DIR``            output folder (not distributed, see below)

The steps are numbered ``01`` and ``02``; their run order is the numeric order.
Filters, discharge rates and motor-unit maps are in ``_emg.py``, the figure
builders in ``_tracking.py`` and the drawing in ``_plot.py``.

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
                that define the cue of the figures

Everything a step writes goes to ``results/`` next to the scripts (git-ignored).

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

FIG2_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(FIG2_DIR, "results")
CONFIG_PATH = os.path.join(FIG2_DIR, "sessions.local.json")

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

# Key of EXTRA_DISCARDED_CHANNELS per recording day: electrodes discarded on
# top of the file's own discardChannelsVec. Only P01 S2 has an entry.
ATTRIBUTE = {
    "P01-S1": "P01_1",
    "P01-S2": "P01_2",
    "P02-S1": "P02",
    "P03-S1": "P03",
    "P04-S1": "P04",
    "P05-S1": "P05",
}

# Channels to discard on top of a file's own discardChannelsVec, per recording
# day ("attribute"). The electrodes DEMUSE did not flag but that are broken on
# inspection. 0-based (row, col) on the raster of GRIDS.
EXTRA_DISCARDED_CHANNELS = {
    "P01_2": [
        # grid 1
        (3, 0), (3, 2), (4, 1), (7, 0), (10, 4), (11, 4), (12, 4),
        # grid 2
        (1, 7), (2, 8), (3, 7), (4, 6), (4, 8), (6, 7), (6, 9), (9, 6),
        (11, 6), (12, 8), (9, 7),
        # grid 3
        (12, 12),
    ],
}

# Electrode layout of the high-density grid used for the motor-unit heatmap.
# The decomposition stores channels on a 13 x 17 raster made of three 13 x 5
# grids; columns 6 and 12 are the (empty) gaps between the three physical grids.
# Rows / cols are 1-indexed to match the DEMUSE convention.
GRIDS = [
    {"rows": range(1, 14), "cols": range(1, 6)},    # grid 1: cols 1-5
    {"rows": range(1, 14), "cols": range(7, 12)},   # grid 2: cols 7-11
    {"rows": range(1, 14), "cols": range(13, 18)},  # grid 3: cols 13-17
]

# Rest/activation threshold on the kinematic trigger, which is stored normalised
# to 0..1. Only used when a gesture has no hold labels and its cue periods are
# taken from the trigger instead.
REST_THRESHOLD = 0.05


# ---------------------------------------------------------------- configuration
def load_config() -> dict:
    """Read ``sessions.local.json`` (data root + files of every session)."""
    if not os.path.isfile(CONFIG_PATH):
        raise FileNotFoundError(
            f"Missing {CONFIG_PATH}\n"
            "The Figure 2 steps read the participant recordings, which are not distributed with this "
            "repository (identifiable EMG of a vulnerable participant group; available from the authors on "
            "reasonable request). Copy sessions.example.json to sessions.local.json and fill in the paths.")
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


def attribute(session):
    """Extra-discard key of a recording day."""
    return ATTRIBUTE[session]


def out_dir(*parts):
    """A sub-folder of RESULTS_DIR, created if missing."""
    path = os.path.join(RESULTS_DIR, *parts)
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------- logging
# Root logger of the pipeline; every module logger is a child of it, so one
# set_level() call covers all of them.
ROOT_LOGGER_NAME = "figure2"
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


def recording_day_blocks(raw_and_decomp_pkl):
    """
    The gesture segments of a tracking file, taken from the recording day it
    was concatenated from.

    The tracking ``.mat`` holds the tasks of the ``*_raw_and_decomp.pkl`` one
    after the other, in the order the pkl stores them -- so every task's
    length is where its gesture sits in the tracking file.

    Parameters
    ----------
    raw_and_decomp_pkl : str
        Path to the unified recording day.

    Returns
    -------
    list of dict
        One per task, in pkl order: ``start`` / ``stop`` (0-based, half-open),
        ``label`` (the gesture, e.g. ``"Power Grasp"``), ``task`` (the full
        task key), ``fsamp`` and ``labels`` (the task's per-sample
        rest/activation labels, or None -- the cues of the tracking figures).
    """
    recording = load_recording_day(raw_and_decomp_pkl)
    blocks = []
    start = 0
    for task_key, task in iter_tasks(recording):
        length = int(np.asarray(task["emg"]).shape[1])
        blocks.append({"start": start,
                       "stop": start + length,
                       "label": task_key.split("_trial")[0],
                       "task": task_key,
                       "fsamp": float(task["fsamp"]),
                       "labels": task.get("labels")})
        start += length
    log.info("Segments from %s: %s", os.path.basename(raw_and_decomp_pkl),
             ", ".join(f"{b['label']} {b['start']}-{b['stop']}" for b in blocks))
    return blocks


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
    returned in column-major order of the grid, i.e. the order the electrode
    positions have in ``discardChannelsVec``.

    Parameters
    ----------
    sig : ndarray of object, shape (rows, cols)
        The ``SIG`` grid of a decomposition.
    discard : ndarray or None, optional
        ``discardChannelsVec`` of the same grid shape; 1 = channel discarded.
    drop_discarded : bool, optional
        Leave the channels flagged in ``discard`` out (default). A channel
        DEMUSE discarded is broken, not quiet: keeping it in would lift the RMS
        of every gesture it appears in.

    Returns
    -------
    ndarray, shape (n_channels, n_samples)
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
                # to the full length. Say so rather than let numpy refuse the
                # stack with a shape error further down.
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


def equal_blocks(gestures, n_samples, n_mu=0):
    """
    Cut a signal into one equally long block per gesture, in the given order.

    For the tracking files that carry no block boundaries of their own: the
    gestures were recorded for the same duration each, so ``n`` gesture names
    mean ``n`` equal pieces of the signal, in the order they are named.

    Parameters
    ----------
    gestures : sequence of str
        The gestures in the file, in recording order, e.g.
        ``["Power Grasp", "Pinch", "Tripod Pinch"]``.
    n_samples : int
        Length of the concatenated signal.
    n_mu : int, optional
        Number of motor units in the file. Written into every block: in a
        tracking file the *same* units run through all gestures, so no block
        owns a subset of them.

    Returns
    -------
    list of dict
        Blocks as :func:`load_tracking` returns them (``start`` / ``stop``
        0-based and half-open).
    """
    gestures = [str(gesture) for gesture in gestures]
    block_len = n_samples // len(gestures)
    remainder = n_samples - block_len * len(gestures)
    if remainder:
        # Either a gesture was not recorded for the same time as the others, or
        # the list does not match the file. The last samples are left out
        # rather than stretched.
        log.warning("%d gesture(s) do not divide %d samples evenly -- the last "
                    "%d sample(s) are left out of every gesture.",
                    len(gestures), n_samples, remainder)

    return [{"start": i * block_len,
             "stop": (i + 1) * block_len,
             "gesture": gesture,
             "name": "",
             "label": gesture,
             "n_mu": int(n_mu),
             "mu_offset": 0}
            for i, gesture in enumerate(gestures)]


def load_tracking(tracking_mat, blocks_mat="auto", drop_discarded=True,
                  gestures=None):
    """
    Load a concatenated decomposition together with its block boundaries.

    Parameters
    ----------
    tracking_mat : str
        Path to the concatenated decomposition ``.mat``.
    blocks_mat : str or None, optional
        Path to the companion ``*_blocks.mat``. ``"auto"`` (default) looks for
        it next to ``tracking_mat``; ``None`` skips it. Without it -- and
        without ``gestures`` -- the whole file is treated as **one** unnamed
        block.
    drop_discarded : bool, optional
        Passed to :func:`grid_to_channels` (default True).
    gestures : sequence of str or None, optional
        The gestures in the file, in recording order. The gestures are equally
        long, so the signal is cut into that many equal pieces
        (:func:`equal_blocks`). Takes precedence over a block file's
        boundaries -- the block file is then still read for its ``trigger``.

    Returns
    -------
    dict
        ``path``, ``name``, ``fsamp``,
        ``emg`` -- ``(channels, samples)``, the flattened ``SIG``,
        ``emg_grid`` -- ``SIG`` as stored, ``discardChannelsVec``, ``PNR``,
        ``MUPulses`` -- object array, the discharge samples of every unit in
        the file's own (1-based MATLAB) time base,
        ``ref_signal`` -- DEMUSE's reference signal, or None,
        ``trigger`` -- the kinematic ground truth of the whole file, or None,
        ``blocks`` -- one dict per gesture, in file order: ``label``,
        ``gesture``, ``name``, ``start`` / ``stop`` (0-based, half-open, into
        the ``emg`` / ``trigger`` time base), ``n_mu`` and ``mu_offset``.
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
        if gestures is None:
            log.warning("No block file next to the tracking MAT (%s) -- the "
                        "gestures inside it cannot be told apart. Pass "
                        "gestures=[...] to cut it into equal pieces instead.",
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

    # Given gestures win over a block file.
    if gestures is not None and len(gestures):
        if blocks:
            log.info("gestures=%s given -- the %d boundary/-ies in the block "
                     "file are ignored and the signal is cut into equal "
                     "pieces.", list(gestures), len(blocks))
        blocks = equal_blocks(gestures, n_samples, n_mu=int(mu_pulses.size))

    if not blocks:
        blocks = [{"start": 0, "stop": n_samples, "gesture": "", "name": "",
                   "label": os.path.basename(tracking_mat),
                   "n_mu": int(mu_pulses.size), "mu_offset": 0}]

    log.info("Tracking: %s -- %d channel(s), %d MU(s), %d block(s) (%s), "
             "fsamp %g Hz", os.path.basename(tracking_mat), emg.shape[0],
             mu_pulses.size, len(blocks),
             ", ".join(block["label"] for block in blocks), fsamp)
    log_array(log, "tracking.emg", emg)
    if trigger is None:
        log.info("No trigger next to this tracking file -- the gesture cues "
                 "cannot be shaded.")

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


def load_tracking_grid(tracking_mat):
    """
    Load a tracking ``.mat`` in grid layout, for the heatmaps and mean MUAPs.

    Unlike :func:`load_tracking`, the electrodes are kept at their grid
    position -- the gap columns between the physical grids become NaN -- and
    the discharges are shifted to 0-based python indices.

    Parameters
    ----------
    tracking_mat : str
        Path to the DEMUSE ``.mat`` (``fsamp`` / ``SIG`` / ``MUPulses`` /
        ``MUIDs``, optionally ``discardChannelsVec`` / ``PNR``).

    Returns
    -------
    dict
        ``fsamp``, ``sig`` -- ``(rows, cols, samples)``, ``discard`` -- bool
        ``(rows, cols)``, True for discarded or empty electrodes, ``pulses`` --
        list of 0-based int arrays, ``ids`` -- list of str, ``pnr`` -- ndarray
        (NaN if the file has none).
    """
    if not os.path.isfile(tracking_mat):
        raise FileNotFoundError(f"Tracking MAT not found: {tracking_mat}")
    d = scipy.io.loadmat(tracking_mat)
    fs = float(d["fsamp"].squeeze())
    sig_cell = d["SIG"]
    nr, nc = sig_cell.shape
    n = max(sig_cell[i, j].size for i in range(nr) for j in range(nc))
    sig = np.full((nr, nc, n), np.nan)
    for i in range(nr):
        for j in range(nc):
            if sig_cell[i, j].size:
                sig[i, j] = sig_cell[i, j].ravel().astype(float)
    discard = d.get("discardChannelsVec", np.zeros((nr, nc))).astype(bool)
    discard |= np.isnan(sig[..., 0])
    pulses = [p.ravel().astype(int) - 1 for p in d["MUPulses"][0]]  # MATLAB -> 0-based
    if "MUIDs" in d:
        ids = [str(np.squeeze(x)) for x in d["MUIDs"][0]]
    else:
        ids = [str(k + 1) for k in range(len(pulses))]
    pnr = d["PNR"].ravel() if "PNR" in d else np.full(len(pulses), np.nan)
    log.info("Tracking grid: %s -- %d x %d electrodes (%d usable), %d MU(s), "
             "%d samples, fsamp %g Hz", os.path.basename(tracking_mat), nr, nc,
             int((~discard).sum()), len(pulses), n, fs)
    return {"fsamp": fs, "sig": sig, "discard": discard, "pulses": pulses,
            "ids": ids, "pnr": pnr}


# ---------------------------------------------------------------- hold labels
def task_gesture(task_key):
    """'Tripod Pinch_trial1_left' -> 'tripod_pinch' (the label spelling)."""
    return task_key.split("_")[0].strip().lower().replace(" ", "_")


def label_mask(label_task, task_key, n_samples, session, quality="good"):
    """0/1 mask from the good, gesture-matching segments + segment rows."""
    if int(label_task["n_samples"]) != n_samples:
        raise ValueError(f"{session} {task_key}: labels have "
                         f"{label_task['n_samples']} samples, block {n_samples}")
    g = task_gesture(task_key)
    mask = np.zeros(n_samples)
    rows = []
    for k, s in enumerate(label_task["segments"], start=1):
        used = s["quality"] == quality and s["gesture"] == g
        a, b = int(s["start_sample"]), int(s["end_sample"])
        if not 0 <= a < b <= n_samples:
            raise ValueError(f"{session} {task_key} segment {k}: "
                             f"{a}-{b} outside 0-{n_samples}")
        if used:
            mask[a:b] = 1.0
        rows.append(dict(session=session, task=task_key, segment=k,
                         start_s=s["start_s"], end_s=s["end_s"],
                         start_sample=a, end_sample=b, gesture=s["gesture"],
                         quality=s["quality"], used=used))
    return mask, rows


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
