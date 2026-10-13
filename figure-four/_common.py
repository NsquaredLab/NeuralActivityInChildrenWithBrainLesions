"""
_common.py
==========

Shared definitions for the Figure 4 pipeline (offline decoding of cued hand
tasks with leave-one-repetition-out cross-validation).

    - ``SESSIONS`` / ``TASKS`` / ``EXCLUDED_TASKS`` / ``REST_BLOCK`` /
      ``CUE_CORRECTION``         the analysed sessions, recordings and the
                                 analysis decisions made per session
    - ``load_config`` / ``session_path``
                                 the private mapping from session names to the
                                 data files (``sessions.local.json``)
    - ``analysed_tasks`` / ``session_channels`` / ``rest_signal`` /
      ``mean_rms_envelope``      data helpers shared by steps 01, 02 and 05
    - ``DATA_DIR`` / ``RESULTS_DIR``
                                 output folders (not distributed, see below)

The steps are numbered ``01`` … ``08``; their run order is the numeric order.
Filters, RMS features and classifier settings are in ``_emg.py``.

Data
----
No participant data are distributed with this repository (see the top-level
README). The real file names encode recording dates and participant initials,
so they are not part of the code either: ``sessions.local.json`` (git-ignored)
maps the de-identified session names used here to the files; its layout is
shown in ``sessions.example.json``. Everything a step writes goes to
``data/`` and ``results/`` next to the scripts, both git-ignored.

Author:  Pauline Wittermann (pauline.wittermann@fau.de) and Dominik I. Braun (dome.braun@fau.de)
"""

import json
import os
import pickle

import numpy as np

import _emg as E

FIG4_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(FIG4_DIR, "data")
LABEL_DIR = os.path.join(DATA_DIR, "labels")
RESULTS_DIR = os.path.join(FIG4_DIR, "results")
CONFIG_PATH = os.path.join(FIG4_DIR, "sessions.local.json")

# Analysed sessions, in the order used everywhere (figure columns, statistics,
# random-number streams). P01 was recorded twice (S1, S2).
SESSIONS = ["P01-S1", "P01-S2", "P02", "P03", "P04", "P05"]

# Task recordings analysed per session (keys of the recording-day pickle; one
# instructed gesture per recording, five cues each). Channel quality control
# runs over all of them, also over excluded ones.
TASKS = {
    "P01-S1": ["Power Grasp_trial91_right", "Pinch_trial1_right", "Tripod Pinch_trial1_right"],
    "P01-S2": ["Power Grasp_trial3_right", "Pinch_trial2_right"],
    "P02": ["Power Grasp_trial1_left", "Pinch_trial1_left"],
    "P03": ["Power Grasp_trial1_left", "Pinch_trial1_left", "Tripod Pinch_trial1_left"],
    "P04": ["Power Grasp_trial2_right", "Pinch_trial2_right", "Tripod Pinch_trial2_right"],
    "P05": ["Power Grasp_trial2_left", "Pinch_trial2_left", "Tripod Pinch_trial2_left"],
}

# Task recordings whose detected attempts are labelled but not analysed
# (quality "reject"), with the reason written into each segment's note.
EXCLUDED_TASKS = {
    "P04": {"Pinch_trial2_right": "P04: only power grasp analysed",
            "Tripod Pinch_trial2_right": "P04: only power grasp analysed"},
}

# Forearm channel block of the separate rest recording that matches the
# session's gesture recordings (verified against a gesture recording of the
# same session).
BLOCKS = {"0-191": slice(0, 192), "192-383": slice(192, 384), "0-383": slice(0, 384)}
REST_BLOCK = {"P01-S1": "0-383", "P01-S2": "0-191", "P02": "0-191",
              "P03": "192-383", "P04": "0-191", "P05": "192-383"}

# Task recordings whose stored cue trace is misaligned: two 1-s sections of the
# raw recording are not contained in the recording-day file, but its stored cue
# is the complete raw cue resampled onto the shorter recording. The cue is
# rebuilt from the raw recording of that task (``01_auto_labels.corrected_cue``);
# the raw file is named in the config.
CUE_CORRECTION = {"P02": "Pinch_trial1_left"}

GESTURES = E.GESTURES                   # canonical class order
JSON_TO_GESTURE = {"power_grasp": "Power Grasp", "pinch": "Pinch", "tripod_pinch": "Tripod Pinch"}
GESTURE_TO_JSON = {v: k for k, v in JSON_TO_GESTURE.items()}
ADC_LIMIT = 32767                       # 16-bit full scale: a channel reaching it is saturated


# ---------------------------------------------------------------- configuration
def load_config() -> dict:
    """Read ``sessions.local.json`` (data root + file of every session)."""
    if not os.path.isfile(CONFIG_PATH):
        raise FileNotFoundError(
            f"Missing {CONFIG_PATH}\n"
            "The Figure 4 steps read the participant recordings, which are not distributed with this "
            "repository (identifiable EMG of a vulnerable participant group; available from the authors on "
            "reasonable request). Copy sessions.example.json to sessions.local.json and fill in the paths.")
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def session_path(cfg: dict, session: str, key: str, task: str | None = None) -> str:
    """Absolute path of a session's file: ``recording``, ``rest_recording`` or a
    raw task recording (``raw_task_recordings[task]``)."""
    entry = cfg["sessions"][session]
    rel = entry["raw_task_recordings"][task] if key == "raw_task_recordings" else entry[key]
    path = os.path.join(cfg["data_root"], rel)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{session}: {key} not found: {path}")
    return path


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------- data helpers
def analysed_tasks(day: dict, session: str) -> list[str]:
    """The session's analysed task keys, in the order of the recording-day pickle."""
    missing = [t for t in TASKS[session] if t not in day["tasks"]]
    if missing:
        raise KeyError(f"{session}: task recordings not found: {missing}")
    return [t for t in day["tasks"] if t in TASKS[session]]


def task_bad_channels(td: dict) -> set[int]:
    """Channels (0-indexed) of one task recording that are saturated or were
    discarded in the motor-unit decomposition (DEMUSE ``discardChannelsVec``).
    The decomposition stores channels on its electrode grid; each grid cell is
    matched to its EMG row by the first 64 samples."""
    emg, grid, disc = td["emg"], td["emg_grid"], td.get("discardChannelsVec")
    bad = set(np.where((np.abs(emg) >= ADC_LIMIT).any(1))[0].tolist())
    if disc is None:
        return bad
    key = {emg[c, :64].tobytes(): c for c in range(emg.shape[0])}
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if disc[i, j] and isinstance(v, np.ndarray) and v.size:
                c = key.get(v[:64].astype(np.float32).tobytes())
                if c is not None:
                    bad.add(c)
    return bad


def rest_block(rest_raw: dict, session: str) -> np.ndarray:
    """The session's forearm block of the rest recording, (channels x samples) float32."""
    return E.flatten_biosignal(rest_raw["biosignal"])[BLOCKS[REST_BLOCK[session]]]


def session_channels(day: dict, rest_raw: dict, session: str) -> tuple[np.ndarray, set[int]]:
    """Analysed channels of a session: all channels except those bad in any
    analysed task recording or saturated in the rest recording.
    Returns (kept channels 0-indexed, bad channel set)."""
    tasks = analysed_tasks(day, session)
    n_ch = day["tasks"][tasks[0]]["emg"].shape[0]
    bad = set()
    for t in tasks:
        bad |= task_bad_channels(day["tasks"][t])
    bad |= set(np.where((np.abs(rest_block(rest_raw, session)) >= ADC_LIMIT).any(1))[0].tolist())
    return np.array([c for c in range(n_ch) if c not in bad]), bad


def rest_signal(rest_raw: dict, session: str) -> np.ndarray:
    """Rest-recording block with channel means removed, exactly like the
    gesture recordings in the recording-day pickles (float32)."""
    rest = rest_block(rest_raw, session).astype(np.float64)
    return (rest - rest.mean(1, keepdims=True)).astype(np.float32)


def mean_rms_envelope(emg: np.ndarray, kept: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """Channel-averaged RMS envelope with the pipeline's filters and RMS window.
    Returns (frame-centre times in s, envelope)."""
    x = E.apply_filters(emg[kept].astype(np.float32), fs, E.BANDPASS_HZ, E.BANDSTOP_HZ, E.BUTTER_ORDER)
    env = E.rms_moving_window(x, E.WINDOW_SAMPLES, E.STRIDE_SAMPLES).mean(0)
    te = (np.arange(len(env)) * E.STRIDE_SAMPLES + E.WINDOW_SAMPLES / 2) / fs
    return te, env


def label_path(session: str) -> str:
    return os.path.join(LABEL_DIR, f"{session}.labels_auto_v2.json")


def envelope_path(session: str) -> str:
    return os.path.join(LABEL_DIR, f"{session}_envelopes.npz")


def load_labels(session: str) -> dict:
    path = label_path(session)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found - run 01_auto_labels.py first")
    with open(path, encoding="utf-8") as f:
        return json.load(f)
