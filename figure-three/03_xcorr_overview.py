"""
03_xcorr_overview.py
====================

Step 3 of the Figure 3 pipeline: group overview of the motor-unit
cross-correlation at lag 0 with the trigger, by gesture and recording day
(Fig3_MU_xcorr_overview).

    top panel     bar = mean of ``xcorr_lag0_hold_mean`` across all MUs that
                  have a value; error bar = SD across those MUs (ddof = 1; only
                  drawn for n >= 2); dots = the individual MUs
    bottom panel  MU status per gesture and session, stacked:
                    filled  = MU with a valid hold cross-correlation
                    hatched = MU active (xcorr_global_lag0 exists), but no valid
                              hold (too few firings, or active only during a ramp)
                    grey    = MU not active during this gesture
                    label   = valid / total
A gesture that was not recorded in a session leaves an empty slot. Only
existing values are used; nothing is imputed.

Input / Output
--------------
    in : data/<S>_overview.csv                 (step 01)
    out: results/Fig3_MU_xcorr_overview.svg
         results/Fig3_MU_xcorr_overview_values.csv   (per gesture and recording day)

Needs no raw data.

Dependencies
------------
    numpy, pandas, matplotlib

Usage
-----
    uv run python figure-three/03_xcorr_overview.py

Author:  Pauline Wittermann (pauline.wittermann@fau.de) and Dominik I. Braun (dome.braun@fau.de)
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")           # save only, no window
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.lines import Line2D   # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import _common as C               # noqa: E402
from _style import GCOL, GESTURES  # noqa: E402

OUT_NAME = "Fig3_MU_xcorr_overview"

INK, MUTED, GRID, INACTIVE = "#1a1a1a", "#6b6b6b", "#dcdcdc", "#e4e4e4"

BAR_W = 0.72          # bar width (one session slot = 1.0)
GROUP_GAP = 1.1       # extra space between gesture groups
FIG_W_MM, FIG_H_MM = 180, 95
SEED = 7              # jitter of the dots only

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "hatch.linewidth": 0.6,
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
}


def main():
    mpl.rcParams.update(RC)
    out_dir = C.results_dir()

    # ---------------------------------------------------------------- data
    files = sorted(glob.glob(os.path.join(C.DATA_DIR, "P*_overview.csv")))
    if not files:
        sys.exit(f"No *_overview.csv found in {C.DATA_DIR} -- run "
                 f"01_xcorr_trigger.py first")
    df = pd.concat([pd.read_csv(f, sep=";", decimal=",", encoding="utf-8-sig")
                    for f in files], ignore_index=True)
    df["gesture"] = df["task"].str.split("_").str[0]
    df["valid"] = df["xcorr_lag0_hold_mean"].notna()
    df["active_nohold"] = ~df["valid"] & df["xcorr_global_lag0"].notna()
    df["inactive"] = ~df["valid"] & df["xcorr_global_lag0"].isna()

    sessions = (df[["patient_id", "visit"]].drop_duplicates()
                .sort_values(["patient_id", "visit"]).itertuples(index=False))
    sessions = [(p, d) for p, d in sessions]

    def session_label(p, d):
        same = [s for s in sessions if s[0] == p]
        return f"{p}\nS{same.index((p, d)) + 1}" if len(same) > 1 else p

    # ---------------------------------------------------------------- figure
    fig, (ax, axn) = plt.subplots(
        2, 1, figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4), sharex=True,
        gridspec_kw=dict(height_ratios=[3, 1.1], hspace=0.10))
    rng = np.random.default_rng(SEED)

    xticks, xlabels, group_spans, summary = [], [], [], []
    n_s = len(sessions)
    for gi, g in enumerate(GESTURES):
        x0 = gi * (n_s + GROUP_GAP)
        group_spans.append((x0 - 0.5, x0 + n_s - 0.5))
        for si, (p, d) in enumerate(sessions):
            x = x0 + si
            xticks.append(x)
            xlabels.append(session_label(p, d))
            sub = df[(df.patient_id == p) & (df.visit == d) & (df.gesture == g)]
            if sub.empty:                   # gesture not recorded -> empty slot
                continue
            v = sub.loc[sub.valid, "xcorr_lag0_hold_mean"].to_numpy()
            n_v = len(v)
            n_a = int(sub.active_nohold.sum())
            n_i = int(sub.inactive.sum())
            mean = v.mean() if n_v else np.nan
            sd = v.std(ddof=1) if n_v > 1 else np.nan
            summary.append(dict(gesture=g, patient_id=p, visit=d, n_valid=n_v,
                                n_active_no_hold=n_a, n_inactive=n_i,
                                mean=mean, sd=sd))

            if n_v:
                ax.bar(x, mean, BAR_W, color=GCOL[g], lw=0, zorder=2)
                if n_v > 1:
                    ax.errorbar(x, mean, yerr=sd, color=INK, lw=0.7,
                                capsize=1.8, capthick=0.7, zorder=4)
                jit = rng.uniform(-BAR_W * 0.28, BAR_W * 0.28, n_v)
                ax.scatter(x + jit, v, s=6, facecolor="white", edgecolor=INK,
                           lw=0.45, zorder=5)

            # MU status, stacked bottom -> top: valid, active w/o hold, inactive
            axn.bar(x, n_v, BAR_W, color=GCOL[g], lw=0, zorder=2)
            if n_a:
                axn.bar(x, n_a, BAR_W, bottom=n_v, facecolor="white",
                        edgecolor=GCOL[g], hatch="//////", lw=0.6, zorder=2)
            if n_i:
                axn.bar(x, n_i, BAR_W, bottom=n_v + n_a, color=INACTIVE, lw=0,
                        zorder=2)
            axn.text(x, n_v + n_a + n_i + 0.5, f"{n_v}/{n_v + n_a + n_i}",
                     ha="center", va="bottom", fontsize=5.5, color=INK)

    # axes styling
    ax.axhline(0, color=INK, lw=0.6, zorder=3)
    ax.set_ylim(-1, 1)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_ylabel("Cross-correlation at lag 0\n(mean ± SD across MUs)")
    axn.set_ylim(0, 15)
    axn.set_yticks([0, 6, 12])
    axn.set_ylabel("MUs (n)")
    for a in (ax, axn):
        a.spines[["top", "right"]].set_visible(False)
        a.yaxis.grid(True, color=GRID, lw=0.4, zorder=0)
        a.set_axisbelow(True)
        a.tick_params(axis="x", length=0)
    axn.set_xticks(xticks)
    axn.set_xticklabels(xlabels, linespacing=0.95)
    axn.set_xlim(group_spans[0][0] - 0.3, group_spans[-1][1] + 0.3)

    # gesture headers: name + thin colour rule above each group
    for g, (a0, a1) in zip(GESTURES, group_spans):
        ax.plot([a0 + 0.08, a1 - 0.08], [1.06, 1.06], color=GCOL[g], lw=2.2,
                solid_capstyle="butt", transform=ax.get_xaxis_transform(),
                clip_on=False)
        ax.text((a0 + a1) / 2, 1.10, g, ha="center", va="bottom", fontsize=7.5,
                fontweight="bold", transform=ax.get_xaxis_transform())

    legend = [
        Line2D([], [], marker="o", ls="", mfc="white", mec=INK, mew=0.45, ms=3,
               label="Individual MU"),
        Patch(facecolor="white", edgecolor=MUTED, hatch="//////", lw=0.6,
              label="MU active, no valid hold"),
        Patch(color=INACTIVE, label="MU not active"),
    ]
    axn.legend(handles=legend, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, -0.42), frameon=False, fontsize=6,
               handlelength=1.3, columnspacing=1.6)

    fig.align_ylabels([ax, axn])
    fig.savefig(os.path.join(out_dir, f"{OUT_NAME}.svg"), bbox_inches="tight",
                metadata={"Date": None})   # no creation date in the file
    plt.close(fig)
    pd.DataFrame(summary).to_csv(os.path.join(out_dir, f"{OUT_NAME}_values.csv"),
                                 sep=";", decimal=",", index=False, float_format="%.4f")
    print(pd.DataFrame(summary).to_string())
    print(f"Wrote {OUT_NAME}.svg and {OUT_NAME}_values.csv to {out_dir}")


if __name__ == "__main__":
    main()
