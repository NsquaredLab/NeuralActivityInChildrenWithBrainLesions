# Hidden motor neuron activity restores grasp in children with paralysed hand — analysis & figure code

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21269465.svg)](https://doi.org/10.5281/zenodo.21269465)

Analysis and figure-generation code accompanying the study:

> **Hidden motor neuron activity restores grasp in children with paralysed hand**
>
> Pauline Wittermann<sup>1†</sup>, Dominik I. Braun<sup>1†</sup>, Nico G. M. Weber<sup>2</sup>, Dörte Wartke<sup>1</sup>, Pınar Güneş<sup>1</sup>, Felix Wachter<sup>3</sup>, Paula Corcosa<sup>3</sup>, Lina Tan<sup>3,4,5</sup>, Henriette Grieshaber-Bouyer Mandelbaum<sup>3</sup>, Jonas Walter<sup>2</sup>, Jörg Franke<sup>2</sup>, Ferdinand Knieling<sup>3</sup>, and Alessandro Del Vecchio<sup>1\*</sup>
>
> Manuscript under review (2026). Article DOI to be added on publication.
>
> Analysis & figure code archived on Zenodo: [10.5281/zenodo.21269465](https://doi.org/10.5281/zenodo.21269465).
>
> † These authors contributed equally. \* Corresponding author: <alessandro.del.vecchio@fau.de>
>
> **Affiliations**
> 1. Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU), Department of Artificial Intelligence in Biomedical Engineering, Professur für Neurophysiology and Neural Interfacing; Erlangen, Germany.
> 2. Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU), Department of Mechanical Engineering, Institute for Factory Automation and Production Systems; Erlangen, Germany.
> 3. Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU), Department of Paediatrics and Adolescent Medicine, Paediatric Experimental and Translational Imaging Laboratory; Erlangen, Germany.
> 4. Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU), Department of Medicine 3 – Rheumatology and Immunology; Erlangen, Germany.
> 5. Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU), Deutsches Zentrum für Immuntherapie (DZI); Erlangen, Germany.

This repository reproduces the quantitative panels of Figures 2, 3, 4 and 5,
the per-session offline-decoding Supplementary Figures S2b–S7b and
Supplementary Table S2 from the high-density surface-electromyography (EMG)
recordings and the EMG-bracelet exergame recording analysed in the study.

---

## Data availability (please read first)

The EMG recordings analysed in this study were acquired from children with hand
paralysis after early brain lesions, including two children after hemispherotomy
— a vulnerable participant group. To protect their
privacy and to honour the consent given by the families and the approval by our research ethics
board, no participant data — neither the raw recordings nor any derived
features — are distributed with this repository. All data are available
from the authors on reasonable request for legitimate scientific use, under a data-use agreement.

This repository provides the complete analysis and figure code so that the
methodology is fully transparent and the published results can be reproduced
once data access has been granted.

**Contact for data requests:** Alessandro Del Vecchio (corresponding author) — <alessandro.del.vecchio@fau.de>

---

## Repository structure

```
.
├── figure-two/                         # Figure 2: motor-unit tracking per recording day
│   ├── _common.py                      #   recording days, data readers, hold labels
│   ├── _emg.py                         #   filters, discharge rate, MUAP maps
│   ├── _tracking.py, _plot.py          #   figure builders, drawing
│   ├── _style.py                       #   colours, SVG export
│   ├── sessions.example.json           #   layout of the private data-file mapping
│   ├── 01_tracking_rates.py            #   step 1  RMS + discharge rates per tracked MU
│   └── 02_heatmaps_muap.py             #   step 2  P2P heatmaps + mean MUAPs
├── figure-three/                       # Figure 3: MU discharge rate vs. cue
│   ├── _common.py                      #   recording days, by-eye verdict, data readers
│   ├── _xcorr.py                       #   discharge rate, normalised cross-correlation
│   ├── _style.py                       #   gesture colours
│   ├── sessions.example.json           #   layout of the private data-file mapping
│   ├── 01_xcorr_trigger.py             #   step 1  xcorr with the recorded trigger
│   ├── 02_xcorr_labels.py              #   step 2  xcorr with the hold labels
│   ├── 03_xcorr_overview.py            #   step 3  Fig3_MU_xcorr_overview
│   └── 04_mu_overview.py               #   step 4  Fig3_MU_overview (panels a-c)
├── figure-four/                        # Figure 4: offline decoding of cued hand tasks
    ├── _common.py                      #   sessions, analysis decisions, shared helpers
    ├── _emg.py                         #   filters, RMS features, CatBoost settings
    ├── _style.py, _svg.py              #   figure style, editable-SVG export
    ├── sessions.example.json           #   layout of the private data-file mapping
    ├── 01_auto_labels.py               #   step 1  rule-based attempt labels
    ├── 02_dataset.py                   #   step 2  labelled datasets + RMS features
    ├── 03_loro_cv.py                   #   step 3  leave-one-repetition-out CatBoost CV
    ├── 04_statistics.py                #   step 4  permutation tests, bootstrap CIs, Holm
    ├── 05_figure.py                    #   step 5  Figure 4 (composite + panel SVGs)
    ├── 06_attempt_summary.py           #   step 6  attempts per cue, cue coupling, cue timing
    ├── 07_supplementary_figures.py     #   step 7  Supplementary Figs. S2b–S7b (one per session)
    └── 08_table_s2.py                  #   step 8  Supplementary Table S2
└── figure-five/                        # Figure 5: one run of the bracelet exergame
    ├── _common.py                      #   run, cue windows, recording reader
    ├── _metrics.py                     #   latency, accuracy, sample/event F1
    ├── _plot.py                        #   run plot, event-based confusion matrix
    ├── sessions.example.json           #   layout of the private data-file mapping
    ├── 01_exergame_run.py              #   step 1  run plot + confusion matrix
    └── 02_exergame_metrics.py          #   step 2  accuracy, latency, F1
```

In every figure folder the scripts are numbered (`01`, `02`, …) to make their
run order explicit; the modules starting with `_` hold the definitions they
share.

---

## Installation

Requires **Python 3.11 or 3.12**.

### With [uv](https://docs.astral.sh/uv/) (recommended)

```bash
uv sync
```

### With pip

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate
pip install "numpy>=1.24,<2" "scipy>=1.11,<2" "matplotlib>=3.9,<4" "pandas>=2.0,<4" \
            "scikit-learn>=1.3,<2" "catboost>=1.2,<2" "umap-learn>=0.5.12,<0.6"
```

`uv sync` installs the exact versions pinned in [`uv.lock`](uv.lock), with which
the published figures were produced; prefer it for a bit-exact reproduction.

---

## Reproducing the results

All analysis steps operate on the participant data, which is not distributed
with this repository but is available from the authors on reasonable request
(see the data note above). Every script also documents its exact inputs
and usage in its module docstring (top of the file), and prints an explanatory
message if a required input is missing.

| Script | Produces |
| --- | --- |
| `figure-two/01_tracking_rates.py` | **Figure 2** tracking panels: RMS, discharge rates and pulses per tracked MU (`results/tracking_rates/`) |
| `figure-two/02_heatmaps_muap.py` | **Figure 2** P2P heatmaps and mean MUAPs per tracked MU (`results/heatmaps/`) |
| `figure-three/01_xcorr_trigger.py` | xcorr of the MU discharge rates with the trigger (`data/<S>_*.csv`) |
| `figure-three/02_xcorr_labels.py` | xcorr of the MU discharge rates with the hold labels (`data/labels_*.csv`, `data/mu_corr_labels_*.csv`) |
| `figure-three/03_xcorr_overview.py` | **Figure 3** `Fig3_MU_xcorr_overview` (`results/`) |
| `figure-three/04_mu_overview.py` | **Figure 3** `Fig3_MU_overview`, panels a–c (`results/`) |
| `figure-four/01_auto_labels.py` | rule-based attempt labels (`data/labels/<S>.labels_auto_v2.json`) |
| `figure-four/02_dataset.py` | labelled datasets + RMS features (`data/<S>_rms_features.npz`) |
| `figure-four/03_loro_cv.py` | leave-one-repetition-out predictions + metrics (`results/loro_results.json`) |
| `figure-four/04_statistics.py` | permutation tests, AUROC intervals, Holm correction (`results/loro_stats.json`) |
| `figure-four/05_figure.py` | **Figure 4** (`results/figure4/`, composite + panel SVGs) |
| `figure-four/06_attempt_summary.py` | attempts per cue, cue coupling and timing of the presented cue (`results/attempt_summary*.csv`, `results/cue_timing.csv`) |
| `figure-four/07_supplementary_figures.py` | **Supplementary Figs. S2b–S7b** (`results/supplementary/SuppFig_<S>.*`) |
| `figure-four/08_table_s2.py` | **Supplementary Table S2** (`results/table_s2*.csv`, `results/table_s2.md`) |
| `figure-five/01_exergame_run.py` | **Figure 5** run plot and event-based confusion matrix (`results/`) |
| `figure-five/02_exergame_metrics.py` | **Figure 5** accuracy, reaction latency, sample/event F1 (`results/Fig5_exergame_*_values.csv`) |

Every figure folder reads the participant files from its own private
`sessions.local.json` (git-ignored); copy the folder's `sessions.example.json`
and set `data_root`. The example names the files each step imports, with their
pseudonymised names (`<S>` = `P01-S1`, `P01-S2`, `P02-S1`, …):
`<S>/<S>_raw_and_decomp.pkl` (recording day: EMG, trigger, labels,
decomposition per task), `<S>/<S>_tracking.mat` (concatenated DEMUSE
decomposition of all gestures of the day), `<S>/<S>_raw_and_decomp.labels_auto_v2.json`
(hold labels) and `exergame/P01-S1_exergame.pkl` (bracelet exergame run).
Everything the steps write goes to `data/` and `results/` inside the folder,
both git-ignored.

### Figure 2, Figure 3 and Figure 5

```bash
uv run python figure-two/01_tracking_rates.py
uv run python figure-two/02_heatmaps_muap.py

uv run python figure-three/01_xcorr_trigger.py
uv run python figure-three/02_xcorr_labels.py
uv run python figure-three/03_xcorr_overview.py   # needs no raw data
uv run python figure-three/04_mu_overview.py      # needs no raw data

uv run python figure-five/01_exergame_run.py
uv run python figure-five/02_exergame_metrics.py
```

The cue of Figure 2 and of the label-based Figure 3 analysis is the hold
labels (`quality == "good"`, gesture matching the task). The by-eye verdict on
cue modulation (Figure 3, panels b/c) is fixed in
[`figure-three/_common.py`](figure-three/_common.py), the cue windows of the
exergame run (transcribed from the protocol; the older myocontrol software
stored no cue track) in [`figure-five/_common.py`](figure-five/_common.py).
Figures are written as SVG with text as text.

### Figure 4, Supplementary Figures S2b–S7b and Table S2

The Figure 4 steps run in numeric order (`01` → `08`). Filters, RMS features and
CatBoost settings are defined in [`figure-four/_emg.py`](figure-four/_emg.py); the analysis decisions
(analysed recordings, excluded tasks, rest-recording blocks, the P02 cue
correction) are fixed in [`figure-four/_common.py`](figure-four/_common.py), and the
labelling rule with its parameters is in
[`figure-four/01_auto_labels.py`](figure-four/01_auto_labels.py). No segment is
labelled by hand. Because the real file names encode recording dates and participant
initials, the steps read them from a private `figure-four/sessions.local.json`
(git-ignored; layout in [`sessions.example.json`](figure-four/sessions.example.json)).
Everything the steps write goes to `figure-four/data/` and `figure-four/results/`,
both git-ignored.

CatBoost runs on the CPU with a fixed seed, the permutation tests and bootstrap
intervals use fixed seeds, and the UMAP embeddings of the supplementary figures a
fixed `random_state`; with the versions in `uv.lock` the whole chain is therefore
bit-for-bit reproducible. Step 3 takes about 45 min and runs its folds in
restartable worker processes. Steps 4, 6 and 8 need no raw data; steps 5 and 7
read the raw recordings again only for the EMG traces they show.

```bash
uv run python figure-four/01_auto_labels.py
uv run python figure-four/02_dataset.py
uv run python figure-four/03_loro_cv.py
uv run python figure-four/04_statistics.py
uv run python figure-four/05_figure.py
uv run python figure-four/06_attempt_summary.py
uv run python figure-four/07_supplementary_figures.py
uv run python figure-four/08_table_s2.py
```

(Use `python …` instead of `uv run python …` if you installed with pip.)

Figures are written as editable SVG (text as text, named groups) and as PDF/PNG.
The final typesetting of Figure 4 and of the supplementary figures (panel
headlines, positions, spacing) was done on these SVGs without changing their
content.

---

## How to cite

If you use this software, please cite both the article and
this software archive. Citation metadata is provided in
[`CITATION.cff`](./CITATION.cff).

The software archive is deposited on Zenodo and can be cited by its DOI:

> Wittermann, P., Braun, D. I., Weber, N. G. M., Wartke, D., Güneş, P., Wachter, F.,
> Corcosa, P., Tan, L., Grieshaber-Bouyer Mandelbaum, H., Walter, J., Franke, J.,
> Knieling, F., & Del Vecchio, A. (2026). *Analysis and figure code for "Hidden
> motor neuron activity restores grasp in children with paralysed hand"*.
> Zenodo. https://doi.org/10.5281/zenodo.21269465

DOI: [10.5281/zenodo.21269465](https://doi.org/10.5281/zenodo.21269465)

---

## License

Source code in this repository is released under the [MIT License](./LICENSE).
The license covers the code only — the raw human EMG data are not part of
this repository and are governed by a separate data-use agreement (see the data
note above).

---

## Authors and contact

Developed in the [n-squared lab](https://www.nsquared.tf.fau.de/), Department of
Artificial Intelligence in Biomedical Engineering, Friedrich-Alexander-Universität
Erlangen-Nürnberg (FAU).

- Code & analysis: **Pauline Wittermann** and **Dominik I. Braun**
- Data requests / correspondence: **Alessandro Del Vecchio** (corresponding author) — <alessandro.del.vecchio@fau.de>
- Principal investigator: **Alessandro Del Vecchio**
