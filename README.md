# ASWAXS Pipeline V2 Live Demo

This folder is a demo project for the next pipeline design. It now includes the
core reduction scripts from the previous pipeline under `src/aswaxs_live/core/`,
so the repository can run without a separate checkout of the older project.

The demo uses a manifest replay as a stand-in for real acquisition:

1. A frame arrives.
2. The frame is immediately integrated to a 1D SAXS curve.
3. When all frames for one `(energy, group)` are present, the group average starts.
4. When all required groups for one energy are averaged, the per-energy ASAXS
   correction starts.

Use `--analysis-mode asaxs` for the current ASAXS workflow. Use
`--analysis-mode saxs` for normal SAXS mode, where the pipeline stops after 1D
reduction and group averages.

This lets us test the event logic with existing data before connecting it to a
real file watcher or Bluesky event stream.

## Project Layout

```text
ASWAXS_reduction_pipeline_v2_demo/
  docs/                  project notes
  scripts/               run launchers; no install step needed
  src/aswaxs_live/       reducer, GUI, viewer, and copied reduction core
  outputs/               ignored local analysis output
```

Run commands from `C:\Users\jiajtian\Documents\Playground`. The project is not
installable yet; the launcher scripts add `src/` to Python's path.

## Run a Small Real-Data Smoke Test

From `C:\Users\jiajtian\Documents\Playground`:

```powershell
python .\ASWAXS_reduction_pipeline_v2_demo\scripts\run_reducer.py `
  --manifest .\ASWAXS_reduction_pipeline\outputs\FC_AuSiO2NP_60uL_min_Eig1M_reduced_no_fluorescence\sequence_manifest.csv `
  --poni "Y:\aswaxs\bera\Apr2026\Commissioning\FC_AgBH_11_919keV\Eig1M\calib.poni" `
  --mask "Y:\aswaxs\bera\Apr2026\Commissioning\FC_AgBH_11_919keV\Eig1M\mask.msk" `
  --output-dir .\ASWAXS_reduction_pipeline_v2_demo\outputs\real_data_smoke `
  --sample-name FC_AuSiO2NP_ASWAXS_60uL_min `
  --gc-group 1 --air-group 2 --empty-group 3 --water-group 4 --sample-group 5 `
  --analysis-mode asaxs `
  --limit-energies 1 `
  --limit-frames-per-group 2
```

The output directory contains:

- `live_events.jsonl`: ordered stage-trigger log.
- `<sample_name>_analysis.h5`: analysis/provenance HDF5 written by the current pipeline helpers.
- `group_summary.csv`: group-average summary table.

The v2 default is HDF5-only for reduced curves. Legacy `.dat` curve files are
written only when `--write-text-output` is enabled.
This smoke command intentionally replays only 2 frames from each group. Remove
`--limit-energies` and `--limit-frames-per-group` when you want every collected
frame written into the live single-frame table.

Each single-frame 1D reduction is appended immediately to
`/entry/realtime/process_01_reduction/frames` inside the same analysis HDF5 file.
Group-average and ASAXS result groups are appended later as their trigger
conditions are met.

The live frame table includes a `qc_status` dataset:

- `pending_group_qc`: this frame has been reduced to 1D, but its full
  `(energy, group)` has not reached the group-average trigger yet.
- `accepted`: this frame was kept when the group average was calculated.
- `rejected_total_intensity`: this frame was dropped from the group average by
  the total-intensity outlier filter.

## Watch a Live Acquisition Folder

For real acquisition, run this script in a second terminal while Bluesky writes
raw HDF5 files into the sample folder. The watcher assigns files by arrival order:

```text
energy 1, group 1, frame 1
energy 1, group 1, frame 2
...
energy 1, group 2, frame 1
...
energy 2, group 1, frame 1
```

Example:

```powershell
python .\ASWAXS_reduction_pipeline_v2_demo\scripts\run_reducer.py `
  --watch-dir "\\chemmat-c51\data_rw\aswaxs\bera\Apr2026\Commissioning\FC_AuSiO2NP_ASWAXS_60uL_min\Eig1M" `
  --pattern "*.h5" `
  --num-energies 15 `
  --num-groups 5 `
  --num-frames 100 `
  --poni ".\FC_ASWAXS\FC_AgBH_11_919keV\Eig1M\calib.poni" `
  --mask ".\FC_ASWAXS\FC_AgBH_11_919keV\Eig1M\mask.msk" `
  --output-dir .\ASWAXS_reduction_pipeline_v2_demo\outputs\live_run `
  --sample-name FC_AuSiO2NP_ASWAXS_60uL_min `
  --analysis-mode asaxs `
  --gc-group 1 --air-group 2 --empty-group 3 --water-group 4 --sample-group 5
```

The watcher waits for file size to stop changing, opens the HDF5 file read-only,
checks that `entry/data/data` exists, then starts the reduction trigger chain.
The detector type is normally inferred from the acquisition file and folder
name. Use `--detector Eig1M` or `--detector Pil300K` only as a manual override
for unusual files where auto-detection is ambiguous.
When `--once` is used, the watcher performs a single pass over files already in
the folder and does not use poll or settle timing. In the GUI, checking
`Watcher once` disables `Poll seconds` and `Settle seconds`.

Existing-output behavior has two modes:

- `resume` is the default. If the reducer is stopped and started again with the
  same output directory/sample analysis HDF5, it reads the existing live frame
  table, skips already reduced source files, advances to the next sequence
  position, and rebuilds any unfinished group from the saved frame curves.
- `restart` starts from scratch in the same location. It removes the existing
  analysis HDF5 before writing new results and replaces `live_events.jsonl`
  from the first new log line. Use `--restart` from the command line, or choose
  `restart` in Window 0.

For normal SAXS-only reduction, use the same command shape with
`--analysis-mode saxs`. In that mode the role options such as `--gc-group` and
`--sample-group` are not required unless you want them recorded in metadata.

## Show Live 1D Curves

Run the live curve viewer in another terminal and point it at the same reducer
output directory:

```powershell
python .\ASWAXS_reduction_pipeline_v2_demo\scripts\run_viewer.py `
  --output-dir .\ASWAXS_reduction_pipeline_v2_demo\outputs\live_run
```

The viewer reads the batch analysis HDF5 file by default. It first looks for
`*_analysis.h5` in the output folder, then falls back to `analysis.h5` for old
test runs. It also keeps `.dat` folder support for compatibility.
The plot source menu has the three views needed during acquisition:

- `h5 single frames`: every individual frame after 1D reduction.
- `h5 group averages`: one averaged curve per `(energy, group)`.
- `h5 final`: final ASAXS-reduced curves.

For `h5 single frames`, use the energy and group selectors instead of browsing a
long flat list. The raw-frame plot modes are:

- `latest`: show the newest frame in the selected energy/group.
- `single frame`: use the frame slider to inspect one frame.
- `last N`: overlay the newest N frames in the group.
- `all in group`: overlay all frames for that group with a compact status legend.
- `average + frames`: show all raw frames lightly with the group average bold.
- `heatmap`: show frame order versus q with intensity as color.

For group-average and final-curve views, click to plot one curve, Ctrl-click to
add or remove curves, and Shift-click to select a range. Large raw-frame overlays
use compact status legends instead of one legend entry per frame.

## Run the V2 GUI

The integrated GUI has three windows:

- Window 0: parameter setup and reducer launch.
- Window 1: acquisition/reduction process monitor.
- Window 2: available 1D file list and selected-curve plot.

Start it with:

```powershell
python .\ASWAXS_reduction_pipeline_v2_demo\scripts\run_gui.py
```

The GUI can run either folder-watcher mode for live acquisition or manifest
replay mode for already collected data.
Window 0 remembers the last-used parameters in
`.aswaxs_live_gui_settings.json` at the project root. That local settings file
is ignored by git.

For simultaneous detector acquisition, set `Detector jobs` to `Pil300K + Eig1M`.
The GUI launches two reducer processes in parallel:

- Pil300K watches the Pil300K folder and writes its live working files to
  `output_dir/Pil300K`.
- Eig1M watches the Eig1M folder and writes its live working files to
  `output_dir/Eig1M`.

The GUI coordinator keeps one public batch analysis HDF5 named
`<sample_name>_analysis.h5` unless the `Analysis HDF5` field is set explicitly.
That combined file is organized as:

```text
/entry
  /Pil300K              # copied Pil300K analysis record
  /Eig1M                # copied Eig1M analysis record
  /stitched_averages    # stitched detector group averages
```

The two reducer processes keep their own detector working HDF5 files so they do
not write to the same HDF5 at the same time. The GUI is the only writer for the
combined HDF5: it refreshes `/entry/Pil300K`, `/entry/Eig1M`, and
`/entry/stitched_averages` as new matching group averages appear.

## Current Demo Boundary

This demo proves live orchestration with either manifest replay or a polling
folder watcher. A future production version can replace the polling folder
watcher with a Bluesky/Kafka event consumer while keeping the same stage-trigger
logic.
