# Project Structure

This project is intentionally not packaged for installation yet. The source is
organized in a standard `src/` layout, and the `scripts/` folder contains small
launchers that add `src/` to `sys.path`.

```text
ASWAXS_reduction_pipeline_v2_demo/
  README.md
  .gitignore
  docs/
    project_structure.md
  scripts/
    run_gui.py
    run_reducer.py
    run_viewer.py
  src/
    aswaxs_live/
      __init__.py
      core/
        __init__.py
        analysis_h5.py
        reduce_aswaxs_sequence.py
        reduce_sequence.py
        reduction_pipeline.py
      gui.py
      reducer.py
      stitcher.py
      viewer.py
  outputs/              ignored, local generated data
```

Main modules:

- `aswaxs_live.reducer`: manifest replay, folder watcher, HDF5 analysis writing,
  resume/restart behavior.
- `aswaxs_live.core`: copied reduction science code from the previous pipeline,
  kept inside this repository so v2 can run by itself.
- `aswaxs_live.viewer`: live curve viewer for single frames, group averages, and
  final ASAXS curves.
- `aswaxs_live.stitcher`: combines detector-named analysis records into one
  batch HDF5 and writes stitched averages.
- `aswaxs_live.gui`: three-window GUI launcher around the reducer and viewer.

For dual-detector runs, the user-facing batch file is
`outputs/<run>/<sample_name>_analysis.h5` with `/entry/Pil300K`,
`/entry/Eig1M`, and `/entry/stitched_averages` in the same HDF5 file.
